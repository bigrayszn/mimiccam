# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Virtual-camera output.

``pyvirtualcam`` writes frames into the shared-memory buffer that a DirectShow
virtual-camera driver exposes to Zoom, Teams, OBS, Telegram and friends. Two
backends are supported on Windows:

* ``obs``          - the OBS Virtual Camera filter, installed by OBS Studio.
                     OBS itself does **not** need to be running.
* ``unitycapture`` - Unity Capture's MIT-licensed filter, the backend we ship
                     for machines without OBS.

Frames travel by shared memory, so there is no encode step and no added latency.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    import pyvirtualcam

    _IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - native DLL loads fail as ImportError
    # Windows refuses to load a .pyd when the full path approaches MAX_PATH.
    # Freezing the app deep inside a user folder is an easy way to hit that, so
    # importing lazily lets the app start and explain the fix instead of dying
    # with a traceback and a PyInstaller error dialog.
    pyvirtualcam = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)

BACKENDS = ("obs", "unitycapture")
BACKEND_LABELS = {
    "obs": "OBS Virtual Camera",
    "unitycapture": "Unity Capture",
}


class VirtualCameraError(RuntimeError):
    pass


def import_error() -> str:
    """Non-empty when the virtual-camera driver could not be loaded at all."""
    return _IMPORT_ERROR


def _require_pyvirtualcam():
    if pyvirtualcam is None:
        hint = ""
        if "too long" in _IMPORT_ERROR.lower():
            hint = (
                "\n\nWindows could not load the driver because MimicCam's folder path "
                "is too deep. Move the MimicCam folder somewhere short such as "
                "C:\\MimicCam and start it again."
            )
        raise VirtualCameraError(f"Virtual-camera driver failed to load: {_IMPORT_ERROR}{hint}")
    return pyvirtualcam


def available_backends() -> list[str]:
    """Return the backends that can currently be opened on this machine."""
    pvc = _require_pyvirtualcam()
    usable: list[str] = []
    for backend in BACKENDS:
        cam = None
        try:
            cam = pvc.Camera(
                width=640, height=360, fps=5, fmt=pvc.PixelFormat.RGB,
                backend=backend, print_fps=False,
            )
            usable.append(backend)
        except Exception:
            pass
        finally:
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
    return usable


class VirtualCamera:
    """A resilient wrapper: reopens itself if the backend drops a frame."""

    def __init__(self, on_log=None) -> None:
        self._cam: Optional[pyvirtualcam.Camera] = None
        self._log = on_log or (lambda _m: None)
        self.backend: str = ""
        self.device: str = ""
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frames_sent = 0
        self.reopens = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self, width: int, height: int, fps: int, backend: str = "") -> bool:
        pvc = _require_pyvirtualcam()
        candidates = [backend] if backend else list(BACKENDS)
        for candidate in candidates:
            if candidate not in BACKENDS:
                continue
            try:
                cam = pvc.Camera(
                    width=width,
                    height=height,
                    fps=fps,
                    fmt=pvc.PixelFormat.RGB,
                    backend=candidate,
                    print_fps=False,
                )
            except Exception as exc:  # noqa: BLE001 - backend probing is best effort
                self._log(f"{BACKEND_LABELS.get(candidate, candidate)} unavailable: {exc}")
                continue
            self._cam = cam
            self.backend = candidate
            self.device = getattr(cam, "device", "") or BACKEND_LABELS.get(candidate, candidate)
            self.width, self.height, self.fps = width, height, fps
            self._log(f"Virtual camera ready: {self.device} ({width}x{height}@{fps})")
            return True

        raise VirtualCameraError(
            "No virtual camera backend found. Install OBS Studio (which provides "
            "the OBS Virtual Camera) or the bundled Unity Capture driver."
        )

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    @property
    def is_open(self) -> bool:
        return self._cam is not None

    # -- frame in ----------------------------------------------------------
    def send(self, frame_rgb: np.ndarray) -> bool:
        cam = self._cam
        if cam is None:
            return False
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("virtual camera expects an HxWx3 uint8 RGB frame")
        try:
            cam.send(frame_rgb)
            self.frames_sent += 1
            return True
        except Exception as exc:  # noqa: BLE001 - reopen on any backend failure
            self._log(f"Virtual camera send failed ({exc}); reopening")
            self.close()
            time.sleep(0.5)
            try:
                self.open(self.width or 1280, self.height or 720, self.fps or 30, self.backend)
                self.reopens += 1
            except Exception as reopen_exc:  # noqa: BLE001
                self._log(f"Virtual camera reopen failed: {reopen_exc}")
            return False

    def sleep_until_next_frame(self) -> None:
        if self._cam is not None:
            try:
                self._cam.sleep_until_next_frame()
                return
            except Exception:
                pass
        time.sleep(1.0 / max(1, self.fps))
