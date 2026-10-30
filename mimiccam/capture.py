"""Webcam capture with glitch recovery.

One thread owns the camera. Frames are converted to RGB once and published as
"latest frame wins", so a slow consumer can never build a queue and add latency.
If the device stalls (common when another app grabs it, or a USB camera sleeps)
the reader reopens it automatically instead of freezing the call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np

MAX_CONSECUTIVE_FAILURES = 30
REOPEN_DELAY_S = 1.0

# A webcam's first frames after opening are almost always black: the sensor is
# still ramping its auto-exposure. Judging a device on frame #1 mislabels a
# perfectly good camera as "black - IR camera?", which is exactly how a working
# camera ends up looking broken in the picker.
PROBE_ATTEMPTS = 24
PROBE_BUDGET_S = 1.2
PROBE_LIT_STD = 12.0
PROBE_DARK_LUMA = 8.0


@dataclass
class CameraEntry:
    index: int
    width: int
    height: int
    mean_luma: float

    @property
    def label(self) -> str:
        note = " (no light - IR camera or covered lens)" if self.mean_luma < PROBE_DARK_LUMA else ""
        return f"Camera {self.index} - {self.width}x{self.height}{note}"


class _ClosedCapture:
    """A device that could not be opened, shaped like a closed VideoCapture."""

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2's API
        return False

    def release(self) -> None:
        return None

    def read(self) -> tuple[bool, None]:
        return False, None

    def get(self, _prop: int) -> float:
        return 0.0


def _open_capture(index: int) -> Any:
    """Open a DirectShow capture, falling back to the default backend.

    The device is asked for nothing it can pick for itself: no width, no height,
    no frame rate. A webcam opened with its own defaults gives the sharpest
    picture it can and avoids the driver picking a mode it does not really
    support; the pipeline normalises frames to the resolution the virtual camera
    publishes, so nothing downstream needs the camera to be told a number.

    DirectShow raises C++ exceptions ("raised unknown C++ exception") for
    transient device states - a camera another app just let go of, or one still
    re-enumerating after a reopen. That must never escape: a caller can only do
    something sensible with a closed capture, and one flaky index must not abort
    a scan of the others.
    """
    backend = getattr(cv2, "CAP_DSHOW", 700)
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return cap
        # MJPG keeps 720p30 feasible on USB 2.0 webcams; ignored when unsupported.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    except Exception:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        return _ClosedCapture()


def read_best_frame(cap, attempts: int = PROBE_ATTEMPTS,
                    budget_s: float = PROBE_BUDGET_S):
    """Read until the sensor has exposed something, and keep the clearest frame.

    Returns ``None`` only when the device never produced a frame at all, which is
    the one case that really means "no usable camera here".
    """
    best = None
    best_std = -1.0
    deadline = time.perf_counter() + budget_s
    for _ in range(max(1, attempts)):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            deviation = float(frame.std())
            if deviation > best_std:
                best, best_std = frame, deviation
            if best_std >= PROBE_LIT_STD:
                break
        if time.perf_counter() >= deadline:
            break
        time.sleep(0.02)
    return best


def order_entries(entries: list[CameraEntry]) -> list[CameraEntry]:
    """Lit cameras first: the IR/covered device should never be the default."""
    return sorted(entries, key=lambda entry: entry.mean_luma < PROBE_DARK_LUMA)


def probe_cameras(max_index: int = 6) -> list[CameraEntry]:
    """Try each capture index and report which ones actually deliver frames."""
    found: list[CameraEntry] = []
    for index in range(max_index):
        try:
            cap = _open_capture(index)
        except Exception:  # noqa: BLE001 - one bad index must not hide the rest
            continue
        try:
            if not cap.isOpened():
                continue
            frame = read_best_frame(cap)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            found.append(
                CameraEntry(
                    index=index,
                    width=int(width),
                    height=int(height),
                    mean_luma=float(frame.mean()),
                )
            )
        finally:
            cap.release()
    return order_entries(found)


class CameraCapture:
    """Threaded reader that always exposes the freshest RGB frame."""

    def __init__(
        self,
        index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.index = index
        # The size the pipeline publishes at, not a request made of the camera:
        # the device is opened with its own defaults and frames are normalised
        # downstream. Kept here because callers pass the configured values.
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self._on_log = on_log or (lambda _msg: None)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self.frames = 0
        self.measured_fps = 0.0
        self.reopens = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        self._stop.clear()
        try:
            self._cap = _open_capture(self.index)
        except Exception as exc:  # noqa: BLE001 - a driver failure is not a crash
            self._on_log(f"Could not open camera {self.index}: {exc}")
            self._cap = None
            return False
        if self._cap is None or not self._cap.isOpened():
            self._on_log(f"Could not open camera {self.index}")
            self._cap = None
            return False
        actual = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH), self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._on_log(f"Camera {self.index} opened at {int(actual[0])}x{int(actual[1])}")
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # -- reader loop -------------------------------------------------------
    def _loop(self) -> None:
        failures = 0
        window_start = time.perf_counter()
        window_frames = 0

        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(REOPEN_DELAY_S)
                continue

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    self.reopens += 1
                    self._on_log("Camera stalled - reopening device")
                    self._release()
                    time.sleep(REOPEN_DELAY_S)
                    if self._stop.is_set():
                        return
                    self._cap = _open_capture(self.index)
                    failures = 0
                else:
                    time.sleep(0.005)
                continue

            failures = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mirror:
                rgb = cv2.flip(rgb, 1)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)

            with self._lock:
                self._frame = rgb
                self._frame_ts = time.perf_counter()
            self.frames += 1
            window_frames += 1

            now = time.perf_counter()
            if now - window_start >= 1.0:
                self.measured_fps = window_frames / (now - window_start)
                window_start = now
                window_frames = 0

    # -- consumer API ------------------------------------------------------
    def read(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame, self._frame_ts

    def stale_for(self) -> float:
        with self._lock:
            if self._frame_ts == 0.0:
                return float("inf")
            return time.perf_counter() - self._frame_ts
