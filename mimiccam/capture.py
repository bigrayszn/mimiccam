# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Webcam capture with glitch recovery.

One thread owns the camera. Frames are converted to RGB once and published as
"latest frame wins", so a slow consumer can never build a queue and add latency.
If the device stalls (common when another app grabs it, or a USB camera sleeps)
the reader reopens it automatically instead of freezing the call.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import sys
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
# Devices are opened in parallel: opening one costs 0.4-1.2 s of driver time and
# nothing else in the scan is shared, so three at a time turns a seven device
# scan from eight seconds into three. Three, not one per device: UVC cameras on
# the same USB controller share bandwidth, and a queue of opens can make a real
# camera look dead when it is only slow.
PROBE_WORKERS = 3
# A scan opens every camera, which takes seconds and briefly takes the devices
# away from whatever else is using them. Reopening Settings, or coming back to
# the window, has no reason to pay that again: the answer is kept for a minute
# and thrown away the moment the device list itself changes.
SCAN_CACHE_S = 60.0


@dataclass
class CameraEntry:
    index: int
    width: int
    height: int
    mean_luma: float
    name: str = ""

    @property
    def label(self) -> str:
        """The device's own name, exactly as Windows reports it.

        A camera is called "OBS Virtual Camera" or "Logitech BRIO" by everyone
        else on the machine; inventing "Camera 1 - 1280x720" made the picker a
        guessing game. The index is only a fallback for the rare device whose
        friendly name cannot be read.
        """
        return self.name or f"Camera {self.index}"


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

    The budget stays per device even though the scan now opens devices in
    parallel: a sensor that opens black and ramps is exactly what the first
    core of frames looks like, and cutting the wait short on "nothing is
    changing yet" mislabels working cameras.
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


# -- device names ----------------------------------------------------------
# OpenCV can enumerate capture sources but cannot name them, while every other
# app on the machine calls the camera what Windows calls it. Those names live in
# DirectShow, behind COM, in the same order OpenCV's DSHOW backend numbers the
# devices - so the names are read here and matched to the probe by index.
_CLSCTX_INPROC_SERVER = 0x1
_COINIT_APARTMENTTHREADED = 0x2
_CLSID_SYSTEM_DEVICE_ENUM = "62be5d10-60eb-11d0-bd3b-00a0c911ce86"
_IID_CREATE_DEV_ENUM = "29840822-5b84-11d0-bd3b-00a0c911ce86"
_CLSID_VIDEO_INPUT_CATEGORY = "860bb310-5d01-11d0-bd3b-00a0c911ce86"
_IID_PROPERTY_BAG = "55272a00-42cb-11ce-8135-00aa004bb851"
_VT_BSTR = 8


def _guid(text: str) -> bytes:
    """A GUID string as the 16 bytes COM expects in memory."""
    fields = text.replace("{", "").replace("}", "").split("-")
    raw = b"".join(bytes.fromhex(part) for part in fields)
    return raw[3::-1] + raw[5:3:-1] + raw[7:5:-1] + raw[8:]


def _method(pointer, slot: int, restype, argtypes):
    """Call vtable entry ``slot`` of a COM interface pointer."""
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, *argtypes)(table[slot])


def _release(pointer) -> None:
    if pointer:
        _method(pointer, 2, ctypes.c_ulong, [ctypes.c_void_p])(pointer)


def dshow_device_names() -> list[str]:
    """Capture device names in DirectShow order, empty when it cannot be read.

    Never raises: a names lookup that fails just leaves the picker with the
    index fallback instead of taking the camera scan down with it.
    """
    if sys.platform != "win32":
        return []
    try:
        return _dshow_names()
    except Exception:  # noqa: BLE001 - a cosmetic lookup must not break capture
        return []


def _dshow_names() -> list[str]:
    ole32 = ctypes.windll.ole32
    oleaut32 = ctypes.windll.oleaut32
    hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    # S_OK / S_FALSE mean this call owns the apartment; RPC_E_CHANGED_MODE means
    # the thread already has one and it must be left alone.
    owns_apartment = hr in (0, 1)
    try:
        source = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(ctypes.create_string_buffer(_guid(_CLSID_SYSTEM_DEVICE_ENUM), 16)),
            None,
            _CLSCTX_INPROC_SERVER,
            ctypes.byref(ctypes.create_string_buffer(_guid(_IID_CREATE_DEV_ENUM), 16)),
            ctypes.byref(source),
        )
        if hr != 0 or not source:
            return []
        try:
            enumerator = ctypes.c_void_p()
            hr = _method(source, 3, ctypes.c_long, [ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_void_p),
                                                    ctypes.c_ulong])(
                source,
                ctypes.create_string_buffer(_guid(_CLSID_VIDEO_INPUT_CATEGORY), 16),
                ctypes.byref(enumerator),
                0,
            )
            # S_FALSE is "category empty", which is not an error to report.
            if hr not in (0, 1) or not enumerator:
                return []
            names: list[str] = []
            try:
                fetched = ctypes.c_ulong()
                while True:
                    moniker = ctypes.c_void_p()
                    fetched.value = 0
                    hr = _method(enumerator, 3, ctypes.c_long,
                                 [ctypes.c_void_p, ctypes.c_ulong,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(ctypes.c_ulong)])(
                        enumerator, 1, ctypes.byref(moniker), ctypes.byref(fetched)
                    )
                    if hr != 0 or fetched.value != 1 or not moniker:
                        break
                    try:
                        names.append(_friendly_name(moniker, oleaut32))
                    finally:
                        _release(moniker)
            finally:
                _release(enumerator)
            return names
        finally:
            _release(source)
    finally:
        if owns_apartment:
            ole32.CoUninitialize()


def _friendly_name(moniker, oleaut32) -> str:
    """The device's own name, or "" when this moniker does not expose one."""
    bag = ctypes.c_void_p()
    # IMoniker is an IPersistStream, so BindToStorage sits at slot 9 behind
    # GetClassID, IsDirty, Load, Save and GetSizeMax.
    hr = _method(moniker, 9, ctypes.c_long,
                 [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)])(
        moniker,
        None,
        None,
        ctypes.create_string_buffer(_guid(_IID_PROPERTY_BAG), 16),
        ctypes.byref(bag),
    )
    if hr != 0 or not bag:
        return ""
    try:
        # VARIANT is 24 bytes on x64: the type at offset 0, the value at 8.
        variant = (ctypes.c_byte * 24)()
        hr = _method(bag, 3, ctypes.c_long,
                     [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                      ctypes.c_void_p])(bag, "FriendlyName", variant, None)
        if hr != 0:
            return ""
        try:
            if ctypes.c_ushort.from_buffer(variant).value != _VT_BSTR:
                return ""
            pointer = ctypes.c_void_p.from_buffer(variant, 8).value
            return ctypes.wstring_at(pointer).strip() if pointer else ""
        finally:
            oleaut32.VariantClear(ctypes.byref(variant))
    finally:
        _release(bag)


def order_entries(entries: list[CameraEntry]) -> list[CameraEntry]:
    """Lit cameras first: the IR/covered device should never be the default."""
    return sorted(entries, key=lambda entry: entry.mean_luma < PROBE_DARK_LUMA)


def probe_cameras(max_index: int = 6) -> list[CameraEntry]:
    """Try every capture index the machine has, and report which ones work.

    ``max_index`` is only the fallback bound for a machine whose device list
    cannot be read: with DirectShow's list in hand the scan covers exactly the
    cameras that exist. A fixed ceiling of six hid every device after it - on
    this laptop that was "OBS Virtual Camera", the one device the app's own
    instructions tell people to select.
    """
    names = dshow_device_names()
    # The list is the whole truth when it can be read: indices past its end are
    # not video inputs, and probing them only spends a probe budget each to
    # learn nothing. The old bound is for machines where the lookup fails.
    count = len(names) if names else max(0, int(max_index))
    if count <= 0:
        return []
    if count == 1:
        entries = [_probe_index(0, names)]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(PROBE_WORKERS, count), thread_name_prefix="camera-probe"
        ) as pool:
            entries = list(pool.map(lambda index: _probe_index(index, names), range(count)))
    return order_entries([entry for entry in entries if entry is not None])


_scan_cache: Optional[tuple[float, tuple[str, ...], list["CameraEntry"]]] = None
_scan_lock = threading.Lock()


def clear_scan_cache() -> None:
    """Forget the last scan (tests, and a user who unplugged a camera)."""
    global _scan_cache
    with _scan_lock:
        _scan_cache = None


def cached_cameras(max_age_s: float = SCAN_CACHE_S) -> Optional[list["CameraEntry"]]:
    """The last scan, when it is still fresh and no device has come or gone."""
    names = tuple(dshow_device_names())
    with _scan_lock:
        cached = _scan_cache
    if cached is None:
        return None
    stamp, signature, entries = cached
    if names != signature or (time.perf_counter() - stamp) > max_age_s:
        return None
    return list(entries)


def scan_cameras(force: bool = False, max_age_s: float = SCAN_CACHE_S) -> list["CameraEntry"]:
    """Devices on this machine, from the cache when it is still good.

    ``force`` is what the Refresh button uses: it is the one place a user asks
    for the machine to be looked at again.
    """
    global _scan_cache
    if not force:
        cached = cached_cameras(max_age_s)
        if cached is not None:
            return cached
    signature = tuple(dshow_device_names())
    entries = probe_cameras()
    with _scan_lock:
        _scan_cache = (time.perf_counter(), signature, list(entries))
    return entries


def _probe_index(index: int, names: list[str]) -> Optional[CameraEntry]:
    """One device: open it, read until it shows something, close it again."""
    try:
        cap = _open_capture(index)
    except Exception:  # noqa: BLE001 - one bad index must not hide the rest
        return None
    try:
        if not cap.isOpened():
            return None
        frame = read_best_frame(cap)
        if frame is None:
            return None
        height, width = frame.shape[:2]
        return CameraEntry(
            index=index,
            width=int(width),
            height=int(height),
            mean_luma=float(frame.mean()),
            name=names[index] if index < len(names) else "",
        )
    finally:
        cap.release()


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
