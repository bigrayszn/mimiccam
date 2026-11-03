# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The runtime pipeline: camera -> engine -> virtual camera.

Three daemon threads, each with one job:

``capture``  reads the webcam (inside :class:`CameraCapture`) and hands the
             newest frame to the engine.
``output``   pulls the newest engine frame and writes it to the virtual camera
             on a fixed cadence, so callers see a steady 30 fps even when the
             model delivers 24, 25 or an occasional 40 frames per second.
``stats``    publishes counters once per second for the UI.

The output thread holds the last good frame when the engine goes quiet. That is
the difference between a one-second network hiccup and a camera that goes black
mid-call.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from .capture import CameraCapture
from .engine import Engine, PassthroughEngine
from .vcam import VirtualCamera

PREVIEW_INTERVAL_S = 1 / 15
STALE_HOLD_S = 0.35


class Pipeline:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        mirror: bool,
        on_log: Callable[[str], None],
        on_preview: Callable[[np.ndarray], None],
        on_stats: Callable[[dict], None],
        vcam_backend: str = "",
        on_input: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._log = on_log
        self._on_preview = on_preview
        self._on_input = on_input
        self._on_stats = on_stats
        self.vcam_backend = vcam_backend

        self.capture = CameraCapture(camera_index, width, height, fps, mirror, on_log=on_log)
        self.vcam = VirtualCamera(on_log=on_log)
        self.engine: Engine = PassthroughEngine()

        self._engine_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_preview = 0.0
        self._last_input_preview = 0.0
        self._last_frame_ts = 0.0
        self._last_luma = 0.0
        self._last_luma_at = 0.0
        self._last_grid: list[float] = []
        self._out_fps = 0.0
        self.started_at = 0.0
        self.held_frames = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self, engine: Optional[Engine] = None) -> bool:
        if not self.capture.start():
            return False
        try:
            if not self.vcam.is_open:
                self.vcam.open(self.width, self.height, self.fps, self.vcam_backend)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Virtual camera error: {exc}")
            self.capture.stop()
            return False

        self.set_engine(engine or PassthroughEngine())
        self._stop.clear()
        self.started_at = time.time()

        self._threads = [
            threading.Thread(target=self._input_loop, name="mimic-input", daemon=True),
            threading.Thread(target=self._output_loop, name="mimic-output", daemon=True),
            threading.Thread(target=self._stats_loop, name="mimic-stats", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._log(f"Pipeline running: {self.vcam.device} -> {self.engine.name}")
        return True

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        self.capture.stop()
        try:
            self.engine.stop()
        except Exception:  # noqa: BLE001
            pass
        self.vcam.close()
        self._on_preview_blank()

    def _on_preview_blank(self) -> None:
        # A single black frame is friendlier than a frozen face in the preview.
        try:
            self._on_preview(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        except Exception:
            pass

    def set_engine(self, engine: Engine) -> None:
        with self._engine_lock:
            old = self.engine
            self.engine = engine
            self._last_frame_ts = 0.0
        try:
            old.stop()
        except Exception:  # noqa: BLE001
            pass
        engine.start()

    # -- worker: camera -> engine -----------------------------------------
    def _input_loop(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            frame, ts = self.capture.read()
            if frame is not None and ts != last_ts:
                last_ts = ts
                # Your own camera, before any model touches it. The UI shows it
                # on request so framing and lighting can be checked while the
                # engine is mid-swap - the output alone cannot tell you whether
                # the camera or the model is the reason something looks off.
                if self._on_input is not None:
                    now = time.perf_counter()
                    if now - self._last_input_preview >= PREVIEW_INTERVAL_S:
                        self._last_input_preview = now
                        try:
                            self._on_input(frame)
                        except Exception:  # noqa: BLE001
                            pass
                with self._engine_lock:
                    engine = self.engine
                try:
                    engine.push(frame, ts)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"Engine rejected frame: {exc}")
            else:
                time.sleep(0.002)

    # -- worker: engine -> virtual camera ---------------------------------
    def _output_loop(self) -> None:
        interval = 1.0 / max(1, self.fps)
        next_frame = time.perf_counter()
        frames = 0
        window_start = next_frame
        last_sent: Optional[np.ndarray] = None

        while not self._stop.is_set():
            with self._engine_lock:
                engine = self.engine
            frame, ts = engine.latest()

            if frame is None:
                frame = last_sent
            if frame is not None:
                # Some webcams ignore the requested size (this one delivers 1080p
                # whatever you ask for). The virtual camera is fixed at the
                # configured resolution, so normalise here, once.
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                now = time.perf_counter()
                if ts and ts != self._last_frame_ts:
                    self._last_frame_ts = ts
                elif now - ts > STALE_HOLD_S and last_sent is not None:
                    # Engine is quiet; repeat the last good frame rather than
                    # letting viewers see a frozen or black feed.
                    self.held_frames += 1
                last_sent = frame
                self.vcam.send(frame)

                if now - self._last_preview >= PREVIEW_INTERVAL_S:
                    self._last_preview = now
                    try:
                        self._on_preview(frame)
                    except Exception:
                        pass
                if now - self._last_luma_at >= 1.0:
                    # Content signature used by tests and the status bar. A coarse
                    # 8x8 luma grid survives exposure changes (this laptop's
                    # webcam auto-exposure swings widely) while still catching a
                    # black feed, a stalled buffer or a channel/geometry mix-up.
                    self._last_luma_at = now
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    self._last_luma = float(gray.mean())
                    grid = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
                    self._last_grid = [round(float(v), 1) for v in grid.ravel()]

            frames += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                self._out_fps = frames / (now - window_start)
                frames = 0
                window_start = now

            next_frame += interval
            sleep_for = next_frame - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind (e.g. compositor stall) - resync instead of
                # sprinting to catch up.
                next_frame = time.perf_counter()

    # -- worker: stats -----------------------------------------------------
    def _stats_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(1.0)
            with self._engine_lock:
                engine = self.engine
            latency = 0.0
            getter = getattr(engine, "latency_ms", None)
            if callable(getter):
                latency = float(getter())
            self._on_stats(
                {
                    "engine": engine.name,
                    "camera_fps": round(self.capture.measured_fps, 1),
                    "output_fps": round(self._out_fps, 1),
                    "frames_sent": self.vcam.frames_sent,
                    "latency_ms": round(latency),
                    "held_frames": self.held_frames,
                    "reopens": self.capture.reopens + self.vcam.reopens,
                    "luma": round(self._last_luma, 1),
                    "grid": list(self._last_grid),
                }
            )
