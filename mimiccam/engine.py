"""Frame engines.

An engine takes the newest camera frame and produces the newest output frame.
Two implementations exist:

* :class:`PassthroughEngine` - no AI, sends the camera straight through. It is
  how users validate their camera + virtual camera before spending any credits.
* :class:`mimiccam.decart_engine.DecartEngine` - publishes frames to Decart's
  Lucy 2.5 realtime session and returns the transformed stream.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class Engine:
    name = "engine"

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def push(self, frame_rgb: np.ndarray, ts: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        """Newest output frame and when this engine produced it.

        The timestamp must be a ``time.perf_counter()`` reading taken when the
        frame became available, because the pipeline compares it against its own
        ``perf_counter`` to decide whether the model has gone quiet and the last
        frame needs holding.
        """
        raise NotImplementedError


class PassthroughEngine(Engine):
    name = "passthrough"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self.frames_in = 0
        self.frames_out = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def push(self, frame_rgb: np.ndarray, ts: float) -> None:
        with self._lock:
            self._frame = frame_rgb
            # A passthrough frame is available the moment it arrives.
            self._ts = time.perf_counter()
        self.frames_in += 1

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            frame = self._frame
        if frame is not None:
            self.frames_out += 1
        return frame, self._ts

    def latency_ms(self) -> float:
        return 0.0
