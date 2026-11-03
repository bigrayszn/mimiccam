# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Lucy 2.5 engine.

Pipeline inside this module:

    numpy RGB frame
        -> livekit VideoSource.capture_frame (30 fps, 1280x720)
        -> Decart realtime session over WebRTC
        -> RemoteVideoTrack, decoded as RGB24
        -> numpy RGB frame handed back to the pipeline

The heavy generative work happens in Decart's datacentre; the local NVIDIA GPU
handles capture, scaling, colour conversion and the virtual-camera feed. The
session reconnects with exponential backoff so a dropped connection never
requires restarting the app mid-call.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from decart import (
    DecartClient,
    ModelState,
    Prompt,
    RealtimeClient,
    RealtimeConnectOptions,
    SetInput,
    models,
)
from livekit import rtc

from . import __version__, credits, prompting, reference, tryon
from .engine import Engine

WATCHDOG_TIMEOUT_S = 12.0

# Decart bills realtime per second of *active generation*, so the meter runs the
# whole time a session is connected - not per frame, not per resolution. Standard
# mode is $0.02/s; speed="fast" is billed at 2x for lower latency, not better
# quality. https://docs.platform.decart.ai/getting-started/pricing
USD_PER_SECOND = credits.USD_PER_SECOND
FAST_USD_PER_SECOND = credits.FAST_USD_PER_SECOND
usd_per_second = credits.usd_per_second

# Billing can only be stopped by ending the session: the API has no pause. So the
# engine parks itself when the camera has shown nothing for a while, and switches
# the virtual camera to the raw feed until someone moves again.
IDLE_PAUSE_S = 45.0
# A session shorter than this is not worth interrupting: every reconnect answers
# with its own first billed seconds, so pausing too eagerly saves nothing.
MIN_SESSION_BEFORE_PAUSE_S = 90.0
# Motion is detected on a 48x27 grey thumbnail. Webcam noise and auto-exposure
# settle well under this; a hand moving across the frame is far above it.
MOTION_THUMBNAIL = (48, 27)
MOTION_THRESHOLD = 1.5
MOTION_SAMPLE_INTERVAL_S = 0.1


def load_reference_image(path: str) -> Optional[bytes]:
    """Normalise a character image to JPEG bytes Lucy can use as a reference.

    The geometry lives in :mod:`mimiccam.reference`: pad to the output aspect
    rather than crop, at least 512 px on the short side and at most 1280 px on
    the long one, per Decart's reference-image guidance.
    """
    if not path:
        return None
    return reference.prepare_reference(path)


class _IdlePause(RuntimeError):
    """Raised inside a session to end it because nothing is moving."""


class DecartEngine(Engine):
    name = "Lucy 2.5"

    def __init__(
        self,
        api_key: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        prompt: str = "",
        preset_key: str = "character",
        reference_image: Optional[str] = None,
        character_notes: str = "",
        garment_image: Optional[str] = None,
        model: str = "lucy-2.5",
        fast_mode: bool = False,
        pause_when_idle: bool = True,
        idle_after_s: float = IDLE_PAUSE_S,
        on_log=None,
    ) -> None:
        self.api_key = api_key
        self.width = width
        self.height = height
        self.fps = fps
        self.preset_key = preset_key
        self.character_notes = (character_notes or "").strip()
        # Try-on mode: the reference image is a garment, not a face.
        self.garment_image = garment_image or ""
        self.try_on = bool(self.garment_image)
        self.appearance: Optional[reference.Appearance] = None
        self.prompt = prompt
        self.model_name = model
        self.fast_mode = fast_mode
        self.pause_when_idle = pause_when_idle
        self.idle_after_s = max(10.0, float(idle_after_s))
        self._log = on_log or (lambda _m: None)

        # Billing: seconds come from the server's own generation ticks, so the
        # meter shows what Decart will invoice rather than what we hope it is.
        self._billed_before = 0.0
        self._session_billed = 0.0
        self.parked = False
        self.pauses = 0
        self._last_motion_ts = time.perf_counter()
        self._last_thumb: Optional[np.ndarray] = None
        self._last_thumb_at = 0.0
        self._session_opened_at = 0.0

        self._in_lock = threading.Lock()
        self._in_frame: Optional[np.ndarray] = None
        self._in_ts = 0.0

        self._out_lock = threading.Lock()
        self._out_frame: Optional[np.ndarray] = None
        self._out_ts = 0.0

        self._pub_times: collections.deque[float] = collections.deque(maxlen=120)
        self._latency_ms = 0.0
        self._last_output_at = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[RealtimeClient] = None
        self._source: Optional[rtc.VideoSource] = None
        self._consumer_task: Optional[asyncio.Task] = None

        if self.try_on:
            # A garment is not a face: no appearance analysis and no character
            # template, because the reference image *is* the outfit.
            self.name = "Lucy try-on"
            self._reference_bytes = tryon.prepare_garment(self.garment_image)
            self.prompt = tryon.as_prompt(self.prompt)
            self._log(f"Try-on garment: {self.garment_image}")
            self._log(f"Try-on prompt: {self.prompt}")
        else:
            self._reference_bytes = load_reference_image(reference_image) if reference_image else None
            if reference_image:
                try:
                    self.appearance = reference.analyse(reference_image)
                except Exception as exc:  # noqa: BLE001 - a photo we cannot read is not fatal
                    self._log(f"Could not read the character photo: {exc}")
            if self.preset_key == "character":
                # The whole-person wording plus the character's own description: an
                # image alone is under-used, and the model otherwise fills the body
                # with whatever the camera is already wearing.
                self.prompt = prompting.character_prompt(
                    self.prompt,
                    appearance=self.appearance,
                    notes=self.character_notes,
                )
            if self.appearance is not None and self.appearance.framing.hint:
                self._log(f"Character photo: {self.appearance.framing.hint}")
        self.connected = threading.Event()
        self.reconnects = 0
        self.frames_published = 0
        self.frames_received = 0

    # -- Engine interface --------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, name="decart-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.parked = False
        client = self._client
        loop = self._loop
        if client is not None and loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(client.disconnect(), loop).result(timeout=4)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=6.0)
            self._thread = None
        self.connected.clear()

    def push(self, frame_rgb: np.ndarray, ts: float) -> None:
        with self._in_lock:
            self._in_frame = frame_rgb
            self._in_ts = ts
        self._track_motion(frame_rgb)
        if self.parked:
            # Billing is off, so show the raw camera instead of freezing the last
            # transformed frame: the call keeps working, it just stops paying.
            with self._out_lock:
                self._out_frame = frame_rgb
                self._out_ts = time.perf_counter()

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._out_lock:
            return self._out_frame, self._out_ts

    def latency_ms(self) -> float:
        return self._latency_ms

    # -- billing -----------------------------------------------------------
    @property
    def billed_seconds(self) -> float:
        return self._billed_before + self._session_billed

    def billing(self) -> dict:
        rate = usd_per_second(self.fast_mode)
        seconds = self.billed_seconds
        return {
            "seconds": round(seconds, 1),
            "usd": round(seconds * rate, 3),
            "usd_per_min": round(rate * 60, 2),
            "parked": self.parked,
            "pauses": self.pauses,
        }

    def _on_generation_tick(self, message) -> None:
        try:
            seconds = float(getattr(message, "seconds", 0))
        except (TypeError, ValueError):
            return
        self._session_billed = max(self._session_billed, seconds)

    # -- idle detection ----------------------------------------------------
    def _track_motion(self, frame_rgb: np.ndarray) -> None:
        """Note whether anything has moved recently, cheaply.

        Runs on the capture thread at ~10 Hz on a 48x27 thumbnail, so it costs
        well under a millisecond and cannot become the frame budget.
        """
        now = time.perf_counter()
        if now - self._last_thumb_at < MOTION_SAMPLE_INTERVAL_S:
            return
        self._last_thumb_at = now
        try:
            # Stride first, then resize: resizing 1280x720 straight down to 48x27
            # spends ~4.5 ms averaging blocks nobody looks at. Every 12th pixel
            # still covers the whole frame and costs a fraction of that.
            step = max(1, frame_rgb.shape[1] // 120)
            small = np.ascontiguousarray(frame_rgb[::step, ::step])
            thumb = cv2.resize(small, MOTION_THUMBNAIL, interpolation=cv2.INTER_AREA)
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        except (cv2.error, ValueError):
            return
        previous = self._last_thumb
        self._last_thumb = thumb
        if previous is None:
            self._last_motion_ts = now
            return
        if float(np.mean(cv2.absdiff(thumb, previous))) > MOTION_THRESHOLD:
            self._last_motion_ts = now

    def should_park(self, now: Optional[float] = None) -> bool:
        """True when billing should stop to protect the user's credits."""
        if not self.pause_when_idle or self.parked:
            return False
        now = time.perf_counter() if now is None else now
        if now - self._session_opened_at < MIN_SESSION_BEFORE_PAUSE_S:
            return False
        return now - self._last_motion_ts > self.idle_after_s

    # -- prompt updates ----------------------------------------------------
    def set_prompt(self, text: str, reference_image: Optional[str] = None) -> None:
        """Change the look mid-session without reconnecting."""
        self.prompt = text
        if reference_image:
            self._reference_bytes = load_reference_image(reference_image)
        client, loop = self._client, self._loop
        if client is None or loop is None or not loop.is_running():
            return
        payload = SetInput(
            prompt=text,
            enhance=True,
            image=self._reference_bytes,
        )

        async def _apply() -> None:
            await client.set(payload)

        try:
            asyncio.run_coroutine_threadsafe(_apply(), loop)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Prompt update failed: {exc}")

    def set_garment(self, path: str, prompt: Optional[str] = None) -> None:
        """Switch outfits mid-call without reconnecting.

        The try-on model applies whatever garment is in the reference image, so a
        new photo plus its sentence is the whole update. Nothing to tear down.
        """
        if not self.try_on:
            raise RuntimeError("This session is not a try-on session")
        self._reference_bytes = tryon.prepare_garment(path)
        self.garment_image = path
        text = self.prompt if prompt is None else tryon.as_prompt(prompt)
        self._log(f"New garment: {Path(path).name}")
        self.set_prompt(text)

    # -- thread / event loop ----------------------------------------------
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._supervise())
        except Exception as exc:  # noqa: BLE001
            self._log(f"Decart engine stopped: {exc}")
        finally:
            try:
                loop.run_until_complete(self._cleanup())
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _cleanup(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def _supervise(self) -> None:
        backoff = 1.5
        first = True
        while not self._stop.is_set():
            had_session = False
            paused = False
            try:
                await self._session()
                had_session = self.connected.is_set()
                if self._stop.is_set():
                    break
                self._log("Decart session ended")
            except _IdlePause:
                paused = True
            except Exception as exc:  # noqa: BLE001
                self._log(f"Decart error: {exc}")
            had_session = had_session or self.connected.is_set()
            self.connected.clear()
            if self._stop.is_set():
                break
            if paused:
                # Billing is off until something happens in front of the camera.
                self.parked = True
                self.pauses += 1
                self._log(
                    f"Billing paused after {self.idle_after_s:.0f}s without movement. "
                    "The camera keeps working; the session resumes on movement."
                )
                await self._wait_for_motion()
                self.parked = False
                if self._stop.is_set():
                    break
                self._log("Movement detected - reconnecting Lucy, billing resumes")
                self._last_motion_ts = time.perf_counter()
                await asyncio.sleep(0.2)
                continue
            if not first:
                self.reconnects += 1
            first = False
            if had_session:
                # A session that worked deserves a fast retry, not the backoff
                # accumulated by earlier failures hours ago.
                backoff = 1.5
            self._log(f"Reconnecting in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 20.0)

    async def _wait_for_motion(self) -> None:
        parked_at = time.perf_counter()
        while not self._stop.is_set():
            if self._last_motion_ts > parked_at:
                return
            await asyncio.sleep(0.25)

    # -- one Decart session ------------------------------------------------
    async def _session(self) -> None:
        client_config = DecartClient(api_key=self.api_key)
        try:
            model = models.realtime(self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unknown model '{self.model_name}': {exc}") from exc
        self._log(
            f"Connecting to {model.name} ({model.width}x{model.height} @ {model.fps} fps)"
        )
        self._log(f"Lucy prompt: {self.prompt}")

        self._source = rtc.VideoSource(self.width, self.height)
        local_track = rtc.LocalVideoTrack.create_video_track("mimiccam-camera", self._source)

        state = ModelState(prompt=Prompt(text=self.prompt or "Keep the person as they are.", enhance=True))
        if self._reference_bytes:
            state.image = self._reference_bytes

        options = RealtimeConnectOptions(
            model=model,
            on_remote_stream=self._handle_remote_stream,
            initial_state=state,
            preferred_video_codec="h264",
            speed="fast" if self.fast_mode else None,
        )

        client = await RealtimeClient.connect(
            base_url=client_config.realtime_base_url,
            api_key=self.api_key,
            local_track=local_track,
            options=options,
            integration=f"mimiccam/{__version__}",
        )
        self._client = client
        try:
            client.on("generation_tick", self._on_generation_tick)
        except Exception:  # noqa: BLE001 - a missing tick only costs the meter
            pass
        self.connected.set()
        self._last_output_at = time.perf_counter()
        self._session_opened_at = time.perf_counter()
        self._last_motion_ts = self._session_opened_at
        self._session_billed = 0.0
        rate = FAST_USD_PER_SECOND if self.fast_mode else USD_PER_SECOND
        self._log(
            f"Lucy session started ({'fast' if self.fast_mode else 'standard'} mode, "
            f"${rate:.2f}/s = ${rate * 60:.2f}/min)"
        )

        # Explicit tasks so a pause (the publisher raising) does not leave the
        # watchdog running against a dead session.
        workers = [
            asyncio.create_task(self._publisher(), name="lucy-publisher"),
            asyncio.create_task(self._watchdog(), name="lucy-watchdog"),
        ]
        try:
            done, _pending = await asyncio.wait(
                workers, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            try:
                await client.disconnect()
            except Exception:
                pass
            self._client = None
            try:
                await self._source.aclose()
            except Exception:
                pass
            self._source = None
            # The server's last tick is authoritative for this session.
            self._billed_before += self._session_billed
            self._session_billed = 0.0

    def _handle_remote_stream(self, track) -> None:
        self._log("Receiving transformed video")
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        self._consumer_task = asyncio.run_coroutine_threadsafe(self._consume(track), loop)

    async def _consume(self, track) -> None:
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        try:
            async for event in stream:
                if self._stop.is_set():
                    return
                frame = event.frame
                expected = frame.width * frame.height * 3
                data = bytes(frame.data)
                if len(data) < expected:
                    continue
                array = np.frombuffer(data[:expected], dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                with self._out_lock:
                    self._out_frame = array
                    # Arrival time, not the input timestamp: the pipeline uses this
                    # to tell "the model is still delivering" from "the model went
                    # quiet and the last frame must be held".
                    self._out_ts = time.perf_counter()
                self._last_output_at = time.perf_counter()
                self.frames_received += 1
                if self._pub_times:
                    self._latency_ms = (time.perf_counter() - self._pub_times.popleft()) * 1000
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    async def _publisher(self) -> None:
        interval = 1.0 / max(1, min(self.fps, 30))
        last_sent_ts = 0.0
        while not self._stop.is_set():
            if self.should_park():
                raise _IdlePause()
            with self._in_lock:
                frame, ts = self._in_frame, self._in_ts
            source = self._source
            if frame is not None and ts != last_sent_ts and source is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                try:
                    source.capture_frame(
                        rtc.VideoFrame(
                            self.width,
                            self.height,
                            rtc.VideoBufferType.RGB24,
                            frame.tobytes(),
                        )
                    )
                    self._pub_times.append(time.perf_counter())
                    self.frames_published += 1
                    last_sent_ts = ts
                except Exception as exc:  # noqa: BLE001
                    self._log(f"Publish failed: {exc}")
                    return
            await asyncio.sleep(interval)

    async def _watchdog(self) -> None:
        """Trigger a reconnect when the remote stream goes silent."""
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if self._stop.is_set():
                return
            silence = time.perf_counter() - self._last_output_at
            if silence > WATCHDOG_TIMEOUT_S:
                raise RuntimeError(f"no video from Lucy for {silence:.0f}s")
