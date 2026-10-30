"""The free engine: the model runs on this machine, not on a server.

This is the "no credits, no subscription" path. It replaces Decart's Lucy 2.5
session with three small on-device networks:

* YuNet finds the face and its 5 landmarks,
* ArcFace turns the character photo into one 512-d identity vector (computed
  once, not per frame),
* the swap network renders your face wearing that identity at 128x128, which is
  glued back on with a soft face mask.

Two threads share the work. The render thread runs the swap network as fast as
the GPU allows (~16 swaps a second on a laptop 4070 through DirectML); the frame
thread detects landmarks every frame and re-projects the newest texture through
them, so the face keeps moving at the full 30 fps while the detail refreshes
underneath it. Rendering and compositing stay separate for exactly that reason.

The identity models run on the NVIDIA GPU through DirectML, with the adapter
pinned explicitly (see ``gpu.directml_providers``); the matte and the room
inpainting stay on the CPU, where they measured faster. No CUDA toolkit, no API
key, and no network traffic after the one-time model download.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from . import body, face, gpu, hair, looks, matte as matting, models, studio
from .engine import Engine

TEXTURE_TTL_S = 1.0
# How long a single dropped detection is bridged with the last good pose. One
# missed frame used to fall through to the untouched camera for 33 ms, which is
# exactly what a swap that "bounces off" looks like; inside this window the
# last landmarks are reused and the face keeps its character.
POSE_HOLD_S = 0.40
# Weight of the newest render when it is folded into the displayed texture. The
# network re-renders at ~16 Hz and each render is a slightly different picture;
# hard-swapping them pops. A 0.6/0.4 mix keeps the detail moving without the
# face flickering between two renders.
TEXTURE_BLEND = 0.60
# 720 px keeps landmarks precise enough to place the paste without eating a
# third of the frame budget: 960 measured 12 ms, 640 measured 5 ms, 720 lands at
# ~7 ms in the engine's four-thread configuration. FaceDetector can refine each
# face in a native-resolution crop for sharper landmarks (4 px against 25 px on
# a 1280x720 frame), but that pass costs ~13 ms and the engine drops below its
# 25 fps floor with it, so video runs the coarse pass and only one-off detections
# (the character photo) are worth refining.
DETECT_MAX_SIDE = 720
# Run the body pass on every frame. Skipping alternate frames halves the cost
# but makes the skin tone pulse at 15 Hz whenever the correction is large, and a
# pulsing arm is worse than two fewer frames per second.
BODY_EVERY = 1
# Queue the person matte every N frames. 1 keeps the silhouette in lockstep
# with the video; 2 halves the heaviest thread's work in exchange for an outline
# that is one frame old. Off by default - tests/verify_matte_lag.py measures what
# the lag costs at the hairline, and the composite reuses the last alpha anyway
# whenever the matte thread is behind.
MATTE_EVERY = 1
# OpenCV hands every operation to a pool the size of the logical core count (32
# on this laptop). The engine already runs three of its own threads next to
# onnxruntime's, so the extra workers only fight them for cores: the studio
# loop measured 26 fps with 4 workers against 23 fps with 32.
CV_THREADS = 4


class LocalEngineError(RuntimeError):
    """Raised when the local engine cannot start (missing models, no face...)."""


class LocalEngine(Engine):
    name = "Local model (offline)"

    def __init__(
        self,
        reference_image: Optional[str] = None,
        preset_key: str = "character",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        on_log=None,
        look: Optional[str] = None,
        providers: Optional[list[str]] = None,
        max_side: int = DETECT_MAX_SIDE,
        max_faces: int = 1,
        body_pass: bool = True,
        studio_background: bool = False,
        studio_image: Optional[str] = None,
        matte_every: int = MATTE_EVERY,
    ) -> None:
        self.reference_image = reference_image or ""
        self.width = width
        self.height = height
        self.fps = fps
        self.look = look or looks.for_preset(preset_key)
        # None means "pick the NVIDIA adapter and insist on it"; an explicit list
        # is the caller's business (tools and tests run the models on the CPU).
        self.providers = providers
        self._strict_gpu = providers is None
        self.max_side = max_side
        self.max_faces = max_faces
        self.body_pass = body_pass
        self.studio_background = bool(studio_background)
        # Defaults to the character photo: the room and the person come from
        # the same reference unless the user picks a separate room.
        self.studio_image = studio_image or self.reference_image
        self._log = on_log or (lambda _m: None)
        self._smoother = face.LandmarkSmoother()
        self._body_tick = 0
        self.matte_every = max(1, int(matte_every))
        self._matte_tick = 0
        self._matte: Optional[matting.MatteModel] = None
        self._studio: Optional[studio.StudioScene] = None
        self._last_face_box: Optional[tuple[int, int, int, int]] = None
        self._last_landmarks: Optional[np.ndarray] = None
        self._last_pose_ts = 0.0
        # Frame counter for the landmark filter. It is fed a frame clock, not a
        # wall clock: the camera delivers 30 pictures a second whatever the
        # engine's processing rate, and a filter timed in seconds changes its
        # lag when the machine is busy. tests/verify_motion.py caught exactly
        # that: the same pan lagged a frame after a still take.
        self._frame_no = 0
        # Where the last frame pasted the character, in frame pixels. Kept for
        # tests/verify_motion.py, which compares it with the face the detector
        # actually saw; None on frames with no paste.
        self._paste_landmarks: Optional[np.ndarray] = None
        self._hair: Optional[hair.HairPass] = None
        self._studio_warned = False

        self._in_lock = threading.Lock()
        self._in_frame: Optional[np.ndarray] = None
        self._in_ts = 0.0

        self._out_lock = threading.Lock()
        self._out_frame: Optional[np.ndarray] = None
        self._out_ts = 0.0

        self._texture_lock = threading.Lock()
        self._texture: Optional[np.ndarray] = None
        self._texture_ts = 0.0

        self._pending_lock = threading.Lock()
        self._pending: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._pending_event = threading.Event()

        # The matte runs on its own thread, in parallel with face detection and
        # the swap paste. It is ~16 ms; doing it inline would cost more than the
        # whole composite budget.
        self._matte_in_lock = threading.Lock()
        self._matte_in: Optional[np.ndarray] = None
        self._matte_in_ts = 0.0
        self._matte_out_lock = threading.Lock()
        self._matte_out: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._matte_out_ts = 0.0
        self._matte_event = threading.Event()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._matte_thread: Optional[threading.Thread] = None
        self.ready = threading.Event()
        self.error: str = ""

        self._detector: Optional[face.FaceDetector] = None
        self._embedder: Optional[face.FaceEmbedder] = None
        self._swapper: Optional[face.FaceSwapper] = None
        self._embedding: Optional[np.ndarray] = None
        self._provider = "CPU"
        self._cv_threads_before = 0

        self.frames_out = 0
        self.frames_swapped = 0
        self.frames_no_face = 0
        self.texture_refreshes = 0
        self._latency_ms = 0.0
        self._fps = 0.0
        self._render_fps = 0.0
        self.swapping = False

    # -- Engine interface --------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        required = list(models.CORE_KEYS)
        if self.studio_background:
            required += list(models.STUDIO_KEYS)
        absent = models.missing(required)
        if absent:
            raise LocalEngineError(models.describe_missing(absent))
        self._stop.clear()
        self.error = ""
        self.ready.clear()
        self._thread = threading.Thread(target=self._run, name="local-engine", daemon=True)
        self._thread.start()
        # Surface setup failures (corrupt model, unusable photo) to the caller
        # instead of leaving a camera running with a dead engine behind it.
        self.ready.wait(timeout=180)
        if self.error:
            self.stop()
            raise LocalEngineError(self.error)

    def stop(self) -> None:
        self._stop.set()
        self._pending_event.set()
        self._matte_event.set()
        for thread in (self._render_thread, self._matte_thread, self._thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self._render_thread = None
        self._matte_thread = None
        self._thread = None
        if self._cv_threads_before and CV_THREADS > 0:
            cv2.setNumThreads(self._cv_threads_before)
            self._cv_threads_before = 0

    def push(self, frame_rgb: np.ndarray, ts: float) -> None:
        with self._in_lock:
            self._in_frame = frame_rgb
            self._in_ts = ts

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._out_lock:
            return self._out_frame, self._out_ts

    def latency_ms(self) -> float:
        return self._latency_ms

    @property
    def paste_landmarks(self) -> Optional[np.ndarray]:
        """The landmarks the last frame's paste was aligned to (copy)."""
        return None if self._paste_landmarks is None else self._paste_landmarks.copy()

    @property
    def measured_fps(self) -> float:
        return self._fps

    @property
    def render_fps(self) -> float:
        return self._render_fps

    @property
    def provider(self) -> str:
        return self._provider

    # -- setup -------------------------------------------------------------
    def _setup(self) -> None:
        started = time.perf_counter()
        if self.providers is None:
            # Strict on purpose: a silent CPU fallback would turn a 30 fps call
            # into a slideshow the user never agreed to, and a silent integrated-
            # GPU fallback would do the same a little more quietly.
            try:
                self.providers = gpu.directml_providers(strict=True)
            except RuntimeError as exc:
                raise LocalEngineError(str(exc)) from exc
        self._detector = face.FaceDetector(
            models.path("detector"), max_side=self.max_side, refine=False
        )
        self._embedder = face.FaceEmbedder(models.path("identity"), self.providers)
        self._swapper = face.FaceSwapper(models.path("swapper"), self.providers)
        self._provider = self._swapper.session.get_providers()[0]
        if self._strict_gpu and self._provider != "DmlExecutionProvider":
            raise LocalEngineError(
                "The models did not land on the NVIDIA GPU (they are running on "
                f"{self._provider}).\n\nMimicCam only runs the free engine on the "
                "NVIDIA card. Update the NVIDIA driver and check that Windows has "
                "not disabled the device, then restart."
            )
        adapter = gpu.adapter_name(self.providers)
        where = f"{self._provider} on {adapter}" if adapter != "CPU" else self._provider
        self._log(
            f"Local models ready in {time.perf_counter() - started:.1f}s "
            f"({where}, look: {self.look}, detect at {self.max_side}px, "
            f"body pass {'on' if self.body_pass else 'off'})"
        )

        if self.studio_background and self.studio_image:
            self._setup_studio()
        elif self.studio_background:
            self._log("Studio background needs a reference image - showing the camera instead")

        if self.reference_image:
            # A bad character photo must not take the whole camera down: the
            # user would see a modal error and no picture at all. Degrade to the
            # look-only path and say exactly what to fix.
            try:
                photo = face.load_image(self.reference_image)
                reference_face = face.largest(self._detector.detect(photo))
                if reference_face is None:
                    raise ValueError(
                        "No face found in that image. Use a clear, front-facing "
                        "photo of the character's face."
                    )
                self._embedding = self._embedder.embedding(photo, reference_face.landmarks)
                # The same detection seeds the hair pass: its landmarks are the
                # ruler the reference head is measured in.
                self._hair = hair.HairPass(photo, reference_face.landmarks, on_log=self._log)
                if self._hair.active:
                    self._log(
                        f"Reference hair will follow the head "
                        f"({self._hair.coverage * 100:.0f}% of the head crop)"
                    )
                self.swapping = True
                height, width = photo.shape[:2]
                self._log(f"Character locked from {self.reference_image} ({width}x{height})")
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                self.swapping = False
                self._embedding = None
                self._log(
                    f"Character image unusable ({exc}) - running without a swap. "
                    "Choose a clear, front-facing photo of one face."
                )
        else:
            self._log("No character image: applying the look without swapping identity")

    def _setup_studio(self) -> None:
        """Build (or load) the reference room and the video matte."""
        started = time.perf_counter()
        try:
            # The matte stays on the CPU by measurement, not by habit: on this
            # laptop the 15 MB network takes 8.9 ms here against 12.2 ms on the
            # 4070, and the full matte 17.0 ms against 23.1 ms, because the
            # per-frame upload costs more than the convolution saves.
            self._matte = matting.shared(models.path("matte"), ["CPUExecutionProvider"])
            # LaMa's CPU session takes a few seconds to build. Only build it when
            # this reference photo has not been cleaned before.
            inpaint = None
            if not studio.is_plate_cached(self.studio_image, self.width, self.height):
                inpaint = studio.shared_inpaint(models.path("inpaint"))
            plate, reference_face = studio.load_plate(
                self.studio_image,
                self.width,
                self.height,
                matte_model=self._matte,
                inpaint_model=inpaint,
                on_log=self._log,
            )
            self._studio = studio.StudioScene(plate, reference_face, on_log=self._log)
            self._log(
                f"Studio background ready in {time.perf_counter() - started:.1f}s "
                f"(matte on {self._matte.provider}, plate {self.studio_image})"
            )
        except Exception as exc:  # noqa: BLE001 - a background must not kill the call
            self._studio = None
            self._matte = None
            self._log(f"Studio background unavailable: {exc}")

    def _run(self) -> None:
        self._cv_threads_before = cv2.getNumThreads()
        if CV_THREADS > 0:
            cv2.setNumThreads(CV_THREADS)
        try:
            self._setup()
        except Exception as exc:  # noqa: BLE001 - reported to the UI thread
            self.error = str(exc)
            self._log(f"Local engine could not start: {exc}")
            self.ready.set()
            return
        self.ready.set()
        if self.swapping:
            self._render_thread = threading.Thread(
                target=self._render_loop, name="local-render", daemon=True
            )
            self._render_thread.start()
        if self._studio is not None and self._matte is not None:
            self._matte_thread = threading.Thread(
                target=self._matte_loop, name="local-matte", daemon=True
            )
            self._matte_thread.start()
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self._log(f"Local engine stopped: {exc}")

    # -- render thread: run the network on the newest face -----------------
    def _render_loop(self) -> None:
        window_start = time.perf_counter()
        produced = 0
        while not self._stop.is_set():
            self._pending_event.wait(timeout=0.2)
            self._pending_event.clear()
            if self._stop.is_set():
                return
            with self._pending_lock:
                pending, self._pending = self._pending, None
            if pending is None or self._swapper is None or self._embedding is None:
                continue
            frame, landmarks = pending
            try:
                texture = self._swapper.render(frame, landmarks, self._embedding)
            except Exception as exc:  # noqa: BLE001 - keep the feed alive
                self._log(f"Render failed: {exc}")
                continue
            self._store_texture(texture)
            self.texture_refreshes += 1

            produced += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                self._render_fps = produced / (now - window_start)
                produced = 0
                window_start = now

    # -- matte thread ------------------------------------------------------
    def _matte_loop(self) -> None:
        while not self._stop.is_set():
            self._matte_event.wait(timeout=0.2)
            self._matte_event.clear()
            if self._stop.is_set():
                return
            with self._matte_in_lock:
                frame = self._matte_in
                self._matte_in = None
            if frame is None or self._matte is None:
                continue
            try:
                alpha, foreground = self._matte.matte(frame)
            except Exception as exc:  # noqa: BLE001 - keep the feed alive
                self._log(f"Matte failed: {exc}")
                continue
            with self._matte_out_lock:
                self._matte_out = (alpha, foreground)
                self._matte_out_ts = time.perf_counter()

    def _queue_matte(self, frame_bgr: np.ndarray) -> None:
        if self._matte_thread is None:
            return
        with self._matte_in_lock:
            self._matte_in = frame_bgr.copy()
        self._matte_event.set()

    def _current_matte(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        with self._matte_out_lock:
            if self._matte_out is None:
                return None, None, 0.0
            return self._matte_out[0], self._matte_out[1], self._matte_out_ts

    def _queue_render(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
        with self._pending_lock:
            self._pending = (frame_bgr.copy(), landmarks.copy())
        self._pending_event.set()

    def _current_texture(self) -> tuple[Optional[np.ndarray], float]:
        with self._texture_lock:
            return self._texture, self._texture_ts

    def _blend_texture(self, texture: np.ndarray) -> np.ndarray:
        """Fold a fresh render into the displayed one instead of popping.

        Called under no lock: the previous texture is read and the new one is
        stored by ``_store_texture``. A texture older than half a second (the
        face just came back, or the character changed) is taken as-is rather
        than smeared onto stale pixels.
        """
        with self._texture_lock:
            old, ts = self._texture, self._texture_ts
        if (old is None or old.shape != texture.shape
                or (time.perf_counter() - ts) > 0.5):
            return texture
        return cv2.addWeighted(old, 1.0 - TEXTURE_BLEND, texture, TEXTURE_BLEND, 0.0)

    def _store_texture(self, texture: np.ndarray) -> None:
        with self._texture_lock:
            self._texture = texture
            self._texture_ts = time.perf_counter()

    # -- frame thread ------------------------------------------------------
    def _loop(self) -> None:
        last_ts = 0.0
        window_start = time.perf_counter()
        produced = 0

        while not self._stop.is_set():
            with self._in_lock:
                frame, ts = self._in_frame, self._in_ts
            if frame is None or ts == last_ts:
                time.sleep(0.003)
                continue
            last_ts = ts

            began = time.perf_counter()
            try:
                out = self._process(frame)
            except Exception as exc:  # noqa: BLE001 - never kill the feed
                self._log(f"Frame skipped: {exc}")
                continue

            with self._out_lock:
                self._out_frame = out
                # perf_counter, not the camera clock: the pipeline uses this to
                # decide whether the engine has gone quiet.
                self._out_ts = time.perf_counter()
            self.frames_out += 1

            spent = (self._out_ts - began) * 1000.0
            self._latency_ms = spent if not self._latency_ms else 0.8 * self._latency_ms + 0.2 * spent

            produced += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                self._fps = produced / (now - window_start)
                produced = 0
                window_start = now

    def _process(self, frame_rgb: np.ndarray) -> np.ndarray:
        self._frame_no += 1
        height, width = frame_rgb.shape[:2]
        if (width, height) != (self.width, self.height):
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._studio is not None and self._matte_thread is not None:
            # Start the matte now so it overlaps detection and the paste.
            if self._matte_tick % self.matte_every == 0:
                self._queue_matte(bgr)
            self._matte_tick += 1

        if self._embedding is not None and self._swapper is not None and self._detector is not None:
            detections = self._detector.detect(bgr)
            if detections:
                detections.sort(key=lambda d: d.area, reverse=True)
                face_det = detections[0]
                # Smooth the transform, not the detection: the paste must follow
                # the face without inheriting detector jitter. The clock is in
                # frames, so the filter behaves the same at 20 fps and 30 fps.
                landmarks = self._smoother.update(
                    face_det.landmarks, now=self._frame_no / max(1.0, float(self.fps))
                )
                self._last_landmarks = landmarks.copy()
                self._last_pose_ts = time.perf_counter()
                self._last_face_box = face_det.box
            elif (self._last_landmarks is not None
                  and (time.perf_counter() - self._last_pose_ts) <= POSE_HOLD_S):
                # YuNet drops the face for a frame or two on a blink, a fast
                # turn or a hand in front of it. Pasting the last good pose
                # bridges the gap; resetting here would flash the real face and
                # is what makes the swap look like it is bouncing off.
                landmarks = self._last_landmarks
                face_det = None
            else:
                self._smoother.reset()
                self._last_landmarks = None
                self.frames_no_face += 1
                landmarks = None

            self._paste_landmarks = None if landmarks is None else landmarks.copy()
            if landmarks is not None:
                texture, texture_ts = self._current_texture()
                stale = texture is None or (time.perf_counter() - texture_ts) > TEXTURE_TTL_S
                if stale:
                    # First face, or it just came back: render once inline so the
                    # user never sees their own face where the character's should be.
                    texture = self._blend_texture(
                        self._swapper.render(bgr, landmarks, self._embedding)
                    )
                    self._store_texture(texture)
                    self.texture_refreshes += 1
                else:
                    self._queue_render(bgr, landmarks)
                if self._hair is not None:
                    # Under the face paste: the character's hair frames the jaw
                    # and hairline, then the swapped face sits on top of it.
                    bgr = self._hair.apply(bgr, landmarks)
                bgr = self._swapper.paste(bgr, texture, landmarks)
                if self.body_pass and face_det is not None and self._body_tick % BODY_EVERY == 0:
                    # Neck, chest and arms take the swapped face's skin tone, so
                    # the body and the face look like the same person.
                    tone = body.face_skin_tone(bgr, face_det.box)
                    if tone is not None:
                        bgr = body.harmonize(bgr, face_det.box, tone)
                self._body_tick += 1
                self.frames_swapped += 1

        if self._studio is not None and self._matte is not None:
            if self._studio.seat is not None or self._last_face_box is not None:
                # The matte covers head, shoulders, chest and arms: everything
                # the camera can see, so nothing of the original room survives.
                alpha, edge, ts = self._current_matte()
                age = time.perf_counter() - ts if ts else 0.0
                # Only pay for a matte on this thread when there is none, or
                # when the matte thread has produced nothing for half a second
                # (it is stuck or died). Re-running it inline the moment the
                # alpha got 300 ms old used to be a trap: the inline pass
                # doubled the frame time, which made the next alpha late again,
                # so a busy machine fell from 30 fps to 3. Half a second is
                # several matte passes of slack, and still short enough that
                # the silhouette never lags far enough to see.
                if alpha is None or age > 0.5:
                    alpha, edge = self._matte.matte(bgr)
                    with self._matte_out_lock:
                        self._matte_out = (alpha, edge)
                        self._matte_out_ts = time.perf_counter()
                bgr = self._studio.composite(bgr, alpha, self._last_face_box, edge)
            elif not self._studio_warned:
                self._studio_warned = True
                self._log("Studio background is waiting for a face to seat you in the room")

        bgr = looks.apply(bgr, self.look)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
