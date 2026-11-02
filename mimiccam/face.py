"""Faces: detect, align, embed and swap - all on-device.

A dependency-light reimplementation of the InsightFace inference path: YuNet
finds the face and its 5 landmarks, ArcFace turns a photo into a 512-d identity
vector, and the swap network renders the target face wearing that identity. No
package that needs a C compiler, no CUDA toolkit, no network call.

Frames are OpenCV BGR ``uint8`` arrays and landmarks are ``float32`` of shape
``(5, 2)`` in *arcface* order (eye, eye, nose, mouth corner, mouth corner, as
seen left-to-right in the image).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from . import gpu

# The 112x112 template InsightFace aligns to before recognition / swapping.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# -- alignment -------------------------------------------------------------
def similarity_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale + rotation + translation) as a 2x3 matrix.

    This is the closed form Umeyama estimate that InsightFace uses. OpenCV's
    ``estimateAffinePartial2D`` would do, but it is RANSAC-based and therefore
    non-deterministic on 5 points, which makes alignment tests flaky.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("src and dst must be matching landmark sets")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # (2, 2) covariance of the two point clouds.
    cov = dst_demean.T @ src_demean / len(src)

    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0

    u, singular, vt = np.linalg.svd(cov)
    rotation = u @ np.diag(d) @ vt
    variance = src_demean.var(axis=0).sum()
    if variance <= 0:
        raise ValueError("degenerate landmarks: all points are identical")
    scale = float((singular @ d) / variance)

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = dst_mean - (scale * rotation) @ src_mean
    return matrix[:2, :].astype(np.float32)


def norm_matrix(landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Matrix that maps image landmarks onto InsightFace's crop template."""
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST.astype(np.float64) * ratio
    dst[:, 0] += diff_x
    return similarity_matrix(landmarks, dst)


def crop(img_bgr: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(img_bgr, matrix, (size, size), borderValue=(0, 0, 0))


class LandmarkSmoother:
    """Alpha-beta tracking of the 5 landmarks.

    A plain exponential average is stable on a still face but lags a moving one:
    at 0.45 per frame it sits about 1.2 frames behind, which on a pan measured
    16 px of drift in tests/verify_motion.py - the swapped face sliding off the
    head. This filter carries a velocity term, so a face moving at a steady speed
    is predicted instead of chased and the steady-state lag is zero, while
    detector noise on a still face is still averaged away (alpha shrinks when the
    measured motion is small).

    A genuine jump (head turn, camera cut, a face swap in the shot) resets the
    state rather than dragging a ghost across the frame.
    """

    def __init__(self, alpha: float = 0.62, beta: float = 0.42,
                 jump_ratio: float = 0.30, max_speed: float = 6.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jump_ratio = float(jump_ratio)
        # Velocity clamp in eye-distances per second: a hand-held webcam cannot
        # swing the face across six face widths in a second, and clamping stops
        # one bad detection from launching the prediction.
        self.max_speed = float(max_speed)
        self._state: Optional[np.ndarray] = None
        self._velocity: Optional[np.ndarray] = None
        self._span = 1.0
        self._last_t = 0.0

    @staticmethod
    def _spacing(points: np.ndarray) -> float:
        return float(np.linalg.norm(points[0] - points[1])) or 1.0

    def update(self, landmarks: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        current = time.perf_counter() if now is None else float(now)
        span = self._spacing(points)
        if self._state is None or self._velocity is None:
            self._state = points.copy()
            self._velocity = np.zeros((5, 2), dtype=np.float32)
            self._span = span
            self._last_t = current
            return points

        dt = float(np.clip(current - self._last_t, 1e-3, 0.25))
        self._last_t = current
        self._span = 0.7 * self._span + 0.3 * span

        predicted = self._state + self._velocity * dt
        residual = points - predicted
        if float(np.linalg.norm(residual.mean(axis=0))) > self.jump_ratio * span:
            self._state = points.copy()
            self._velocity = np.zeros((5, 2), dtype=np.float32)
            return points

        # Faster motion is followed harder; a still face is averaged more.
        speed = float(np.linalg.norm(residual.mean(axis=0))) / (dt * span)
        alpha = float(np.clip(self.alpha + 0.35 * min(speed, 3.0), 0.25, 0.92))

        self._state = predicted + alpha * residual
        velocity = self._velocity + (self.beta * alpha / dt) * residual
        limit = self.max_speed * self._span
        magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity = velocity * np.minimum(1.0, limit / np.maximum(magnitude, 1e-3))
        self._velocity = velocity
        return self._state.copy()

    @property
    def current(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def reset(self) -> None:
        self._state = None
        self._velocity = None


def match_tone(texture: np.ndarray, reference: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Match a rendered face's colour statistics to the frame it came from.

    The network re-lights the face it generates. Pasting a studio-bright face
    onto a dim room (or the reverse) is what makes a swap read as a sticker, so
    the render is pulled toward the lighting of the live crop before compositing.
    """
    if texture.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (texture.shape[1], texture.shape[0]))
    height, width = texture.shape[:2]
    box = (slice(int(height * 0.2), int(height * 0.8)), slice(int(width * 0.2), int(width * 0.8)))

    tex_lab = cv2.cvtColor(texture, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = tex_lab.copy()
    for channel in range(3):
        patch = tex_lab[box][:, :, channel]
        target = ref_lab[box][:, :, channel]
        tex_mean, tex_std = float(patch.mean()), float(patch.std()) + 1e-3
        ref_mean, ref_std = float(target.mean()), float(target.std()) + 1e-3
        gain = float(np.clip(ref_std / tex_std, 0.75, 1.25))
        adjusted = (tex_lab[:, :, channel] - tex_mean) * gain + ref_mean
        out[:, :, channel] = tex_lab[:, :, channel] + strength * (adjusted - tex_lab[:, :, channel])
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite(frame_bgr: np.ndarray, texture: np.ndarray, mask: np.ndarray,
              matrix: np.ndarray) -> np.ndarray:
    """Warp a 128px face texture back onto the frame, inside its own bounding box.

    Warping two full 1080p planes costs more than the swap network itself, so
    the paste is clipped to the face's footprint. ``matrix`` is the frame->crop
    transform the texture was rendered with.
    """
    height, width = frame_bgr.shape[:2]
    size = texture.shape[0]
    inverse = cv2.invertAffineTransform(matrix)

    corners = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    projected = cv2.transform(corners.reshape(1, -1, 2), inverse).reshape(-1, 2)
    pad = max(4, size // 16)
    x0 = max(0, int(np.floor(projected[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(projected[:, 1].min())) - pad)
    x1 = min(width, int(np.ceil(projected[:, 0].max())) + pad)
    y1 = min(height, int(np.ceil(projected[:, 1].max())) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame_bgr

    shift = np.eye(3, dtype=np.float32)
    shift[0, 2] = -x0
    shift[1, 2] = -y0
    roi_matrix = (shift @ np.vstack([inverse, [0.0, 0.0, 1.0]]))[:2]
    roi_size = (x1 - x0, y1 - y0)

    warped = cv2.warpAffine(texture, roi_matrix, roi_size, flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(mask, roi_matrix, roi_size, flags=cv2.INTER_LINEAR)
    # uint8 SIMD blending: the float-plane version cost 4.8 ms for a 465 px
    # face on this laptop, this one 0.7 ms. The 1/255 quantisation of the soft
    # edge is below the camera's own noise floor.
    if alpha.dtype != np.uint8:
        alpha = cv2.convertScaleAbs(alpha, alpha=255.0)
    weight = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

    out = frame_bgr.copy()
    target = out[y0:y1, x0:x1]
    front = cv2.multiply(warped, weight, scale=1.0 / 255.0)
    back = cv2.multiply(target, cv2.bitwise_not(weight), scale=1.0 / 255.0)
    cv2.add(front, back, dst=target)
    return out


# -- detection -------------------------------------------------------------
@dataclass
class Detection:
    """One face in a frame."""

    box: tuple[int, int, int, int]  # x, y, w, h in image pixels
    score: float
    landmarks: np.ndarray  # (5, 2) float32, arcface order
    index: int = 0

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


# Window sizes the second pass may run at. YuNet's landmark precision is a
# function of how many pixels the face occupies, not of the resolution it is
# told about: on a 1280x720 frame measured against a full-resolution fit, the
# same face lands 25 px off at a 720 px coarse pass, 10 px at 960, and 3.5 px
# when the detector sees it at native size. A padded crop at one of these few
# sizes keeps the face large without rebuilding a session for every frame.
REFINE_SIZES = (160, 192, 256, 320, 384, 448, 512, 640, 896)
REFINE_PAD = 1.35  # crop side, as a multiple of the detected face box


class FaceDetector:
    """YuNet, the tiny detector bundled with OpenCV's model zoo."""

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6,
                 max_side: int = 960, refine: bool = True) -> None:
        self.model_path = str(model_path)
        self.score_threshold = float(score_threshold)
        self.max_side = int(max_side)
        self.refine = bool(refine)
        self._detector = None
        self._input_size: tuple[int, int] | None = None
        # A second session for the crops. Sharing one with the frame pass would
        # rebuild a detector twice per frame (frame size, crop size, frame size)
        # and measure 63 ms a frame instead of 15.
        self._crop_detector = None
        self._crop_size: int | None = None

    def _factory(self):
        return getattr(cv2, "FaceDetectorYN_create", None) or cv2.FaceDetectorYN.create

    def _for_size(self, width: int, height: int):
        if self._detector is None or self._input_size != (width, height):
            self._detector = self._factory()(
                self.model_path, "", (width, height),
                self.score_threshold, 0.3, 50,
            )
            self._input_size = (width, height)
        return self._detector

    def _choose_size(self, needed: int, width: int, height: int) -> Optional[int]:
        """Pick a crop size, sticking with the session already built.

        The face box breathes by a few pixels a frame, so choosing purely by
        ``needed`` flips between two ladder sizes and rebuilds a YuNet session
        every few frames - measured at 40 ms a rebuild, which halved the engine's
        frame rate. Hysteresis keeps one size until the face grows or shrinks
        past 20% of it.
        """
        if self._crop_size is not None and 0.75 * self._crop_size <= needed <= self._crop_size:
            if self._crop_size <= width and self._crop_size <= height:
                return self._crop_size
        for candidate in REFINE_SIZES:
            if candidate >= needed and candidate <= width and candidate <= height:
                return candidate
        return None

    def _crop_for_size(self, size: int):
        if self._crop_detector is None or self._crop_size != size:
            self._crop_detector = self._factory()(
                self.model_path, "", (size, size),
                self.score_threshold, 0.3, 50,
            )
            self._crop_size = size
        return self._crop_detector

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Faces in the frame, in frame pixels.

        Two passes. The whole frame is scanned at ``max_side`` to find faces
        cheaply; each face is then re-detected in a square, un-resized crop
        around it, where YuNet sees it at native size and its landmarks are
        precise. The crop is centred on the coarse box rather than on the last
        frame's face, so it always contains exactly one face and cannot latch
        onto a neighbour.
        """
        found = self._detect_frame(frame_bgr)
        if not self.refine or not found:
            return found
        return [self._refine(frame_bgr, detection) for detection in found]

    def _refine(self, frame_bgr: np.ndarray, detection: Detection) -> Detection:
        """Re-detect one face in a padded native-resolution crop.

        Falls back to the coarse detection when the crop cannot be made (face
        at the frame edge on a small frame) or when the refined box does not
        overlap the coarse one - a second face behind the first, or noise.
        """
        height, width = frame_bgr.shape[:2]
        x, y, w, h = detection.box
        if w < 8 or h < 8:
            return detection
        needed = int(max(w, h) * REFINE_PAD)
        size = self._choose_size(needed, width, height)
        if size is None:
            return detection

        cx, cy = x + w / 2.0, y + h / 2.0
        x0 = max(0, min(width - size, int(round(cx - size / 2.0))))
        y0 = max(0, min(height - size, int(round(cy - size / 2.0))))
        crop = frame_bgr[y0:y0 + size, x0:x0 + size]
        try:
            refined = largest(self._run(crop, self._crop_for_size(size)))
        except cv2.error:
            return detection
        if refined is None:
            return detection

        bx, by, bw, bh = refined.box
        fx, fy = bx + x0, by + y0  # back to frame pixels
        # The refined box must be the same face: its centre has to fall inside
        # the coarse box, allowing for the detector's own box jitter.
        rx, ry = fx + bw / 2.0, fy + bh / 2.0
        if not (x - w * 0.2 <= rx <= x + w * 1.2 and y - h * 0.2 <= ry <= y + h * 1.2):
            return detection
        return Detection(
            box=(fx, fy, bw, bh),
            score=refined.score,
            landmarks=refined.landmarks + np.array([x0, y0], dtype=np.float32),
            index=detection.index,
        )

    def _detect_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        scale = 1.0
        longest = max(width, height)
        if self.max_side and longest > self.max_side:
            scale = self.max_side / float(longest)
        work = frame_bgr
        if scale != 1.0:
            work = cv2.resize(
                frame_bgr,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        found = self._run(work)
        if not found or scale == 1.0:
            return found

        out: list[Detection] = []
        inverse = 1.0 / scale
        for item in found:
            bx, by, bw, bh = item.box
            out.append(Detection(
                box=(int(bx * inverse), int(by * inverse),
                     int(bw * inverse), int(bh * inverse)),
                score=item.score,
                landmarks=item.landmarks * inverse,
                index=item.index,
            ))
        return out

    def _run(self, work: np.ndarray, detector=None) -> list[Detection]:
        work_h, work_w = work.shape[:2]
        if detector is None:
            detector = self._for_size(work_w, work_h)
        try:
            _, faces = detector.detect(work)
        except cv2.error:
            # A stale detector bound to another resolution: rebuild next call.
            self._detector = None
            self._crop_detector = None
            self._input_size = None
            self._crop_size = None
            return []
        if faces is None:
            return []

        out: list[Detection] = []
        for index, row in enumerate(faces):
            row = np.asarray(row, dtype=np.float32).ravel()
            if row.size < 15:
                continue
            score = float(row[14])
            if score < self.score_threshold:
                continue
            box = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            points = row[4:14].reshape(5, 2)
            out.append(Detection(box=box, score=score, landmarks=points, index=index))
        return out


def largest(detections: Sequence[Detection]) -> Optional[Detection]:
    return max(detections, key=lambda d: d.area) if detections else None


# -- onnx plumbing ---------------------------------------------------------
def default_providers(prefer_gpu: bool = True) -> list:
    """Providers for every model session, pinned to the NVIDIA adapter.

    DirectML would otherwise take adapter 0, which is the discrete card on most
    machines and not on all of them. ``gpu.directml_providers`` names the adapter
    explicitly and returns CPU-only when there is no NVIDIA card, which the local
    engine refuses (see ``LocalEngine._setup``).
    """
    if prefer_gpu:
        return gpu.directml_providers()
    return ["CPUExecutionProvider"]


def _session(model_path: str | Path, providers: Optional[list[str]] = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The pipeline already has capture and output threads; keep the inference
    # threads from fighting them for cores.
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers or default_providers(),
    )


# -- identity --------------------------------------------------------------
class FaceEmbedder:
    """ArcFace: a face crop becomes a 512-d unit vector."""

    def __init__(self, model_path: str | Path, providers: Optional[list[str]] = None) -> None:
        self.session = _session(model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[3] or 112)

    def embedding(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        aligned = crop(frame_bgr, norm_matrix(landmarks, self.size), self.size)
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True
        )
        vector = self.session.run(None, {self.input_name: blob})[0].ravel().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


# -- swap ------------------------------------------------------------------
# How much of the live mouth to carry over. A full copy hands the speaker's own
# lips to ArcFace, which reads the mouth as identity: on the motion harness
# (tests/verify_motion.py) similarity to the character fell from 0.767 to 0.491
# at 1.0 - below the 0.50 the engine requires - and 0.5 holds at 0.602 on a
# still and 0.533 with the head moving, while still carrying the articulation.
MOUTH_BLEND = 0.50

# Eyes carry expression too: a blink from the render is a blink that never
# happened. They also carry more identity than the lips - at 0.50 the motion
# harness fell to 0.460 while panning, below the 0.50 the engine requires,
# because ArcFace aligns on the eyes. 0.30 holds at 0.511 and still darkens the
# lids enough for a blink to read.
EYE_BLEND = 0.30


def mouth_alpha(size: int, landmarks: np.ndarray,
                matrix: np.ndarray) -> Optional[np.ndarray]:
    """Soft mask over the lips, in the same 128px crop space as the texture.

    inswapper renders the whole face from the character photo, mouth included.
    With five landmarks and no mouth tracking, the swapped lips sit in roughly
    the right place but stop matching what is being said, which is the first
    thing anyone notices on a call. Copying the live crop's mouth back over the
    render restores articulation the network never saw.

    The mouth is the one feature the five landmarks pin down exactly, which is
    why this works without a 68- or 106-point model: the two corners give the
    centre, the width and the roll of the head.
    """
    points = cv2.transform(
        np.asarray(landmarks, np.float32).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)
    if points.shape[0] < 5:
        return None
    left, right = points[3], points[4]
    width = float(np.hypot(*(right - left)))
    if not np.isfinite(width) or width < size * 0.08:
        return None
    centre = (left + right) * 0.5
    # The corners sit on the lip line; the lips themselves hang below it.
    centre[1] += width * 0.10
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    axes = (max(2, int(width * 0.58)), max(2, int(width * 0.44)))
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(
        mask, (int(round(centre[0])), int(round(centre[1]))), axes, angle, 0, 360, 255, -1
    )
    k = max(3, int(width * 0.30) | 1)
    soft = cv2.GaussianBlur(mask, (k, k), 0)
    return cv2.convertScaleAbs(soft, alpha=MOUTH_BLEND)


def eye_alpha(size: int, landmarks: np.ndarray,
              matrix: np.ndarray) -> Optional[np.ndarray]:
    """Soft mask over both eyes, in crop space.

    The two eye landmarks give the interocular span, which sets the size and the
    tilt of both patches - so a head roll moves the eyes with the mouth instead
    of smearing them. Kept inside the lids: cheeks and brows stay rendered.
    """
    points = cv2.transform(
        np.asarray(landmarks, np.float32).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)
    if points.shape[0] < 5:
        return None
    left, right = points[0], points[1]
    span = float(np.hypot(*(right - left)))
    if not np.isfinite(span) or span < size * 0.10:
        return None
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    axes = (max(2, int(span * 0.24)), max(2, int(span * 0.17)))
    mask = np.zeros((size, size), np.uint8)
    for eye in (left, right):
        cv2.ellipse(
            mask, (int(round(eye[0])), int(round(eye[1]))), axes, angle, 0, 360, 255, -1
        )
    k = max(3, int(span * 0.16) | 1)
    soft = cv2.GaussianBlur(mask, (k, k), 0)
    return cv2.convertScaleAbs(soft, alpha=EYE_BLEND)


def expression_alpha(size: int, landmarks: np.ndarray,
                     matrix: np.ndarray) -> Optional[np.ndarray]:
    """Everything that moves when you speak: both eyes and the mouth."""
    parts = [
        alpha
        for alpha in (
            eye_alpha(size, landmarks, matrix),
            mouth_alpha(size, landmarks, matrix),
        )
        if alpha is not None
    ]
    if not parts:
        return None
    combined = parts[0]
    for extra in parts[1:]:
        combined = cv2.max(combined, extra)
    return combined


def blend_alpha(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Put ``top`` over ``base`` where the uint8 ``alpha`` is opaque."""
    weight = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    front = cv2.multiply(top, weight, scale=1.0 / 255.0)
    back = cv2.multiply(base, cv2.bitwise_not(weight), scale=1.0 / 255.0)
    return cv2.add(front, back)


class FaceSwapper:
    """The 128x128 identity swap network.

    ``emap`` ships inside the .onnx file as its last initialiser: it projects an
    ArcFace embedding into the swap network's latent space. Extracting it here
    is what lets us avoid depending on the InsightFace Python package, which
    needs a C compiler on Windows.
    """

    def __init__(self, model_path: str | Path, providers: Optional[list[str]] = None,
                 keep_expression: bool = True) -> None:
        self.model_path = str(model_path)
        self.session = _session(self.model_path, providers)
        # See expression_alpha: lips and eyes come from the live frame.
        self.keep_expression = bool(keep_expression)
        self.input_names = [item.name for item in self.session.get_inputs()]
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[2] or 128)
        self.emap = self._load_emap()

        mask = self._oval_mask(self.size)
        self._mask = mask

    def _load_emap(self) -> np.ndarray:
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:  # pragma: no cover - onnx is a hard dependency
            raise RuntimeError(
                "The 'onnx' package is required to read the swap model. "
                "Install it with: pip install onnx"
            ) from exc
        model = onnx.load(self.model_path, load_external_data=False)
        initializers = model.graph.initializer
        if not initializers:
            raise RuntimeError("swap model has no initialisers; file looks corrupt")
        # fp32 keeps emap last; the fp16 conversion can append cast scalars, so
        # look it up by name first and fall back to the documented position.
        by_name = {init.name: init for init in initializers}
        chosen = by_name.get("emap") or initializers[-1]
        emap = numpy_helper.to_array(chosen).astype(np.float32)
        if emap.ndim != 2 or emap.shape[0] != emap.shape[1]:
            raise RuntimeError(f"unexpected emap shape {emap.shape}")
        return emap

    @staticmethod
    def _oval_mask(size: int) -> np.ndarray:
        """Soft face-shaped mask in crop space, kept off the crop border.

        A plain square mask (what InsightFace itself blends with) leaves visible
        corners on the jaw and forehead at webcam resolution.
        """
        mask = np.zeros((size, size), dtype=np.float32)
        centre = (size // 2, int(size * 0.52))
        # Reach the hairline and the jaw. A mask that stops at the cheekbones
        # leaves the original jaw visible underneath - the "floating face".
        axes = (int(size * 0.39), int(size * 0.47))
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
        k = max(5, (size // 7) | 1)
        return cv2.GaussianBlur(mask, (k, k), 0)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        projected = vector @ self.emap
        norm = float(np.linalg.norm(projected))
        if norm:
            projected = projected / norm
        return projected.astype(np.float32)

    def render(self, frame_bgr: np.ndarray, landmarks: np.ndarray,
               embedding: np.ndarray) -> np.ndarray:
        """The swapped face as a 128x128 BGR texture, in crop coordinates."""
        matrix = norm_matrix(landmarks, self.size)
        aligned = crop(frame_bgr, matrix, self.size)
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 255.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True
        )
        outputs = self.session.run(
            None, {self.input_names[0]: blob, self.input_names[1]: self.latent(embedding)}
        )
        rendered = outputs[0][0].transpose(1, 2, 0)
        texture = np.clip(255.0 * rendered, 0, 255).astype(np.uint8)[:, :, ::-1]
        # Re-light the render to the live crop's colour statistics: the network
        # normalises its output, which reads as a bright sticker in a dim room.
        return match_tone(texture, aligned)

    def paste(self, frame_bgr: np.ndarray, texture: np.ndarray,
              landmarks: np.ndarray) -> np.ndarray:
        """Glue a rendered texture onto this frame using *current* landmarks.

        Separating render from paste is what keeps the face moving at frame
        rate while the network refreshes the detail a few times a second. The
        mouth is the exception: it is taken from this frame every time, because
        a texture rendered a third of a second ago is already the wrong shape.
        """
        matrix = norm_matrix(landmarks, self.size)
        if self.keep_expression:
            alpha = expression_alpha(self.size, landmarks, matrix)
            if alpha is not None:
                live = crop(frame_bgr, matrix, self.size)
                texture = blend_alpha(texture, live, alpha)
        return self._composite(frame_bgr, texture, matrix)

    def swap(self, frame_bgr: np.ndarray, landmarks: np.ndarray,
             embedding: np.ndarray) -> np.ndarray:
        matrix = norm_matrix(landmarks, self.size)
        aligned = crop(frame_bgr, matrix, self.size)
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 255.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True
        )
        outputs = self.session.run(
            None, {self.input_names[0]: blob, self.input_names[1]: self.latent(embedding)}
        )
        rendered = outputs[0][0].transpose(1, 2, 0)
        fake = np.clip(255.0 * rendered, 0, 255).astype(np.uint8)[:, :, ::-1]
        # Goes through paste() so a one-shot swap keeps the mouth too.
        return self.paste(frame_bgr, fake, landmarks)

    def _composite(self, frame_bgr: np.ndarray, texture: np.ndarray,
                   matrix: np.ndarray) -> np.ndarray:
        return composite(frame_bgr, texture, self._mask, matrix)


# -- helpers ---------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, tolerating non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def identity_from_photo(image_bgr: np.ndarray, detector: FaceDetector,
                        embedder: FaceEmbedder) -> np.ndarray:
    """The 512-d embedding of the largest face in a reference photo."""
    face = largest(detector.detect(image_bgr))
    if face is None:
        raise ValueError(
            "No face found in that image. Use a clear, front-facing photo of "
            "the character's face."
        )
    return embedder.embedding(image_bgr, face.landmarks)
