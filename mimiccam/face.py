# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
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

from . import exposure, gpu

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


# The texture is matched to the *skin* of the live crop, not to the whole crop.
# A crop's corners hold background - a bright wall behind a backlit head - and
# averaging that in is how a swap ends up as a pale mask on a dark face: the
# network's own face is pulled toward the wall instead of toward the person.
# This soft ellipse sits over the cheeks and nose, inside the jaw and below the
# eyes, and is identical in the texture and in the frame.
TONE_DX = 0.30
TONE_DY = 0.30
TONE_CY = 0.56
# Lightness may be stretched a little to follow the frame's own contrast. More
# than this and a flat render gains grain it never had, or a lit face goes grey.
TONE_GAIN = (0.85, 1.20)


def tone_patch(size: int) -> np.ndarray:
    """Skin region of an aligned crop, as a boolean mask (cached per size)."""
    cached = _tone_patches.get(size)
    if cached is None:
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, (size // 2, int(size * TONE_CY)),
                    (max(2, int(size * TONE_DX)), max(2, int(size * TONE_DY))),
                    0, 0, 360, 255, -1)
        cached = mask > 0
        _tone_patches[size] = cached
    return cached


_tone_patches: dict[int, np.ndarray] = {}


def match_tone(texture: np.ndarray, reference: np.ndarray,
               strength: float = 0.85) -> np.ndarray:
    """Match a rendered face's colour to the skin of the frame it came from.

    The network re-lights the face it generates, and in a backlit room it hands
    back a studio-bright face. Pasting that unchanged is the "pale mask glued to
    a dark person" look: the face is 20 levels brighter than the neck under it.
    The render is therefore pulled toward the live face's own skin - lightness
    with a clipped gain, chroma by offset only, so a warm lamp or a cool screen
    carries over without turning skin green or pink.
    """
    if texture.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (texture.shape[1], texture.shape[0]))
    size = texture.shape[0]
    patch = tone_patch(size)

    tex_lab = cv2.cvtColor(texture, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    tex_pixels = tex_lab[patch]
    ref_pixels = ref_lab[patch]
    if tex_pixels.shape[0] < 32 or ref_pixels.shape[0] < 32:
        return texture
    tex_mean = tex_pixels.mean(axis=0)
    ref_mean = ref_pixels.mean(axis=0)
    tex_std = tex_pixels.std(axis=0)
    ref_std = ref_pixels.std(axis=0)

    out = tex_lab.copy()
    gain = float(np.clip(ref_std[0] / max(1.0, float(tex_std[0])),
                         TONE_GAIN[0], TONE_GAIN[1]))
    lightness = (tex_lab[:, :, 0] - tex_mean[0]) * gain + ref_mean[0]
    out[:, :, 0] = tex_lab[:, :, 0] + strength * (lightness - tex_lab[:, :, 0])
    for channel in (1, 2):
        shifted = tex_lab[:, :, channel] + (ref_mean[channel] - tex_mean[channel])
        out[:, :, channel] = tex_lab[:, :, channel] + strength * (shifted - tex_lab[:, :, channel])
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def brighten_for_network(aligned: np.ndarray) -> np.ndarray:
    """Lift a dark crop so the swap network can see the face in it.

    The 128 px network returns a near-silhouette when the face it is handed sits
    under about luma 50, so a backlit frame used to be lifted whole - and the
    published frame with it, which is what turns a dim room into a washed-out
    one. The lift belongs to the network's input alone: the render is tone
    matched back to the unlifted crop straight afterwards, so the frame keeps the
    camera's own exposure.
    """
    luma = float(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).mean())
    gamma = exposure.gamma_for_luma(luma, exposure.TARGET_LUMA, exposure.GAMMA_MIN)
    if gamma is None:
        return aligned
    key = round(gamma, 3)
    table = _lift_tables.get(key)
    if table is None:
        table = exposure.lut_for(key)
        if len(_lift_tables) > 64:
            _lift_tables.clear()
        _lift_tables[key] = table
    return cv2.LUT(aligned, table)


_lift_tables: dict[float, np.ndarray] = {}


# The ramp of the face mask, in 0..255 alpha. Between these the paste is
# partially transparent, which is exactly where a tone step between the render
# and the frame reads as a seam around the jaw and the hairline.
SEAM_BAND = (10.0, 245.0)
# A correction larger than this is not a lighting offset but a different face in
# a different place; leave it alone rather than paint a bright halo.
SEAM_LIMIT = 48.0
# The ramp runs along the jaw, and on a backlit frame part of it is the wall the
# camera metered for rather than the person. Measuring there dragged the whole
# face toward the wall: on a real dim frame the paste came out at luma 56 where
# the frame's own face sat at 46, and the face's level rode the wall's: sd 2.1
# levels over a run whose source face moved by 0.4. So the ramp is read only
# where the frame there is plausibly the face's own skin - within this fraction
# of the face's interior, plus a few levels of slack so a flat face is not gated
# away - and the gate is skipped when there is no interior to calibrate against.
SEAM_GATE = (0.55, 1.35)
SEAM_GATE_SLACK = 12
# The offset describes a constant colour step, but the pixels it is read from
# change every frame as the head moves, so a per-frame reading pumps. Measured
# on a moving head in a dim room, the offset's own range was 11 levels against a
# source face that moved by 1.5 - a face that brightens and darkens with the
# head, which is invisible in a 720p still and obvious in a 640x360 tile. The
# lock closes this share of the gap to each new reading and never moves faster
# than SEAM_SLEW levels a frame: ~0.25 s to follow a lamp being switched on at
# 30 fps, and slow enough that a frame's noise cannot show.
SEAM_LOCK = 0.12
SEAM_SLEW = 1.2


def seam_band(target: np.ndarray, alpha: np.ndarray) -> Optional[np.ndarray]:
    """The ramp pixels worth measuring: the blend's edge, minus background.

    The ramp runs along the jaw, and on a backlit frame part of it is the wall
    the camera metered for rather than the person. Reading there dragged the
    whole face toward the wall: on a real dim frame the paste came out at luma 56
    where the frame's own face sat at 46. So the ramp is kept only where the
    frame there is plausibly the face's own skin - within SEAM_GATE of the face's
    interior, plus a few levels of slack - and the gate is skipped when there is
    no interior to calibrate against (a face half off the frame), which leaves
    the old whole-ramp behaviour and its cap in place.
    """
    if alpha.dtype != np.uint8:
        alpha = cv2.convertScaleAbs(alpha, alpha=255.0)
    band = cv2.inRange(alpha, SEAM_BAND[0], SEAM_BAND[1])
    if cv2.countNonZero(band) < 64:
        return None
    interior = (alpha >= 250).astype(np.uint8)
    if cv2.countNonZero(interior) >= 200:
        grey = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        skin = float(cv2.mean(grey, mask=interior)[0])
        if skin > 1.0:
            keep = cv2.inRange(grey, int(max(0.0, skin * SEAM_GATE[0])),
                               int(min(255.0, skin * SEAM_GATE[1] + SEAM_GATE_SLACK)))
            band = cv2.bitwise_and(band, keep)
            if cv2.countNonZero(band) < 64:
                return None
    return band


def seam_offset(warped: np.ndarray, target: np.ndarray,
                alpha: np.ndarray) -> Optional[np.ndarray]:
    """The colour step between a warped face and the frame, in the ramp.

    ``match_tone`` matches the render's statistics to the live crop, but it does
    so in the crop's own coordinates and only partially. What is left over is a
    colour step at the mask boundary: the soft ramp then paints a ring of one
    skin tone fading into another, which is the pale blotch around the jaw in a
    dim room.

    Returns the mean per-channel difference (BGR, float32), or None when the
    ramp holds too few usable pixels to trust - a face half off the frame, or a
    ramp whose pixels are all background.
    """
    # inRange + mean(mask=...) rather than boolean indexing: the fancy-index
    # version allocates a float copy of the whole ROI and cost 8.7 ms a paste,
    # which is a quarter of the frame budget for a colour offset.
    band = seam_band(target, alpha)
    if band is None:
        return None
    difference = cv2.subtract(target, warped, dtype=cv2.CV_16S)
    values = cv2.mean(difference, mask=band)[:3]
    offset = np.array(values, dtype=np.float32)
    if not np.all(np.isfinite(offset)):
        return None
    return np.clip(offset, -SEAM_LIMIT, SEAM_LIMIT)


def apply_offset(texture: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Move a whole texture by a constant colour offset."""
    shifted = texture.astype(np.int16) + np.rint(offset).astype(np.int16)
    return np.clip(shifted, 0, 255).astype(np.uint8)





class ToneLock:
    """Holds the paste's colour offset steady from one frame to the next.

    The seam offset is a measurement of a constant, and it is read every frame
    from a ramp that slides over the jaw as the head moves. Left raw, the face's
    brightness follows that noise at 30 Hz. This averages it with a bounded
    speed, so a real change of light is followed in a quarter of a second and a
    single frame's reading is not a visible step.

    One lock belongs to one swapper: it is the state of the face being pasted,
    so a new character photo or a face that left the frame starts it over.
    """

    def __init__(self, rate: float = SEAM_LOCK, slew: float = SEAM_SLEW) -> None:
        self.rate = float(rate)
        self.slew = float(slew)
        self._value: Optional[np.ndarray] = None

    def smooth(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float32).reshape(-1)
        if self._value is None or self._value.shape != measured.shape:
            self._value = measured.copy()
            return self._value
        step = np.clip(self.rate * (measured - self._value), -self.slew, self.slew)
        self._value = self._value + step
        return self._value

    def reset(self) -> None:
        self._value = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._value is None else self._value.copy()


def seam_shift(warped: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Offset a warped face so its edge sits at the frame's own colour.

    The one-shot form: measure the step in the ramp and add it back to the whole
    texture. At the ramp the two then agree, so the boundary disappears; the
    interior keeps every bit of its structure because a constant offset moves
    the whole face, not its detail. The engine pastes through a ``ToneLock``
    instead, so the offset cannot pump from frame to frame.

    Returns the texture unchanged when the ramp cannot be trusted or when the
    difference is smaller than a level.
    """
    offset = seam_offset(warped, target, alpha)
    if offset is None or float(np.abs(offset).max()) < 1.0:
        return warped
    return apply_offset(warped, offset)


def composite(frame_bgr: np.ndarray, texture: np.ndarray, mask: np.ndarray,
              matrix: np.ndarray, lock: Optional["ToneLock"] = None) -> np.ndarray:
    """Warp a face texture back onto the frame, inside its own bounding box.

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
    offset = seam_offset(warped, frame_bgr[y0:y1, x0:x1], alpha)
    if offset is not None:
        if lock is not None:
            offset = lock.smooth(offset)
        if float(np.abs(offset).max()) >= 1.0:
            warped = apply_offset(warped, offset)
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


def _session(model_path: str | Path, providers: Optional[list[str]] = None,
             optimize: str = "all"):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if optimize == "basic"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
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
# How stale an occlusion reading may be before XSeg is asked again. The pass
# runs on the render thread (see FaceSwapper._refresh_occlusion) at ~20 ms a
# call, so it is refreshed once per render at most: at 0.12 s a hand arriving
# is caught within about three frames.
OCCLUSION_MAX_AGE = 0.12

MOUTH_BLEND = 0.50

# Both weights sit on a measured cliff, and the detail pass moved the cliff.
# Averaging two different mouths peaks in the middle: with the 512 px render the
# Laplacian variance inside the mouth window is 264 with no blend, 135 at 0.50,
# 149 at 0.75 and 214 at 1.00, so the lips are the softest patch on the face at
# exactly the weight articulation wants. Raising it is visibly cleaner - 0.75
# reads as a real mouth - but the motion harness then measures 0.481 (0.65) and
# 0.459 (0.70) against the character, under the 0.50 the engine requires: at
# 1.00 similarity collapses to 0.491 because ArcFace reads the lips as identity.
# So 0.50 is a ceiling, not a preference, and it is left alone.
#
# Eyes are the opposite: the same harness passes at anything from 0.15 to 0.50,
# so the only reason for the weight is that a blink has to read. With the sharp
# render a permanent live-eye mix is a permanent smear - variance 803 with no
# blend, 677 at 0.15, 571 at 0.30, 493 at 0.45 - for a lid that is shut about one
# frame in a hundred. 0.22 keeps the blink and gives back a tenth of the detail.
EYE_BLEND = 0.22


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
    """Put ``top`` over ``base`` where the uint8 ``alpha`` is opaque.

    Blended inside the mask's own bounding box only: the lips and eyes cover a
    few per cent of the texture, so mixing whole 512 px planes to change that
    few per cent is a millisecond a frame spent on pixels the mask zeroes.
    """
    rows = np.flatnonzero(alpha.max(axis=1))
    cols = np.flatnonzero(alpha.max(axis=0))
    if rows.size == 0 or cols.size == 0:
        return base
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    out = base.copy()
    region = out[y0:y1, x0:x1]
    weight = cv2.cvtColor(alpha[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    front = cv2.multiply(top[y0:y1, x0:x1], weight, scale=1.0 / 255.0)
    back = cv2.multiply(region, cv2.bitwise_not(weight), scale=1.0 / 255.0)
    cv2.add(front, back, dst=region)
    return out


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
        # The graph is read before the session is built, for two reasons: the
        # emap lives in its initialisers, and an fp16 graph needs a different
        # DirectML optimiser setting - see below.
        graph = self._read_graph()
        self._fp16 = self._graph_is_fp16(graph)
        chosen = providers or default_providers()
        # Measured on this machine: DirectML runs the fused fp16 form of
        # HyperSwap incorrectly - the output ignores the source identity
        # (cosine 0.164 to the character, where CPU scores 0.769). The basic
        # optimiser does not fuse it that way and scores 0.548 in 20 ms instead
        # of 0.164 in 25 ms, so it is both correct and faster. The 128 px fp32
        # network is unaffected and keeps the full optimiser.
        self.session = _session(
            self.model_path, chosen,
            optimize="basic" if (self._fp16 and any("Dml" in p for p in chosen)) else "all",
        )
        # See expression_alpha: lips and eyes come from the live frame.
        self.keep_expression = bool(keep_expression)
        inputs = self.session.get_inputs()
        self.input_names = [item.name for item in inputs]
        # Architectures differ in input *order*: inswapper takes (target image,
        # latent), while HyperSwap's export takes (source embedding, target
        # image) - so pick by rank, not by position.
        image_input = next((item for item in inputs if len(item.shape) == 4), None)
        latent_input = next((item for item in inputs if len(item.shape) == 2), None)
        if image_input is None or latent_input is None:
            raise RuntimeError(
                f"{self.model_path} does not look like a swap network: "
                f"inputs are {[(i.name, i.shape) for i in inputs]}"
            )
        self.image_input = image_input.name
        self.latent_input = latent_input.name
        self.size = int(image_input.shape[2] or 128)
        self.emap = self._emap_from(graph)
        # HyperSwap normalises to mean/std 0.5 and ends in tanh; the inswapper
        # family takes 0..1 and ends near 0..1. Named rather than sniffed, since
        # a wrong constant here reads as a washed-out or clipped face.
        name = Path(self.model_path).name.lower()
        self.mean, self.std = (0.5, 0.5) if "hyperswap" in name else (0.0, 1.0)
        self.mask_output = next(
            (item.name for item in self.session.get_outputs()
             if len(item.shape) == 4 and int(item.shape[1] or 0) == 1),
            None,
        )
        # Decided from the first render's own range: HyperSwap/FluxRT export
        # through tanh, inswapper does not, and guessing wrong costs a frame.
        self._tanh: Optional[bool] = None
        self._model_mask: Optional[np.ndarray] = None
        self.occluder = None
        self._occlusion: Optional[np.ndarray] = None
        self._occlusion_ts = 0.0

        mask = self._oval_mask(self.size)
        self._mask = mask
        # Masks for detail-enhanced textures, which are pasted larger than the
        # network's own 128 px render.
        self._masks: dict[int, np.ndarray] = {self.size: mask}
        # The paste's colour offset, kept steady across frames (see ToneLock).
        self._tone_lock = ToneLock()

    def _read_graph(self):
        try:
            import onnx
        except ImportError as exc:  # pragma: no cover - onnx is a hard dependency
            raise RuntimeError(
                "The 'onnx' package is required to read the swap model. "
                "Install it with: pip install onnx"
            ) from exc
        return onnx.load(self.model_path, load_external_data=False)

    @staticmethod
    def _graph_is_fp16(graph) -> bool:
        """True when the weights are float16, which DirectML fuses badly."""
        try:
            from onnx import TensorProto
        except ImportError:  # pragma: no cover
            return False
        if graph is None:
            return False
        return any(
            init.data_type == TensorProto.FLOAT16
            for init in graph.graph.initializer[:64]
        )

    def _emap_from(self, graph) -> Optional[np.ndarray]:
        if graph is None:
            return None
        from onnx import numpy_helper

        initializers = graph.graph.initializer
        if not initializers:
            raise RuntimeError("swap model has no initialisers; file looks corrupt")
        # fp32 keeps emap last; the fp16 conversion can append cast scalars, so
        # look it up by name first. HyperSwap and ReSwapper consume the ArcFace
        # vector directly and carry no emap at all - that is not an error.
        by_name = {init.name: init for init in initializers}
        chosen = by_name.get("emap")
        if chosen is None:
            squares = [
                init for init in initializers
                if len(init.dims) == 2 and init.dims[0] == init.dims[1] == 512
            ]
            chosen = squares[0] if len(squares) == 1 else None
        if chosen is None:
            return None
        emap = numpy_helper.to_array(chosen).astype(np.float32)
        if emap.ndim != 2 or emap.shape[0] != emap.shape[1]:
            return None
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
        if self.emap is None:
            # HyperSwap takes the identity vector itself, unit-normed
            # ("embedding_norm" in FaceFusion's model metadata).
            norm = float(np.linalg.norm(vector))
            if norm:
                vector = vector / norm
            return vector.astype(np.float32)
        projected = vector @ self.emap
        norm = float(np.linalg.norm(projected))
        if norm:
            projected = projected / norm
        return projected.astype(np.float32)

    def render(self, frame_bgr: np.ndarray, landmarks: np.ndarray,
               embedding: np.ndarray) -> np.ndarray:
        """The swapped face as a BGR texture, in crop coordinates.

        The crop handed to the network is brightened when the room is dim - the
        network needs to see a face to render one - but the texture comes back
        matched to the *unlifted* crop, so the pasted face sits in the room's own
        light instead of glowing out of it.
        """
        matrix = norm_matrix(landmarks, self.size)
        aligned = crop(frame_bgr, matrix, self.size)
        seen = brighten_for_network(aligned)
        scaled = seen.astype(np.float32) / 255.0
        if self.mean or self.std != 1.0:
            scaled = (scaled - self.mean) / self.std
        # swapRB: the graphs were trained on RGB, the frame is BGR.
        blob = np.ascontiguousarray(scaled[:, :, ::-1].transpose(2, 0, 1)[None])
        outputs = self.session.run(
            None, {self.image_input: blob, self.latent_input: self.latent(embedding)}
        )
        rendered = outputs[0][0].transpose(1, 2, 0)
        if self._tanh is None:
            self._tanh = bool(rendered.min() < -0.02)
        if self._tanh:
            rendered = (rendered + 1.0) * 127.5
        else:
            rendered = rendered * 255.0
        texture = np.clip(rendered, 0, 255).astype(np.uint8)[:, :, ::-1]
        self._keep_model_mask(outputs)
        self._refresh_occlusion(aligned)
        return match_tone(texture, aligned)

    def mask_for(self, size: int) -> np.ndarray:
        """The face-shaped mask for a texture of ``size`` px."""
        size = int(size)
        cached = self._masks.get(size)
        if cached is None:
            cached = self._oval_mask(size)
            self._masks[size] = cached
        return cached

    def matrix_for(self, landmarks: np.ndarray, size: int) -> np.ndarray:
        """The frame->crop transform for a texture of ``size`` px.

        The 128 px template is what the network renders into; a detail pass
        returns a larger texture of the same crop, so its transform is that one
        scaled up. ``norm_matrix(landmarks, size)`` cannot do this job: it only
        agrees with the 128 template for multiples of 112.
        """
        matrix = norm_matrix(landmarks, self.size)
        if int(size) == self.size:
            return matrix
        scale = float(size) / float(self.size)
        zoom = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], dtype=np.float32)
        return zoom @ np.vstack([matrix, [0.0, 0.0, 1.0]])

    def paste(self, frame_bgr: np.ndarray, texture: np.ndarray,
              landmarks: np.ndarray) -> np.ndarray:
        """Glue a rendered texture onto this frame using *current* landmarks.

        Separating render from paste is what keeps the face moving at frame
        rate while the network refreshes the detail a few times a second. The
        mouth is the exception: it is taken from this frame every time, because
        a texture rendered a third of a second ago is already the wrong shape.

        The texture may be the network's own 128 px render or a larger one from
        the detail pass; the transform, the live crop and the expression masks
        all follow its size, so both land on the same pixels.
        """
        size = int(texture.shape[0])
        matrix = self.matrix_for(landmarks, size)
        # One crop per paste, shared by the expression blend and the occluder.
        live = crop(frame_bgr, matrix, size)
        if self.keep_expression:
            # Built in the network's own crop space, which is where the mask
            # geometry was tuned, then scaled up: a soft oval over the lips does
            # not gain anything from 512 px of it.
            alpha = expression_alpha(self.size, landmarks, norm_matrix(landmarks, self.size))
            if alpha is not None:
                if size != self.size:
                    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_LINEAR)
                texture = blend_alpha(texture, live, alpha)
        return self._composite(frame_bgr, texture, matrix, self.paste_mask(size, live))

    def swap(self, frame_bgr: np.ndarray, landmarks: np.ndarray,
             embedding: np.ndarray) -> np.ndarray:
        # Goes through render() and paste() so a one-shot swap gets the same
        # exposure handling and keeps the mouth too.
        return self.paste(frame_bgr, self.render(frame_bgr, landmarks, embedding), landmarks)

    @property
    def tone_lock(self) -> ToneLock:
        """The paste's colour lock, created on first use.

        A swapper can be built without ``__init__`` - the paste tests do that to
        skip loading a model - so the lock cannot assume it was set there.
        """
        lock = self.__dict__.get("_tone_lock")
        if lock is None:
            lock = ToneLock()
            self._tone_lock = lock
        return lock

    def reset_tone(self) -> None:
        """Forget the locked colour offset: a new face starts from a fresh read."""
        lock = self.__dict__.get("_tone_lock")
        if lock is not None:
            lock.reset()

    def _keep_model_mask(self, outputs: list) -> None:
        """Store the swap network's own face mask, when it exports one.

        HyperSwap returns a second tensor: the network's own idea of which
        pixels are the face it just drew. It follows the hairline and the jaw
        better than any ellipse, so it replaces the oval wherever it exists -
        the oval is still multiplied in, because the network's mask reaches past
        the jaw on a turned head and that is where a paste shows.
        """
        name = self.__dict__.get("mask_output")
        if not name:
            return
        index = next(
            (i for i, item in enumerate(self.session.get_outputs()) if item.name == name),
            None,
        )
        if index is None or index >= len(outputs):
            return
        mask = np.squeeze(np.asarray(outputs[index])).astype(np.float32)
        if mask.ndim != 2:
            return
        if mask.shape != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        self._model_mask = np.clip(mask, 0.0, 1.0)

    def _refresh_occlusion(self, aligned: np.ndarray) -> None:
        """Read XSeg on the render thread, where the swap graph already runs.

        Running it from the paste thread crashed the engine: two DirectML
        sessions executing at once from different threads took the process down
        (intermittently between runs, and the app's own selftest died in 2 of 2
        attempts once the mask was attached). The CPU is not an alternative -
        measured 630 ms a pass against 20 ms on the GPU - so the pass moved to
        the thread that already renders, and the paste warps the cached result.

        The face is rendered far less often than it is pasted, so the reading is
        reused for a few frames (OCCLUSION_MAX_AGE): a hand that arrives is
        caught at the next render, within ~0.1 s.
        """
        occluder = self.__dict__.get("occluder")
        if occluder is None:
            return
        if (time.perf_counter() - self.__dict__.get("_occlusion_ts", 0.0)) <= OCCLUSION_MAX_AGE:
            return
        try:
            mask = occluder.protection(aligned)
        except Exception:
            return
        self._occlusion = mask
        self._occlusion_ts = time.perf_counter()

    def occlusion_mask(self, live: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """The cached XSeg reading. Read on the render thread; never run here.

        ``live`` is accepted so the paste keeps one call signature, but it is
        deliberately not used: the mask belongs to the render's own crop.
        """
        if self.__dict__.get("occluder") is None:
            return None
        return self.__dict__.get("_occlusion")

    def paste_mask(self, size: int, live: Optional[np.ndarray] = None) -> np.ndarray:
        """The blend mask for a paste: face shape x occlusion.

        Three sources combine here, in the order they should: the network's own
        mask if it has one (jaw and hairline), the oval (a hard ceiling on how
        far the paste may reach), and XSeg (hands, hair, a microphone in front of
        the face). Any one of them can veto a pixel.
        """
        size = int(size)
        base = self.mask_for(size)
        model_mask = self.__dict__.get("_model_mask")
        if model_mask is not None:
            if model_mask.shape != (size, size):
                model_mask = cv2.resize(model_mask, (size, size),
                                        interpolation=cv2.INTER_LINEAR)
            base = base * model_mask
        if live is not None:
            protect = self.occlusion_mask(live)
            if protect is not None and protect.size:
                if protect.shape != (size, size):
                    protect = cv2.resize(protect, (size, size),
                                         interpolation=cv2.INTER_LINEAR)
                base = base * protect
        return base

    def _composite(self, frame_bgr: np.ndarray, texture: np.ndarray,
                   matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            mask = self.mask_for(texture.shape[0])
        return composite(frame_bgr, texture, mask, matrix, lock=self.tone_lock)


class FaceEnhancer:
    """GFPGAN v1.4 over the swapped 128 px texture.

    inswapper renders 128 px, which is soft once it is pasted at webcam scale.
    GFPGAN rebuilds eyes, lashes, beard strands and skin texture from that crop
    without inventing a different face, because the crop already carries the
    identity it is restoring.

    The ONNX export takes a fixed 512x512 **BGR** tensor in ``[-1, 1]``. RGB was
    tried first and turned skin orange; the channels are deliberately untouched.
    Measured 44 ms a call on a 4070 laptop through DirectML, so it runs on the
    render cadence rather than on every pasted frame.
    """

    SIZE = 512
    # How much of the enhanced texture survives the identity blend. GFPGAN
    # rebuilds eyes, lashes and skin texture, but it also drifts the face away
    # from the identity the swap network rendered: measured on a real frame
    # against the character photo, the pasted face scored **0.637** identity at
    # 1.0, **0.677** at 0.7 and 0.692 at 0.55. 0.7 keeps nearly all of the
    # detail and gives a quarter of the identity back.
    KEEP = 0.70

    def __init__(self, model_path: str | Path, providers: Optional[list[str]] = None,
                 session=None) -> None:
        self.session = session if session is not None else _session(str(model_path), providers)
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def prepare(cls, texture_bgr: np.ndarray) -> np.ndarray:
        """BGR swap crop -> the NCHW float blob GFPGAN expects."""
        big = cv2.resize(texture_bgr, (cls.SIZE, cls.SIZE), interpolation=cv2.INTER_CUBIC)
        scaled = big.astype(np.float32) / 255.0
        scaled = (scaled - 0.5) / 0.5
        return np.ascontiguousarray(scaled.transpose(2, 0, 1)[None])

    @classmethod
    def restore(cls, output: np.ndarray) -> np.ndarray:
        """Network output -> a BGR texture at the enhancer's own resolution.

        Deliberately *not* shrunk back to 128: shrinking throws away every
        pixel GFPGAN just invented, and the paste can use the larger texture
        directly (``FaceSwapper.paste`` scales its transform to the texture).
        """
        return np.clip((output[0].transpose(1, 2, 0) + 1.0) * 127.5,
                       0, 255).astype(np.uint8)

    def enhance(self, texture_bgr: np.ndarray) -> np.ndarray:
        """Return a sharper, larger texture, or the input if the graph misbehaves."""
        blob = self.prepare(texture_bgr)
        output = self.session.run(None, {self.input_name: blob})[0]
        texture = self.restore(output)
        if texture.shape[:2] != (self.SIZE, self.SIZE):
            return texture_bgr
        if float(texture.std()) < 1.0:
            # A graph that returns a flat plane has failed, not enhanced.
            return texture_bgr
        # Keep a share of the network's own render under the enhanced one. The
        # detail pass sharpens the eyes and skin but drifts identity; a 0.7/0.3
        # mix measured 0.677 against the character photo where the pure
        # enhanced texture measured 0.637, and at webcam scale the difference in
        # sharpness is not visible.
        base = cv2.resize(texture_bgr, (self.SIZE, self.SIZE),
                          interpolation=cv2.INTER_CUBIC)
        return cv2.addWeighted(texture, self.KEEP, base, 1.0 - self.KEEP, 0.0)


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
