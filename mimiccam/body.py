"""Body and arm pass: make the visible skin belong to the new face.

A 128 px face network only owns the face. The neck, chest and arms keep the
original person's skin, which is the giveaway that something is pasted on. This
module finds skin pixels below the face box (neck, chest, hands, forearms) and
moves their chroma and brightness toward the tone of the *swapped* face, so the
body reads as one person.

It deliberately does not touch clothing: recolouring a shirt without knowing
where the shirt ends produces blotches. That is a generative job, not a colour
transform, and the honest place for it is Lucy mode.

Colour maths is in CIELAB (perceptually uniform), the mask is YCrCb skin
detection (cheap and stable on webcam footage), and both are feathered so the
transition at the jaw and sleeve edges has no visible seam.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# YCrCb skin bounds that hold up across skin tones on webcam video.
SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)

TORSO_WIDTHS = 3.6      # torso window, as a multiple of the face width
TORSO_HEIGHTS = 3.0     # how far below the face box to look
FACE_PAD = 0.18         # ignore skin inside the (slightly grown) face box


def _torso_window(shape: tuple[int, int], face_box: tuple[int, int, int, int]):
    height, width = shape
    x, y, w, h = face_box
    cx = x + w // 2
    half = int(w * TORSO_WIDTHS / 2)
    top = max(0, y + h // 2)
    bottom = min(height, int(y + h * (1.0 + TORSO_HEIGHTS)))
    left = max(0, cx - half)
    right = min(width, cx + half)
    return left, top, right, bottom


MASK_SCALE = 0.5  # skin is a smooth region; half resolution is plenty
WORK_SCALE = 4    # grid the body correction is computed on (quarter of the frame)


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def skin_mask(bgr: np.ndarray, scale: float = MASK_SCALE) -> np.ndarray:
    """Soft 0..1 skin probability, computed at half resolution.

    The mask only ever modulates a colour gradient, so the extra precision of a
    full-resolution inRange + morphology buys nothing and costs ~3 ms a frame.
    ``scale`` downscales the input first; callers that already work on a coarse
    grid pass 1.0 so the detector keeps its resolution without an extra resize.
    """
    height, width = bgr.shape[:2]
    if scale == 1.0:
        small = bgr
    else:
        small = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32) / 255.0
    return _resize(soft, (width, height))


def body_mask(bgr: np.ndarray, face_box: tuple[int, int, int, int],
              scale: float = MASK_SCALE, window_div: int = 4) -> np.ndarray:
    """Skin below/around the face: neck, chest, arms and hands, not the face."""
    height, width = bgr.shape[:2]
    mask = skin_mask(bgr, scale=scale)

    x, y, w, h = face_box
    pad_x, pad_y = int(w * FACE_PAD), int(h * FACE_PAD)
    fx0, fy0 = max(0, x - pad_x), max(0, y - pad_y)
    fx1, fy1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    mask[fy0:fy1, fx0:fx1] = 0.0

    # Build the torso window small and blur it small: a 9 px feather on a
    # quarter-size grid is a 36 px gradient on the frame, for a tenth of the cost.
    sw, sh = max(1, width // window_div), max(1, height // window_div)
    window = np.zeros((sh, sw), dtype=np.float32)
    left, top, right, bottom = _torso_window((height, width), (x, y, w, h))
    window[top // window_div:max(top // window_div + 1, bottom // window_div),
           left // window_div:max(left // window_div + 1, right // window_div)] = 1.0
    window = cv2.GaussianBlur(window, (0, 0), 3.0)
    return mask * _resize(window, (width, height))


def mean_lab(bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Average CIELAB colour of a region, or None when the region is empty."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None:
        return lab.reshape(-1, 3).mean(axis=0)
    weights = mask.reshape(-1)
    total = float(weights.sum())
    if total < 64.0:  # a handful of pixels is noise, not a skin tone
        return None
    flat = lab.reshape(-1, 3)
    return (flat * weights[:, None]).sum(axis=0) / total


def face_skin_tone(bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """The tone of the face we just pasted - the target for the body."""
    x, y, w, h = face_box
    height, width = bgr.shape[:2]
    inset_x, inset_y = int(w * 0.2), int(h * 0.2)
    patch = bgr[max(0, y + inset_y):min(height, y + h - inset_y),
                max(0, x + inset_x):min(width, x + w - inset_x)]
    if patch.size == 0:
        return None
    # A mean is insensitive to resolution: halving the patch turns a 1.6 ms
    # conversion into 0.4 ms and lands on the same tone to a fraction of a level.
    return mean_lab(cv2.resize(patch, (max(1, patch.shape[1] // 2), max(1, patch.shape[0] // 2)),
                               interpolation=cv2.INTER_AREA))


def harmonize(bgr: np.ndarray, face_box: tuple[int, int, int, int],
              target_lab: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Shift the body's skin colour toward ``target_lab`` (the swapped face).

    The correction is a smooth field, so mask, colour statistics and the LAB
    maths run on a quarter-resolution grid and the result is interpolated back
    up. Half resolution cost 14 ms a frame at 720p; this costs about 4 ms and
    the skin detection keeps its resolution (``scale=1.0``, ``window_div=2``
    put the skin map on the frame/4 grid and the torso feather on frame/8, the
    same two grids the half-resolution version used).
    """
    height, width = bgr.shape[:2]
    left, top, right, bottom = _torso_window((height, width), face_box)
    pad = 12
    left, top = max(0, left - pad), max(0, top - pad)
    right, bottom = min(width, right + pad), min(height, bottom + pad)
    x0, y0 = left // WORK_SCALE, top // WORK_SCALE
    x1, y1 = max(x0 + 2, right // WORK_SCALE), max(y0 + 2, bottom // WORK_SCALE)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return bgr

    roi = bgr[y0 * WORK_SCALE:y1 * WORK_SCALE, x0 * WORK_SCALE:x1 * WORK_SCALE]
    small = cv2.resize(roi, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    box = (face_box[0] // WORK_SCALE - x0, face_box[1] // WORK_SCALE - y0,
           max(1, face_box[2] // WORK_SCALE), max(1, face_box[3] // WORK_SCALE))
    mask = body_mask(small, box, scale=1.0, window_div=2)
    if float(mask.sum()) < 16.0:
        return bgr

    current = mean_lab(small, mask)
    if current is None:
        return bgr

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    blend = (mask * strength)[:, :, None]

    # Chroma: move toward the face's hue/saturation, not a full overwrite, so
    # shading and texture detail in the arm survive.
    delta = np.asarray(target_lab, dtype=np.float32)[1:] - current[1:]
    lab[:, :, 1:] += delta[None, None, :] * blend

    # Lightness: gentle ratio, clamped. A studio-lit reference must not blow out
    # a dim bedroom.
    ratio = float(np.clip(float(target_lab[0]) / max(1.0, float(current[0])), 0.82, 1.22))
    lab[:, :, 0] *= 1.0 + (ratio - 1.0) * blend[:, :, 0]

    corrected = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = bgr.copy()
    out[y0 * WORK_SCALE:y1 * WORK_SCALE, x0 * WORK_SCALE:x1 * WORK_SCALE] = cv2.resize(
        corrected, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return out
