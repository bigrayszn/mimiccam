# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Hair: the character's head shape travels with the swapped face.

The 128 px swap network only owns the face inside InsightFace's crop, so the
hairline, temples and the space above the forehead keep the original person's
hair. The face then reads as a mask sitting on someone else's head - the swap is
right in the middle and wrong at the edges.

This pass closes that gap without a second network. The character photo is
detected once at setup; from its five landmarks we take a box around the whole
head, mark the pixels that are not skin and sit above the eyes (the hair), and
warp that patch onto the live frame with the same similarity transform that
aligns the two faces. The result is drawn *under* the swapped face, so it frames
the jaw and hairline instead of fighting it.

Everything is keyed off the eye distance, so it scales with the head, works at
any webcam distance and costs about a millisecond: a 200 px patch warp, not a
full-frame one.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import body, face

# Head box, in multiples of the eye separation.
HALF_WIDTH = 1.30
TOP = 1.95          # above the eye line: forehead plus the crown of the hair
BOTTOM = 0.30       # below it: keeps the mask off the eyes and eyebrows    # Head ellipse, same units, centred above the eyes. Wide and tall enough to
# reach the temples and the sides above the ears: clipping the hair at the
# crown is what leaves the live person's hair visible around a swapped face.
ELLIPSE_DX = 1.28
ELLIPSE_DY = 1.34
ELLIPSE_CY = 0.52   # the ellipse centre sits this far above the eye line
# The face itself is excluded from the hair mask: a downward ellipse over the
# eyes, nose and cheeks, so side hair can come in without painting the face.
FACE_DX = 0.66
FACE_DY = 0.92
FACE_CY = 0.48      # centre below the eye line
SIDE_BOTTOM = 0.34  # hair may reach this far below the eyes at the sides

FEATHER = 0.08      # feather on the hair edge, as a fraction of the eye distance
TINT_REFRESH = 8    # frames between recolour measurements of the leftover hair
SCALP_DX = 1.00     # remaining-hair tint: half width, in eye distances
SCALP_DY = 1.15
SCALP_CY = 0.50
TINT = 0.75         # how far the leftover live hair moves toward the pasted hair
ALPHA = 0.95        # a trace of the live hair through the pasted hair reads as
                    # "same lighting" instead of a wig
MIN_COVERAGE = 0.004  # less hair than this in the photo: nothing worth pasting


def eye_span(landmarks: np.ndarray) -> float:
    """Distance between the eye centres - the unit the head is measured in."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    return float(np.linalg.norm(points[0] - points[1])) or 1.0


def head_crop(landmarks: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Box around the whole head (hair included) that fits inside the image."""
    height, width = shape[:2]
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    span = eye_span(points)
    eye_y = float((points[0][1] + points[1][1]) / 2.0)
    cx = float((points[0][0] + points[1][0]) / 2.0)
    x0 = max(0, int(round(cx - HALF_WIDTH * span)))
    x1 = min(width, int(round(cx + HALF_WIDTH * span)))
    y0 = max(0, int(round(eye_y - TOP * span)))
    y1 = min(height, int(round(eye_y + BOTTOM * span)))
    return x0, y0, x1, y1


def foreground(crop_bgr: np.ndarray, landmarks: np.ndarray,
               origin: tuple[int, int] = (0, 0)) -> Optional[np.ndarray]:
    """Person-versus-room mask for the reference photo, without a model.

    OpenCV's GrabCut is seeded from the face (certainly the person) and the crop
    border (certainly the room) and settles the hair in between. Without this
    step a pale wall above a shaved head counts as "not skin", and the pass
    would paste a wall-shaped helmet. A photo it cannot segment yields ``None``
    and the hair pass simply does not run.
    """
    height, width = crop_bgr.shape[:2]
    if height < 16 or width < 16:
        return None
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2).copy()
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    span = eye_span(points)
    if span < 8.0:
        return None
    eye_y = float((points[0][1] + points[1][1]) / 2.0)
    cx = float((points[0][0] + points[1][0]) / 2.0)

    grab = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)
    grab[:3, :] = cv2.GC_PR_BGD
    grab[-3:, :] = cv2.GC_PR_BGD
    grab[:, :3] = cv2.GC_PR_BGD
    grab[:, -3:] = cv2.GC_PR_BGD
    # Cheeks and jaw are certainly the person: that is the seed GrabCut grows
    # the hair and the head shape from.
    cv2.ellipse(grab, (int(round(cx)), int(round(eye_y + 0.1 * span))),
                (max(2, int(round(0.6 * span))), max(2, int(round(0.75 * span)))),
                0, 0, 360, cv2.GC_FGD, -1)
    try:
        cv2.grabCut(crop_bgr, grab, None,
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.logical_or(grab == cv2.GC_FGD, grab == cv2.GC_PR_FGD).astype(np.float32)
    return cv2.GaussianBlur(fg, (0, 0), 1.5)


def hair_mask(crop_bgr: np.ndarray, landmarks: np.ndarray,
              origin: tuple[int, int] = (0, 0),
              person: Optional[np.ndarray] = None) -> np.ndarray:
    """Soft 0..1 mask of the hair inside ``crop_bgr``.

    Hair is "not skin, inside the head ellipse, above the eyes, and part of the
    "person". That covers dark, red and grey hair alike and leaves the
    forehead, cheeks and ears to the face pass. Skin detection is the same YCrCb
    test the body pass uses, so the two agree about where the person ends and
    the hair begins; the person mask is what keeps the room out of it.
    """
    height, width = crop_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    if height < 8 or width < 8:
        return mask

    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2).copy()
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    span = eye_span(points)
    eye_y = float((points[0][1] + points[1][1]) / 2.0)
    cx = float((points[0][0] + points[1][0]) / 2.0)
    if span < 6.0:
        return mask

    skin = body.skin_mask(crop_bgr, scale=1.0)
    hair = np.clip(1.0 - skin, 0.0, 1.0)
    if person is not None:
        hair *= np.clip(person, 0.0, 1.0)

    ellipse = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        ellipse,
        (int(round(cx)), int(round(eye_y - ELLIPSE_CY * span))),
        (max(2, int(round(ELLIPSE_DX * span))), max(2, int(round(ELLIPSE_DY * span)))),
        0, 0, 360, 1.0, -1,
    )
    hair *= ellipse

    # The face is not hair: cut an oval over the eyes, nose and cheeks, then
    # anything below the ears. The sides survive, which is the point - a
    # hairstyle that stops at the crown reads as a wig balanced on the head.
    face_oval = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        face_oval,
        (int(round(cx)), int(round(eye_y + FACE_CY * span))),
        (max(2, int(round(FACE_DX * span))), max(2, int(round(FACE_DY * span)))),
        0, 0, 360, 1.0, -1,
    )
    hair *= 1.0 - cv2.GaussianBlur(face_oval, (0, 0), max(2.0, 0.04 * span))

    cutoff = int(round(eye_y + SIDE_BOTTOM * span))
    if cutoff < height:
        hair[max(0, cutoff):, :] = 0.0

    return cv2.GaussianBlur(hair, (0, 0), max(2.0, FEATHER * span))


class HairPass:
    """The character's hair, aligned to the live face every frame.

    Built once from the reference photo. ``apply`` is a no-op when the photo has
    no usable hair (a shaved head, a hat that reads as skin, a helmet) or when
    the face is too small for the warp to mean anything.
    """

    def __init__(self, reference_bgr: np.ndarray, landmarks: np.ndarray,
                 on_log=None) -> None:
        self._log = on_log or (lambda _m: None)
        self._crop: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        self._origin = (0, 0)
        self.coverage = 0.0
        self._tint_gain: Optional[np.ndarray] = None
        self._tint_weight: Optional[np.ndarray] = None
        self._tint_size: Optional[tuple[int, int]] = None
        self._tint_tick = 0

        height, width = reference_bgr.shape[:2]
        x0, y0, x1, y1 = head_crop(self._landmarks, (height, width))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        crop = np.ascontiguousarray(reference_bgr[y0:y1, x0:x1])
        person = foreground(crop, self._landmarks, origin=(x0, y0))
        if person is None or float(person.mean()) < 0.05:
            self._log("Could not separate the person from the photo; leaving hair alone")
            return
        mask = hair_mask(crop, self._landmarks, origin=(x0, y0), person=person)
        # Solid hair only: the feathered fringe is not what decides whether there
        # is a hairstyle worth pasting.
        self.coverage = float((mask > 0.5).mean())
        if self.coverage < MIN_COVERAGE:
            self._log("Reference photo has no usable hair; leaving hair alone")
            return
        self._crop = crop
        self._mask = mask
        self._origin = (x0, y0)

    @property
    def active(self) -> bool:
        return self._crop is not None and self._mask is not None

    def apply(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Wrap the reference hair onto the live head; returns the frame."""
        if not self.active:
            return frame_bgr
        assert self._crop is not None and self._mask is not None

        live = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        span = eye_span(live)
        if span < 8.0:
            return frame_bgr

        # Reference pixels -> live pixels: the same similarity the face uses.
        try:
            matrix = face.similarity_matrix(self._landmarks, live)
        except ValueError:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        crop_h, crop_w = self._crop.shape[:2]
        ox, oy = self._origin
        # The patch is stored with its top-left at the crop origin, so the
        # reference->live similarity needs that offset composed back in.
        corners = np.array(
            [[ox, oy], [ox + crop_w - 1, oy],
             [ox + crop_w - 1, oy + crop_h - 1], [ox, oy + crop_h - 1]],
            dtype=np.float32,
        )
        projected = cv2.transform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
        x0 = max(0, int(np.floor(projected[:, 0].min())))
        y0 = max(0, int(np.floor(projected[:, 1].min())))
        x1 = min(width, int(np.ceil(projected[:, 0].max())))
        y1 = min(height, int(np.ceil(projected[:, 1].max())))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return frame_bgr

        origin = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]],
                          dtype=np.float32)
        shift = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]],
                         dtype=np.float32)
        warp = (shift @ np.vstack([matrix, [0.0, 0.0, 1.0]]) @ origin)[:2]
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpAffine(self._crop, warp, size, flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(self._mask, warp, size, flags=cv2.INTER_LINEAR)
        if float(alpha.max()) < 0.05:
            return frame_bgr
        alpha = np.clip(alpha * ALPHA, 0.0, 1.0)

        target = frame_bgr[y0:y1, x0:x1].astype(np.float32)
        blended = warped.astype(np.float32) * alpha[:, :, None] + target * (1.0 - alpha[:, :, None])
        blended = self._tint_leftover_hair(blended, alpha, live, (x0, y0), size)
        out = frame_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def _leftover_weights(self, blended: np.ndarray, alpha: np.ndarray,
                          live: np.ndarray, offset: tuple[int, int],
                          size: tuple[int, int]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Where the live hair still shows, at a third of the frame resolution.

        The character's hairstyle can be smaller than the live head: braids, a
        fringe or a ponytail stick out past the warped patch, and the patch then
        reads as a hairline glued on. These weights mark that leftover hair so
        it can be pulled toward the pasted hair colour.

        Only *dark* pixels inside the scalp ellipse count: a shiny forehead
        fails the YCrCb skin test and must never be painted as hair.
        """
        span = eye_span(live)
        width, height = size
        if width < 12 or height < 12:
            return None, None
        scale = 3
        sw, sh = max(4, width // scale), max(4, height // scale)

        small = cv2.resize(np.clip(blended, 0, 255).astype(np.uint8), (sw, sh),
                           interpolation=cv2.INTER_AREA)
        alpha_small = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        covered = alpha_small > 0.5
        if not bool(covered.any()):
            return None, None

        span_s = span / scale
        eyes_y = (float((live[0][1] + live[1][1]) / 2.0) - offset[1]) / scale
        cx = (float((live[0][0] + live[1][0]) / 2.0) - offset[0]) / scale
        scalp = np.zeros((sh, sw), dtype=np.float32)
        cv2.ellipse(scalp, (int(round(cx)), int(round(eyes_y - SCALP_CY * span_s))),
                    (max(2, int(round(SCALP_DX * span_s))), max(2, int(round(SCALP_DY * span_s)))),
                    0, 0, 360, 1.0, -1)
        cutoff = int(round(eyes_y - 0.05 * span_s))
        if cutoff < sh:
            scalp[max(0, cutoff):, :] = 0.0
        scalp = cv2.GaussianBlur(scalp, (0, 0), max(1.5, min(9.0, FEATHER * span_s)))

        leftover = 1.0 - body.skin_mask(small, scale=1.0)
        luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        reference_luma = float(luma[covered].mean())
        dark = np.clip((reference_luma * 1.25 + 12.0 - luma.astype(np.float32)) / 24.0, 0.0, 1.0)
        weight = scalp * leftover * dark * (1.0 - alpha_small)
        if float(weight.sum()) < 24.0:
            return None, None

        hair_pixels = small[(weight > 0.25) & (alpha_small < 0.2)]
        pasted_pixels = small[covered]
        if hair_pixels.size < 64 or pasted_pixels.size < 64:
            return None, None
        return weight, np.array([pasted_pixels.reshape(-1, 3).mean(axis=0),
                                 hair_pixels.reshape(-1, 3).mean(axis=0)], dtype=np.float32)

    def _tint_leftover_hair(self, blended: np.ndarray, alpha: np.ndarray,
                            live: np.ndarray, offset: tuple[int, int],
                            size: tuple[int, int]) -> np.ndarray:
        """Nudge the leftover live hair toward the pasted hair's colour.

        The colour shift is a per-channel gain computed from the two means a few
        times a second, then applied with integer SIMD ops: a full LAB
        conversion per frame along with the blend measured 13 ms, this is about
        a tenth of that and the eye cannot see the difference on hair.
        """
        self._tint_tick += 1
        # The cached mask is tied to one ROI: the head moves, the box the paste
        # covers changes size, and a stale mask would not even broadcast.
        if self._tint_size != size:
            self._tint_gain = self._tint_weight = None
            self._tint_size = size
        if self._tint_tick % TINT_REFRESH == 0 or self._tint_gain is None:
            weight, means = self._leftover_weights(blended, alpha, live, offset, size)
            if weight is None or means is None:
                self._tint_gain = self._tint_weight = None
                return blended
            pasted, hair = means[0], means[1]
            # A gain clipped to a sensible range: hair that is already close must
            # not flicker, and a blown-out highlight must not become grey.
            gain = np.clip((pasted + 16.0) / (hair + 16.0), 0.55, 1.85).astype(np.float32)
            self._tint_gain = gain
            self._tint_weight = np.clip(
                cv2.resize(weight, size, interpolation=cv2.INTER_LINEAR) * TINT, 0.0, 1.0
            )
        if self._tint_gain is None or self._tint_weight is None:
            return blended

        tinted = cv2.merge([
            cv2.convertScaleAbs(blended[:, :, c], alpha=float(self._tint_gain[c]))
            for c in range(3)
        ])
        weight = cv2.cvtColor((self._tint_weight * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        keep = cv2.bitwise_not(weight)
        front = cv2.multiply(tinted, weight, scale=1.0 / 255.0)
        back = cv2.multiply(np.clip(blended, 0, 255).astype(np.uint8), keep, scale=1.0 / 255.0)
        cv2.add(front, back, dst=front)
        return front.astype(np.float32)
