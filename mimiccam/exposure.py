"""Lift an underexposed face before the swap gets hold of it.

The swap network is unforgiving about exposure. Fed a face sitting at luma 35 -
a backlit room where the camera has metered for the window - it returns a face
at luma 35, the paste colour-matches it to the frame it came from, and the
character ends up invisible against a near-silhouette. YuNet stops finding the
face at all below roughly 50 luma at its 0.6 threshold, measured on a portrait
darkened in steps:

    face luma   170   ->  19,387 pixels visibly changed by the swap
    face luma    85   ->   5,872
    face luma    51   ->   not detected

So a dark room does not degrade the swap gracefully; it silently drops it.

The fix is a gamma lift, not a gain. A gain would push the window behind the
subject - already at 255 - further into clipping and lose the edge that makes a
silhouette readable. A gamma curve keeps 0 and 255 where they are and moves only
the shadows and mid-tones, which is where an underexposed face lives. The lift is
applied to the whole frame rather than to a brightened patch of face: pasting a
well-lit oval into a dark room looks exactly like a face-shaped sticker.

Both the detection and the render see the lifted frame, and the frame that is
published is the lifted one - "put the result back" has to mean the output, or
the character is darkened back down by the paste's own colour matching.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# Where a face reads as a face to both YuNet and the swap network. Chosen on a
# real backlit frame rather than by taste: lifting it to 80, 95, 110 and 130
# scored the pasted character +0.234, +0.341, +0.268 and +0.201, and the frames
# at 110 and 130 were visibly washed out. 95 is both the most recognisable and
# the gentlest lift.
TARGET_LUMA = 95.0
# The darkest a lift may go. gamma 0.33 takes luma 10 to ~97 and luma 25 to ~120;
# past this the shadows turn to sensor noise faster than they turn into a face.
GAMMA_MIN = 0.33
# Only worth a second detection pass when the frame itself is this dark. A
# backlit subject in a bright room has a mean well above this and is caught by
# the face measurement instead, so the extra detection is not paid on every
# faceless frame.
SCENE_MEAN_TRIGGER = 80.0
# What a dark frame is lifted towards for that second look, before any face is
# known. Deliberately below TARGET_LUMA: it only has to be enough for YuNet.
SCENE_TARGET_LUMA = 100.0
# The share of a detection box measured for the face's own brightness. The outer
# part of a box on a turned head is hair and background, and on a silhouette it
# is the bright wall that made the frame look correctly exposed in the first
# place.
INNER = 0.6


def face_luma(bgr: np.ndarray, box: tuple[int, int, int, int]) -> Optional[float]:
    """Mean luma of the middle of a face box, or None if the box is unusable."""
    height, width = bgr.shape[:2]
    x, y, w, h = (int(v) for v in box[:4])
    if w <= 2 or h <= 2:
        return None
    shrink_x = int(w * (1.0 - INNER) * 0.5)
    shrink_y = int(h * (1.0 - INNER) * 0.5)
    x0 = max(0, min(width - 1, x + shrink_x))
    y0 = max(0, min(height - 1, y + shrink_y))
    x1 = max(x0 + 1, min(width, x + w - shrink_x))
    y1 = max(y0 + 1, min(height, y + h - shrink_y))
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(grey.mean())


def gamma_for_luma(luma: float, target: float = TARGET_LUMA,
                   gamma_min: float = GAMMA_MIN) -> Optional[float]:
    """The gamma that maps ``luma`` to ``target``, or None when no lift is needed.

    ``out = 255 * (in / 255) ** gamma``, so gamma is the log ratio of the two
    normalised luminances. Values below 1 brighten.
    """
    if luma is None or not math.isfinite(luma) or luma <= 0.0:
        return None
    if luma >= target:
        return None
    normalised = min(1.0, luma / 255.0)
    wanted = min(1.0, target / 255.0)
    gamma = math.log(wanted) / math.log(normalised)
    if not math.isfinite(gamma) or gamma >= 1.0:
        return None
    return max(gamma_min, gamma)


def lut_for(gamma: float) -> np.ndarray:
    """A 256-entry table applying that gamma to every channel."""
    ramp = np.arange(256, dtype=np.float32) / 255.0
    values = np.power(np.clip(ramp, 0.0, 1.0), float(gamma)) * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


class Normaliser:
    """Decides whether a frame needs lifting, and lifts it.

    Holds the 256-entry tables so a moving exposure between frames does not
    rebuild one per frame; the cache is tiny (a dozen gammas at most) and keyed
    on a rounded value.
    """

    def __init__(
        self,
        target: float = TARGET_LUMA,
        gamma_min: float = GAMMA_MIN,
        scene_trigger: float = SCENE_MEAN_TRIGGER,
        scene_target: float = SCENE_TARGET_LUMA,
    ) -> None:
        self.target = float(target)
        self.gamma_min = float(gamma_min)
        self.scene_trigger = float(scene_trigger)
        self.scene_target = float(scene_target)
        self.lifts = 0
        # What the last measured face sat at, for the engine's log line.
        self.last_luma: Optional[float] = None
        self._luts: dict[float, np.ndarray] = {}

    # -- decisions ---------------------------------------------------------
    def gamma_for(self, luma: Optional[float], target: Optional[float] = None) -> Optional[float]:
        return gamma_for_luma(
            luma if luma is not None else 0.0,
            self.target if target is None else target,
            self.gamma_min,
        )

    def retry_gamma(self, bgr: np.ndarray) -> Optional[float]:
        """A gamma worth re-running detection at, when nothing was found.

        Returns None when the frame is not dark, so an empty bright room is
        still detected exactly once.
        """
        mean = float(cv2.cvtColor(cv2.resize(bgr, (64, 36),
                                             interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY).mean())
        if mean >= self.scene_trigger:
            return None
        return self.gamma_for(mean, target=self.scene_target)

    # -- the lift itself ---------------------------------------------------
    def apply(self, bgr: np.ndarray, gamma: float) -> np.ndarray:
        key = round(float(gamma), 3)
        table = self._luts.get(key)
        if table is None:
            table = lut_for(key)
            if len(self._luts) > 64:
                self._luts.clear()
            self._luts[key] = table
        return cv2.LUT(bgr, table)

    def lift_for_face(self, bgr: np.ndarray,
                      box: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
        """Lift a frame whose face is underexposed. Returns (frame, gamma used)."""
        luma = face_luma(bgr, box)
        self.last_luma = luma
        if luma is None or luma >= self.target:
            return bgr, 1.0
        gamma = self.gamma_for(luma)
        if gamma is None:
            return bgr, 1.0
        self.lifts += 1
        return self.apply(bgr, gamma), gamma
