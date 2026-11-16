# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The body pass must fade out of the face, not stop at a line under the jaw.

Zeroing the face box left the sliver of neck under the chin at the original
person's skin tone while the face above belonged to the character - the pale
blotch along the jawline. These tests hold the shape of that exclusion: no
correction on the face, full correction on the neck, and a ramp between.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimiccam import body  # noqa: E402

FRAME = (320, 320)          # height, width
FACE_BOX = (110, 60, 100, 120)  # x, y, w, h


def skin_frame(size: tuple[int, int] = FRAME) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:, :] = (150, 170, 205)  # BGR light skin
    return frame


class FaceExclusionTests(unittest.TestCase):

    def test_the_face_itself_is_left_alone(self) -> None:
        mask = body.body_mask(skin_frame(), FACE_BOX)
        x, y, w, h = FACE_BOX
        self.assertLess(float(mask[y + h // 2, x + w // 2]), 0.05)

    def test_the_neck_under_the_jaw_is_corrected(self) -> None:
        mask = body.body_mask(skin_frame(), FACE_BOX)
        x, y, w, h = FACE_BOX
        just_below = float(mask[y + h + 8, x + w // 2])
        self.assertGreater(just_below, 0.35, "the neck under the jaw is still excluded")

    def test_the_change_is_a_ramp_not_a_step(self) -> None:
        mask = body.body_mask(skin_frame(), FACE_BOX)
        x, y, w, h = FACE_BOX
        column = mask[y + h - 20:y + h + 40, x + w // 2]
        jumps = np.abs(np.diff(column))
        self.assertLess(float(jumps.max()), 0.35, "the exclusion still has a hard edge")

    def test_the_correction_does_not_soften_the_body(self) -> None:
        """The pass must shift colour, not resample the neck and chest.

        Carrying the quarter-resolution result back over the ROI replaced the
        torso with a 4x upscale of itself: on a real studio frame the Laplacian
        variance under the jaw fell from 11.0 to 6.0, which is the smeared,
        blotchy neck. Only the difference is carried back now.
        """
        import cv2

        frame = skin_frame((360, 480))
        # Texture in the chest area: stubble, fabric weave, sensor noise.
        rng = np.random.default_rng(7)
        area = frame[250:340, 130:350]
        area[:] = np.clip(area.astype(np.int16) + rng.integers(-18, 19, area.shape), 0, 255)
        box = (190, 60, 100, 120)
        out = body.harmonize(frame, box, np.array([150.0, 140.0, 145.0], np.float32))

        def sharpness(img: np.ndarray) -> float:
            grey = cv2.cvtColor(img[250:340, 130:350], cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(grey, cv2.CV_64F).var())

        self.assertGreater(sharpness(out), sharpness(frame) * 0.9,
                           "the body pass resampled the torso and lost its detail")

    def test_a_dim_neck_still_counts_as_this_persons_skin(self) -> None:
        """The global YCrCb window is a daylight rule; the face is the reference.

        On a real 1080p frame the neck sat at Cr 131 against a floor of 133 and
        the whole area dropped out of the mask, leaving the original tone under
        a swapped jaw. Pixels within CHROMA_TOLERANCE of the face's own chroma
        now count even when the window rejects them.
        """
        import cv2

        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:, :] = (150, 170, 190)  # BGR skin, Cr 139.6 - just inside the window
        box = (190, 60, 100, 120)
        # The neck: the face's own chroma pulled 8 levels down, which is Cr 131
        # against a global floor of 133. Measured on a real 1080p frame: face Cr
        # 135.1, neck 131.3.
        neck = frame[220:330, 150:330]
        ycrcb = cv2.cvtColor(neck, cv2.COLOR_BGR2YCrCb).astype(np.int16)
        ycrcb[:, :, 1] -= 8
        neck[:] = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        self.assertLess(float(body.skin_mask(frame)[220:330, 150:330].mean()), 0.2,
                        "the fixture is meant to sit outside the global window")
        mask = body.body_mask(frame, box)
        self.assertGreater(float(mask[220:330, 150:330].mean()), 0.3,
                           "the neck was dropped from the mask")

    def test_a_grey_shirt_is_not_skin(self) -> None:
        frame = skin_frame((360, 480))
        frame[220:340, 120:360] = (150, 150, 150)  # neutral, no chroma at all
        mask = body.body_mask(frame, (190, 60, 100, 120))
        self.assertLess(float(mask[220:340, 120:360].mean()), 0.15)

    def test_it_still_needs_skin(self) -> None:
        frame = np.zeros((*FRAME, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 30)  # a black jumper, not skin
        mask = body.body_mask(frame, FACE_BOX)
        self.assertLess(float(mask.max()), 0.2)


if __name__ == "__main__":
    unittest.main()
