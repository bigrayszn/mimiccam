"""Tests for the wardrobe: the character's clothes, transplanted.

Synthetic images only - the geometry and the cut-out rules are what can go wrong
silently; whether a suit looks right is a measurement, not a unit test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import body, wardrobe  # noqa: E402

FACE = (70, 20, 60, 70)          # x, y, w, h in the reference image
CHIN = FACE[1] + FACE[3]         # 90
GARMENT = (190, 120, 70)         # a BGR colour the skin detector rejects


class _Matte:
    """A matte model that returns whatever alpha it was given."""

    def __init__(self, alpha: np.ndarray) -> None:
        self.alpha_frame = alpha

    def alpha(self, _frame_bgr: np.ndarray) -> np.ndarray:
        return self.alpha_frame


def _reference(size: tuple[int, int] = (300, 200)):
    """A person wearing a plain garment from the chin down."""
    image = np.full((size[0], size[1], 3), 40, np.uint8)
    image[CHIN - 5:, 30:170] = GARMENT
    alpha = np.zeros((size[0], size[1]), np.float32)
    alpha[CHIN - 5:, 30:170] = 1.0
    return image, alpha


class TorsoMaskTests(unittest.TestCase):
    def test_the_face_and_hair_are_not_cloth(self) -> None:
        image, alpha = _reference()
        mask = wardrobe.torso_mask(image, FACE, alpha)
        self.assertEqual(float(mask[:CHIN - 10].sum()), 0.0)

    def test_skin_is_not_cloth(self) -> None:
        image, alpha = _reference()
        # Paint a forearm across the garment: skin has to stay the live person's.
        skin = (120, 150, 190)  # BGR, inside the YCrCb skin box
        image[200:260, 40:160] = skin
        mask = wardrobe.torso_mask(image, FACE, alpha)
        self.assertLess(float(mask[200:260, 40:160].mean()), 0.15)
        self.assertGreater(float(mask[120:160, 40:160].mean()), 0.5)

    def test_no_torso_means_no_wardrobe(self) -> None:
        image = np.full((300, 200, 3), 40, np.uint8)
        empty = np.zeros((300, 200), np.float32)
        held = wardrobe.Wardrobe(image, FACE, _Matte(empty))
        self.assertFalse(held.ready)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        image, alpha = _reference()
        self.held = wardrobe.Wardrobe(image, FACE, _Matte(alpha))
        self.assertTrue(self.held.ready)

    def test_the_chins_and_centres_line_up(self) -> None:
        live_face = (300, 120, 120, 140)  # twice the reference face width
        matrix, scale = self.held.matrix_for(live_face)
        self.assertAlmostEqual(scale, 2.0, places=3)
        px, py = self.held.origin
        rx, ry, rw, rh = FACE
        # The reference chin, expressed in patch coordinates.
        point = np.array([[[rx + rw / 2.0 - px, ry + rh - py]]], np.float32)
        mapped = cv2.transform(point, matrix)[0, 0]
        self.assertAlmostEqual(float(mapped[0]), 300 + 60.0, delta=0.5)
        self.assertAlmostEqual(float(mapped[1]), 120 + 140.0, delta=0.5)

    def test_the_scale_is_clamped(self) -> None:
        tiny, scale = self.held.matrix_for((0, 0, 1, 1))
        self.assertEqual(scale, wardrobe.MIN_SCALE)
        huge, scale = self.held.matrix_for((0, 0, 4000, 1000))
        self.assertEqual(scale, wardrobe.MAX_SCALE)


def _two_tone_alpha(size: tuple[int, int] = (300, 200)) -> tuple[np.ndarray, np.ndarray]:
    """A garment whose left half is dark and right half is bright.

    Two tones make the seam between them findable in the output, which is how the
    tracking tests see where the garment actually landed.
    """
    image = np.full((size[0], size[1], 3), 40, np.uint8)
    image[CHIN - 5:, 30:100] = (70, 70, 70)
    image[CHIN - 5:, 100:170] = (170, 170, 170)
    alpha = np.zeros((size[0], size[1]), np.float32)
    alpha[CHIN - 5:, 30:170] = 1.0
    return image, alpha


def _silhouette(columns: tuple[int, int], rows: tuple[int, int],
                shape: tuple[int, int] = (400, 300)) -> np.ndarray:
    alpha = np.zeros(shape, np.float32)
    alpha[rows[0]:rows[1], columns[0]:columns[1]] = 1.0
    return alpha


class TorsoQuadTests(unittest.TestCase):
    def test_the_quad_is_the_silhouette_below_the_chin(self) -> None:
        alpha = _silhouette((80, 280), (100, 400))
        quad = wardrobe.torso_quad(alpha, (110, 60, 60, 70))
        self.assertIsNotNone(quad)
        self.assertAlmostEqual(float(quad[0, 0]), 80.0, delta=2.0)
        self.assertAlmostEqual(float(quad[1, 0]), 279.0, delta=2.0)
        # Top edge at the chin, bottom edge a face-relative depth below it - not
        # at the bottom of the frame, which would make a webcam crop and a
        # full-length portrait incomparable.
        self.assertAlmostEqual(float(quad[0, 1]), 130.0, delta=1.0)
        self.assertAlmostEqual(float(quad[3, 1]), 242.0, delta=2.0)

    def test_the_same_rule_puts_the_rows_in_the_same_place_in_both_photos(self) -> None:
        portrait = _silhouette((30, 170), (85, 300), shape=(300, 200))
        webcam = _silhouette((80, 280), (100, 400))
        a = wardrobe.torso_quad(portrait, FACE)
        b = wardrobe.torso_quad(webcam, (110, 60, 60, 70))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        for quad, (chin, height) in ((a, (CHIN, FACE[3])), (b, (130, 70))):
            self.assertAlmostEqual(float(quad[0, 1]), float(chin), delta=1.0)
            self.assertAlmostEqual(float(quad[3, 1]) - float(quad[0, 1]),
                                   height * wardrobe.QUAD_DEPTH, delta=2.0)

    def test_a_sliver_is_not_a_torso(self) -> None:
        alpha = _silhouette((140, 170), (100, 400))
        self.assertIsNone(wardrobe.torso_quad(alpha, (110, 60, 60, 70)))

    def test_nobody_there_is_no_quad(self) -> None:
        alpha = np.zeros((400, 300), np.float32)
        self.assertIsNone(wardrobe.torso_quad(alpha, (110, 60, 60, 70)))

    def test_a_gap_under_the_jaw_does_not_push_the_collar_down(self) -> None:
        alpha = _silhouette((80, 280), (150, 400))  # matte starts 20 px below the chin
        quad = wardrobe.torso_quad(alpha, (110, 60, 60, 70))
        self.assertIsNotNone(quad)
        self.assertAlmostEqual(float(quad[0, 1]), 130.0, delta=1.0)


class TrackingTests(unittest.TestCase):
    """The clothes follow the body, not the head."""

    LIVE_FACE = (110, 60, 60, 70)

    def setUp(self) -> None:
        image, alpha = _two_tone_alpha()
        self.held = wardrobe.Wardrobe(image, FACE, _Matte(alpha))
        self.assertTrue(self.held.ready)
        self.matrix, _scale = self.held.matrix_for(self.LIVE_FACE)

    def test_the_reference_torso_quad_lands_on_the_live_one(self) -> None:
        alpha = _silhouette((80, 280), (100, 400))
        transform = self.held.transform_for(self.LIVE_FACE, alpha, self.matrix)
        px, py = self.held.origin
        source = (self.held.quad - np.array([px, py], np.float32)).astype(np.float32)
        mapped = cv2.perspectiveTransform(source.reshape(1, -1, 2), transform)[0]
        for got, want in zip(mapped, ((80, 130), (279, 130), (279, 242), (80, 242))):
            self.assertAlmostEqual(float(got[0]), float(want[0]), delta=2.0)
            self.assertAlmostEqual(float(got[1]), float(want[1]), delta=2.0)

    def test_a_leaning_body_skews_the_garment(self) -> None:
        # Shoulders right of the hips: the person is leaning towards their left.
        alpha = np.zeros((400, 300), np.float32)
        alpha[130:200, 120:260] = 1.0
        alpha[200:340, 60:300] = 1.0
        transform = self.held.transform_for(self.LIVE_FACE, alpha, self.matrix)
        px, py = self.held.origin
        source = (self.held.quad - np.array([px, py], np.float32)).astype(np.float32)
        mapped = cv2.transform(source.reshape(1, -1, 2), transform)[0]
        # The lower edge slides left of the shoulders instead of staying under
        # them: that is the shear the affine fit is here to carry. It shares the
        # fit with the width change, so it is a fraction of the lean, not all of it.
        self.assertLess(float(mapped[3, 0]) - float(mapped[0, 0]), -5.0,
                        "the garment stayed rectangular on a leaning body")
        self.assertGreater(float(mapped[1, 0]) - float(mapped[0, 0]), 100.0)

    def test_the_garment_lands_on_an_offset_body_not_under_the_face(self) -> None:
        # Same head position, body leaning to the right.
        alpha = _silhouette((80, 280), (100, 400))
        frame = np.full((400, 300, 3), 10, np.uint8)
        out = self.held.dress(frame, self.LIVE_FACE, alpha)
        row = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[180]
        bright = np.flatnonzero(row > 120)
        self.assertGreater(bright.size, 20, "nothing bright was pasted")
        seam = float(bright[0])
        # The fallback transform would have put the seam on the face centre (140).
        self.assertGreater(seam, 160.0)
        self.assertLess(seam, 210.0)

    def test_a_sliver_of_matte_falls_back_to_the_face(self) -> None:
        alpha = _silhouette((140, 170), (100, 400))
        transform = self.held.transform_for(self.LIVE_FACE, alpha, self.matrix)
        expected = np.eye(3, dtype=np.float32)
        expected[:2] = self.matrix
        np.testing.assert_allclose(transform, expected, atol=1e-5)

    def test_nobody_in_frame_falls_back_to_the_face(self) -> None:
        alpha = np.zeros((400, 300), np.float32)
        transform = self.held.transform_for(self.LIVE_FACE, alpha, self.matrix)
        expected = np.eye(3, dtype=np.float32)
        expected[:2] = self.matrix
        np.testing.assert_allclose(transform, expected, atol=1e-5)


class DressTests(unittest.TestCase):
    def setUp(self) -> None:
        image, alpha = _reference()
        self.held = wardrobe.Wardrobe(image, FACE, _Matte(alpha))
        self.frame = np.full((400, 300, 3), 10, np.uint8)
        self.live_face = (110, 60, 60, 70)

    def test_the_garment_lands_on_the_body(self) -> None:
        alpha = np.ones((400, 300), np.float32)
        out = self.held.dress(self.frame, self.live_face, alpha)
        chin = self.live_face[1] + self.live_face[3]
        patch = out[chin + 10:chin + 40, 130:170]
        self.assertGreater(float(patch.mean()), 60.0)
        # Above the chin it is still the empty frame.
        self.assertEqual(float(out[0:40].mean()), 10.0)

    def test_nobody_in_frame_means_nothing_pasted(self) -> None:
        empty = np.zeros((400, 300), np.float32)
        out = self.held.dress(self.frame, self.live_face, empty)
        self.assertEqual(int(out.mean()), 10)

    def test_the_live_skin_stays_on_top(self) -> None:
        alpha = np.ones((400, 300), np.float32)
        chin = self.live_face[1] + self.live_face[3]
        frame = self.frame.copy()
        frame[chin + 10:chin + 40, 130:170] = (120, 150, 190)  # a hand over the chest
        self.assertGreater(float(body.skin_mask(frame)[chin + 15:chin + 35,
                                                       135:165].mean()), 0.5)
        out = self.held.dress(frame, self.live_face, alpha)
        kept = out[chin + 15:chin + 35, 135:165].astype(int)
        self.assertLess(abs(float(kept.mean()) - float(np.array((120, 150, 190)).mean())), 40.0)


if __name__ == "__main__":
    unittest.main()
