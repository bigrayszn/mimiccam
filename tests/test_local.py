"""Unit tests for the free, offline engine.

Nothing here needs a network, a camera or a 500 MB model: the alignment maths,
the paste geometry, the colour looks and the model-file bookkeeping are all
pure functions. The end-to-end run lives in ``tests/verify_local.py``.

    python -m pytest tests/test_local.py -q
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimiccam import face, looks, models  # noqa: E402

# A synthetic frontal face: eyes, nose, mouth corners, in image pixels.
FACE_POINTS = np.array(
    [[70, 70], [130, 70], [100, 95], [85, 125], [115, 125]], dtype=np.float32
)


class AlignmentTests(unittest.TestCase):
    def test_template_maps_onto_itself(self) -> None:
        matrix = face.norm_matrix(face.ARCFACE_DST, 112)
        warped = cv2.transform(face.ARCFACE_DST.reshape(1, -1, 2), matrix).reshape(-1, 2)
        np.testing.assert_allclose(warped, face.ARCFACE_DST, atol=1e-3)

    def test_128_crop_template_is_offset(self) -> None:
        # InsightFace's 128 px template is the 112 px one shifted 8 px right.
        matrix = face.norm_matrix(face.ARCFACE_DST, 128)
        warped = cv2.transform(face.ARCFACE_DST.reshape(1, -1, 2), matrix).reshape(-1, 2)
        np.testing.assert_allclose(warped[:, 0], face.ARCFACE_DST[:, 0] + 8, atol=1e-3)
        np.testing.assert_allclose(warped[:, 1], face.ARCFACE_DST[:, 1], atol=1e-3)

    def test_scale_and_translation_are_recovered(self) -> None:
        # Same face shape, twice as large and moved: alignment must normalise it.
        moved = FACE_POINTS * 2.0 + np.array([40.0, 30.0], dtype=np.float32)
        first = face.norm_matrix(FACE_POINTS, 112)
        second = face.norm_matrix(moved, 112)
        a = cv2.transform(FACE_POINTS.reshape(1, -1, 2), first).reshape(-1, 2)
        b = cv2.transform(moved.reshape(1, -1, 2), second).reshape(-1, 2)
        np.testing.assert_allclose(a, b, atol=1e-2)

    def test_degenerate_landmarks_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            face.norm_matrix(np.zeros((5, 2), dtype=np.float32), 112)

    def test_largest_picks_biggest_face(self) -> None:
        small = face.Detection(box=(0, 0, 10, 10), score=0.9, landmarks=FACE_POINTS)
        big = face.Detection(box=(50, 50, 80, 80), score=0.5, landmarks=FACE_POINTS)
        self.assertIs(face.largest([small, big]), big)
        self.assertIsNone(face.largest([]))


class CompositeTests(unittest.TestCase):
    def _matrix(self, points: np.ndarray) -> np.ndarray:
        return face.norm_matrix(points, 128)

    def test_paste_lands_on_the_face(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        texture = np.full((128, 128, 3), 200, dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        out = face.composite(frame, texture, mask, self._matrix(FACE_POINTS))
        self.assertGreater(out[95, 100].mean(), 150)   # inside the face
        self.assertLess(out[5, 5].mean(), 10)          # far away, untouched

    def test_paste_never_touches_distant_pixels(self) -> None:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        before = frame.copy()
        texture = np.full((128, 128, 3), 255, dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        out = face.composite(frame, texture, mask, self._matrix(FACE_POINTS))
        self.assertTrue(np.array_equal(out[200:, :], before[200:, :]))
        self.assertTrue(np.array_equal(out[:, 280:], before[:, 280:]))
        self.assertFalse(np.array_equal(out, before))

    def test_zero_mask_leaves_the_frame_alone(self) -> None:
        frame = np.full((120, 120, 3), 42, dtype=np.uint8)
        texture = np.full((128, 128, 3), 255, dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.float32)
        points = np.array([[30, 30], [50, 30], [40, 40], [33, 55], [47, 55]], np.float32)
        out = face.composite(frame, texture, mask, self._matrix(points))
        self.assertTrue(np.array_equal(out, frame))

    def test_float_and_uint8_masks_paste_the_same(self) -> None:
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        texture = np.full((128, 128, 3), 190, dtype=np.uint8)
        soft = np.zeros((128, 128), np.float32)
        cv2.ellipse(soft, (64, 64), (40, 50), 0, 0, 360, 1.0, -1)
        soft = cv2.GaussianBlur(soft, (0, 0), 4)
        matrix = self._matrix(FACE_POINTS)
        as_float = face.composite(frame, texture, soft, matrix)
        as_byte = face.composite(frame, texture, (soft * 255).astype(np.uint8), matrix)
        self.assertLessEqual(int(np.abs(as_float.astype(np.int16)
                                        - as_byte.astype(np.int16)).max()), 2)

    def test_oval_mask_is_soft_and_centred(self) -> None:
        mask = face.FaceSwapper._oval_mask(128)
        self.assertEqual(mask.shape, (128, 128))
        self.assertAlmostEqual(float(mask[64, 64]), 1.0, places=2)
        self.assertLess(float(mask[0, 0]), 0.05)
        self.assertGreater(float(mask.sum()), 128 * 128 * 0.3)


class LookTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.frame = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)

    def test_every_look_preserves_shape_and_dtype(self) -> None:
        for name in looks.LOOKS:
            out = looks.apply(self.frame, name)
            self.assertEqual(out.shape, self.frame.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_neutral_is_identity(self) -> None:
        self.assertIs(looks.apply(self.frame, "neutral"), self.frame)

    def test_unknown_look_falls_back_to_neutral(self) -> None:
        self.assertIs(looks.apply(self.frame, "nonsense"), self.frame)

    def test_presets_map_to_a_known_look(self) -> None:
        for key in ("character", "identity", "anime", "cinematic", "suit", "custom"):
            self.assertIn(looks.for_preset(key), looks.LOOKS)

    def test_grades_actually_change_the_picture(self) -> None:
        for name in ("cinematic", "cel", "studio"):
            out = looks.apply(self.frame, name)
            self.assertFalse(np.array_equal(out, self.frame), name)


class SmoothingAndToneTests(unittest.TestCase):
    def test_smoother_moves_partway_towards_new_points(self) -> None:
        smoother = face.LandmarkSmoother(alpha=0.5)
        start = FACE_POINTS.copy()
        smoother.update(start)
        moved = start + np.array([10.0, 0.0], dtype=np.float32)
        blended = smoother.update(moved)
        self.assertGreater(float(blended[2, 0]), float(start[2, 0]))
        self.assertLess(float(blended[2, 0]), float(moved[2, 0]))

    def test_smoother_snaps_on_a_big_jump(self) -> None:
        smoother = face.LandmarkSmoother(alpha=0.5)
        smoother.update(FACE_POINTS)
        far = FACE_POINTS + np.array([120.0, 60.0], dtype=np.float32)
        self.assertTrue(np.allclose(smoother.update(far), far, atol=1e-3))

    def test_smoother_still_when_input_is_still(self) -> None:
        smoother = face.LandmarkSmoother()
        jittered = [FACE_POINTS + rng for rng in (np.zeros((5, 2), np.float32),)]
        for points in jittered * 5:
            steady = smoother.update(points)
        self.assertTrue(np.allclose(steady, FACE_POINTS, atol=1e-3))

    def test_smoother_does_not_lag_a_steady_pan(self) -> None:
        """A constant-velocity head must be tracked, not chased one frame behind."""
        smoother = face.LandmarkSmoother()
        points = FACE_POINTS.copy()
        now = 100.0
        for step in range(30):
            moved = points + np.array([6.0 * step, 0.0], dtype=np.float32)
            tracked = smoother.update(moved, now=now)
            now += 1.0 / 30.0
        error = float(np.mean(np.linalg.norm(tracked - moved, axis=1)))
        self.assertLess(error, 2.0)

    def test_smoother_keeps_a_still_face_still(self) -> None:
        smoother = face.LandmarkSmoother()
        now = 50.0
        rng = np.random.default_rng(5)
        for _ in range(30):
            noisy = FACE_POINTS + rng.normal(0.0, 0.8, FACE_POINTS.shape).astype(np.float32)
            tracked = smoother.update(noisy, now=now)
            now += 1.0 / 30.0
        self.assertLess(float(np.abs(tracked - FACE_POINTS).max()), 1.5)

    def test_tone_match_brightens_a_dark_render(self) -> None:
        render = np.full((128, 128, 3), 60, dtype=np.uint8)
        render[..., 2] = 90
        reference = np.full((128, 128, 3), 170, dtype=np.uint8)
        out = face.match_tone(render, reference, strength=1.0)
        self.assertEqual(out.shape, render.shape)
        self.assertGreater(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()),
                           float(cv2.cvtColor(render, cv2.COLOR_BGR2GRAY).mean()))

    def test_mask_reaches_the_jaw(self) -> None:
        mask = face.FaceSwapper._oval_mask(128)
        # 110 px down the crop is below the mouth on InsightFace's template.
        self.assertGreater(float(mask[110, 64]), 0.4)


class BodyPassTests(unittest.TestCase):
    def setUp(self) -> None:
        from mimiccam import body

        self.body_mod = body
        self.frame = np.full((360, 480, 3), 240, dtype=np.uint8)  # bright wall
        self.face_box = (190, 60, 100, 120)
        # A forearm: classic skin BGR, well below the face box.
        self.arm = (slice(250, 330), slice(120, 240))
        self.frame[self.arm] = (120, 150, 205)

    def test_skin_is_detected_and_background_is_not(self) -> None:
        mask = self.body_mod.skin_mask(self.frame)
        self.assertGreater(float(mask[self.arm].mean()), 0.6)
        self.assertLess(float(mask[10:40, 10:40].mean()), 0.05)

    def test_body_mask_excludes_the_face(self) -> None:
        # Put skin inside the face box too: it must be dropped, the arm kept.
        self.frame[80:150, 210:270] = (120, 150, 205)
        mask = self.body_mod.body_mask(self.frame, self.face_box)
        self.assertLess(float(mask[80:150, 210:270].mean()), 0.05)
        self.assertGreater(float(mask[self.arm].mean()), 0.3)

    def test_harmonize_moves_skin_towards_the_face_tone(self) -> None:
        target = np.array([120.0, 150.0, 150.0], dtype=np.float32)  # pinker, duller
        # Measure the arm itself. A mask-weighted whole-frame mean would also
        # count pixels the correction pushed across the skin thresholds.
        before = cv2.cvtColor(self.frame[self.arm], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
        out = self.body_mod.harmonize(self.frame, self.face_box, target, strength=1.0)
        after = cv2.cvtColor(out[self.arm], cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
        self.assertLess(abs(after[1] - target[1]), abs(before[1] - target[1]) - 1.0)
        self.assertLess(abs(after[2] - target[2]), abs(before[2] - target[2]) + 1e-3)

    def test_harmonize_leaves_the_background_alone(self) -> None:
        out = self.body_mod.harmonize(
            self.frame, self.face_box, np.array([120.0, 150.0, 150.0], np.float32)
        )
        self.assertTrue(np.array_equal(out[10:60, 10:80], self.frame[10:60, 10:80]))

    def test_harmonize_is_a_no_op_without_skin(self) -> None:
        wall = np.full((240, 320, 3), 200, dtype=np.uint8)
        out = self.body_mod.harmonize(wall, (100, 40, 80, 90), np.array([90.0, 140.0, 140.0], np.float32))
        self.assertTrue(np.array_equal(out, wall))


class _FakeSwapper:
    """Just enough of FaceSwapper for the frame loop."""

    def render(self, frame, landmarks, embedding):
        return np.full((128, 128, 3), 90, dtype=np.uint8)

    def paste(self, frame, texture, landmarks):
        out = frame.copy()
        out[:8, :8] = 255  # a visible marker: "the swap happened"
        return out


class _FakeDetector:
    def __init__(self) -> None:
        self.faces: list = []

    def detect(self, frame):
        return list(self.faces)


class PoseHoldTests(unittest.TestCase):
    """A dropped detection must not flash the un-swapped face."""

    def setUp(self) -> None:
        from mimiccam.local_engine import LocalEngine

        self.engine = LocalEngine(providers=["CPUExecutionProvider"])
        self.engine.width, self.engine.height = 64, 64
        self.engine._swapper = _FakeSwapper()
        self.engine._detector = _FakeDetector()
        self.engine._embedding = np.zeros((512,), np.float32)
        self.frame = np.full((64, 64, 3), 30, dtype=np.uint8)
        self.face = face.Detection(box=(10, 10, 40, 40), score=0.9, landmarks=FACE_POINTS)

    def test_miss_inside_the_hold_still_swaps(self) -> None:
        self.engine._detector.faces = [self.face]
        self.engine._process(self.frame)
        self.engine._detector.faces = []
        before = self.engine.frames_swapped
        out = self.engine._process(self.frame)
        self.assertEqual(self.engine.frames_swapped, before + 1)
        self.assertEqual(int(out[0, 0].mean()), 255)
        self.assertEqual(self.engine.frames_no_face, 0)

    def test_hold_expires(self) -> None:
        self.engine._detector.faces = [self.face]
        self.engine._process(self.frame)
        self.engine._detector.faces = []
        self.engine._last_pose_ts -= 5.0  # well past POSE_HOLD_S
        out = self.engine._process(self.frame)
        self.assertEqual(self.engine.frames_no_face, 1)
        self.assertEqual(int(out[0, 0].mean()), 30)
        self.assertIsNone(self.engine._last_landmarks)

    def test_new_render_is_blended_not_popped_in(self) -> None:
        old = np.zeros((128, 128, 3), dtype=np.uint8)
        new = np.full((128, 128, 3), 255, dtype=np.uint8)
        self.engine._store_texture(old)
        blended = self.engine._blend_texture(new)
        self.assertGreater(int(blended.mean()), 100)
        self.assertLess(int(blended.mean()), 200)

    def test_a_stale_texture_is_taken_as_is(self) -> None:
        self.engine._store_texture(np.zeros((128, 128, 3), dtype=np.uint8))
        self.engine._texture_ts -= 5.0
        new = np.full((128, 128, 3), 255, dtype=np.uint8)
        self.assertEqual(int(self.engine._blend_texture(new).mean()), 255)


class HairPassTests(unittest.TestCase):
    """The reference hair must land on the live head, and nowhere else."""

    def _reference(self) -> tuple[np.ndarray, np.ndarray]:
        image = np.full((320, 320, 3), 230, dtype=np.uint8)      # bright wall
        cv2.ellipse(image, (100, 100), (58, 74), 0, 0, 360, (24, 26, 30), -1)   # hair
        cv2.ellipse(image, (100, 168), (48, 60), 0, 0, 360, (150, 180, 210), -1)  # face
        points = FACE_POINTS + np.array([0.0, 95.0], dtype=np.float32)
        return image, points

    def test_mask_covers_hair_and_leaves_the_face(self) -> None:
        from mimiccam import hair

        image, points = self._reference()
        x0, y0, x1, y1 = hair.head_crop(points, image.shape[:2])
        mask = hair.hair_mask(image[y0:y1, x0:x1], points, origin=(x0, y0))
        self.assertGreater(float(mask[95 - y0, 100 - x0]), 0.6)      # hair
        self.assertLess(float(mask[168 - y0, 100 - x0]), 0.2)        # cheek
        self.assertLess(float(mask[5, 5]), 0.05)                      # wall

    def test_pass_is_active_and_follows_a_scaled_face(self) -> None:
        from mimiccam import hair

        image, points = self._reference()
        pass_ = hair.HairPass(image, points)
        self.assertTrue(pass_.active)

        live = np.full((480, 640, 3), 128, dtype=np.uint8)
        live_points = points * 2.0
        out = pass_.apply(live, live_points)

        matrix = face.similarity_matrix(points, live_points)
        spot = cv2.transform(np.array([[[100.0, 95.0]]], np.float32), matrix)[0, 0]
        patch = out[int(spot[1]) - 6:int(spot[1]) + 6, int(spot[0]) - 6:int(spot[0]) + 6]
        self.assertLess(float(patch.mean()), 110.0)          # hair arrived
        self.assertGreater(float(out[5, 5].mean()), 120.0)   # wall untouched

    def test_hairless_reference_is_inactive(self) -> None:
        from mimiccam import hair

        image = np.full((320, 320, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (100, 168), (60, 80), 0, 0, 360, (150, 180, 210), -1)
        points = FACE_POINTS + np.array([0.0, 95.0], dtype=np.float32)
        pass_ = hair.HairPass(image, points)
        self.assertFalse(pass_.active)
        live = np.full((240, 240, 3), 90, dtype=np.uint8)
        self.assertTrue(np.array_equal(pass_.apply(live, points), live))

    def test_tiny_face_is_left_alone(self) -> None:
        from mimiccam import hair

        image, points = self._reference()
        pass_ = hair.HairPass(image, points)
        live = np.full((240, 240, 3), 90, dtype=np.uint8)
        tiny = points - np.array([60.0, 60.0], np.float32)
        tiny[:, :] = tiny[0] + np.array([[0, 0], [3, 0], [1.5, 1.5], [0.5, 3], [2.5, 3]], np.float32)
        self.assertTrue(np.array_equal(pass_.apply(live, tiny), live))


class MatteTests(unittest.TestCase):
    def test_refine_trims_the_uncertain_rim(self) -> None:
        from mimiccam import matte

        alpha = np.full((40, 40), 0.04, dtype=np.float32)  # only background noise
        self.assertLess(float(matte.refine(alpha).max()), 0.001)
        solid = np.full((40, 40), 0.99, dtype=np.float32)
        self.assertGreater(float(matte.refine(solid).mean()), 0.9)

    def test_unmix_recovers_a_dark_hair_edge_from_a_bright_wall(self) -> None:
        from mimiccam import matte

        background = np.full((60, 60, 3), 230, dtype=np.uint8)
        # Half-covered hair: 0.5 * 30 + 0.5 * 230 = 130
        frame = np.full((60, 60, 3), 130, dtype=np.uint8)
        alpha = np.full((60, 60), 0.5, dtype=np.float32)
        out = matte.unmix(frame, alpha, background)
        self.assertLess(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()), 90)

    def test_decontaminate_paints_the_rim_with_the_person_not_the_wall(self) -> None:
        from mimiccam import matte

        frame = np.full((120, 120, 3), 235, dtype=np.uint8)   # bright camera wall
        frame[30:90, 30:90] = 20                              # dark hair
        alpha = np.zeros((120, 120), np.float32)
        alpha[30:90, 30:90] = 1.0
        alpha[26:34, 26:94] = 0.5                             # soft edge band
        out = matte.decontaminate(frame, alpha)
        # The soft ring is now the hair's colour, not a pale grey of wall + hair.
        self.assertLess(float(out[30, 60].mean()), 60)
        self.assertLess(float(out[60, 30].mean()), 60)

    def test_decontaminate_leaves_the_inside_and_the_background_alone(self) -> None:
        from mimiccam import matte

        rng = np.random.default_rng(11)
        frame = rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)
        alpha = np.zeros((90, 90), np.float32)
        alpha[30:60, 30:60] = 1.0
        out = matte.decontaminate(frame, alpha)
        self.assertTrue(np.array_equal(out[35:55, 35:55], frame[35:55, 35:55]))
        self.assertTrue(np.array_equal(out[0:20, 0:20], frame[0:20, 0:20]))
        self.assertEqual(out.shape, frame.shape)

    def test_background_estimate_ignores_the_person(self) -> None:
        from mimiccam import matte

        frame = np.full((160, 160, 3), 200, dtype=np.uint8)
        frame[50:130, 50:110] = 20  # a dark person in the middle
        alpha = np.zeros((160, 160), np.float32)
        alpha[50:130, 50:110] = 1.0
        estimate = matte.background_estimate(frame, alpha)
        self.assertGreater(float(estimate[5:20, 5:20].mean()), 170)
        self.assertGreater(float(estimate[145, 145].mean()), 170)
        self.assertGreater(float(estimate[20, 145].mean()), 170)


class StudioTests(unittest.TestCase):
    def test_cover_resize_fills_the_frame_without_bars(self) -> None:
        from mimiccam import studio

        image = np.zeros((300, 500, 3), dtype=np.uint8)
        out = studio.cover_resize(image, 320, 240)
        self.assertEqual(out.shape, (240, 320, 3))

    def test_person_mask_is_a_dilated_silhouette(self) -> None:
        from mimiccam import studio

        alpha = np.zeros((100, 100), np.float32)
        alpha[30:70, 30:70] = 1.0
        mask = studio.person_mask(alpha)
        self.assertEqual(mask.max(), 255)
        self.assertGreater(float(mask.sum()), 40 * 40 * 255)

    def test_seat_is_bottom_anchored_and_never_shrinks(self) -> None:
        from mimiccam.studio import StudioScene

        plate = np.full((720, 1280, 3), 30, dtype=np.uint8)
        scene = StudioScene(plate, reference_face=(567, 70, 246, 317))
        seat = scene.seat_person((470, 466, 229, 254))
        self.assertGreaterEqual(seat.scale, 1.0)
        self.assertLessEqual(seat.scale, 1.55)
        self.assertAlmostEqual(seat.ty, 720 * (1 - seat.scale), places=3)
        self.assertLessEqual(seat.tx, 0.0)
        self.assertGreaterEqual(seat.tx, 1280 * (1 - seat.scale))
        # Measured once: head movement must not rescale the body.
        self.assertIs(scene.seat_person((300, 300, 100, 100)), seat)

    def test_close_up_camera_shrinks_into_the_reference_seat(self) -> None:
        from mimiccam.studio import StudioScene

        plate = np.full((720, 1280, 3), 30, dtype=np.uint8)
        scene = StudioScene(plate, reference_face=(567, 70, 250, 317))
        # A webcam close-up: the face is twice the width of the reference face.
        seat = scene.seat_person((300, 420, 500, 520))
        self.assertLess(seat.scale, 1.0)
        self.assertGreaterEqual(seat.scale, 0.55)
        self.assertAlmostEqual(seat.ty, 720 * (1 - seat.scale), places=3)
        # The face still lands on the reference chair, not off the frame.
        face_centre = seat.scale * (300 + 250) + seat.tx
        self.assertAlmostEqual(face_centre, 567 + 125, delta=2.0)

    def test_offscreen_face_cannot_push_the_subject_out(self) -> None:
        from mimiccam.studio import StudioScene

        plate = np.full((720, 1280, 3), 30, dtype=np.uint8)
        scene = StudioScene(plate, reference_face=(567, 70, 250, 317))
        # Face hard against the right edge of a close-up frame.
        seat = scene.seat_person((1150, 420, 500, 520))
        self.assertLess(seat.scale, 1.0)
        face_centre = seat.scale * (1150 + 250) + seat.tx
        self.assertGreaterEqual(face_centre, 1280 * 0.24 - 1)
        self.assertLessEqual(face_centre, 1280 * 0.76 + 1)

    def test_composite_keeps_the_room_where_there_is_no_person(self) -> None:
        from mimiccam.studio import StudioScene

        plate = np.full((180, 320, 3), 70, dtype=np.uint8)
        plate[40:120, 100:200] = 200
        scene = StudioScene(plate, reference_face=(120, 20, 60, 70))
        frame = np.full((180, 320, 3), 255, dtype=np.uint8)
        alpha = np.zeros((180, 320), np.float32)
        alpha[60:180, 90:230] = 1.0
        out = scene.composite(frame, alpha, face_box=(140, 40, 50, 60))
        # The top-right corner is outside the person and outside the cast
        # shadow's reach: it must still be exactly the room.
        self.assertTrue(np.array_equal(out[0:6, 310:320], plate[0:6, 310:320]))
        self.assertGreater(float(out[150, 160].mean()), 200)

    def test_relight_keeps_the_persons_detail(self) -> None:
        from mimiccam.studio import StudioScene

        rng = np.random.default_rng(11)
        # A busy texture stands in for hair and skin detail: the relight must
        # recolour it, not blur it. Building the correction on a quarter-size
        # copy and interpolating back dropped Laplacian variance 589 -> 13.
        person = rng.integers(60, 200, (240, 200, 3), dtype=np.uint8)
        person[::2] = (person[::2].astype(np.int16) // 2).astype(np.uint8)
        plate = np.full((240, 200, 3), 210, dtype=np.uint8)
        scene = StudioScene(plate, reference_face=(80, 20, 60, 70))
        alpha = np.ones((240, 200), np.float32)
        out = scene._relight(person, alpha)

        grey = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        source_detail = float(cv2.Laplacian(grey(person), cv2.CV_32F).var())
        relit_detail = float(cv2.Laplacian(grey(out), cv2.CV_32F).var())
        self.assertGreater(relit_detail, source_detail * 0.8)
        # And it did do something: the person is pulled towards the bright room.
        self.assertGreater(float(out.mean()), float(person.mean()))

    def test_composite_alpha_zero_returns_the_plate(self) -> None:
        from mimiccam.studio import StudioScene

        plate = np.full((120, 160, 3), 42, dtype=np.uint8)
        scene = StudioScene(plate, reference_face=(60, 20, 40, 50))
        scene.seat_person((60, 60, 40, 50))
        out = scene.composite(np.zeros((120, 160, 3), np.uint8),
                              np.zeros((120, 160), np.float32), None)
        self.assertTrue(np.array_equal(out, plate))


class MatteCadenceTests(unittest.TestCase):
    """The matte can run every second frame, but must default to every frame."""

    def test_matte_cadence_defaults_to_every_frame(self) -> None:
        from mimiccam.local_engine import LocalEngine

        self.assertEqual(LocalEngine().matte_every, 1)
        self.assertEqual(LocalEngine(matte_every=2).matte_every, 2)
        # A zero or negative cadence would divide by zero in the frame loop.
        self.assertEqual(LocalEngine(matte_every=0).matte_every, 1)


class CameraProbeTests(unittest.TestCase):
    """A camera judged on its first (dark) frame looks broken - this is the fix."""

    class FakeCapture:
        def __init__(self, frames):
            self.frames = list(frames)

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

    def _frame(self, level: int, noise: int = 0) -> np.ndarray:
        frame = np.full((48, 64, 3), level, dtype=np.uint8)
        if noise:
            rng = np.random.default_rng(level)
            frame = np.clip(frame.astype(int) + rng.integers(-noise, noise, frame.shape), 0, 255)
            frame = frame.astype(np.uint8)
        return frame

    def test_waits_for_exposure_to_ramp_up(self) -> None:
        from mimiccam import capture

        dark = [self._frame(0) for _ in range(6)]
        lit = self._frame(140, noise=40)
        best = capture.read_best_frame(self.FakeCapture(dark + [lit]))
        self.assertIsNotNone(best)
        self.assertGreater(float(best.mean()), 100)

    def test_returns_dark_frame_when_never_lit(self) -> None:
        from mimiccam import capture

        best = capture.read_best_frame(self.FakeCapture([self._frame(3)] * 4))
        self.assertIsNotNone(best)
        self.assertLess(float(best.mean()), 8)

    def test_device_without_frames_is_reported_as_none(self) -> None:
        from mimiccam import capture

        self.assertIsNone(capture.read_best_frame(self.FakeCapture([]), attempts=3, budget_s=0.05))

    def test_lit_cameras_are_ordered_first(self) -> None:
        from mimiccam.capture import CameraEntry, order_entries

        entries = [
            CameraEntry(index=0, width=1280, height=720, mean_luma=0.2),
            CameraEntry(index=1, width=1920, height=1080, mean_luma=120.0),
        ]
        self.assertEqual([e.index for e in order_entries(entries)], [1, 0])

    def test_black_camera_is_labelled(self) -> None:
        from mimiccam.capture import CameraEntry

        self.assertIn("no light", CameraEntry(0, 1280, 720, 0.1).label)
        self.assertNotIn("no light", CameraEntry(1, 1280, 720, 90.0).label)


class ModelBookkeepingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original = models.ENV_MODELS_DIR
        self._set_dir(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._set_dir(None)
        models.ENV_MODELS_DIR = self._original
        self._tmp.cleanup()

    def _set_dir(self, path: Path | None) -> None:
        import os

        if path is None:
            os.environ.pop(models.ENV_MODELS_DIR, None)
        else:
            os.environ[models.ENV_MODELS_DIR] = str(path)

    def test_empty_dir_reports_every_model_missing(self) -> None:
        # The default check is the core engine. The studio models are optional
        # until the user asks for the reference room.
        self.assertEqual(len(models.missing()), len(models.CORE_KEYS))
        self.assertFalse(models.is_ready())
        self.assertGreater(models.total_missing_bytes(), 500 * 1024 * 1024)
        self.assertIn("Missing", models.describe_missing(models.missing()))
        self.assertEqual(len(models.missing(models.STUDIO_KEYS)), len(models.STUDIO_KEYS))
        self.assertTrue(models.is_ready(models.STUDIO_KEYS) is False)

    def test_truncated_file_is_not_accepted(self) -> None:
        spec = models.BY_KEY["swapper"]
        models.path("swapper").write_bytes(b"0" * 1024)
        self.assertFalse(models.present("swapper"))
        self.assertIn(spec, models.missing())

    def test_correctly_sized_file_is_accepted(self) -> None:
        spec = models.BY_KEY["detector"]
        # Sparse file of exactly the right size: no disk cost, correct check.
        with open(models.path("detector"), "wb") as handle:
            handle.truncate(spec.size_bytes)
        self.assertTrue(models.present("detector"))

    def test_download_rejects_short_files(self) -> None:
        # A mirror that returns an error page instead of weights must not be
        # stored as a model: the size check is what protects the engine.
        source = Path(self._tmp.name) / "short.onnx"
        source.write_bytes(b"x" * 10)
        original_specs = models.SPECS
        original = models.BY_KEY["detector"]
        broken = replace(original, url=source.as_uri())
        models.SPECS = tuple(broken if s.key == "detector" else s for s in original_specs)
        models.BY_KEY["detector"] = broken
        try:
            with self.assertRaises(models.ModelError):
                models.download("detector")
            self.assertFalse(models.path("detector").with_suffix(".onnx.part").exists())
        finally:
            models.SPECS = original_specs
            models.BY_KEY["detector"] = original

    def test_licence_summary_names_the_research_models(self) -> None:
        text = models.licence_summary()
        self.assertIn("inswapper_128.onnx", text)
        self.assertIn("non-commercial research", text)


class MouthMaskTests(unittest.TestCase):
    """The lips are the one feature taken from the camera, not the render.

    inswapper has no mouth tracking, so a swapped face keeps the character's
    mouth shape through every sentence. These tests pin the geometry that puts
    the live lips back: centred on the mouth, scaled to its width, rolled with
    the head, and never reaching as high as the eyes.
    """

    SIZE = 128

    def _landmarks(self, width: float = 24.0, angle: float = 0.0,
                   centre: tuple[float, float] = (64.0, 92.0)) -> np.ndarray:
        half = width / 2.0
        dx, dy = np.cos(angle) * half, np.sin(angle) * half
        return np.array(
            [
                [40.0, 40.0], [88.0, 40.0], [64.0, 66.0],
                [centre[0] - dx, centre[1] - dy],
                [centre[0] + dx, centre[1] + dy],
            ],
            np.float32,
        )

    def _alpha(self, landmarks: np.ndarray) -> np.ndarray:
        return face.mouth_alpha(self.SIZE, landmarks, np.eye(2, 3, dtype=np.float32))

    def test_mask_sits_on_the_lips_and_away_from_the_eyes(self) -> None:
        alpha = self._alpha(self._landmarks())
        self.assertIsNotNone(alpha)
        ys, xs = np.nonzero(alpha > 40)
        self.assertGreater(ys.size, 40)
        self.assertAlmostEqual(float(xs.mean()), 64.0, delta=3.0)
        # The corners sit on the lip line; the lips hang below it.
        self.assertGreater(float(ys.mean()), 92.0)
        self.assertLess(float(ys.mean()), 104.0)
        self.assertEqual(int(alpha[:50].max()), 0)

    def test_a_wider_mouth_gets_a_wider_mask(self) -> None:
        narrow = self._alpha(self._landmarks(width=16.0))
        wide = self._alpha(self._landmarks(width=32.0))
        self.assertGreater((wide > 40).sum(), 2.5 * (narrow > 40).sum())

    def test_mask_rolls_with_the_head(self) -> None:
        for degrees in (30.0, -30.0):
            alpha = self._alpha(self._landmarks(angle=float(np.radians(degrees))))
            moments = cv2.moments((alpha > 40).astype(np.uint8), binaryImage=True)
            theta = 0.5 * np.degrees(
                np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
            )
            error = ((theta - degrees + 90) % 180) - 90
            self.assertAlmostEqual(float(error), 0.0, delta=8.0)

    def test_degenerate_landmarks_are_ignored(self) -> None:
        self.assertIsNone(self._alpha(self._landmarks(width=0.0)))
        self.assertIsNone(
            face.mouth_alpha(
                self.SIZE, np.zeros((3, 2), np.float32), np.eye(2, 3, dtype=np.float32)
            )
        )

    def test_blend_takes_the_live_pixels_where_the_mask_is_opaque(self) -> None:
        base = np.full((8, 8, 3), 10, np.uint8)
        top = np.full((8, 8, 3), 200, np.uint8)
        alpha = np.zeros((8, 8), np.uint8)
        alpha[:, :4] = 255
        out = face.blend_alpha(base, top, alpha)
        self.assertTrue((out[:, :4] == 200).all())
        self.assertTrue((out[:, 4:] == 10).all())


class ExpressionTests(unittest.TestCase):
    """Eyes and mouth both come from the camera: the face has to blink."""

    SIZE = 128

    def setUp(self) -> None:
        self.landmarks = np.array(
            [
                [40.0, 46.0], [88.0, 46.0], [64.0, 72.0],
                [52.0, 96.0], [76.0, 96.0],
            ],
            np.float32,
        )
        self.matrix = np.eye(2, 3, dtype=np.float32)

    def test_eye_mask_lands_on_the_eyes_only(self) -> None:
        alpha = face.eye_alpha(self.SIZE, self.landmarks, self.matrix)
        self.assertIsNotNone(alpha)
        ys, xs = np.nonzero(alpha > 40)
        self.assertGreater(ys.size, 20)
        self.assertLess(float(ys.mean()), 60.0)
        # Both eyes, not one wide smear between them.
        self.assertGreater(float(xs.min()), 25.0)
        self.assertLess(float(xs.max()), 103.0)

    def test_a_blink_filtered_mask_covers_eyes_and_mouth_but_not_the_nose(self) -> None:
        combined = face.expression_alpha(self.SIZE, self.landmarks, self.matrix)
        self.assertIsNotNone(combined)
        self.assertGreater(int(combined[46, 40]), 40)  # left eye
        self.assertGreater(int(combined[46, 88]), 40)  # right eye
        self.assertGreater(int(combined[98, 64]), 40)  # below the lip line
        self.assertLess(int(combined[70, 64]), 40)  # bridge of the nose stays rendered

    def test_a_single_eye_still_gets_a_mask(self) -> None:
        far_eye = self.landmarks.copy()
        far_eye[1] = [200.0, 46.0]  # one landmark dragged out of the crop
        combined = face.expression_alpha(self.SIZE, far_eye, self.matrix)
        self.assertIsNotNone(combined)

    def test_degenerate_landmarks_return_nothing(self) -> None:
        squashed = self.landmarks.copy()
        squashed[1] = squashed[0]
        self.assertIsNone(face.eye_alpha(self.SIZE, squashed, self.matrix))
        self.assertIsNone(
            face.expression_alpha(self.SIZE, np.zeros((2, 2), np.float32), self.matrix)
        )


if __name__ == "__main__":
    unittest.main()
