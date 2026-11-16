# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the GFPGAN detail pass.

No 340 MB download and no GPU: a stand-in session proves the tensor contract -
shape, range, channel order and the fall-back when the graph returns garbage.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import face, models  # noqa: E402


class _EchoSession:
    """Stands in for the ONNX graph: returns whatever it was given."""

    def __init__(self, transform=None) -> None:
        self.seen: np.ndarray | None = None
        self.transform = transform

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _names, feeds):
        blob = feeds["input"]
        self.seen = blob
        return [blob if self.transform is None else self.transform(blob)]


def _texture(size: int = 128) -> np.ndarray:
    # Distinct channel means make a channel swap visible to the assertions.
    texture = np.zeros((size, size, 3), np.uint8)
    texture[:, :, 0] = 10  # B
    texture[:, :, 1] = 20  # G
    texture[:, :, 2] = 30  # R
    return texture


class DetailTensorTests(unittest.TestCase):
    def test_prepare_matches_the_graph_contract(self) -> None:
        blob = face.FaceEnhancer.prepare(_texture())
        self.assertEqual(blob.shape, (1, 3, 512, 512))
        self.assertEqual(blob.dtype, np.float32)
        self.assertGreaterEqual(float(blob.min()), -1.0)
        self.assertLessEqual(float(blob.max()), 1.0)
        # BGR is deliberate: RGB turned skin orange when measured.
        self.assertAlmostEqual(float(blob[0, 0].mean()), (10 / 255 - 0.5) / 0.5, places=4)
        self.assertAlmostEqual(float(blob[0, 1].mean()), (20 / 255 - 0.5) / 0.5, places=4)
        self.assertAlmostEqual(float(blob[0, 2].mean()), (30 / 255 - 0.5) / 0.5, places=4)

    def test_restore_returns_the_networks_own_resolution(self) -> None:
        # Shrinking back to 128 would throw away the pixels GFPGAN invented.
        output = np.zeros((1, 3, 512, 512), np.float32)
        restored = face.FaceEnhancer.restore(output)
        self.assertEqual(restored.shape, (512, 512, 3))
        self.assertEqual(restored.dtype, np.uint8)

    def test_enhance_returns_a_512_texture_in_colour_order(self) -> None:
        session = _EchoSession()
        enhancer = face.FaceEnhancer("unused.onnx", session=session)
        out = enhancer.enhance(_texture())
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(session.seen.shape, (1, 3, 512, 512))
        self.assertGreater(float(out[:, :, 2].mean()), float(out[:, :, 0].mean()))

    def test_the_enhanced_texture_keeps_a_share_of_the_render(self) -> None:
        """GFPGAN drifts identity; the raw render is mixed back in underneath.

        Measured on a real frame against the character photo: identity 0.637
        with the pure enhanced texture, 0.677 with the mix.
        """
        # An echo graph that answers brighter than it was asked: a stand-in for
        # a detail pass that has changed every pixel.
        session = _EchoSession(transform=lambda blob: blob * 0.5 + 0.5)
        enhancer = face.FaceEnhancer("unused.onnx", session=session)
        out = enhancer.enhance(_texture())
        blue = float(out[:, :, 0].mean())
        enhanced_only = float(np.clip((-0.9216 * 0.5 + 0.5 + 1.0) * 127.5, 0, 255))
        self.assertGreater(blue, 10.0, "the detail pass still shows")
        self.assertLess(blue, enhanced_only - 5.0, "the render is mixed back in")

    def test_a_wrongly_shaped_output_is_rejected(self) -> None:
        # A graph that answers with a different resolution is not usable.
        session = _EchoSession(transform=lambda blob: np.ones((1, 3, 64, 64), np.float32))
        enhancer = face.FaceEnhancer("unused.onnx", session=session)
        source = _texture()
        self.assertIs(enhancer.enhance(source), source)

    def test_a_flat_output_is_treated_as_a_failure(self) -> None:
        # A broken graph returns a constant plane; that must not reach the camera.
        session = _EchoSession(transform=lambda blob: np.zeros_like(blob))
        enhancer = face.FaceEnhancer("unused.onnx", session=session)
        source = _texture()
        self.assertIs(enhancer.enhance(source), source)


class EnhancerPasteTests(unittest.TestCase):
    """A 512 px texture must land exactly where the 128 px one would."""

    def setUp(self) -> None:
        self.swapper = face.FaceSwapper.__new__(face.FaceSwapper)
        self.swapper.size = 128
        self.swapper.keep_expression = False
        self.swapper._mask = self.swapper._oval_mask(128)
        self.swapper._masks = {128: self.swapper._mask}
        self.frame = np.full((240, 320, 3), 40, dtype=np.uint8)

    def test_the_transform_scales_with_the_texture(self) -> None:
        base = face.norm_matrix(FACE_POINTS, 128)
        big = self.swapper.matrix_for(FACE_POINTS, 512)
        np.testing.assert_allclose(big, np.vstack([base * 4.0, [0.0, 0.0, 1.0]])[:2], atol=1e-3)

    def test_a_512_texture_pastes_where_the_128_one_does(self) -> None:
        small = np.zeros((128, 128, 3), np.uint8)
        small[:, :] = 200
        large = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
        a = self.swapper.paste(self.frame, small, FACE_POINTS)
        b = self.swapper.paste(self.frame, large, FACE_POINTS)
        self.assertEqual(a.shape, self.frame.shape)
        # Same footprint and the same blended value: only sampling differs.
        self.assertLess(float(np.abs(a.astype(int) - b.astype(int)).mean()), 1.0)
        self.assertGreater(float(b[45:100, 90:180].mean()), 100.0)
        self.assertLess(float(b[0:10, 0:10].mean()), 45.0)


class _FakeSwapper:
    """The 128 px network, replaced by a constant render."""

    size = 128

    def render(self, frame, landmarks, embedding):
        return np.full((128, 128, 3), 60, dtype=np.uint8)

    def paste(self, frame, texture, landmarks):
        out = frame.copy()
        out[:8, :8] = texture[3, 3]
        return out


class _Brightener:
    """Paints the texture white, so "the detail pass ran" is visible."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def enhance(self, texture):
        self.calls += 1
        if self.fail:
            raise RuntimeError("graph unavailable")
        return np.full_like(texture, 255)


class _FakeDetector:
    def __init__(self) -> None:
        self.faces: list = []

    def detect(self, frame):
        return list(self.faces)


FACE_POINTS = np.array(
    [[70, 70], [130, 70], [100, 95], [85, 125], [115, 125]], dtype=np.float32
)


class DetailEngineTests(unittest.TestCase):
    """The first frame must be as sharp as the ones the render thread makes."""

    def _engine(self, enhancer=None, detail: bool = True):
        from mimiccam.local_engine import LocalEngine

        engine = LocalEngine(providers=["CPUExecutionProvider"], detail=detail)
        engine.width, engine.height = 128, 128
        engine._swapper = _FakeSwapper()
        engine._detector = _FakeDetector()
        engine._detector.faces = [
            face.Detection(box=(20, 20, 108, 108), score=0.9, landmarks=FACE_POINTS)
        ]
        engine._embedding = np.zeros((512,), np.float32)
        engine._enhancer = enhancer
        return engine

    def test_the_first_render_is_enhanced_before_it_is_pasted(self) -> None:
        enhancer = _Brightener()
        engine = self._engine(enhancer)
        out = engine._process(np.full((128, 128, 3), 30, dtype=np.uint8))
        self.assertEqual(enhancer.calls, 1)
        # The paste carries the enhanced texture, not the raw render.
        self.assertEqual(int(out[0, 0].mean()), 255)

    def test_a_broken_enhancer_costs_detail_and_nothing_else(self) -> None:
        enhancer = _Brightener(fail=True)
        engine = self._engine(enhancer)
        out = engine._process(np.full((128, 128, 3), 30, dtype=np.uint8))
        self.assertEqual(int(out[0, 0].mean()), 60)  # the raw render
        self.assertIsNone(engine._enhancer)  # dropped, not retried every frame
        self.assertEqual(engine.frames_no_face, 0)

    def test_detail_off_never_loads_the_network(self) -> None:
        engine = self._engine(detail=False)
        engine._enhancer = None
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[models.ENV_MODELS_DIR] = tmp
            try:
                engine._swapper = _FakeSwapper()
                engine._load_enhancer()
            finally:
                os.environ.pop(models.ENV_MODELS_DIR, None)
        self.assertIsNone(engine._enhancer)
        self.assertEqual(int(engine._process(np.full((128, 128, 3), 30, np.uint8))[0, 0, 0]), 60)


class DetailRegistryTests(unittest.TestCase):
    def test_the_model_is_registered_as_optional_apache_2(self) -> None:
        spec = models.BY_KEY["enhancer"]
        self.assertEqual(spec.filename, "gfpgan_1.4.onnx")
        self.assertIn("Apache-2.0", spec.licence)
        self.assertGreater(spec.size_bytes, 300 * 1024 * 1024)
        self.assertEqual(models.DETAIL_KEYS, ("enhancer",))
        # Optional means it is not part of the mandatory first-run download.
        self.assertNotIn("enhancer", models.CORE_KEYS)

    def test_it_is_reported_missing_until_the_file_exists(self) -> None:
        original = os.environ.get(models.ENV_MODELS_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[models.ENV_MODELS_DIR] = tmp
            try:
                self.assertFalse(models.present("enhancer"))
                self.assertIn(
                    "gfpgan_1.4.onnx", models.describe_missing(models.missing(models.DETAIL_KEYS))
                )
            finally:
                if original is None:
                    os.environ.pop(models.ENV_MODELS_DIR, None)
                else:
                    os.environ[models.ENV_MODELS_DIR] = original


class SeamToneTests(unittest.TestCase):
    """The ramp is read where the person is, and the reading is held steady.

    Both were measured on a real backlit frame with the head panning. The ramp
    runs along the jaw, so part of it is the bright wall the camera metered for:
    reading there pulled the whole pasted face toward the wall (luma 56 where the
    frame's own face sat at 46) and rode the wall's noise as the head moved (sd
    2.1 levels against a source face's 0.4). Gating the ramp to skin and locking
    the offset across frames brought the face to 44.6 and its swing to 4.5 levels
    against the source's 1.5.
    """

    def _ramp_scene(self):
        """A face at 60 with the wall at 220 across one side of the ramp."""
        warped = np.full((96, 96, 3), 60, np.uint8)
        target = np.full((96, 96, 3), 60, np.uint8)
        target[24:72, 60:72] = 220  # the wall, beyond the jaw, inside the ramp
        alpha = np.zeros((96, 96), np.uint8)
        alpha[24:72, 24:72] = 128  # the ramp
        alpha[36:60, 36:60] = 255  # the face's interior
        return warped, target, alpha

    def test_the_ramp_is_read_where_the_frame_shows_skin(self) -> None:
        from unittest import mock

        warped, target, alpha = self._ramp_scene()
        gated = face.seam_offset(warped, target, alpha)
        self.assertIsNotNone(gated)
        self.assertLess(float(np.abs(gated).max()), 6.0,
                        f"the wall still reached the face ({gated})")
        with mock.patch.object(face, "SEAM_GATE", (0.0, 99.0)):
            whole = face.seam_offset(warped, target, alpha)
        self.assertGreater(float(whole.max()), 30.0,
                           "the ungated reading no longer sees the wall to fix")
        self.assertGreater(float(whole.max()), float(np.abs(gated).max()) * 3,
                           "the gate has to be what removes the wall, not luck")

    def test_a_ramp_of_pure_background_is_not_used_at_all(self) -> None:
        # The mask has slid off the jaw: every ramp pixel is wall, not skin.
        warped = np.full((96, 96, 3), 60, np.uint8)
        target = np.full((96, 96, 3), 60, np.uint8)
        alpha = np.zeros((96, 96), np.uint8)
        alpha[24:72, 24:72] = 128
        alpha[36:60, 36:60] = 255
        target[24:72, 24:72] = 220
        target[36:60, 36:60] = 60
        self.assertIsNone(face.seam_offset(warped, target, alpha))

    def test_a_single_reading_cannot_step_the_face(self) -> None:
        lock = face.ToneLock()
        readings = [np.array([4.0, 6.0, 5.0], np.float32),
                    np.array([4.0, -6.0, 5.0], np.float32)]
        values = []
        for index in range(40):
            values.append(lock.smooth(readings[index % 2]).copy())
        steps = np.abs(np.diff(np.array(values), axis=0))
        self.assertLessEqual(float(steps.max()), face.SEAM_SLEW + 1e-6,
                             "one frame moved the face more than the slew allows")
        self.assertLess(abs(float(values[-1][1])), 2.0,
                        f"the lock drifted with the reading (ended at {values[-1][1]:.1f})")

    def test_the_lock_still_follows_a_change_of_light(self) -> None:
        lock = face.ToneLock()
        for _ in range(10):
            lock.smooth(np.zeros(3, np.float32))
        for _ in range(30):
            lock.smooth(np.array([0.0, 20.0, 0.0], np.float32))
        self.assertAlmostEqual(float(lock.value[1]), 20.0, delta=1.5,
                               msg="a lamp being switched on has to land within a second")

    def test_a_new_face_starts_the_lock_over(self) -> None:
        lock = face.ToneLock()
        lock.smooth(np.array([0.0, 12.0, 0.0], np.float32))
        lock.reset()
        np.testing.assert_allclose(lock.smooth(np.array([0.0, -4.0, 0.0], np.float32)),
                                   [0.0, -4.0, 0.0])

    def test_two_pastes_of_one_texture_do_not_jump(self) -> None:
        """A frame later, on a frame that needs no correction, the face holds."""
        size = 200
        frame = np.full((size, size, 3), 60, np.uint8)
        texture = np.full((128, 128, 3), 90, np.uint8)
        matrix = np.array([[1.0, 0.0, -36], [0.0, 1.0, -36]], np.float32)
        mask = face.FaceSwapper._oval_mask(128)
        inside = mask >= 0.99
        region = (slice(36, 36 + 128), slice(36, 36 + 128))

        def face_level(image):
            return float(image[region][inside].astype(np.float32).mean())

        lock = face.ToneLock()
        first = face_level(face.composite(frame, texture, mask, matrix, lock=lock))
        self.assertIsNotNone(lock.value)
        matched = frame.copy()
        matched[region[0], region[1]] = texture  # this frame needs no correction
        held = face_level(face.composite(matched, texture, mask, matrix, lock=lock))
        plain = face_level(face.composite(matched, texture, mask, matrix))
        self.assertLess(abs(held - first), 3.0,
                        f"the face stepped instead of holding ({first:.0f} -> {held:.0f})")
        self.assertGreater(abs(plain - first), 15.0,
                           "the unlocked paste no longer shows the step to fix")


class SeamTests(unittest.TestCase):
    """No ring of one skin tone fading into another around the jaw.

    The render is matched to the frame's statistics only partially, so a
    constant colour step survives at the mask boundary and the soft ramp paints
    it as a pale blotch. These tests paste a crop of the frame back onto itself
    with a known tone shift: a correct composite keeps the shift inside the face
    and loses it at the edge.
    """

    ORIGIN = 56
    # A multiplicative shift, not a constant one: with a constant offset the
    # seam correction could undo the whole difference and the test would pass
    # without proving anything.
    GAIN = 0.72
    LIFT = 18.0

    def _scene(self):
        size = 240
        ramp = np.linspace(90.0, 140.0, size, dtype=np.float32)
        frame = np.empty((size, size, 3), np.float32)
        frame[:, :, 0] = 70.0 + ramp * 0.3   # B
        frame[:, :, 1] = 80.0 + ramp * 0.5   # G
        frame[:, :, 2] = 100.0 + ramp * 0.7  # R
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        o = self.ORIGIN
        crop = frame[o:o + 128, o:o + 128].astype(np.float32)
        texture = np.clip(crop * self.GAIN + self.LIFT, 0, 255).astype(np.uint8)
        return frame, texture

    def _matrix(self):
        # frame -> crop is the inverse of where the texture belongs, so the
        # texture lands at +ORIGIN in the frame.
        o = self.ORIGIN
        return np.array([[1.0, 0.0, -o], [0.0, 1.0, -o]], np.float32)

    def _composite(self):
        frame, texture = self._scene()
        mask = face.FaceSwapper._oval_mask(128)
        return frame, face.composite(frame, texture, mask, self._matrix()), mask

    def _in_frame(self, mask):
        region = np.zeros((240, 240), bool)
        o = self.ORIGIN
        region[o:o + 128, o:o + 128] = mask
        return region

    def test_the_edge_carries_no_colour_step(self) -> None:
        frame, out, mask = self._composite()
        alpha = face.cv2.convertScaleAbs(mask, alpha=255.0)
        band = self._in_frame((alpha > face.SEAM_BAND[0]) & (alpha < face.SEAM_BAND[1]))
        step = np.abs(out.astype(np.float32) - frame.astype(np.float32))[band].mean()
        self.assertLess(step, 6.0, f"the paste boundary still steps by {step:.1f} levels")

    def test_without_the_seam_correction_the_edge_would_show(self) -> None:
        """The fix has to be what removes the step, not the synthetic scene."""
        from unittest import mock

        frame, texture = self._scene()
        mask = face.FaceSwapper._oval_mask(128)
        with mock.patch.object(face, "seam_offset", lambda warped, target, alpha: None):
            naive = face.composite(frame, texture, mask, self._matrix())
        alpha = face.cv2.convertScaleAbs(mask, alpha=255.0)
        band = self._in_frame((alpha > face.SEAM_BAND[0]) & (alpha < face.SEAM_BAND[1]))
        step = np.abs(naive.astype(np.float32) - frame.astype(np.float32))[band].mean()
        self.assertGreater(step, 10.0, "the synthetic scene no longer has a seam to fix")

    def test_the_face_keeps_its_detail_inside_the_mask(self) -> None:
        """A DC correction must move the face, not flatten it onto the frame.

        The scene is nearly uniform, so matching the boundary also matches the
        centre's mean - what has to survive is the *structure*. The face is
        drawn from a texture with noise on it: the composite still has to carry
        that noise, at a correlation close to 1.
        """
        frame, texture = self._scene()
        rng = np.random.default_rng(7)
        texture = np.clip(
            texture.astype(np.float32) + rng.normal(0.0, 6.0, texture.shape), 0, 255
        ).astype(np.uint8)
        mask = face.FaceSwapper._oval_mask(128)
        out = face.composite(frame, texture, mask, self._matrix())

        alpha = face.cv2.convertScaleAbs(mask, alpha=255.0)
        inside = self._in_frame(alpha >= 250)
        o = self.ORIGIN
        drawn = texture[alpha >= 250].astype(np.float32)
        # out is 3-channel, drawn 3-channel: correlate the whole patch.
        got = out[o:o + 128, o:o + 128][alpha >= 250].astype(np.float32)
        corr = float(np.corrcoef(got.reshape(-1), drawn.reshape(-1))[0, 1])
        self.assertGreater(corr, 0.9, f"the pasted face lost its detail (corr {corr:.2f})")

    def test_a_face_half_off_the_frame_is_left_alone(self) -> None:
        flat = np.full((64, 64, 3), 200, np.uint8)
        empty = np.zeros((64, 64), np.uint8)
        shifted = face.seam_shift(flat, np.zeros((64, 64, 3), np.uint8), empty)
        self.assertTrue((shifted == 200).all(), "a ramp with no pixels must not be trusted")

        frame, texture = self._scene()
        mask = face.FaceSwapper._oval_mask(128)
        # Mirrored and pushed off the edge: the ramp is mostly outside the frame.
        matrix = np.array([[-1.0, 0.0, 20.0], [0.0, 1.0, 20.0]], np.float32)
        face.composite(frame, texture, mask, matrix)  # must not raise

    def test_the_correction_is_capped(self) -> None:
        warped = np.full((32, 32, 3), 10, np.uint8)
        target = np.full((32, 32, 3), 250, np.uint8)
        ramp = np.zeros((32, 32), np.uint8)
        ramp[8:24, 8:24] = 128  # a wide ramp band
        out = face.seam_shift(warped, target, ramp)
        self.assertLessEqual(float(out.max()), 10 + face.SEAM_LIMIT)


if __name__ == "__main__":
    unittest.main()
