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


if __name__ == "__main__":
    unittest.main()
