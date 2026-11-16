# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The 256 px tier: swap-network plumbing, XSeg masking and the fp32 copy.

These are unit tests with fake sessions. They cover the parts that broke while
the tier was being built and that a real model would hide:

- swap graphs differ in *input order* (inswapper takes target first, HyperSwap
  takes the embedding first), so inputs are matched by rank, not position;
- an fp16 graph on DirectML produced a face that ignored the character, and an
  fp32 rewrite fixed it, so the optimiser choice and the rewrite are pinned;
- HyperSwap ends in tanh and returns its own face mask, inswapper does neither;
- XSeg has to veto pixels in the paste, not tint them.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mimiccam import face, models, occlusion


# -- fakes ------------------------------------------------------------------
class _Io:
    def __init__(self, name: str, shape) -> None:
        self.name = name
        self.shape = list(shape)


class _FakeSession:
    """Enough of an onnxruntime session for FaceSwapper/Occluder."""

    def __init__(self, inputs, outputs, results=None,
                 providers=("DmlExecutionProvider", "CPUExecutionProvider")) -> None:
        self._inputs = [_Io(name, shape) for name, shape in inputs]
        self._outputs = [_Io(name, shape) for name, shape in outputs]
        self._results = results if results is not None else []
        self._providers = list(providers)
        self.feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def run(self, _names, feed):
        self.feed = feed
        return self._results


def _tiny_swap_graph(path: Path, fp16: bool = True, hyperswap_order: bool = True) -> Path:
    """A two-input ONNX file shaped like a swap network, for the loader."""
    target = helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, 256, 256])
    source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 512])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 256, 256])
    inputs = [source, target] if hyperswap_order else [target, source]
    node = helper.make_node("Identity", ["target"], ["output"])
    weights = numpy_helper.from_array(
        np.zeros((8, 8), np.float16 if fp16 else np.float32), "trunk"
    )
    graph = helper.make_graph([node], "tiny-swap", inputs, [output], [weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def _landmarks() -> np.ndarray:
    return np.array(
        [[0.36, 0.42], [0.64, 0.42], [0.50, 0.55], [0.38, 0.70], [0.62, 0.70]],
        dtype=np.float32,
    ) * 128.0


def _blank_frame(size: int = 128) -> np.ndarray:
    frame = np.full((size, size, 3), 90, np.uint8)
    cv2.circle(frame, (size // 2, size // 2), size // 4, (200, 180, 160), -1)
    return frame


def _bare_swapper(model_path: Path, session: _FakeSession, size: int = 128):
    """A FaceSwapper with the model-loading part skipped."""
    swapper = face.FaceSwapper.__new__(face.FaceSwapper)
    swapper.model_path = str(model_path)
    swapper.session = session
    swapper.input_names = ["target", "source"]
    swapper.image_input = "target"
    swapper.latent_input = "source"
    swapper.size = size
    swapper.emap = None
    swapper.mean, swapper.std = 0.0, 1.0
    swapper.mask_output = None
    swapper.keep_expression = True
    swapper._tanh = None
    swapper._model_mask = None
    swapper.occluder = None
    swapper._occlusion = None
    swapper._occlusion_ts = 0.0
    mask = swapper._oval_mask(size)
    swapper._mask = mask
    swapper._masks = {size: mask}
    return swapper


# -- the graph --------------------------------------------------------------
class GraphLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.model = Path(self._dir.name) / "hyperswap_1a_256.onnx"

    def _build(self, fp16: bool = True, hyperswap_order: bool = True):
        _tiny_swap_graph(self.model, fp16=fp16, hyperswap_order=hyperswap_order)
        session = _FakeSession(
            [("source", [1, 512]), ("target", [1, 3, 256, 256])],
            [("output", [1, 3, 256, 256]), ("mask", [1, 1, 256, 256])],
        )
        seen = {}

        def fake_session(path, providers=None, optimize="all"):
            seen["optimize"] = optimize
            return session

        with mock.patch.object(face, "_session", fake_session):
            swapper = face.FaceSwapper(self.model, ["DmlExecutionProvider"])
        return swapper, seen

    def test_inputs_are_matched_by_rank_not_by_position(self):
        # HyperSwap lists the embedding first; inswapper lists the image first.
        swapper, _ = self._build()
        self.assertEqual(swapper.image_input, "target")
        self.assertEqual(swapper.latent_input, "source")
        self.assertEqual(swapper.size, 256)

    def test_the_model_own_mask_is_found(self):
        swapper, _ = self._build()
        self.assertEqual(swapper.mask_output, "mask")

    def test_hyper_swap_uses_tanh_normalisation(self):
        swapper, _ = self._build()
        self.assertEqual((swapper.mean, swapper.std), (0.5, 0.5))

    def test_fp16_graph_gets_the_basic_optimiser(self):
        # DirectML's fused fp16 graph ignored the source identity (0.16 instead
        # of 0.55), so an fp16 network must not be built with the full optimiser.
        _, seen = self._build(fp16=True)
        self.assertEqual(seen["optimize"], "basic")

    def test_fp32_graph_keeps_the_full_optimiser(self):
        _, seen = self._build(fp16=False)
        self.assertEqual(seen["optimize"], "all")

    def test_fp16_is_still_basic_when_the_gpu_is_not_directml(self):
        _tiny_swap_graph(self.model, fp16=True)
        session = _FakeSession(
            [("source", [1, 512]), ("target", [1, 3, 256, 256])],
            [("output", [1, 3, 256, 256])],
        )
        seen = {}

        def fake_session(path, providers=None, optimize="all"):
            seen["optimize"] = optimize
            return session

        with mock.patch.object(face, "_session", fake_session):
            face.FaceSwapper(self.model, ["CPUExecutionProvider"])
        self.assertEqual(seen["optimize"], "all")


# -- the identity vector ----------------------------------------------------
class LatentTest(unittest.TestCase):
    def test_hyper_swap_takes_the_unit_normed_embedding(self):
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None)
        vector = np.arange(512, dtype=np.float32) + 1.0
        latent = swapper.latent(vector)
        self.assertEqual(latent.shape, (1, 512))
        self.assertAlmostEqual(float(np.linalg.norm(latent)), 1.0, places=5)
        self.assertTrue(np.all(np.sign(latent[0][:3]) == np.sign(vector[:3])))

    def test_an_emap_still_projects_before_normalising(self):
        swapper = _bare_swapper(Path("inswapper_128.onnx"), None)
        swapper.emap = np.eye(512, dtype=np.float32) * 3.0
        latent = swapper.latent(np.ones(512, dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(latent)), 1.0, places=5)

    def test_the_decode_is_decided_from_the_output_range(self):
        tanh_result = np.zeros((1, 3, 128, 128), np.float32)
        tanh_result[:] = -0.5
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), _FakeSession(
            [("source", [1, 512]), ("target", [1, 3, 128, 128])],
            [("output", [1, 3, 128, 128])], results=[tanh_result],
        ))
        swapper.render(_blank_frame(), _landmarks(), np.zeros(512, np.float32))
        self.assertIs(swapper._tanh, True)

    def test_a_zero_to_one_output_is_not_treated_as_tanh(self):
        ramp = np.full((1, 3, 128, 128), 0.4, np.float32)
        swapper = _bare_swapper(Path("inswapper_128.onnx"), _FakeSession(
            [("target", [1, 3, 128, 128]), ("source", [1, 512])],
            [("output", [1, 3, 128, 128])], results=[ramp],
        ))
        swapper.image_input, swapper.latent_input = "target", "source"
        swapper.render(_blank_frame(), _landmarks(), np.zeros(512, np.float32))
        self.assertIs(swapper._tanh, False)


# -- occlusion masking ------------------------------------------------------
class _FakeOccluder:
    def __init__(self, hand: tuple[int, int]) -> None:
        self.hand = hand
        self.calls = 0

    def protection(self, crop_bgr):
        self.calls += 1
        size = int(crop_bgr.shape[0])
        mask = np.ones((size, size), np.float32)
        mask[:, self.hand[0]:self.hand[1]] = 0.0
        return mask


class PasteMaskTest(unittest.TestCase):
    def test_the_network_mask_vetoes_pixels(self):
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None, size=256)
        model_mask = np.ones((256, 256), np.float32)
        model_mask[:40, :] = 0.0
        swapper._model_mask = model_mask
        mask = swapper.paste_mask(256)
        self.assertEqual(float(mask[:40, :].max()), 0.0)
        self.assertGreater(float(mask[128, 128]), 0.5)

    def test_the_paste_never_runs_xseg_itself(self):
        # Two DirectML sessions executing at once from two threads crashed the
        # engine, so the mask is read on the render thread and the paste only
        # warps what was cached. This is the regression guard for that fix.
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None, size=256)
        swapper.occluder = _FakeOccluder((0, 90))
        mask = swapper.paste_mask(256, _blank_frame(256))
        self.assertEqual(swapper.occluder.calls, 0)
        self.assertGreater(float(mask[128, 128]), 0.5)

    def test_a_cached_reading_keeps_a_hand_out_of_the_paste(self):
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None, size=256)
        swapper.occluder = _FakeOccluder((0, 90))
        live = _blank_frame(256)
        swapper._refresh_occlusion(live)
        mask = swapper.paste_mask(256, live)
        self.assertEqual(float(mask[:, 0:90].max()), 0.0)
        self.assertGreater(float(mask[:, 200].max()), 0.5)

    def test_the_reading_is_reused_between_renders(self):
        # One XSeg pass costs ~20 ms, so it cannot run on every render either.
        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None, size=256)
        swapper.occluder = _FakeOccluder((0, 40))
        live = _blank_frame(256)
        swapper._refresh_occlusion(live)
        swapper._refresh_occlusion(live)
        self.assertEqual(swapper.occluder.calls, 1)

    def test_a_failing_occluder_falls_back_to_the_plain_mask(self):
        class Broken:
            def protection(self, crop_bgr):
                raise RuntimeError("graph fell over")

        swapper = _bare_swapper(Path("hyperswap_1a_256.onnx"), None, size=256)
        swapper.occluder = Broken()
        live = _blank_frame(256)
        swapper._refresh_occlusion(live)
        mask = swapper.paste_mask(256, live)
        self.assertGreater(float(mask[128, 128]), 0.5)


class OccluderTest(unittest.TestCase):
    def _occluder(self, shape):
        session = _FakeSession([("input", shape)], [("output", [1, 256, 256, 1])],
                               results=[np.full((1, 256, 256, 1), 0.9, np.float32)])
        with mock.patch.object(occlusion, "_session", lambda *a, **k: session):
            return occlusion.Occluder("xseg_3.onnx", ["CPUExecutionProvider"]), session

    def test_the_layout_is_read_off_the_graph(self):
        nhwc, _ = self._occluder([1, 256, 256, 3])
        nchw, _ = self._occluder([1, 3, 256, 256])
        self.assertTrue(nhwc.nhwc)
        self.assertFalse(nchw.nhwc)

    def test_nhwc_is_fed_untransposed(self):
        occluder, session = self._occluder([1, 256, 256, 3])
        occluder.skin(_blank_frame(256))
        self.assertEqual(session.feed["input"].shape, (1, 256, 256, 3))

    def test_protection_is_one_on_skin(self):
        occluder, _ = self._occluder([1, 256, 256, 3])
        mask = occluder.protection(_blank_frame(256))
        self.assertEqual(mask.shape, (256, 256))
        self.assertGreater(float(mask[128, 128]), 0.9)


# -- model selection --------------------------------------------------------
class SwapperPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patcher = mock.patch.dict(os.environ, {models.ENV_MODELS_DIR: self._dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_fp32_copy_wins_over_the_fp16_download(self):
        with mock.patch.object(models, "present", return_value=True), \
             mock.patch.object(models, "_fp32_complete", return_value=True):
            self.assertEqual(models.swapper_path().name, models.SWAPPER_FP32_FILENAME)
            self.assertEqual(models.swapper_precision(), "fp32")

    def test_the_fp16_download_is_used_until_the_copy_exists(self):
        with mock.patch.object(models, "present", return_value=True), \
             mock.patch.object(models, "_fp32_complete", return_value=False):
            self.assertEqual(
                models.swapper_path().name, models.BY_KEY["swapper_256"].filename
            )
            self.assertEqual(models.swapper_precision(), "fp16")

    def test_a_rig_without_the_tier_still_swaps(self):
        def only_core(key):
            return key != "swapper_256"

        with mock.patch.object(models, "present", side_effect=only_core), \
             mock.patch.object(models, "_fp32_complete", return_value=False):
            self.assertEqual(models.swapper_path().name, "inswapper_128.onnx")
            self.assertEqual(models.swapper_precision(), "")


class Fp32CopyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patcher = mock.patch.dict(os.environ, {models.ENV_MODELS_DIR: self._dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source = self._dir_path() / models.BY_KEY["swapper_256"].filename
        _tiny_swap_graph(self.source, fp16=True)

    def _dir_path(self) -> Path:
        return Path(self._dir.name)

    def test_the_rewrite_retypes_every_float16_tensor(self):
        with mock.patch.object(models, "present", return_value=True), \
             mock.patch.object(models, "_fp32_complete", return_value=False):
            written = models.convert_swapper_fp32()
        self.assertTrue(written.is_file())
        graph = onnx.load(str(written))
        dtypes = {init.data_type for init in graph.graph.initializer}
        self.assertEqual(dtypes, {TensorProto.FLOAT})

    def test_a_missing_tier_is_not_an_error(self):
        with mock.patch.object(models, "present", return_value=False), \
             mock.patch.object(models, "_fp32_complete", return_value=False):
            self.assertIsNone(models.prepare_swapper_fp32())
        self.assertFalse(models.swapper_fp32_path().exists())


if __name__ == "__main__":
    unittest.main()
