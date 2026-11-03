# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the low-light lift.

The numbers in the docstrings were measured on the machine this project is
developed on (RTX 4070 Laptop, 8 GB) by darkening a well-lit portrait in steps
and running it through the real YuNet, inswapper and ArcFace models:

    face luma   170  ->  19,387 pixels visibly changed by the swap
    face luma    85  ->   5,872
    face luma    51  ->   not detected at all

No model is loaded here: the curve, the measurement window and the engine's two
recovery paths are checked against stand-in sessions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import exposure, face  # noqa: E402

FACE_POINTS = np.array(
    [[70, 70], [130, 70], [100, 95], [85, 125], [115, 125]], dtype=np.float32
)


def _frame(luma: int, size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), luma, dtype=np.uint8)


class GammaCurveTests(unittest.TestCase):
    def test_the_curve_lands_the_face_on_the_target(self) -> None:
        # luma 35 is the backlit room the engine was tested in; 60 and 85 are the
        # dark-but-visible range the same measurement covered. Below ~20 the
        # floor takes over, and that is the next test.
        for luma in (25, 35, 60, 85):
            gamma = exposure.gamma_for_luma(float(luma))
            self.assertIsNotNone(gamma, luma)
            lifted = exposure.lut_for(gamma)[luma]
            self.assertAlmostEqual(float(lifted), exposure.TARGET_LUMA, delta=4.0)

    def test_the_floor_stops_the_lift_short_rather_than_overshooting(self) -> None:
        # gamma 0.33 is the floor for anything under about 13 luma.
        for luma in (2, 5, 10):
            gamma = exposure.gamma_for_luma(float(luma))
            self.assertEqual(gamma, exposure.GAMMA_MIN)
            self.assertLess(float(exposure.lut_for(gamma)[luma]), exposure.TARGET_LUMA)

    def test_a_face_that_is_already_lit_is_left_alone(self) -> None:
        self.assertIsNone(exposure.gamma_for_luma(exposure.TARGET_LUMA))
        self.assertIsNone(exposure.gamma_for_luma(170.0))
        self.assertIsNone(exposure.gamma_for_luma(255.0))

    def test_a_black_frame_is_not_lifted_forever(self) -> None:
        # luma 1 would ask for gamma 0.16; the floor stops the shadows turning
        # into sensor noise. A frame of pure black is left alone entirely: gamma
        # on zero is still zero, so the pass would be pure cost.
        self.assertEqual(exposure.gamma_for_luma(1.0), exposure.GAMMA_MIN)
        self.assertIsNone(exposure.gamma_for_luma(0.0))

    def test_black_is_black_and_white_is_white(self) -> None:
        lut = exposure.lut_for(0.4)
        self.assertEqual(int(lut[0]), 0)
        self.assertEqual(int(lut[255]), 255)
        # A gamma curve never inverts: every step is darker or equal.
        self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0))


class FaceLumaTests(unittest.TestCase):
    def test_the_middle_of_the_box_is_what_gets_measured(self) -> None:
        # A box whose border is the bright wall that fooled the camera's own
        # metering, with a dark face inside it.
        frame = np.full((200, 200, 3), 250, dtype=np.uint8)
        frame[50:150, 50:150] = 32
        luma = exposure.face_luma(frame, (40, 40, 120, 120))
        self.assertIsNotNone(luma)
        self.assertLess(luma, 55.0)

    def test_a_degenerate_box_measures_nothing(self) -> None:
        self.assertIsNone(exposure.face_luma(_frame(30), (0, 0, 1, 1)))
        self.assertIsNone(exposure.face_luma(_frame(30), (0, 0, 0, 0)))


class NormaliserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normaliser = exposure.Normaliser()

    def test_a_dark_face_is_lifted_and_counted(self) -> None:
        dark = _frame(30)
        lifted, gamma = self.normaliser.lift_for_face(dark, (0, 0, 128, 128))
        self.assertLess(gamma, 1.0)
        self.assertGreater(float(lifted.mean()), float(dark.mean()))
        self.assertEqual(self.normaliser.lifts, 1)
        self.assertLess(self.normaliser.last_luma, 40.0)

    def test_a_lit_face_is_returned_untouched(self) -> None:
        lit = _frame(160)
        lifted, gamma = self.normaliser.lift_for_face(lit, (0, 0, 128, 128))
        self.assertIs(lifted, lit)
        self.assertEqual(gamma, 1.0)
        self.assertEqual(self.normaliser.lifts, 0)

    def test_the_same_gamma_reuses_one_table(self) -> None:
        dark = _frame(30)
        self.normaliser.lift_for_face(dark, (0, 0, 128, 128))
        tables = dict(self.normaliser._luts)
        self.normaliser.lift_for_face(dark, (0, 0, 128, 128))
        self.assertEqual(tables.keys(), self.normaliser._luts.keys())
        self.assertIs(list(tables.values())[0], list(self.normaliser._luts.values())[0])

    def test_only_a_dark_frame_earns_a_second_detection(self) -> None:
        self.assertIsNotNone(self.normaliser.retry_gamma(_frame(30)))
        # A bright room with nobody in it must not pay for a second pass.
        self.assertIsNone(self.normaliser.retry_gamma(_frame(140)))


class _RecordingSwapper:
    """The 128 px network, replaced by a constant render that records its input."""

    size = 128

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def render(self, frame, landmarks, embedding):
        self.seen.append(frame.copy())
        return np.full((128, 128, 3), 60, dtype=np.uint8)

    def paste(self, frame, texture, landmarks):
        out = frame.copy()
        out[:8, :8] = texture[3, 3]
        return out


class _FixedDetector:
    def __init__(self, found: bool = True) -> None:
        self.found = found
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if not self.found:
            return []
        return [face.Detection(box=(20, 20, 108, 108), score=0.9, landmarks=FACE_POINTS)]


class _OneShotDetector:
    """Finds the face once, then loses it: what a turned head looks like."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls > 1:
            return []
        return [face.Detection(box=(20, 20, 108, 108), score=0.9, landmarks=FACE_POINTS)]


class _BrightOnlyDetector:
    """Finds the face only once the frame has been lifted: a dark-room camera."""

    def __init__(self, floor: float = 60.0) -> None:
        self.floor = floor
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if float(frame.mean()) < self.floor:
            return []
        return [face.Detection(box=(20, 20, 108, 108), score=0.9, landmarks=FACE_POINTS)]


class EngineLowLightTests(unittest.TestCase):
    def _engine(self, detector, swapper=None, keep_expression: bool = False):
        from mimiccam.local_engine import LocalEngine

        engine = LocalEngine(providers=["CPUExecutionProvider"], detail=False)
        engine.width, engine.height = 128, 128
        engine._swapper = swapper or _RecordingSwapper()
        engine._detector = detector
        engine._embedding = np.zeros((512,), np.float32)
        engine._enhancer = None
        return engine

    def test_the_swap_and_the_output_see_the_lifted_frame(self) -> None:
        swapper = _RecordingSwapper()
        engine = self._engine(_FixedDetector(), swapper)
        source = _frame(30)
        out = engine._process(source)
        # The renderer was handed the brightened frame, not the dark one...
        self.assertEqual(len(swapper.seen), 1)
        self.assertGreater(float(swapper.seen[0].mean()), 70.0)
        # ...and so was the frame that gets published.
        self.assertGreater(float(out.mean()), 70.0)
        self.assertGreaterEqual(engine.exposure_lifts, 1)

    def test_a_dark_run_asks_the_matte_for_the_fine_pass_once(self) -> None:
        from mimiccam import matte

        notes: list[str] = []
        engine = self._engine(_FixedDetector())
        engine._log = notes.append
        for _ in range(6):
            engine._process(_frame(30))
        self.assertEqual(engine._ratio_choice.ratio, matte.DETAIL_RATIO)
        self.assertEqual(sum("matting at full detail" in note for note in notes), 1)

    def test_a_lit_frame_keeps_the_cheap_matte(self) -> None:
        from mimiccam import matte

        engine = self._engine(_FixedDetector())
        for _ in range(6):
            engine._process(_frame(150))
        self.assertEqual(engine._ratio_choice.ratio, matte.DOWNSAMPLE_RATIO)

    def test_a_lit_frame_is_not_regraded(self) -> None:
        swapper = _RecordingSwapper()
        engine = self._engine(_FixedDetector(), swapper)
        source = _frame(150)
        out = engine._process(source)
        self.assertEqual(int(swapper.seen[0].mean()), 150)
        self.assertEqual(int(out[20, 20, 0]), 150)  # untouched outside the 8x8 paste
        self.assertEqual(engine.exposure_lifts, 0)

    def test_a_dark_room_still_finds_the_face(self) -> None:
        # YuNet returns nothing on the dark frame; the retry on the lifted one
        # is what keeps the swap running instead of falling back to the camera.
        detector = _BrightOnlyDetector()
        engine = self._engine(detector)
        engine._process(_frame(25))
        self.assertEqual(detector.calls, 2)
        self.assertEqual(engine.frames_swapped, 1)

    def test_an_empty_bright_room_is_detected_once(self) -> None:
        detector = _FixedDetector(found=False)
        engine = self._engine(detector)
        engine._process(_frame(150))
        self.assertEqual(detector.calls, 1)
        self.assertEqual(engine.frames_no_face, 1)

    def test_a_held_pose_keeps_the_lift_too(self) -> None:
        # The pose hold bridges a dropped detection; if it pasted an unlifted
        # frame the picture would step between bright and dark as the detector
        # flickered.
        engine = self._engine(_OneShotDetector())
        first = engine._process(_frame(30))
        second = engine._process(_frame(30))
        self.assertGreater(float(first.mean()), 70.0)
        self.assertEqual(int(second[20, 20, 0]), int(first[20, 20, 0]))
        self.assertEqual(engine.frames_no_face, 0)

    def test_the_lift_is_reported_once_not_per_frame(self) -> None:
        notes: list[str] = []
        engine = self._engine(_FixedDetector())
        engine._log = notes.append
        for _ in range(4):
            engine._process(_frame(30))
        self.assertEqual(sum("Low light" in note for note in notes), 1)


if __name__ == "__main__":
    unittest.main()
