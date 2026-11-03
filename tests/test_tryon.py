# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Lucy try-on (``lucy-vton-latest``).

No key, no session, no network: the parts that decide whether the model gets a
usable garment and a usable instruction are pure image and text work, and those
are the parts that fail quietly on a call.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import config, tryon  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_a_bare_description_becomes_an_instruction(self) -> None:
        text = tryon.as_prompt("black leather bomber jacket")
        self.assertTrue(text.startswith("Substitute the current top with"))
        self.assertIn("black leather bomber jacket", text)

    def test_the_users_own_instruction_is_left_alone(self) -> None:
        for sentence in (
            "Add a wide-brimmed straw hat to the person's head",
            "Replace the shirt with a sports jersey",
            "Change the jacket to deep red leather",
        ):
            self.assertEqual(tryon.as_prompt(sentence), sentence)

    def test_empty_text_falls_back_to_the_documented_default(self) -> None:
        self.assertEqual(tryon.as_prompt(""), tryon.DEFAULT_PROMPT)
        self.assertEqual(tryon.as_prompt("   \n "), tryon.DEFAULT_PROMPT)
        self.assertIn("reference image", tryon.DEFAULT_PROMPT)

    def test_whitespace_is_flattened(self) -> None:
        self.assertEqual(
            tryon.as_prompt("  a   black \n hoodie  "),
            "Substitute the current top with a black hoodie",
        )


def _write(path: Path, image: Image.Image) -> Path:
    image.save(path)
    return path


class GarmentPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _decode(self, data: bytes) -> Image.Image:
        self.assertEqual(data[:2], b"\xff\xd8")  # JPEG, which is what the model takes
        return Image.open(io.BytesIO(data))

    def test_a_small_garment_is_raised_to_the_model_minimum(self) -> None:
        source = _write(
            self.dir / "small.png",
            Image.fromarray(np.full((150, 200, 3), 200, np.uint8), "RGB"),
        )
        image = self._decode(tryon.prepare_garment(source))
        self.assertGreaterEqual(min(image.size), tryon.MIN_SHORT_SIDE)
        self.assertAlmostEqual(image.size[0] / image.size[1], 200 / 150, places=2)

    def test_a_huge_garment_is_brought_down_to_the_limit(self) -> None:
        source = _write(
            self.dir / "big.jpg",
            Image.fromarray(np.full((2000, 3000, 3), 120, np.uint8), "RGB"),
        )
        image = self._decode(tryon.prepare_garment(source))
        self.assertLessEqual(max(image.size), tryon.MAX_LONG_SIDE)

    def test_a_transparent_cutout_is_flattened_onto_white(self) -> None:
        # A product shot is usually a cut-out: JPEG has no alpha, and filling the
        # background with black would hand the model a different garment.
        rgba = np.zeros((600, 600, 4), np.uint8)
        rgba[150:450, 200:400] = (30, 60, 200, 255)
        source = _write(self.dir / "cutout.png", Image.fromarray(rgba, "RGBA"))
        image = self._decode(tryon.prepare_garment(source)).convert("RGB")
        corner = np.asarray(image)[:8, :8].reshape(-1, 3).mean(axis=0)
        self.assertGreater(float(corner.min()), 240.0)

    def test_a_missing_file_is_a_clear_failure(self) -> None:
        with self.assertRaises(FileNotFoundError):
            tryon.prepare_garment(self.dir / "nope.png")


class EngineWiringTests(unittest.TestCase):
    """The engine has to pick the try-on model and treat the photo as clothing."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.garment = Path(self._tmp.name) / "jacket.png"
        Image.fromarray(np.full((512, 512, 3), 90, np.uint8), "RGB").save(self.garment)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _engine(self, **kwargs):
        from mimiccam.decart_engine import DecartEngine
        from decart import models as decart_models

        model = kwargs.pop("model", tryon.MODEL)
        decart_models.realtime(model)  # the SDK must actually know this model
        return DecartEngine(api_key="test", model=model, **kwargs), model

    def test_try_on_shapes_the_prompt_and_loads_the_garment(self) -> None:
        engine, _ = self._engine(
            garment_image=str(self.garment), prompt="black hoodie", preset_key="character"
        )
        self.assertTrue(engine.try_on)
        self.assertEqual(engine.prompt, "Substitute the current top with black hoodie")
        self.assertEqual(engine.name, "Lucy try-on")
        self.assertTrue(engine._reference_bytes.startswith(b"\xff\xd8"))
        # The character-photo path must not run: no appearance analysis.
        self.assertIsNone(engine.appearance)

    def test_switching_garments_needs_no_reconnect(self) -> None:
        engine, _ = self._engine(garment_image=str(self.garment), prompt="a hoodie")
        before = engine._reference_bytes
        second = Path(self._tmp.name) / "shirt.png"
        Image.fromarray(np.full((512, 512, 3), 240, np.uint8), "RGB").save(second)

        engine.set_garment(str(second), "a white linen shirt")
        self.assertEqual(engine.garment_image, str(second))
        self.assertNotEqual(engine._reference_bytes, before)
        self.assertEqual(engine.prompt, "Substitute the current top with a white linen shirt")

    def test_a_character_session_refuses_a_garment(self) -> None:
        engine, _ = self._engine(model="lucy-2.5", prompt="")
        with self.assertRaises(RuntimeError):
            engine.set_garment(str(self.garment))

    def test_settings_remember_the_garment(self) -> None:
        settings = config.Settings()
        self.assertFalse(settings.tryon)
        self.assertEqual(settings.garment_image, "")


if __name__ == "__main__":
    unittest.main()
