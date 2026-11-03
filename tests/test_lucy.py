# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the Lucy 2.5 conditioning path.

These are the parts that decide whether a character swap lands as a whole-body
substitution or as a face paste: the shape of the reference image, and the
sentences sent with it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import prompting, reference  # noqa: E402


def _write(path: Path, array: np.ndarray) -> Path:
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    return path


class ReferenceGeometryTests(unittest.TestCase):
    def test_portrait_is_padded_to_the_output_aspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = _write(Path(tmp) / "portrait.png", np.full((800, 600, 3), 90, np.uint8))
            data = reference.prepare_reference(str(source))
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        height, width = array.shape[:2]
        self.assertAlmostEqual(width / height, 16 / 9, places=2)
        self.assertGreaterEqual(min(width, height), reference.MIN_SHORT_SIDE)
        self.assertLessEqual(max(width, height), reference.MAX_LONG_SIDE)
        self.assertEqual(data[:2], b"\xff\xd8")

    def test_padding_keeps_every_pixel_of_the_subject(self) -> None:
        # A red bar down the left edge: cropping it away would be a silent
        # failure, because the subject would still look plausible.
        image = np.full((720, 720, 3), 40, np.uint8)
        image[:, :60] = (220, 20, 20)
        with tempfile.TemporaryDirectory() as tmp:
            source = _write(Path(tmp) / "edge.png", image)
            data = reference.prepare_reference(str(source))
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        red = (array[:, :, 2] > 150) & (array[:, :, 1] < 90)
        self.assertGreater(red.sum(), 0.03 * array.shape[0] * array.shape[1])

    def test_transparent_png_does_not_become_black(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpha.png"
            rgba = np.zeros((600, 600, 4), np.uint8)
            rgba[150:450, 150:450] = (230, 200, 170, 255)
            Image.fromarray(rgba, "RGBA").save(path)
            data = reference.prepare_reference(str(path))
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(array.mean()), 40.0)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            reference.prepare_reference("definitely-not-here.png")


class AppearanceTests(unittest.TestCase):
    def test_colour_names(self) -> None:
        cases = {
            (12, 12, 14): "black",
            (245, 245, 245): "white",
            (120, 20, 20): "dark red",
            (20, 40, 120): "dark blue",
        }
        for rgb, expected in cases.items():
            self.assertEqual(reference._name(rgb), expected, rgb)

    def test_dominant_colours_find_a_suit_and_a_shirt(self) -> None:
        image = np.zeros((300, 300, 3), np.uint8)
        image[:, :] = (15, 15, 18)      # jacket
        image[100:200, 100:200] = (240, 240, 240)  # shirt
        mask = np.full((300, 300), 255, np.uint8)
        names = reference._dominant_colours(image, mask)
        self.assertIn("black", names)
        self.assertTrue({"white", "light grey"} & set(names), names)

    def test_framing_thresholds(self) -> None:
        tall = reference._framing((10, 10, 200, 700), 720)
        self.assertEqual(tall.kind, "portrait")
        self.assertIn("lower half", tall.hint)

        mid = reference._framing((10, 10, 200, 150), 720)
        self.assertEqual(mid.kind, "half")

        full = reference._framing((10, 10, 40, 50), 720)
        self.assertEqual(full.kind, "full")

        none = reference._framing(None, 720)
        self.assertEqual(none.kind, "unknown")


class PromptTests(unittest.TestCase):
    framing = reference.Framing("portrait", 0.5, "")

    def test_default_prompt_is_a_whole_person_substitution(self) -> None:
        text = prompting.character_prompt(prompting.CHARACTER_TEMPLATE)
        self.assertTrue(text.lower().startswith("substitute the whole person"))
        self.assertIn("head to toe", text)
        self.assertIn("complete outfit", text)
        self.assertNotIn("keep their face, hair", text.lower())

    def test_notes_are_carried_into_the_prompt(self) -> None:
        text = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE,
            notes="black suit, white shirt, black tie, black trousers, black shoes",
        )
        self.assertIn("black trousers", text)

    def test_below_frame_clause_only_when_the_photo_cannot_show_it(self) -> None:
        short = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE, framing=self.framing, notes="a sharp black suit"
        )
        self.assertIn("below the frame", short)

        covered = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE,
            framing=self.framing,
            notes="black trousers and black leather shoes",
        )
        self.assertNotIn("below the frame", covered)

        full_body = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE,
            framing=reference.Framing("full", 0.1, ""),
        )
        self.assertNotIn("below the frame", full_body)

    def test_appearance_phrase_is_used(self) -> None:
        appearance = reference.Appearance(
            hair="black", garments=["black", "white"], framing=self.framing
        )
        text = prompting.character_prompt(prompting.CHARACTER_TEMPLATE, appearance=appearance)
        self.assertIn("black hair", text)
        self.assertIn("black and white clothing", text)

    def test_look_notes_replace_the_sampled_garment_line(self) -> None:
        appearance = reference.Appearance(
            hair="black", garments=["black"], framing=self.framing
        )
        text = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE,
            appearance=appearance,
            notes="a black suit jacket, white shirt, black tie, black trousers, black shoes",
        )
        self.assertIn("They wear a black suit jacket", text)
        self.assertNotIn("mostly black clothing", text)

    def test_prompt_is_trimmed_to_the_server_limit(self) -> None:
        notes = " ".join(["an extremely detailed garment description"] * 40)
        text = prompting.character_prompt(
            prompting.CHARACTER_TEMPLATE, framing=self.framing, notes=notes
        )
        self.assertLessEqual(len(text), prompting.MAX_PROMPT_CHARS)
        self.assertTrue(text.lower().startswith("substitute"))
        self.assertTrue(text.endswith("."))

    def test_legacy_face_locking_prompt_is_replaced(self) -> None:
        from mimiccam import presets

        old = presets.LEGACY_CHARACTER_PROMPT
        self.assertEqual(presets.migrate(old), prompting.CHARACTER_TEMPLATE)
        self.assertEqual(presets.migrate("a prompt I wrote"), "a prompt I wrote")

        # Even if the old text arrives some other way, the builder must not merge
        # "keep their clothing consistent" into a whole-body substitution.
        text = prompting.character_prompt(old, framing=self.framing)
        self.assertTrue(text.lower().startswith("substitute the whole person"))
        self.assertNotIn("keep their face, hair", text.lower())

    def test_one_sentence_template_is_not_duplicated(self) -> None:
        text = prompting.character_prompt("Substitute the person in the video with the chef.")
        self.assertEqual(text.lower().count("substitute"), 1)


if __name__ == "__main__":
    unittest.main()
