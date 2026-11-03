# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression tests for the wiring between the UI, the presets and the engines.

They have to hold on a machine with no camera, no GPU, no API key and no
display, because that is exactly what CI is. The point is to fail in a pull
request rather than the first time a user turns a feature on.

Two things rot quietly, and are what this file covers:

* a module that stops importing. Most of the app loads lazily - the matte, the
  hair pass, the studio room, the offline engine - so a broken import there is
  invisible until that feature is used. The workflow's "Check imports" step only
  covers four modules; this walks the package and imports all of them.
* a preset the combo box offers that the engines cannot turn into a prompt or a
  look. Those failures are silent: the app starts, the call connects, and only
  the picture is wrong.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mimiccam  # noqa: E402
from mimiccam import looks, presets, prompting  # noqa: E402


def discovered_modules() -> list[str]:
    """Every importable module in the package, without importing them first."""
    return sorted(
        info.name
        for info in pkgutil.walk_packages(mimiccam.__path__, f"{mimiccam.__name__}.")
    )


class ModuleImportTests(unittest.TestCase):
    def test_every_module_imports(self) -> None:
        names = discovered_modules()
        # A package that lost its __path__ would otherwise make this pass trivially.
        self.assertGreaterEqual(len(names), 20, f"only discovered {names}")
        for name in names:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertIsNotNone(module.__file__, f"{name} is not a real module")

    def test_the_modules_the_app_cannot_run_without_are_present(self) -> None:
        # Deleting or renaming one of these is a legitimate refactor, but it must
        # be a deliberate edit to this list, not a quiet disappearance.
        present = set(discovered_modules())
        expected = {
            "mimiccam.__main__",
            "mimiccam.capture",
            "mimiccam.config",
            "mimiccam.decart_engine",
            "mimiccam.engine",
            "mimiccam.local_engine",
            "mimiccam.models",
            "mimiccam.pipeline",
            "mimiccam.presets",
            "mimiccam.prompting",
            "mimiccam.reference",
            "mimiccam.tryon",
            "mimiccam.vcam",
            "mimiccam.ui",
            "mimiccam.ui.download",
            "mimiccam.ui.main_window",
        }
        self.assertEqual(sorted(expected - present), [])

    def test_the_launch_paths_are_callable(self) -> None:
        """``python -m mimiccam`` and the frozen exe both go through these."""
        from mimiccam import __main__ as cli
        from mimiccam import ui

        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(ui.run))
        self.assertTrue(callable(ui.MainWindow))


def engine_prompt(preset_key: str, custom: str = "", notes: str = "") -> str:
    """The prompt DecartEngine assembles, without a camera, a key or a session.

    Mirrors its ``__init__``: ``presets.system_prompt`` first, then the
    whole-person builder for the character preset only.
    """
    text = presets.system_prompt(preset_key, custom)
    if presets.get(preset_key).key == "character":
        return prompting.character_prompt(text, notes=notes)
    return text


class PresetPlumbingTests(unittest.TestCase):
    def test_every_preset_has_a_unique_key_and_a_label(self) -> None:
        keys = [preset.key for preset in presets.PRESETS]
        self.assertEqual(len(keys), len(set(keys)), keys)
        for preset in presets.PRESETS:
            self.assertTrue(preset.key.strip())
            self.assertTrue(preset.label.strip())

    def test_every_preset_reaches_the_engine_as_a_prompt_it_can_send(self) -> None:
        for preset in presets.PRESETS:
            with self.subTest(preset=preset.key):
                sent = engine_prompt(preset.key)
                if preset.key == "custom":
                    # Custom with nothing typed is empty on purpose: the engine
                    # substitutes its own default rather than sending "".
                    self.assertEqual(sent, "")
                    continue
                self.assertTrue(sent)
                self.assertLessEqual(len(sent), prompting.MAX_PROMPT_CHARS)

    def test_look_notes_reach_the_character_prompt(self) -> None:
        sent = engine_prompt(
            "character", notes="a black suit jacket, black trousers, black shoes"
        )
        self.assertIn("black trousers", sent)
        self.assertLessEqual(len(sent), prompting.MAX_PROMPT_CHARS)

    def test_an_unknown_saved_preset_behaves_like_the_default(self) -> None:
        # settings.json survives upgrades, so it can name a preset this build
        # no longer ships.
        self.assertEqual(engine_prompt("retired-in-a-later-build"), engine_prompt("character"))

    def test_a_preset_that_keeps_the_person_is_not_treated_as_a_swap(self) -> None:
        """identity/cinematic/suit deliberately preserve the source person."""
        for preset in presets.PRESETS:
            if preset.needs_reference:
                with self.subTest(preset=preset.key):
                    self.assertFalse(prompting.face_only_note(preset.key))

    def test_every_preset_maps_to_a_look_that_exists(self) -> None:
        for preset in presets.PRESETS:
            with self.subTest(preset=preset.key):
                self.assertIn(looks.for_preset(preset.key), looks.LOOKS)
        # A look mapped to a preset key that does not exist is a typo that
        # silently gives those users the neutral look.
        orphans = sorted(set(looks.PRESET_LOOKS) - set(presets.BY_KEY))
        self.assertEqual(orphans, [])


if __name__ == "__main__":
    unittest.main()
