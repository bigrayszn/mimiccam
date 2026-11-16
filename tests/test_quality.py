# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Maximum quality: what the mode is, what it costs, and that it cannot drift.

The point of these tests is the honesty of the mode rather than the widget: the
price per minute is arithmetic that has to match Decart's published rate, fast
mode must never be billed alongside a mode that claims the best picture, and the
same rate has to appear before a session, during it, and in the README.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Qt without a display, decided before PySide6 is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - a checkout without the GUI dependency
    QApplication = None

from mimiccam import config, credits, quality  # noqa: E402


class RateTests(unittest.TestCase):
    def test_the_rate_is_decarts_published_rate_in_minutes(self) -> None:
        # $0.02/s is the unit Decart bills in and what the engine meter reads;
        # $1.20/min is the same money said the way people plan a call.
        self.assertEqual(credits.USD_PER_SECOND * 60, quality.USD_PER_MINUTE)
        self.assertAlmostEqual(quality.per_minute(), 1.20, places=3)
        self.assertAlmostEqual(quality.per_minute(True), 2.40, places=3)

    def test_fast_mode_is_exactly_twice_the_price(self) -> None:
        self.assertAlmostEqual(
            quality.per_minute(True), 2 * quality.per_minute(), places=3
        )

    def test_minutes_a_balance_buys(self) -> None:
        self.assertAlmostEqual(quality.minutes_for(36.0), 30.0, places=2)
        self.assertAlmostEqual(quality.minutes_for(12.0, 6.0), 5.0, places=2)
        self.assertAlmostEqual(quality.minutes_for(2.4, 0.0, fast_mode=True), 1.0, places=2)
        # A balance already spent, or never set, buys nothing - it must not go
        # negative and read as credit.
        self.assertEqual(quality.minutes_for(0.0), 0.0)
        self.assertEqual(quality.minutes_for(5.0, 10.0), 0.0)

    def test_minutes_text_never_rounds_credit_up(self) -> None:
        self.assertEqual(quality.minutes_text(29.9), "29 min left")
        self.assertEqual(quality.minutes_text(0.4), "under a minute left")
        self.assertEqual(quality.minutes_text(0.0), "no minutes left")

    def test_summary_states_the_tier_and_the_price(self) -> None:
        text = quality.summary()
        self.assertIn("standard tier", text)
        self.assertIn(f"{quality.WIDTH}x{quality.HEIGHT}", text)
        self.assertIn("$1.20/min", text)

    def test_the_receipt_has_the_arithmetic_in_it(self) -> None:
        text = quality.receipt(balance_usd=36.0)
        self.assertIn("$1.20/min", text)
        self.assertIn("$72.00 an hour", text)
        self.assertIn("$36.00", text)          # a 30-minute call
        self.assertIn("$2.40/min", text)       # fast mode, and that it is not better
        self.assertIn("about 30 minutes", text)
        self.assertIn("connected", text)       # the meter follows connected time

    def test_the_receipt_asks_for_a_balance_when_there_is_none(self) -> None:
        text = quality.receipt()
        self.assertIn("No balance set", text)
        self.assertIn("Settings", text)

    def test_a_spent_balance_is_not_reported_as_minutes(self) -> None:
        text = quality.receipt(balance_usd=10.0, spent_usd=10.0)
        self.assertIn("already spent", text)
        self.assertNotIn("about 0 minutes", text)


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._original = config.SETTINGS_PATH
        config.SETTINGS_PATH = Path(self.temp.name) / "settings.json"
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        config.SETTINGS_PATH = self._original

    def test_max_quality_is_off_by_default_and_survives_a_reload(self) -> None:
        self.assertFalse(config.Settings().max_quality)
        settings = config.Settings(max_quality=True, fast_mode=False)
        settings.save()
        self.assertTrue(config.Settings.load().max_quality)


class EngineTierTests(unittest.TestCase):
    def test_a_quality_session_cannot_be_billed_at_the_fast_rate(self) -> None:
        """A stale settings file asking for both must not double the bill."""
        from mimiccam.decart_engine import DecartEngine

        log: list[str] = []
        engine = DecartEngine(
            api_key="not-a-real-key",
            max_quality=True,
            fast_mode=True,
            on_log=log.append,
        )
        self.assertFalse(engine.fast_mode)
        self.assertEqual(engine.billing()["usd_per_min"], 1.2)
        self.assertTrue(any("$1.20/min" in line for line in log), log)

    def test_the_tier_is_unchanged_when_the_mode_is_off(self) -> None:
        from mimiccam.decart_engine import DecartEngine

        engine = DecartEngine(api_key="not-a-real-key", fast_mode=True)
        self.assertTrue(engine.fast_mode)
        self.assertEqual(engine.billing()["usd_per_min"], 2.4)


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class MaximumQualityUiTests(unittest.TestCase):
    def setUp(self) -> None:
        from mimiccam.ui import main_window as ui

        self.ui = ui
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._original = config.SETTINGS_PATH
        config.SETTINGS_PATH = Path(self.temp.name) / "settings.json"
        self.addCleanup(self._restore)
        self.app = QApplication.instance() or QApplication([])

    def _restore(self) -> None:
        config.SETTINGS_PATH = self._original

    def _window(self):
        # Cameras are not what this file is about; the scan is a given.
        original_scan = self.ui.scan_cameras
        original_cache = self.ui.cached_cameras
        self.ui.scan_cameras = lambda force=False, max_age_s=60.0: []
        self.ui.cached_cameras = lambda *args, **kwargs: None
        self.addCleanup(setattr, self.ui, "scan_cameras", original_scan)
        self.addCleanup(setattr, self.ui, "cached_cameras", original_cache)
        window = self.ui.MainWindow()
        self.addCleanup(window.close)
        for _ in range(20):
            self.app.processEvents()
        return window

    def test_the_price_is_visible_before_anything_starts(self) -> None:
        window = self._window()
        label = window.price_label.text()
        self.assertIn("$1.20/min", label)
        self.assertIn("$72.00 an hour", label)
        self.assertIn("30 min = $36.00", label)
        # The meter chip quotes the same rate while idle, not only once running.
        window.radio_decart.setChecked(True)
        self.assertIn("$1.20/min", window.credit_box.text())

    def test_ticking_it_selects_lucy_and_the_standard_tier(self) -> None:
        window = self._window()
        window.radio_local.setChecked(True)
        window.max_check.setChecked(True)
        self.assertTrue(window.radio_decart.isChecked())
        self.assertFalse(window.fast_check.isChecked())
        self.assertFalse(window.fast_check.isEnabled(), "fast mode is 2x for the same picture")
        self.assertFalse(window.pause_check.isChecked(), "pausing puts the raw camera back on screen")
        self.assertIn("$1.20/min", window.credit_box.text())
        self.assertIn("$72.00 an hour", window.hint.text())

    def test_leaving_lucy_turns_the_mode_off_again(self) -> None:
        window = self._window()
        window.max_check.setChecked(True)
        window.radio_passthrough.setChecked(True)
        self.assertFalse(window.max_check.isChecked())
        self.assertTrue(window.fast_check.isEnabled())

    def test_a_running_call_is_told_the_change_lands_on_the_next_start(self) -> None:
        window = self._window()
        window._running = True
        try:
            window.max_check.setChecked(True)
            self.assertIn("next start", window.hint.text())
        finally:
            window._running = False

    def test_saving_never_keeps_fast_mode_with_maximum_quality(self) -> None:
        window = self._window()
        window.max_check.setChecked(True)
        settings = window._collect_settings()
        self.assertTrue(settings.max_quality)
        self.assertEqual(settings.mode, "decart")
        self.assertFalse(settings.fast_mode)
        # The vcam must not downscale what Lucy rendered.
        self.assertEqual((settings.width, settings.height, settings.fps),
                         (quality.WIDTH, quality.HEIGHT, quality.FPS))
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertTrue(saved["max_quality"])
        self.assertFalse(saved["fast_mode"])

    def test_the_saved_mode_comes_back_on_the_next_launch(self) -> None:
        config.SETTINGS_PATH.write_text(
            json.dumps({"mode": "local", "max_quality": True}), encoding="utf-8"
        )
        window = self._window()
        for _ in range(40):
            self.app.processEvents()
            if window.max_check.isChecked():
                break
            time.sleep(0.01)
        self.assertTrue(window.max_check.isChecked())
        self.assertTrue(window.radio_decart.isChecked(), "maximum quality runs on Lucy")
        self.assertFalse(window.fast_check.isEnabled())

    def test_no_character_photo_means_no_billed_session(self) -> None:
        window = self._window()
        window.max_check.setChecked(True)
        shown: list[tuple] = []
        # Named before patching: a cleanup that restores the patch would leak it
        # into every later test in this process.
        original_key, original_gpu = self.ui.get_api_key, self.ui.require_nvidia
        self.addCleanup(setattr, self.ui, "get_api_key", original_key)
        self.addCleanup(setattr, self.ui, "require_nvidia", original_gpu)
        self.ui.get_api_key = lambda: "not-a-real-key"  # type: ignore[assignment]
        self.ui.require_nvidia = lambda gpu: ""  # type: ignore[assignment]
        original = self.ui.QMessageBox.information
        self.ui.QMessageBox.information = lambda *args, **kwargs: shown.append(args)
        self.addCleanup(setattr, self.ui.QMessageBox, "information", original)

        window._start()

        self.assertTrue(shown, "it must say why nothing started")
        self.assertFalse(window._running)
        self.assertIsNone(window.pipeline)

    def test_backing_out_of_the_price_starts_nothing(self) -> None:
        window = self._window()
        window.max_check.setChecked(True)
        # The check the dialog sits behind is "a photo is chosen", not "the file
        # still exists": no image is opened before the price is answered.
        window._reference_path = "character.png"
        original_key, original_gpu = self.ui.get_api_key, self.ui.require_nvidia
        self.addCleanup(setattr, self.ui, "get_api_key", original_key)
        self.addCleanup(setattr, self.ui, "require_nvidia", original_gpu)
        self.ui.get_api_key = lambda: "not-a-real-key"  # type: ignore[assignment]
        self.ui.require_nvidia = lambda gpu: ""  # type: ignore[assignment]
        asked: list[str] = []
        original = self.ui.QMessageBox.question
        self.ui.QMessageBox.question = lambda parent, title, text, *a, **k: (
            asked.append(text), self.ui.QMessageBox.StandardButton.Cancel
        )[1]
        self.addCleanup(setattr, self.ui.QMessageBox, "question", original)

        window._start()

        self.assertTrue(asked, "the price must be shown before the meter starts")
        self.assertIn("$1.20/min", asked[0])
        self.assertFalse(window._running)
        self.assertIsNone(window.pipeline)


if __name__ == "__main__":
    unittest.main()
