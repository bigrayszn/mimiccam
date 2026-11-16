# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The camera picker after a scan: live devices stay usable, missing ones grey.

A camera unplugged since the last run is still the user's camera, so it stays in
the list, disabled, with the name Windows gave it. These tests build the real
window offscreen with the scan stubbed out, because the point is the list the
user sees, not the probing.
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

from mimiccam import config  # noqa: E402
from mimiccam.capture import CameraEntry  # noqa: E402

REMEMBERED = [
    {"index": 1, "name": "Integrated Webcam", "width": 1920, "height": 1080, "mean_luma": 130.0},
    {"index": 6, "name": "OBS Virtual Camera", "width": 640, "height": 480, "mean_luma": 54.0},
    {"index": 2, "name": "DroidCam Video", "width": 640, "height": 480, "mean_luma": 90.0},
]


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class CameraListTests(unittest.TestCase):

    def setUp(self) -> None:
        from mimiccam.ui import main_window as ui

        self.ui = ui
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._original_path = config.SETTINGS_PATH
        config.SETTINGS_PATH = Path(self.temp.name) / "settings.json"
        self.addCleanup(self._restore_path)
        self.app = QApplication.instance() or QApplication([])

    def _restore_path(self) -> None:
        config.SETTINGS_PATH = self._original_path

    def _settings(self, *, name: str) -> None:
        config.SETTINGS_PATH.write_text(
            json.dumps({"camera_index": 1, "camera_name": name, "cameras": REMEMBERED}),
            encoding="utf-8",
        )

    def _window(self, live: list[CameraEntry]):
        # The scan is exercised in test_local.py; here it is a given.
        original_scan = self.ui.scan_cameras
        original_cache = self.ui.cached_cameras
        self.ui.scan_cameras = lambda force=False, max_age_s=60.0: live
        self.ui.cached_cameras = lambda *args, **kwargs: None
        self.addCleanup(setattr, self.ui, "scan_cameras", original_scan)
        self.addCleanup(setattr, self.ui, "cached_cameras", original_cache)

        window = self.ui.MainWindow()
        self.addCleanup(window.close)
        deadline = time.time() + 10
        while time.time() < deadline:
            self.app.processEvents()
            if not window._scanning and window.camera_combo.count():
                break
            time.sleep(0.01)
        self.app.processEvents()
        return window

    def _rows(self, window):
        model = window.camera_combo.model()
        return [
            (window.camera_combo.itemText(i), window.camera_combo.itemData(i),
             model.item(i).isEnabled())
            for i in range(window.camera_combo.count())
        ]

    def test_a_camera_that_is_gone_is_greyed_not_dropped(self) -> None:
        self._settings(name="Integrated Webcam")
        window = self._window([
            CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam"),
            CameraEntry(1, 640, 480, 54.0, name="OBS Virtual Camera"),
        ])
        rows = self._rows(window)

        self.assertEqual([row[0] for row in rows],
                         ["Integrated Webcam", "OBS Virtual Camera", "DroidCam Video"])
        self.assertTrue(rows[0][2] and rows[1][2], "a live camera must stay selectable")
        self.assertFalse(rows[2][2], "the unplugged camera should be greyed out")

    def test_the_saved_camera_is_found_again_after_renumbering(self) -> None:
        """Unplugging one device shifts the indices of the rest."""
        self._settings(name="Integrated Webcam")
        window = self._window([
            CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam"),
            CameraEntry(1, 640, 480, 54.0, name="OBS Virtual Camera"),
        ])
        self.assertEqual(window.camera_combo.currentText(), "Integrated Webcam")
        self.assertEqual(window.camera_combo.currentData(), 0)

    def test_the_missing_camera_is_remembered_for_the_next_launch(self) -> None:
        self._settings(name="Integrated Webcam")
        self._window([
            CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam"),
        ])
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertIn("DroidCam Video", [item["name"] for item in saved["cameras"]])

    def test_a_camera_missing_for_three_scans_is_forgotten(self) -> None:
        """Greyed is a grace period, not forever."""
        self._settings(name="Integrated Webcam")
        window = self._window([
            CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam"),
        ])
        self.assertIn("DroidCam Video", self._names(window), "greyed on the first miss")
        live = [CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam")]

        window._scanned_cameras(live)
        self.assertIn("DroidCam Video", self._names(window), "still greyed on the second miss")

        window._scanned_cameras(live)
        self.assertNotIn("DroidCam Video", self._names(window), "forgotten on the third miss")
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("DroidCam Video", [item["name"] for item in saved["cameras"]])

    def test_repainting_the_list_does_not_age_a_camera_out(self) -> None:
        """Opening Settings reuses the cached scan; that is not another miss."""
        self._settings(name="Integrated Webcam")
        window = self._window([
            CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam"),
        ])
        live = [CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam")]
        for _ in range(5):
            window._populate_cameras(live)  # what a cache hit does
            window._show_remembered_cameras()
        self.assertIn("DroidCam Video", self._names(window),
                      "a repaint must not count as a scan")

    def test_a_camera_that_comes_back_starts_over(self) -> None:
        self._settings(name="Integrated Webcam")
        window = self._window([])
        live = [CameraEntry(0, 1920, 1080, 130.0, name="Integrated Webcam")]
        window._scanned_cameras(live)
        window._scanned_cameras(live)
        # Reappears at a new index: its miss count must reset, or it would be
        # forgotten on the next scan while sitting in the list.
        window._scanned_cameras([CameraEntry(3, 640, 480, 90.0, name="DroidCam Video")] + live)
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        by_name = {item["name"]: item["misses"] for item in saved["cameras"]}
        self.assertEqual(by_name["DroidCam Video"], 0)

    def _names(self, window) -> list[str]:
        return [row[0] for row in self._rows(window)]

    def test_names_are_shown_without_decoration(self) -> None:
        self._settings(name="OBS Virtual Camera")
        window = self._window([
            CameraEntry(1, 640, 480, 54.0, name="OBS Virtual Camera"),
        ])
        for text, _data, _enabled in self._rows(window):
            self.assertNotIn("Camera ", text)
            self.assertNotIn("x480", text)


if __name__ == "__main__":
    unittest.main()
