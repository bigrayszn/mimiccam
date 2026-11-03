# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Exercise the live screen headlessly and write a picture of it.

The live screen is the part of the window a caller actually looks at: preview,
one start button, the look selector and the character picker. Everything else
is behind the Settings toggle. This script builds the real window with Qt's
offscreen platform, drives those controls and checks the panel really gives its
space back to the preview, then saves two PNGs to a scratch folder so the layout
can be looked at (they have placeholder glyphs - offscreen Qt has no fonts).

    python tests/verify_live_ui.py

Settings are redirected to a scratch APPDATA, so a developer's own
settings.json is never read or written.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must happen before the app modules import: config.py resolves paths at import.
SCRATCH = tempfile.mkdtemp(prefix="mimiccam-ui-")
os.environ["APPDATA"] = SCRATCH
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mimiccam.ui import main_window as ui  # noqa: E402

# Never touch a real camera: the human may be on a call.
ui.probe_cameras = lambda _count=6: []

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{' - ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def center_color(label):
    """Mid pixel of the label's pixmap, or None when nothing is displayed."""
    pixmap = label.pixmap()
    if pixmap is None or pixmap.isNull():
        return None
    image = pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


def main() -> int:
    app = QApplication([])
    window = ui.MainWindow()
    window.resize(1180, 760)
    window.show()
    app.processEvents()

    check("window built", window.start_btn is not None)
    check(
        "settings open by default",
        window.settings_btn.isChecked() and window.settings_panel.isVisibleTo(window),
        f"button={window.settings_btn.text()!r}",
    )
    wide_with_panel = window.preview.width()

    # Cost controls: both are on the settings panel, and both must tell the truth.
    check(
        "idle billing pause is offered and on by default",
        window.pause_check.isChecked() if hasattr(window, "pause_check") else False,
        window.pause_check.text()[:52] if hasattr(window, "pause_check") else "missing",
    )
    check(
        "fast mode is labelled as a 2x price, not a picture upgrade",
        "2x" in window.fast_check.text() and "$0.04" in window.fast_check.text(),
        window.fast_check.text()[:64],
    )

    # --- the credit box ---------------------------------------------------
    check(
        "credit box is on screen under the preview bar",
        window.credit_box.isVisibleTo(window),
        window.credit_box.text(),
    )
    window.radio_decart.setChecked(True)
    window.fast_check.setChecked(False)
    app.processEvents()
    check(
        "Lucy price is quoted per second",
        "$0.02/s" in window.credit_box.text(),
        window.credit_box.text(),
    )
    window.fast_check.setChecked(True)
    app.processEvents()
    check(
        "fast mode doubles the quoted rate",
        "$0.04/s" in window.credit_box.text(),
        window.credit_box.text(),
    )
    window.fast_check.setChecked(False)
    window.balance_edit.setText("10.00")
    window.balance_set.click()
    app.processEvents()
    check(
        "a typed balance is accepted and shown",
        window.settings.balance_usd == 10.0 and "$10.00" in window.credit_box.text(),
        window.credit_box.text(),
    )

    # Look selector drives the settings preset, and the preset text follows.
    keys = [window.live_look.itemData(i) for i in range(window.live_look.count())]
    target = keys.index("anime")
    window.live_look.setCurrentIndex(target)
    app.processEvents()
    check(
        "live look -> settings preset",
        window.preset_combo.currentData() == "anime",
        f"preset={window.preset_combo.currentData()}",
    )
    check(
        "preset prompt applied",
        "anime" in window.prompt_edit.toPlainText().lower(),
        window.prompt_edit.toPlainText()[:48],
    )

    # ...and the other way round, without an event loop fight.
    index = window.preset_combo.findData("cinematic")
    window.preset_combo.setCurrentIndex(index)
    app.processEvents()
    check(
        "settings preset -> live look",
        window.live_look.currentData() == "cinematic",
        f"live={window.live_look.currentData()}",
    )

    # Hiding the panel gives the preview the width back.
    window.settings_btn.setChecked(False)
    app.processEvents()
    check(
        "settings closed",
        not window.settings_panel.isVisibleTo(window),
        f"button={window.settings_btn.text()!r}",
    )
    wide_without_panel = window.preview.width()
    check(
        "preview grows when settings close",
        wide_without_panel > wide_with_panel,
        f"{wide_with_panel} -> {wide_without_panel} px",
    )

    # Press Start with a stand-in pipeline: the live screen must collapse the
    # settings panel and bring it back on Stop.
    class _VCam:
        device = "Fake Vcam"

    class _FakeMeter:
        """Stands in for the Decart engine's server-fed meter."""

        def __init__(self) -> None:
            self.usd = 0.0
            self.seconds = 0.0
            self.parked = False

        def billing(self) -> dict:
            return {
                "seconds": self.seconds,
                "usd": self.usd,
                "usd_per_min": 1.2,
                "parked": self.parked,
                "pauses": 0,
            }

    class _Pipeline:
        def __init__(self, **_kwargs):
            self.vcam = _VCam()
            self.engine = _FakeMeter()

        def start(self, _engine=None) -> bool:
            return True

        def stop(self) -> None:
            pass

    ui.Pipeline = _Pipeline
    window.radio_passthrough.setChecked(True)
    window.settings.camera_index = 1
    window._start()
    app.processEvents()
    check("start collapses settings", not window.settings_panel.isVisibleTo(window))
    check("start button reads Stop", window.start_btn.text() == "Stop")

    # Two minutes of Lucy at $0.02/s, as the server's ticks would report them.
    window.pipeline.engine.usd = 2.4
    window.pipeline.engine.seconds = 120
    window._update_stats({"camera_fps": 30, "output_fps": 30, "held_frames": 0})
    app.processEvents()
    streaming = window.credit_box.text()
    check(
        "session spend comes from the engine's meter",
        "$2.40 this run" in streaming and "2:00 billed" in streaming,
        streaming,
    )
    check(
        "balance counts down while streaming",
        "$7.60" in streaming,
        streaming,
    )

    window._stop()
    app.processEvents()
    check(
        "stopping commits the spend to the saved balance",
        abs(window.settings.balance_spent - 2.4) < 0.001,
        f"spent={window.settings.balance_spent}",
    )
    check(
        "the idle box still shows what is left",
        "$7.60" in window.credit_box.text(),
        window.credit_box.text(),
    )
    check("stop restores settings", window.settings_panel.isVisibleTo(window))
    check("stop button reads Start camera", window.start_btn.text() == "Start camera")

    # --- your own camera, not just the model's output --------------------
    # A stand-in camera keeps the human's real webcam out of this test.
    class _FakeCam:
        opened: list[int] = []
        instances: list = []

        def __init__(self, index, width, height, fps, mirror, on_log=None):
            self.index = index
            self.stopped = False
            self._frame = np.zeros((height, width, 3), dtype=np.uint8)
            self._frame[:, :, 2] = 220  # blue: unmistakably "camera"
            self._ts = 1.0
            _FakeCam.opened.append(index)
            _FakeCam.instances.append(self)

        def start(self) -> bool:
            return True

        def stop(self) -> None:
            self.stopped = True

        def read(self):
            return self._frame, self._ts

    ui.CameraCapture = _FakeCam
    window.settings.camera_index = 1
    check(
        "preview offers both views",
        window.preview_out_btn.isChecked() and not window.preview_cam_btn.isChecked(),
        window.preview_caption.text(),
    )
    window._choose_preview("camera")
    app.processEvents()
    check("camera view opens the webcam", _FakeCam.opened == [1], f"opened={_FakeCam.opened}")
    window._pump_idle_frame()
    app.processEvents()
    camera_pixel = center_color(window.preview)
    check(
        "idle camera frame is shown",
        camera_pixel is not None and camera_pixel.blue() > 100 and camera_pixel.red() < 40,
        f"pixel={camera_pixel}",
    )
    window._choose_preview("output")
    app.processEvents()
    check(
        "switching to output releases the camera",
        window._idle_cam is None and _FakeCam.instances[-1].stopped,
    )

    # During a session the switch must route frames without a second capture.
    window.settings.camera_index = 1
    window.radio_passthrough.setChecked(True)
    window._start()
    app.processEvents()
    model_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    model_frame[:, :, 0] = 220  # red: model output
    window.bridge.preview.emit(model_frame)
    app.processEvents()
    output_pixel = center_color(window.preview)
    check(
        "output view shows the model",
        output_pixel is not None and output_pixel.red() > 100 and output_pixel.blue() < 40,
        f"pixel={output_pixel}",
    )
    camera_source = np.zeros((180, 320, 3), dtype=np.uint8)
    camera_source[:, :, 2] = 220
    window.bridge.input_preview.emit(camera_source)
    app.processEvents()
    check(
        "camera frames stay hidden in output view",
        center_color(window.preview).red() > 100,
        f"pixel={center_color(window.preview)}",
    )

    # ...but they are not thrown away: the raw camera goes in the corner so the
    # model can be watched without losing sight of your own framing.
    inset = window.preview.inset_size()
    check(
        "camera shows as a small box over the output",
        inset is not None and inset.width() < window.preview.width() // 3,
        f"inset={inset.width() if inset else None}px, pane={window.preview.width()}px",
    )
    grab = window.preview.grab().toImage()
    corner = grab.pixelColor(grab.width() - 30, grab.height() - 30)
    check(
        "the small box is painted in the corner",
        corner.blue() > 100 and corner.red() < 40,
        f"corner={corner.name()}",
    )
    check(
        "the small box never covers the output itself",
        center_color(window.preview).red() > 100,
        f"centre={center_color(window.preview)}",
    )

    # ...and it is an option, on by default, remembered between launches.
    check(
        "picture-in-picture is offered and on by default",
        window.pip_check.isChecked() and window.settings.pip_camera,
        window.pip_check.text(),
    )
    window.pip_check.setChecked(False)
    app.processEvents()
    check("turning it off clears the box", not window.preview.has_inset())
    check("the choice is saved", window.settings.pip_camera is False)
    window.bridge.input_preview.emit(camera_source)
    app.processEvents()
    check(
        "turning it off keeps new camera frames out of the corner",
        not window.preview.has_inset(),
    )
    window.pip_check.setChecked(True)
    window.bridge.input_preview.emit(camera_source)
    app.processEvents()
    check("turning it back on restores the box", window.preview.has_inset())

    window._choose_preview("camera")
    app.processEvents()
    check(
        "camera view mid-session reuses the running camera",
        window._idle_cam is None and len(_FakeCam.opened) == 1,
        f"opened={_FakeCam.opened}",
    )
    window.bridge.input_preview.emit(camera_source)
    app.processEvents()
    pixel = center_color(window.preview)
    check(
        "camera view shows the raw camera",
        pixel is not None and pixel.blue() > 100 and pixel.red() < 40,
        f"pixel={pixel}",
    )
    check(
        "camera view drops the corner box",
        not window.preview.has_inset(),
        f"inset={window.preview.inset_size()}",
    )
    window.bridge.preview.emit(model_frame)
    app.processEvents()
    check(
        "model frames stay hidden in camera view",
        center_color(window.preview).blue() > 100,
        f"pixel={center_color(window.preview)}",
    )
    window._choose_preview("output")
    window.bridge.input_preview.emit(camera_source)
    app.processEvents()
    window._stop()
    app.processEvents()
    check(
        "stopping clears the corner box",
        not window.preview.has_inset(),
        f"inset={window.preview.inset_size()}",
    )

    # Offscreen grabs have no font database, so these are for layout only.
    live_shot = Path(SCRATCH) / "live_screen.png"
    settings_shot = Path(SCRATCH) / "live_screen_settings.png"
    window.grab().save(str(live_shot))
    window.settings_btn.setChecked(True)
    app.processEvents()
    window.grab().save(str(settings_shot))
    check("layout grabs written", live_shot.exists() and settings_shot.exists(), SCRATCH)

    window.close()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
