# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""MimicCam main window.

The UI thread never touches a camera, a socket or a frame loop. All worker
callbacks are emitted as Qt signals from a small bridge object, which Qt queues
onto the GUI thread for us.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__, __copyright__, __url__, __version__, models, tryon
from ..capture import CameraCapture, CameraEntry, probe_cameras
from ..config import Settings, characters_dir, get_api_key, mask_key, set_api_key
from ..credits import clock, money, remaining_usd, usd_per_second
from ..gpu import describe as describe_gpu
from ..gpu import nvidia_adapter, nvidia_info, require_nvidia
from ..local_engine import LocalEngine, LocalEngineError
from ..pipeline import Pipeline
from ..presets import PRESETS, get as get_preset, migrate as migrate_prompt
from ..vcam import BACKEND_LABELS, available_backends
from .download import ModelDownloadDialog

STYLE = """
QWidget { background: #14161a; color: #e8eaed; font-size: 13px; }
QGroupBox {
    border: 1px solid #2a2f37; border-radius: 8px; margin-top: 14px;
    padding: 10px 8px 8px 8px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8ab4f8; }
QLineEdit, QPlainTextEdit, QComboBox {
    background: #1c1f25; border: 1px solid #2f353f; border-radius: 6px; padding: 5px;
    selection-background-color: #3b6fd4;
}
QComboBox QAbstractItemView { background: #1c1f25; selection-background-color: #2f4d86; }
QPushButton {
    background: #232830; border: 1px solid #333a45; border-radius: 6px; padding: 7px 12px;
}
QPushButton:hover { background: #2b323c; }
QPushButton:disabled { color: #6b7280; }
QPushButton#primary { background: #3b6fd4; border: none; font-weight: 700; padding: 11px; }
QPushButton#primary:hover { background: #4a80e6; }
QPushButton#primary[running="true"] { background: #b3261e; }
QLabel#preview { background: #000; border: 1px solid #2a2f37; border-radius: 8px; }
QLabel#creditbox {
    background: #171a20; border: 1px solid #2a2f37; border-radius: 7px;
    padding: 5px 10px; color: #c6d0dc;
}
QLabel#hint { color: #93a1b3; }
QLabel#chip {
    background: #1c1f25; border: 1px solid #2f353f; border-radius: 10px;
    padding: 3px 9px; color: #b9c4d2;
}
QPushButton#seg { padding: 3px 12px; border-radius: 12px; color: #b9c4d2; }
QPushButton#seg:checked {
    background: #3b6fd4; border-color: #3b6fd4; color: #ffffff; font-weight: 600;
}
"""


class PreviewView(QLabel):
    """The preview pane: one big picture, plus a small box of your own camera.

    The big picture shows whichever view is selected. While it is the processed
    output, the raw camera is drawn as a small inset in the bottom-right corner,
    so the model can be watched without losing sight of what you actually look
    like - the one thing the output alone cannot tell you.
    """

    INSET_RATIO = 0.22  # of the pane width
    INSET_MARGIN = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._inset: QPixmap | None = None
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(
            "What callers receive, with your untouched camera in the corner."
        )

    # -- content -----------------------------------------------------------
    def set_inset(self, pixmap: QPixmap | None) -> None:
        if pixmap is not None and pixmap.isNull():
            pixmap = None
        if pixmap is None and self._inset is None:
            return
        self._inset = pixmap
        self.update()

    def inset_box(self) -> QSize:
        """Box the inset is scaled into; ``scaled`` keeps the aspect inside it."""
        width = max(96, int(self.width() * self.INSET_RATIO))
        return QSize(width, width)

    def has_inset(self) -> bool:
        return self._inset is not None

    def inset_size(self) -> QSize | None:
        return self._inset.size() if self._inset is not None else None

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        inset = self._inset
        if inset is None:
            return
        margin = self.INSET_MARGIN
        x = self.width() - inset.width() - margin
        y = self.height() - inset.height() - margin
        if x < margin or y < margin:
            return  # pane too narrow to spare the corner

        path = QPainterPath()
        path.addRoundedRect(
            QRectF(x - 1, y - 1, inset.width() + 2, inset.height() + 2), 7.0, 7.0
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillPath(path, QColor("#0b0d10"))
        painter.setClipPath(path)
        painter.drawPixmap(x, y, inset)
        painter.setClipping(False)
        painter.setPen(QPen(QColor("#5a6373"), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # A short tag, so the small box can never be mistaken for the output.
        tag = QFont(self.font())
        tag.setPointSizeF(max(7.5, tag.pointSizeF() - 3.5))
        tag.setBold(True)
        painter.setFont(tag)
        metrics = painter.fontMetrics()
        text = "YOU"
        chip = QRectF(
            x + 6,
            y + 6,
            metrics.horizontalAdvance(text) + 8,
            metrics.height() + 2,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRoundedRect(chip, 4.0, 4.0)
        painter.setPen(QColor("#e6eaef"))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class Bridge(QObject):
    """Worker-thread callbacks -> GUI-thread signals."""

    preview = Signal(object)
    input_preview = Signal(object)
    log = Signal(str)
    stats = Signal(dict)
    cameras = Signal(list)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings.load()
        self.pipeline: Pipeline | None = None
        self._preview_size = (960, 540)
        self._reference_path: str = ""
        self._garment_path: str = ""
        self._running = False
        self._syncing_look = False
        self._gpu = nvidia_info()
        # The preview pane shows the processed output by default. "camera" shows
        # the raw webcam, which is the only way to check your framing, your
        # lighting, or whether a bad picture is the camera or the model.
        self._preview_source = "output"
        self._idle_cam: CameraCapture | None = None
        self._idle_ts = 0.0
        self._closing = False
        self._resume_preview_after_scan = False
        # Credit accounting. Decart has no balance API, so the balance is the
        # figure the user typed in and MimicCam subtracts what it metered:
        # _spend_base is what earlier runs already took, _run_offset the
        # engine's meter at the moment the balance was last anchored.
        self._spend_base = float(self.settings.balance_spent)
        self._run_usd = 0.0
        self._run_offset = 0.0
        self._last_money_save = 0.0
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(66)  # ~15 Hz, same cadence as the live preview
        self._idle_timer.timeout.connect(self._pump_idle_frame)

        self.bridge = Bridge()
        self.bridge.preview.connect(self._show_frame)
        self.bridge.input_preview.connect(self._show_camera_frame)
        self.bridge.log.connect(self._append_log)
        self.bridge.stats.connect(self._update_stats)
        self.bridge.cameras.connect(self._populate_cameras)

        self.setWindowTitle(f"{__app_name__} {__version__} - real-time character swap webcam")
        self.resize(self.settings.window_width, self.settings.window_height)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh_credit_box()

        QTimer.singleShot(150, self._startup_checks)
        self._scan_cameras()

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(f"<b style='font-size:18px'>{__app_name__}</b>")
        gpu_text = describe_gpu(self._gpu)
        self.gpu_label = QLabel(gpu_text)
        self.gpu_label.setObjectName("hint")
        header.addWidget(title)
        header.addSpacing(12)
        header.addWidget(self.gpu_label)
        header.addStretch(1)
        # Whose app this is, in the app. The licence and the repository are in the
        # tooltip; both come from ``mimiccam.__init__`` so they cannot drift.
        credit = QLabel(f"{__copyright__}")
        credit.setObjectName("hint")
        credit.setToolTip(f"MimicCam, GPL-2.0-or-later.\n{__url__}")
        header.addWidget(credit)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        # --- preview ------------------------------------------------------
        left = QVBoxLayout()
        preview_bar = QHBoxLayout()
        preview_bar.setSpacing(6)
        self.preview_caption = QLabel("Preview - output (what callers see)")
        self.preview_caption.setObjectName("hint")
        preview_bar.addWidget(self.preview_caption)
        preview_bar.addStretch(1)
        self.preview_out_btn = QPushButton("Output")
        self.preview_cam_btn = QPushButton("Camera")
        self.preview_cam_btn.setToolTip(
            "Your webcam, untouched by any model. Use it to check framing and light "
            "before a call, or to see what the model started from mid-call."
        )
        self.preview_out_btn.setToolTip("What the virtual camera sends to Zoom, Teams or OBS.")
        self.preview_group = QButtonGroup(self)
        self.preview_group.setExclusive(True)
        for button in (self.preview_out_btn, self.preview_cam_btn):
            button.setCheckable(True)
            button.setObjectName("seg")
            self.preview_group.addButton(button)
            preview_bar.addWidget(button)
        self.preview_out_btn.setChecked(True)
        self.preview_out_btn.clicked.connect(lambda: self._choose_preview("output"))
        self.preview_cam_btn.clicked.connect(lambda: self._choose_preview("camera"))
        self.pip_check = QCheckBox("Picture-in-picture")
        self.pip_check.setChecked(self.settings.pip_camera)
        self.pip_check.setToolTip(
            "While the output is on screen, show your raw camera as a small box in "
            "the corner. It costs no credits and no extra camera access."
        )
        self.pip_check.toggled.connect(self._pip_changed)
        preview_bar.addWidget(self.pip_check)
        left.addLayout(preview_bar)

        # The credit box: one line, always in the same place, so the burn rate
        # and what is left can be read at a glance mid-call.
        self.credit_box = QLabel("")
        self.credit_box.setObjectName("creditbox")
        self.credit_box.setTextFormat(Qt.TextFormat.RichText)
        self.credit_box.setToolTip(
            "Decart bills per second of active generation. Seconds come from "
            "Decart's own billing ticks, not from a guess; the balance is the "
            "figure you enter in Settings, since Decart has no balance API."
        )
        # Hug the text instead of stretching across the window: it is a chip,
        # not a banner, and a wide empty strip would just push the picture down.
        credit_row = QHBoxLayout()
        credit_row.addWidget(self.credit_box)
        credit_row.addStretch(1)
        left.addLayout(credit_row)

        self.preview = PreviewView()
        self.preview.setText(
            "No picture yet\nPress Start, or choose Camera to check your webcam."
        )
        left.addWidget(self.preview, 1)

        # --- live bar ------------------------------------------------------
        # The one row a caller needs: start, the look, the character photo.
        # Everything else lives behind the Settings toggle, so the preview can
        # take the whole window while a call is running.
        live = QHBoxLayout()
        live.setSpacing(8)
        self.start_btn = QPushButton("Start camera")
        self.start_btn.setObjectName("primary")
        self.start_btn.setMinimumWidth(140)
        self.start_btn.clicked.connect(self._toggle)
        live.addWidget(self.start_btn)

        look_label = QLabel("Look")
        look_label.setObjectName("hint")
        live.addWidget(look_label)
        self.live_look = QComboBox()
        for preset in PRESETS:
            self.live_look.addItem(preset.label, preset.key)
        # Connect only once the items are in: addItem on an empty combo emits
        # currentIndexChanged while the settings widgets do not exist yet.
        self.live_look.currentIndexChanged.connect(self._live_look_changed)
        live.addWidget(self.live_look, 1)

        self.live_char = QPushButton("Character photo...")
        self.live_char.clicked.connect(self._pick_reference)
        live.addWidget(self.live_char)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setCheckable(True)
        self.settings_btn.toggled.connect(self._show_settings)
        live.addWidget(self.settings_btn)
        left.addLayout(live)

        chips = QHBoxLayout()
        self.chip_vcam = QLabel("Virtual camera: idle")
        self.chip_fps = QLabel("fps -")
        self.chip_latency = QLabel("latency -")
        for chip in (self.chip_vcam, self.chip_fps, self.chip_latency):
            chip.setObjectName("chip")
            chips.addWidget(chip)
        chips.addStretch(1)
        left.addLayout(chips)

        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        left.addWidget(self.hint)
        body.addLayout(left, 3)

        # --- controls -----------------------------------------------------
        # Scrolls: on a laptop the groups are taller than the window, and the
        # log pane used to be clipped by the bottom edge.
        self.settings_panel = QScrollArea()
        self.settings_panel.setWidgetResizable(True)
        self.settings_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_panel.setMaximumWidth(400)
        self.settings_panel.setMinimumWidth(340)
        self.settings_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        cam_group = QGroupBox("Camera")
        cam_form = QFormLayout(cam_group)
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._camera_changed)
        cam_form.addRow("Device", self.camera_combo)
        row = QHBoxLayout()
        self.refresh_cameras_btn = QPushButton("Refresh")
        self.refresh_cameras_btn.clicked.connect(self._scan_cameras)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Virtual cam: Auto", "")
        for key, label in BACKEND_LABELS.items():
            self.backend_combo.addItem(label, key)
        row.addWidget(self.refresh_cameras_btn)
        row.addWidget(self.backend_combo, 1)
        cam_form.addRow(row)
        controls.addWidget(cam_group)

        mode_group = QGroupBox("Engine")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.radio_local = QRadioButton("Local model (free, offline - no credits)")
        self.radio_passthrough = QRadioButton("Passthrough (no AI, free - test your setup)")
        self.radio_decart = QRadioButton("Lucy 2.5 via Decart (needs API key, billed per second)")
        for button in (self.radio_local, self.radio_passthrough, self.radio_decart):
            self.mode_group.addButton(button)
            mode_layout.addWidget(button)
        self.radio_local.setToolTip(
            "Runs the character swap on this GPU. One-time model download, then "
            "no API key, no subscription and no network."
        )

        # Decart's fast tier is a price, not a quality, setting: same output,
        # lower latency, billed at 2x. The label has to say so.
        self.fast_check = QCheckBox(
            "Fast mode - same picture, lower latency, 2x credits ($0.04/s)"
        )
        self.fast_check.setChecked(self.settings.fast_mode)
        mode_layout.addWidget(self.fast_check)

        self.pause_check = QCheckBox(
            "Pause billing when nothing moves (camera keeps working, resumes on movement)"
        )
        self.pause_check.setChecked(self.settings.pause_when_idle)
        # The credit box quotes the rate for the engine and speed selected here,
        # so it reads right before Start as well as during a call.
        self.mode_group.buttonToggled.connect(lambda *_args: self._refresh_credit_box())
        self.fast_check.toggled.connect(lambda *_args: self._refresh_credit_box())
        self.pause_check.setToolTip(
            "Decart bills per second of active generation and has no pause API, so "
            "MimicCam ends the session after 45 s without movement and reconnects "
            "when the camera sees something again. The virtual camera shows your raw "
            "feed while paused."
        )
        mode_layout.addWidget(self.pause_check)

        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("Decart API key (stored in Windows Credential Manager)")
        self.key_save = QPushButton("Save")
        self.key_save.clicked.connect(self._save_key)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_save)
        mode_layout.addLayout(key_row)
        self.key_status = QLabel("")
        self.key_status.setObjectName("hint")
        mode_layout.addWidget(self.key_status)

        balance_row = QHBoxLayout()
        self.balance_edit = QLineEdit()
        self.balance_edit.setPlaceholderText("Decart balance in $, e.g. 10.00")
        if self.settings.balance_usd > 0:
            self.balance_edit.setText(f"{self.settings.balance_usd:.2f}")
        self.balance_set = QPushButton("Set")
        self.balance_set.clicked.connect(self._balance_changed)
        balance_row.addWidget(self.balance_edit, 1)
        balance_row.addWidget(self.balance_set)
        mode_layout.addLayout(balance_row)
        self.balance_hint = QLabel(
            "Decart has no balance API, so enter your balance in dollars - the same "
            "unit as the $0.02/s price - and MimicCam subtracts what it meters. "
            "Set it again after a top-up."
        )
        self.balance_hint.setObjectName("hint")
        self.balance_hint.setWordWrap(True)
        mode_layout.addWidget(self.balance_hint)
        controls.addWidget(mode_group)

        char_group = QGroupBox("Character")
        char_layout = QVBoxLayout(char_group)
        img_row = QHBoxLayout()
        self.thumb = QLabel("no reference")
        self.thumb.setFixedSize(96, 96)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("border: 1px dashed #3a424e; border-radius: 6px; color: #6b7280;")
        pick = QVBoxLayout()
        self.pick_btn = QPushButton("Choose image...")
        self.pick_btn.clicked.connect(self._pick_reference)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_reference)
        pick.addWidget(self.pick_btn)
        pick.addWidget(self.clear_btn)
        pick.addStretch(1)
        img_row.addWidget(self.thumb)
        img_row.addLayout(pick, 1)
        char_layout.addLayout(img_row)

        # Lucy sees the clothes that are inside the photo and nothing else. What
        # is below the frame has to be written down or the model keeps the
        # wearer's own trousers, shoes and body.
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText(
            "Look notes: what the character wears, head to toe"
        )
        self.notes_edit.setToolTip(
            "Described in words, the character's outfit is applied to the whole "
            "body - not just the part inside the photo. Example: black suit "
            "jacket, white shirt, black tie, black trousers, black leather shoes."
        )
        char_layout.addWidget(self.notes_edit)

        self.studio_check = QCheckBox("Seat me in the reference photo's room")
        self.studio_check.setToolTip(
            "Removes the reference person from the photo once, then puts your "
            "camera image into their seat with the room's lighting. Adds a "
            "15 MB matte model and a 208 MB inpainting model on first use."
        )
        char_layout.addWidget(self.studio_check)
        self.studio_hint = QLabel(
            "The room is rebuilt from the character image: everything behind you "
            "is replaced, including the chair, wall, art and lamp light."
        )
        self.studio_hint.setObjectName("hint")
        self.studio_hint.setWordWrap(True)
        char_layout.addWidget(self.studio_hint)

        self.mouth_check = QCheckBox("Keep my mouth and eyes (real expression)")
        self.mouth_check.setToolTip(
            "Pastes your own lips and eyes over the swapped face every frame. The "
            "swap network tracks neither, so without this the character's lips "
            "stop matching what you are saying and they never blink."
        )
        char_layout.addWidget(self.mouth_check)

        self.detail_check = QCheckBox("Sharper face (GFPGAN detail, +340 MB model)")
        self.detail_check.setToolTip(
            "Restores eyes, beard and skin texture in the rendered face. Costs "
            "about 44 ms per render on a 4070, so the face refreshes at roughly "
            "10 Hz instead of 16. Free engine only."
        )
        char_layout.addWidget(self.detail_check)

        self.body_swap_check = QCheckBox("Wear the character's clothes (from the photo)")
        self.body_swap_check.setToolTip(
            "Cuts the garment out of the character photo and fits it to the "
            "shoulders and waist of your own silhouette, so it leans and shifts "
            "with you, then pastes it inside that silhouette. Your hands, arms "
            "and neck stay yours and keep the character's skin tone. It does not "
            "change your body shape, and a raised arm carries the sleeve with the "
            "outline rather than bending the elbow - it is a transplant, not a "
            "generative model."
        )
        char_layout.addWidget(self.body_swap_check)

        # Try-on is a different Lucy model, not a different prompt: it takes a
        # garment photo and dresses the person in it, so the character image is
        # not used at all.
        self.tryon_check = QCheckBox("Try-on: wear the garment in a photo (Lucy)")
        self.tryon_check.setToolTip(
            "lucy-vton-latest replaces whatever you are wearing with the garment "
            "in a photo - the item alone on a clean background works best. Needs "
            "Lucy mode (API key, billed per second). The character photo is ignored."
        )
        self.tryon_check.toggled.connect(self._tryon_toggled)
        char_layout.addWidget(self.tryon_check)

        garment_row = QHBoxLayout()
        self.garment_btn = QPushButton("Choose garment...")
        self.garment_btn.clicked.connect(self._pick_garment)
        self.send_garment_btn = QPushButton("Send to call")
        self.send_garment_btn.setToolTip(
            "Applies the garment and the description above to the running call "
            "without reconnecting."
        )
        self.send_garment_btn.clicked.connect(self._send_garment)
        self.garment_label = QLabel("no garment")
        self.garment_label.setObjectName("hint")
        garment_row.addWidget(self.garment_btn)
        garment_row.addWidget(self.send_garment_btn)
        garment_row.addWidget(self.garment_label, 1)
        char_layout.addLayout(garment_row)

        self.preset_combo = QComboBox()
        for preset in PRESETS:
            self.preset_combo.addItem(preset.label, preset.key)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        char_layout.addWidget(self.preset_combo)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the look you want...")
        self.prompt_edit.setFixedHeight(96)
        char_layout.addWidget(self.prompt_edit)
        controls.addWidget(char_group)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setFont(QFont("Consolas", 9))
        controls.addWidget(self.log_edit)
        controls.addStretch(1)
        self.settings_panel.setWidget(panel)
        body.addWidget(self.settings_panel, 2)

        self.status = self.statusBar()
        self.status.showMessage("Idle")
        # Start with the full panel open; the button then reads "Hide settings".
        self.settings_btn.setChecked(True)

    # -- startup -----------------------------------------------------------
    def _startup_checks(self) -> None:
        if self._gpu is None:
            self.bridge.log.emit(
                "No NVIDIA GPU detected. MimicCam needs an NVIDIA driver; "
                "passthrough may still work but performance will not be validated."
            )
        else:
            self.bridge.log.emit(f"GPU: {describe_gpu(self._gpu)}")

        key = get_api_key()
        self.key_status.setText(f"Key: {mask_key(key)}" if key else "No API key saved")
        local_ready = models.is_ready()
        if self.settings.mode == "decart" and key:
            self.radio_decart.setChecked(True)
        elif self.settings.mode == "local" or local_ready:
            # Free path first: once the models are on disk this is the default,
            # because it costs nothing and never needs a key.
            self.radio_local.setChecked(True)
        elif key:
            self.radio_decart.setChecked(True)
        else:
            self.radio_passthrough.setChecked(True)

        if local_ready:
            self.bridge.log.emit("Local models installed - the free engine is ready.")
            adapter = nvidia_adapter()
            if adapter:
                self.bridge.log.emit(
                    f"Free engine pinned to {adapter['name']} "
                    f"(DirectML adapter {adapter['index']}). It will not fall back to "
                    "the CPU or to integrated graphics."
                )
            else:
                self.bridge.log.emit(
                    "No NVIDIA DirectX adapter found: the free engine will refuse to "
                    "start rather than run on the CPU."
                )
        else:
            self.bridge.log.emit(
                "Local models not downloaded yet ("
                f"{models.total_missing_bytes() / 1048576:.0f} MB). "
                "Choose the local engine to fetch them."
            )

        self.prompt_edit.setPlainText(
            migrate_prompt(self.settings.prompt) or get_preset(self.settings.preset).prompt
        )
        self.notes_edit.setText(self.settings.character_notes)
        index = self.preset_combo.findData(self.settings.preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        if self.settings.reference_image:
            self._set_reference(self.settings.reference_image, announce=False)
        self.studio_check.setChecked(self.settings.studio_background)
        self.mouth_check.setChecked(self.settings.keep_expression)
        self.detail_check.setChecked(self.settings.detail)
        self.body_swap_check.setChecked(self.settings.body_swap)
        self._garment_path = self.settings.garment_image
        if self._garment_path:
            self.garment_label.setText(Path(self._garment_path).name)
        self.tryon_check.setChecked(self.settings.tryon)
        self._tryon_toggled(self.settings.tryon)

        backend_index = self.backend_combo.findData(self.settings.vcam_backend)
        if backend_index >= 0:
            self.backend_combo.setCurrentIndex(backend_index)

        threading.Thread(target=self._check_backends, daemon=True).start()

    def _check_backends(self) -> None:
        try:
            backends = available_backends()
        except Exception as exc:  # noqa: BLE001 - report, never crash the GUI thread
            self.bridge.log.emit(str(exc))
            return
        if not backends:
            self.bridge.log.emit(
                "No virtual-camera backend found. Install OBS Studio, then restart MimicCam."
            )
        else:
            names = ", ".join(BACKEND_LABELS.get(b, b) for b in backends)
            self.bridge.log.emit(f"Virtual camera backend(s) available: {names}")

    def _scan_cameras(self) -> None:
        # Probing opens every camera in turn, so drop the idle preview first:
        # holding the device would make the probe report our own camera missing.
        self._resume_preview_after_scan = self._idle_cam is not None
        self._stop_idle_camera()

        def probe() -> None:
            try:
                entries = probe_cameras(6)
            except Exception as exc:  # noqa: BLE001 - a scan failure must still report
                self.bridge.log.emit(f"Camera scan failed: {exc}")
                entries = []
            # Emitting an empty list still updates the picker; a dead thread
            # would leave it saying "No camera found" with no explanation.
            self.bridge.cameras.emit(entries)

        threading.Thread(target=probe, name="camera-probe", daemon=True).start()

    def _camera_changed(self) -> None:
        if self._idle_cam is not None and not self._running:
            self._start_idle_camera()

    def _populate_cameras(self, entries: list[CameraEntry]) -> None:
        self.camera_combo.clear()
        if not entries:
            self.camera_combo.addItem("No camera found", -1)
            self.bridge.log.emit(
                "No camera delivered a single frame. Close any app that is using "
                "the camera (Teams, WhatsApp, OBS preview, another MimicCam) and "
                "press Refresh."
            )
            return
        for entry in entries:
            self.camera_combo.addItem(entry.label, entry.index)
        preferred = self.settings.camera_index
        index = self.camera_combo.findData(preferred)
        lit = next((i for i, entry in enumerate(entries) if entry.mean_luma >= 8), None)
        if index >= 0 and entries[index].mean_luma < 8 and lit is not None:
            # The saved device is dark (an IR camera, a covered lens, or a probe
            # that came back empty last time). A lit camera is strictly better.
            self.bridge.log.emit(
                f"Camera {preferred} currently gives no picture - using "
                f"camera {entries[lit].index} instead."
            )
            index = lit
        elif index < 0:
            # Entries arrive lit-first, so never silently default to a dark one.
            index = lit if lit is not None else 0
        if index >= 0:
            self.camera_combo.setCurrentIndex(index)
        names = ", ".join(entry.label for entry in entries)
        self.bridge.log.emit(f"Cameras: {names}")
        if entries[index].mean_luma < 8:
            self.bridge.log.emit(
                "The selected camera is delivering no light. Open the lens cover or "
                "pick another device above."
            )
        if self._resume_preview_after_scan and self._preview_source == "camera" \
                and not self._running:
            self._resume_preview_after_scan = False
            QTimer.singleShot(150, self._start_idle_camera)

    # -- preview source ----------------------------------------------------
    def _choose_preview(self, source: str) -> None:
        """Show either what callers receive, or your own untouched camera."""
        source = "camera" if source == "camera" else "output"
        button = self.preview_cam_btn if source == "camera" else self.preview_out_btn
        if not button.isChecked():
            button.setChecked(True)
        changed = source != self._preview_source
        self._preview_source = source
        # The big picture is about to change, so a corner box of the same feed
        # would either duplicate it or go stale.
        self.preview.set_inset(None)
        if source == "camera":
            self.preview_caption.setText("Preview - your camera (no model)")
            if not self._running:
                # Re-clicking Camera must not bounce the device open and shut.
                if self._idle_cam is None:
                    self._start_idle_camera()
            elif changed:
                self.bridge.log.emit(
                    "Preview: your raw camera. The call still receives the processed "
                    "feed."
                )
        else:
            self.preview_caption.setText("Preview - output (what callers see)")
            self._stop_idle_camera()
            self._show_preview_placeholder()

    def _pip_changed(self, enabled: bool) -> None:
        self.settings.pip_camera = enabled
        self.settings.save()
        if not enabled:
            self.preview.set_inset(None)
        self.bridge.log.emit(
            "Picture-in-picture on: your camera in the corner of the output."
            if enabled
            else "Picture-in-picture off."
        )

    def _show_preview_placeholder(self) -> None:
        self.preview.set_inset(None)
        if self._running:
            self.preview.setText("")
            return
        self.preview.setPixmap(QPixmap())
        self.preview.setText(
            "No picture yet\nPress Start, or choose Camera to check your webcam."
        )

    def _start_idle_camera(self) -> None:
        """Open the webcam for preview only - no engine, no virtual camera."""
        if self._closing:
            return
        self._stop_idle_camera()
        selected = self.camera_combo.currentData()
        index = int(selected) if selected is not None else -1
        if index < 0:
            index = int(self.settings.camera_index)
        if index < 0:
            self.preview.setText("No camera found\nPress Refresh in Settings.")
            return
        cam = CameraCapture(
            index,
            self.settings.width,
            self.settings.height,
            max(1, min(30, self.settings.fps)),
            self.settings.mirror,
            on_log=self.bridge.log.emit,
        )
        try:
            opened = cam.start()
        except Exception as exc:  # noqa: BLE001 - a camera driver must not kill the UI
            self.bridge.log.emit(f"Camera preview failed: {exc}")
            opened = False
        if not opened:
            self.preview.setText(
                "Camera busy\nAnother app is using it - close that app, then choose "
                "Camera again."
            )
            return
        self._idle_cam = cam
        self._idle_ts = 0.0
        self._idle_timer.start()
        self.bridge.log.emit("Camera preview on: your webcam, no model and no credits.")

    def _stop_idle_camera(self) -> None:
        self._idle_timer.stop()
        cam, self._idle_cam = self._idle_cam, None
        if cam is None:
            return
        try:
            cam.stop()
        except Exception as exc:  # noqa: BLE001
            self.bridge.log.emit(f"Camera preview stop failed: {exc}")

    def _pump_idle_frame(self) -> None:
        cam = self._idle_cam
        if cam is None or self._preview_source != "camera":
            return
        frame, ts = cam.read()
        if frame is None or ts == self._idle_ts:
            return
        self._idle_ts = ts
        self._show_frame(frame, "camera")

    # -- character / key ---------------------------------------------------
    def _pick_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose character image", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            target = characters_dir() / Path(path).name
            try:
                if Path(path).resolve() != target.resolve():
                    shutil.copyfile(path, target)
                self._set_reference(str(target))
            except OSError as exc:
                QMessageBox.warning(self, "Could not copy image", str(exc))

    def _set_reference(self, path: str, announce: bool = True) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._reference_path = path
        self.settings.reference_image = path
        self.thumb.setPixmap(
            pixmap.scaled(
                92, 92,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.thumb.setStyleSheet("border: 1px solid #3a424e; border-radius: 6px;")
        # Tell the user straight away whether the photo can carry a whole-body
        # swap, instead of letting a chest-up crop quietly turn Lucy into a
        # face swap halfway through a call.
        try:
            from ..reference import analyse

            framing = analyse(path).framing
        except Exception:  # noqa: BLE001 - a photo we cannot read is not fatal
            framing = None
        if framing is not None and framing.kind != "full":
            self.hint.setText(framing.hint)
        if announce:
            self.bridge.log.emit(f"Reference image: {Path(path).name}")

    def _clear_reference(self) -> None:
        self._reference_path = ""
        self.settings.reference_image = ""
        self.thumb.setPixmap(QPixmap())
        self.thumb.setText("no reference")
        self.thumb.setStyleSheet("border: 1px dashed #3a424e; border-radius: 6px; color: #6b7280;")

    # -- try-on ------------------------------------------------------------
    def _tryon_toggled(self, on: bool) -> None:
        """Try-on replaces the outfit, so the character photo step is paused."""
        on = bool(on)
        self.settings.tryon = on
        self.garment_btn.setEnabled(on)
        self.send_garment_btn.setEnabled(on and self._running)
        if on:
            self.prompt_edit.setPlaceholderText(
                "Black leather bomber jacket with ribbed cuffs, relaxed fit"
            )
            # A preset prompt left over from character mode would be sent as the
            # garment instruction - "substitute the whole person" dressed as a
            # shirt. Replace it unless the user has typed something of their own.
            current = self.prompt_edit.toPlainText().strip()
            if not current or current in {preset.prompt.strip() for preset in PRESETS}:
                self.prompt_edit.setPlainText(tryon.DEFAULT_PROMPT)
            self.hint.setText(
                "Try-on dresses you in the garment photo: the character image is "
                "not used, only the clothing changes, and it bills per second "
                "like Lucy."
            )

    def _live_decart_engine(self):
        """The running Decart engine, or None when nothing is live."""
        pipeline = getattr(self, "pipeline", None)
        return getattr(pipeline, "engine", None) if pipeline else None

    def _pick_garment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a garment photo", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        target = characters_dir() / Path(path).name
        try:
            if Path(path).resolve() != target.resolve():
                shutil.copyfile(path, target)
        except OSError as exc:
            QMessageBox.warning(self, "Could not copy image", str(exc))
            return
        self._set_garment(str(target))

    def _set_garment(self, path: str) -> None:
        self._garment_path = path
        self.settings.garment_image = path
        self.garment_label.setText(Path(path).name)
        self.send_garment_btn.setEnabled(self.tryon_check.isChecked())
        self._apply_garment()

    def _send_garment(self) -> None:
        """Re-send the outfit, with whatever the user has edited above it."""
        if not self._garment_path:
            self.hint.setText("Choose a garment photo first.")
            return
        self._apply_garment()

    def _apply_garment(self) -> None:
        """Push a garment into the live session; a no-op when none is running."""
        engine = self._live_decart_engine()
        if engine is None or not getattr(engine, "try_on", False):
            return
        try:
            engine.set_garment(
                self._garment_path, self.prompt_edit.toPlainText().strip()
            )
            self.hint.setText("New outfit sent to the call.")
        except Exception as exc:  # noqa: BLE001 - a failed image is not fatal
            QMessageBox.warning(self, "Could not send the garment", str(exc))

    def _preset_changed(self) -> None:
        key = self.preset_combo.currentData()
        preset = get_preset(key)
        if preset.key != "custom":
            self.prompt_edit.setPlainText(preset.prompt)
        if preset.needs_reference and not self._reference_path:
            self.hint.setText("This preset works best with a character image.")
        elif self._running:
            # The look is baked into the engine at start, so say so instead of
            # leaving a selector that appears to do nothing mid-call.
            self.hint.setText(
                "Look changes apply on the next start: press Stop, then Start camera."
            )
        self._sync_look(preset.key)

    # -- live bar ----------------------------------------------------------
    def _sync_look(self, key: str) -> None:
        """Keep the live-bar look in step with the settings preset."""
        index = self.live_look.findData(key)
        if index < 0 or index == self.live_look.currentIndex():
            return
        self._syncing_look = True
        self.live_look.setCurrentIndex(index)
        self._syncing_look = False

    def _live_look_changed(self) -> None:
        if self._syncing_look or not hasattr(self, "preset_combo"):
            return
        index = self.preset_combo.findData(self.live_look.currentData())
        if index < 0:
            return
        self._syncing_look = True
        self.preset_combo.setCurrentIndex(index)
        self._syncing_look = False
        self._preset_changed()

    def _show_settings(self, shown: bool) -> None:
        self.settings_panel.setVisible(shown)
        self.settings_btn.setText("Hide settings" if shown else "Settings")

    def _save_key(self) -> None:
        value = self.key_edit.text().strip()
        if not value:
            return
        if set_api_key(value):
            self.key_status.setText(f"Key: {mask_key(value)}")
            self.bridge.log.emit("API key saved to Windows Credential Manager.")
            self.key_edit.clear()
            self.radio_decart.setChecked(True)
        else:
            QMessageBox.warning(
                self, "Could not save key",
                "Windows Credential Manager rejected the key. Set the DECART_API_KEY "
                "environment variable instead.",
            )

    # -- credits ----------------------------------------------------------
    def _balance_changed(self) -> None:
        """Anchor the balance to the figure the user reads off their dashboard."""
        raw = self.balance_edit.text().strip().replace("$", "").replace(",", "")
        if not raw:
            self.settings.balance_usd = 0.0
            self.settings.balance_spent = 0.0
            self._spend_base = 0.0
            self._run_usd = 0.0
            self.settings.save()
            self.balance_hint.setText(
                "No balance set. Enter the figure your dashboard shows to see it "
                "count down while you stream."
            )
            self._refresh_credit_box()
            return
        try:
            value = float(raw)
        except ValueError:
            QMessageBox.warning(self, "Balance not understood", f"{raw!r} is not a number.")
            return
        if value < 0:
            QMessageBox.warning(self, "Balance not understood", "A balance cannot be negative.")
            return
        self.settings.balance_usd = round(value, 2)
        # A new anchor means nothing has been taken from it yet, including the
        # part of a running session that happened before this moment.
        self.settings.balance_spent = 0.0
        self._spend_base = 0.0
        self._run_usd = 0.0
        engine = getattr(self.pipeline, "engine", None) if self.pipeline is not None else None
        billing = engine.billing() if hasattr(engine, "billing") else None
        self._run_offset = float(billing["usd"]) if billing else 0.0
        self.settings.save()
        self.balance_edit.setText(f"{self.settings.balance_usd:.2f}")
        self.balance_hint.setText(
            "Decart has no balance API, so enter your balance in dollars - the "
            "same unit as the $0.02/s price - and MimicCam subtracts what it "
            "meters. Set it again after a top-up."
        )
        self.bridge.log.emit(
            f"Balance set to {money(self.settings.balance_usd)} - MimicCam will "
            "subtract the seconds it meters from it."
        )
        self._refresh_credit_box()

    def _balance_parts(self, run_usd: float) -> list[str]:
        spent = self._spend_base + run_usd
        if self.settings.balance_usd <= 0:
            return ["no balance set - add one in Settings"]
        left = remaining_usd(self.settings.balance_usd, spent)
        if left <= 0:
            return [f"<b>$0.00</b> left of {money(self.settings.balance_usd)} - used up"]
        return [f"<b>{money(left)}</b> left of {money(self.settings.balance_usd)}"]

    def _selected_mode(self) -> str:
        if not hasattr(self, "radio_decart"):
            return self.settings.mode
        if self.radio_decart.isChecked():
            return "decart"
        if self.radio_local.isChecked():
            return "local"
        if self.radio_passthrough.isChecked():
            return "passthrough"
        return self.settings.mode

    def _refresh_credit_box(self) -> None:
        """One line: rate, what this session cost, and what the balance has left.

        Runs at the stats cadence (1 Hz), never per frame, and reads the engine's
        server-fed meter rather than guessing from wall-clock time.
        """
        if not hasattr(self, "credit_box"):
            return
        fast = self.fast_check.isChecked() if hasattr(self, "fast_check") else self.settings.fast_mode
        rate = usd_per_second(fast)
        mode = self._selected_mode()
        engine = getattr(self.pipeline, "engine", None) if self.pipeline is not None else None
        billing = engine.billing() if hasattr(engine, "billing") else None

        if billing is None:
            # No paid engine has a meter: either it has not started yet, or the
            # free engine is running, which bills nothing by definition.
            self._run_usd = 0.0
            if mode == "decart":
                parts = [f"<b>{money(rate)}/s</b>", "Lucy 2.5 - idle"]
                parts += self._balance_parts(0.0)
            else:
                flavour = "local model" if mode == "local" else "passthrough"
                parts = ["<b>$0.00/s</b>", f"{flavour} - no credits spent"]
                if self.settings.balance_usd > 0:
                    parts += self._balance_parts(0.0)
            self.credit_box.setText(" &nbsp;·&nbsp; ".join(parts))
            return

        # Quote the rate the running engine is actually billed at, not whatever
        # the checkbox says now: flipping it mid-call does not change the session.
        rate = float(billing.get("usd_per_min", rate * 60)) / 60.0
        self._run_usd = max(0.0, round(float(billing["usd"]) - self._run_offset, 3))
        parts = [f"<b>{money(rate)}/s</b>", f"{clock(billing['seconds'])} billed"]
        if billing.get("parked"):
            parts.append("billing paused")
        parts.append(f"{money(self._run_usd)} this run")
        parts += self._balance_parts(self._run_usd)
        self.credit_box.setText(" &nbsp;·&nbsp; ".join(parts))

    def _save_spend_progress(self) -> None:
        """Persist mid-call so a crash cannot hand back credit that was spent."""
        total = round(self._spend_base + self._run_usd, 3)
        if self.settings.balance_usd <= 0:
            return
        if abs(total - self.settings.balance_spent) < 0.01:
            return
        now = time.monotonic()
        if now - self._last_money_save < 15.0:
            return
        self._last_money_save = now
        self.settings.balance_spent = total
        self.settings.save()

    def _commit_spend(self) -> None:
        """Fold this run's spend into the saved figure, once."""
        total = round(self._spend_base + self._run_usd, 3)
        if self.settings.balance_usd > 0 and abs(total - self.settings.balance_spent) >= 0.001:
            self.settings.balance_spent = total
            self.settings.save()
        self._spend_base = total
        self._run_usd = 0.0
        self._run_offset = 0.0

    # -- run control -------------------------------------------------------
    def _collect_settings(self) -> Settings:
        selected = self.camera_combo.currentData()
        # Only remember a real selection. An empty probe result used to save 0,
        # which then silently pointed the next launch at the black IR camera.
        if selected is not None and int(selected) >= 0:
            self.settings.camera_index = int(selected)
        if self.radio_decart.isChecked():
            self.settings.mode = "decart"
        elif self.radio_local.isChecked():
            self.settings.mode = "local"
        else:
            self.settings.mode = "passthrough"
        self.settings.preset = self.preset_combo.currentData()
        self.settings.prompt = self.prompt_edit.toPlainText().strip()
        self.settings.character_notes = self.notes_edit.text().strip()
        self.settings.reference_image = self._reference_path
        self.settings.studio_background = self.studio_check.isChecked()
        self.settings.keep_expression = self.mouth_check.isChecked()
        self.settings.detail = self.detail_check.isChecked()
        self.settings.body_swap = self.body_swap_check.isChecked()
        self.settings.tryon = self.tryon_check.isChecked()
        self.settings.garment_image = self._garment_path
        self.settings.fast_mode = self.fast_check.isChecked()
        self.settings.pause_when_idle = self.pause_check.isChecked()
        self.settings.pip_camera = self.pip_check.isChecked()
        self.settings.vcam_backend = self.backend_combo.currentData() or ""
        self.settings.save()
        return self.settings

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        settings = self._collect_settings()
        if settings.mode == "decart" and not get_api_key():
            QMessageBox.information(
                self, "API key required",
                "Enter your Decart API key, or switch to Passthrough to test the "
                "camera and virtual camera without spending credits.",
            )
            return
        if settings.mode == "decart":
            problem = require_nvidia(self._gpu)
            if problem:
                QMessageBox.critical(self, "NVIDIA GPU required", problem)
                return
        needed = list(models.CORE_KEYS)
        if settings.detail:
            needed += list(models.DETAIL_KEYS)
        if settings.studio_background:
            needed += list(models.STUDIO_KEYS)
        if settings.mode == "local" and models.missing(needed):
            dialog = ModelDownloadDialog(self, keys=needed)
            if not dialog.start():
                self.bridge.log.emit("Local model download cancelled - nothing started.")
                return
        if settings.camera_index < 0:
            QMessageBox.warning(self, "No camera", "No working camera was found.")
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Starting...")

        engine = None
        if settings.mode == "decart":
            if settings.tryon and not self._garment_path:
                self.start_btn.setEnabled(True)
                self.start_btn.setText("Start camera")
                QMessageBox.warning(
                    self, "No garment photo",
                    "Try-on needs a garment photo: choose one under Try-on first.",
                )
                return
            try:
                from ..decart_engine import DecartEngine

                engine = DecartEngine(
                    api_key=get_api_key(),
                    width=settings.width,
                    height=settings.height,
                    fps=settings.fps,
                    prompt=self.prompt_edit.toPlainText().strip(),
                    preset_key=settings.preset,
                    reference_image=None if settings.tryon else (self._reference_path or None),
                    character_notes=settings.character_notes,
                    garment_image=self._garment_path if settings.tryon else None,
                    model=tryon.MODEL if settings.tryon else settings.model,
                    fast_mode=settings.fast_mode,
                    pause_when_idle=settings.pause_when_idle,
                    on_log=self.bridge.log.emit,
                )
            except Exception as exc:  # noqa: BLE001
                self.start_btn.setEnabled(True)
                self.start_btn.setText("Start camera")
                QMessageBox.critical(self, "Could not start Decart", str(exc))
                return
        elif settings.mode == "local":
            if self._gpu is None:
                self.bridge.log.emit(
                    "No NVIDIA GPU detected: the local engine will use whatever "
                    "DirectML device Windows exposes, which may be slow."
                )
            if not self._reference_path:
                self.bridge.log.emit(
                    "No character image: the local engine will apply the look "
                    "without swapping identity."
                )
            try:
                engine = LocalEngine(
                    reference_image=self._reference_path or None,
                    preset_key=settings.preset,
                    width=settings.width,
                    height=settings.height,
                    fps=settings.fps,
                    on_log=self.bridge.log.emit,
                    keep_expression=settings.keep_expression,
                    detail=settings.detail,
                    body_swap=settings.body_swap,
                    studio_background=settings.studio_background,
                    studio_image=(self._reference_path or None)
                    if settings.studio_background else None,
                )
            except Exception as exc:  # noqa: BLE001
                self.start_btn.setEnabled(True)
                self.start_btn.setText("Start camera")
                QMessageBox.critical(self, "Could not start the local engine", str(exc))
                return

        # Credit accounting for this run starts now: the engine's meter begins
        # at zero and the saved spend is the baseline it is measured against.
        self._spend_base = float(self.settings.balance_spent)
        self._run_usd = 0.0
        self._run_offset = 0.0
        self._last_money_save = time.monotonic()

        # One webcam, one owner: release the idle preview before the pipeline
        # opens the same device.
        if self._idle_cam is not None:
            self._stop_idle_camera()
            time.sleep(0.15)  # give DirectShow a moment to hand the device over

        self.pipeline = Pipeline(
            camera_index=settings.camera_index,
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            mirror=settings.mirror,
            on_log=self.bridge.log.emit,
            on_preview=self.bridge.preview.emit,
            on_input=self.bridge.input_preview.emit,
            on_stats=self.bridge.stats.emit,
            vcam_backend=settings.vcam_backend,
        )

        started = False
        engine_error = ""
        try:
            started = self.pipeline.start(engine)
        except LocalEngineError as exc:
            engine_error = str(exc)
            self.bridge.log.emit(f"Local engine: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.bridge.log.emit(f"Start failed: {exc}")

        self.start_btn.setEnabled(True)
        if not started:
            self.start_btn.setText("Start camera")
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
            QMessageBox.critical(
                self, "Could not start",
                engine_error
                or "The camera or virtual camera could not be opened. Check the log pane.",
            )
            if self._preview_source == "camera":
                QTimer.singleShot(300, self._start_idle_camera)
            return

        self._running = True
        # Live screen: the preview gets the whole window while a call runs.
        self.settings_btn.setChecked(False)
        self.start_btn.setText("Stop")
        self.start_btn.setProperty("running", "true")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.chip_vcam.setText(f"Virtual camera: {self.pipeline.vcam.device}")
        self.hint.setText(
            f"In Zoom / Telegram / WhatsApp choose \"{self.pipeline.vcam.device}\" as the camera."
        )
        self.status.showMessage("Running")
        self._refresh_credit_box()

    def _stop(self) -> None:
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Stopping...")
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                self.bridge.log.emit(f"Stop error: {exc}")
            # Last reading before the engine (and its meter) is dropped.
            self._refresh_credit_box()
            self.pipeline = None
        self._commit_spend()
        self._refresh_credit_box()
        self._running = False
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start camera")
        self.start_btn.setProperty("running", "false")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.settings_btn.setChecked(True)
        self.chip_vcam.setText("Virtual camera: idle")
        self.chip_fps.setText("fps -")
        self.chip_latency.setText("latency -")
        self.hint.setText("")
        self.status.showMessage("Idle")
        self._show_preview_placeholder()
        if self._preview_source == "camera":
            # Bring the webcam preview back once the pipeline has let go of it.
            QTimer.singleShot(300, self._start_idle_camera)

    # -- worker callbacks (GUI thread) ------------------------------------
    def _show_camera_frame(self, frame: np.ndarray) -> None:
        self._show_frame(frame, "camera")

    def _show_frame(self, frame: np.ndarray, source: str = "output") -> None:
        if frame is None or frame.size == 0:
            return
        if source == "camera" and self._preview_source != "camera":
            # The big picture is the output, so the camera goes in the corner -
            # unless the option is off, in which case not even the resize runs.
            if self.settings.pip_camera:
                self.preview.set_inset(self._inset_pixmap(frame))
            return
        if source != self._preview_source:
            return
        self.preview.setPixmap(self._to_pixmap(frame, self.preview.size()))

    @staticmethod
    def _to_pixmap(frame: np.ndarray, size: QSize) -> QPixmap:
        height, width = frame.shape[:2]
        image = QImage(
            frame.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()
        return QPixmap.fromImage(image).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _inset_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Shrink with OpenCV first: it is far cheaper than Qt's smooth scale.

        Measured on a 1280x720 frame going to 336 px: OpenCV bilinear 0.11 ms,
        Qt's smooth scale of the full frame 1.46 ms, and INTER_AREA 1.90 ms. The
        inset is small enough that bilinear is indistinguishable from AREA, so
        the cheap filter wins - the pane already paints the main picture at
        15 Hz and this is the extra frame it must not waste.
        """
        box = self.preview.inset_box()
        height, width = frame.shape[:2]
        target_h = max(1, round(height * box.width() / max(1, width)))
        small = cv2.resize(
            frame, (box.width(), target_h), interpolation=cv2.INTER_LINEAR
        )
        return self._to_pixmap(small, QSize(box.width(), target_h))

    def _append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{stamp}] {message}")

    def _update_stats(self, stats: dict) -> None:
        self.chip_fps.setText(
            f"out {stats.get('output_fps', 0)} fps / cam {stats.get('camera_fps', 0)} fps"
        )
        latency = stats.get("latency_ms") or 0
        self.chip_latency.setText(f"latency {latency} ms" if latency else "latency -")
        self._refresh_credit_box()
        self._save_spend_progress()
        if stats.get("held_frames"):
            self.status.showMessage(
                f"Running - {stats['held_frames']} frames held while the model caught up"
            )

    # -- teardown ----------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self._stop_idle_camera()
        if self._running:
            self._stop()
        self._commit_spend()
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.settings.save()
        super().closeEvent(event)


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setApplicationName(__app_name__)
    window = MainWindow()
    window.show()
    return app.exec()
