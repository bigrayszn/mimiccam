"""Model download dialog.

The offline engine needs three weight files the app never bundles: they are
hundreds of megabytes and the InsightFace ones carry a research-only licence, so
the user has to see what is being fetched before it happens. This dialog shows
the total, the current file and a live progress bar, and can be cancelled at any
point (the partial file is deleted, never left half-written).
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from .. import models


class _Worker(QThread):
    progress = Signal(str, int, int)
    done = Signal(bool, str)

    def __init__(self, cancel: threading.Event, keys) -> None:
        super().__init__()
        self._cancel = cancel
        self._keys = keys

    def run(self) -> None:  # noqa: D401 - Qt entry point
        try:
            models.ensure(
                progress=lambda key, done, total: self.progress.emit(key, done, total),
                cancel=self._cancel,
                keys=self._keys,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the dialog
            self.done.emit(False, str(exc))
            return
        self.done.emit(True, "")


class ModelDownloadDialog(QDialog):
    def __init__(self, parent=None, keys=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download local models")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._cancel = threading.Event()
        self._ok = False
        self._keys = list(keys) if keys is not None else list(models.CORE_KEYS)

        missing = models.missing(self._keys)
        total_mb = sum(spec.size_bytes for spec in missing) / 1048576

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"MimicCam runs the model on this PC. It needs {len(missing)} file(s), "
            f"{total_mb:.0f} MB, downloaded once."
        ))
        licence = QLabel(models.licence_summary())
        licence.setWordWrap(True)
        licence.setStyleSheet("color: #93a1b3; font-size: 11px;")
        layout.addWidget(licence)

        self.status = QLabel("Starting...")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignRight)

        self._worker = _Worker(self._cancel, self._keys)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)

    def start(self) -> bool:
        self._worker.start()
        self.exec()
        self._worker.wait(2000)
        return self._ok

    def _on_progress(self, key: str, done: int, total: int) -> None:
        spec = models.BY_KEY.get(key)
        name = spec.filename if spec else key
        share = (done / total) if total else 0.0
        self.bar.setValue(int(share * 100))
        self.status.setText(
            f"{name}: {done / 1048576:.0f} of {total / 1048576:.0f} MB"
        )

    def _on_done(self, ok: bool, error: str) -> None:
        self._ok = ok
        if ok:
            self.bar.setValue(100)
            self.status.setText("Models ready.")
        else:
            self.status.setText(f"Download failed: {error}")
        self.cancel_btn.setText("Close")
        self.accept() if ok else None

    def _on_cancel(self) -> None:
        self._cancel.set()
        self._worker.wait(3000)
        self.reject()
