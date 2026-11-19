"""Hardware / software decoder selection widget with background FPS probing."""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from core.config import DecoderConfig
from core.hwaccel import ProbeResult, list_decoders, probe_decoder
from .i18n import tr


class DecoderPanel(QtWidgets.QGroupBox):
    """Dropdown of available decoders + live probe status indicator."""

    decoderChanged = QtCore.Signal(object)  # DecoderConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._options: List[DecoderConfig] = list_decoders()
        self._video_path: Optional[str] = None
        self._probe_seq = 0
        self._lock = threading.Lock()
        self._pending_result: Optional[Tuple[int, ProbeResult]] = None
        self._last_state: str = "novideo"   # novideo | probing | done
        self._last_result: Optional[ProbeResult] = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(4)
        self.combo = QtWidgets.QComboBox()
        for opt in self._options:
            self.combo.addItem(opt.label)
        self.combo.currentIndexChanged.connect(self._on_index_changed)
        lay.addWidget(self.combo)

        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_probe)

        self.retranslate()

    # ------------------------------------------------------------------ API
    def decoder(self) -> DecoderConfig:
        idx = max(0, min(self.combo.currentIndex(), len(self._options) - 1))
        return self._options[idx]

    def set_by_key(self, kind: str, device: int) -> None:
        for i, opt in enumerate(self._options):
            if opt.kind == kind and opt.device == device:
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(i)
                self.combo.blockSignals(False)
                break
        if self._video_path:
            self._start_probe()

    def set_video(self, path: Optional[str]) -> None:
        self._video_path = path if path else None
        if not self._video_path:
            self._probe_seq += 1
            self._last_state = "novideo"
            self._last_result = None
            self._poll_timer.stop()
            self._render_status()
        else:
            self._start_probe()

    def retranslate(self) -> None:
        self.setTitle(tr("grp_decoder"))
        self._render_status()

    # ------------------------------------------------------------------ probing
    def _on_index_changed(self, _idx: int) -> None:
        if self._video_path:
            self._start_probe()
        self.decoderChanged.emit(self.decoder())

    def _start_probe(self) -> None:
        path = self._video_path
        if not path:
            return
        self._probe_seq += 1
        seq = self._probe_seq
        dec = self.decoder()
        self._last_state = "probing"
        self._last_result = None
        self._render_status()
        if not self._poll_timer.isActive():
            self._poll_timer.start()

        def worker() -> None:
            res = probe_decoder(path, dec, n_frames=30)
            with self._lock:
                if seq == self._probe_seq:
                    self._pending_result = (seq, res)

        threading.Thread(target=worker, name="DecoderProbe", daemon=True).start()

    def _poll_probe(self) -> None:
        with self._lock:
            item = self._pending_result
            self._pending_result = None
        if item is None:
            return
        seq, res = item
        if seq != self._probe_seq:
            return
        self._poll_timer.stop()
        self._last_state = "done"
        self._last_result = res
        self._render_status()

    def _render_status(self) -> None:
        if self._last_state == "novideo":
            self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
            self.lbl_status.setText(tr("probe_novideo"))
            return
        if self._last_state == "probing":
            self.lbl_status.setStyleSheet("color: #3a8ee6; font-size: 11px;")
            self.lbl_status.setText(tr("probing"))
            return
        res = self._last_result
        if res is None:
            return
        if not res.ok:
            self.lbl_status.setStyleSheet("color: #e05252; font-size: 11px;")
            self.lbl_status.setText(tr("probe_fail", msg=res.message))
        elif res.fell_back:
            self.lbl_status.setStyleSheet("color: #d9822b; font-size: 11px;")
            self.lbl_status.setText(tr("probe_fallback", fps=res.fps))
        elif res.hw_active:
            self.lbl_status.setStyleSheet("color: #2ea043; font-size: 11px;")
            self.lbl_status.setText(tr("probe_hw_ok", fps=res.fps))
        else:
            self.lbl_status.setStyleSheet("color: #2ea043; font-size: 11px;")
            self.lbl_status.setText(tr("probe_sw", fps=res.fps))
