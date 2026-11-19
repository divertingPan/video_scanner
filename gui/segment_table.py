"""Editable segment table (start frame | width) with a width curve."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .i18n import tr


class SegmentTable(QtWidgets.QWidget):
    segmentsChanged = QtCore.Signal()
    jumpToFrame = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._busy = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumWidth(0)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        lay.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton()
        self.btn_del = QtWidgets.QPushButton()
        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.remove_rows)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_del)
        lay.addLayout(row)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumWidth(0)
        self.plot.setFixedHeight(90)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [], stepMode="center", pen=pg.mkPen((80, 200, 255), width=2))
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 80, 80), width=1))
        self.plot.addItem(self.cursor)
        lay.addWidget(self.plot)
        self.retranslate()

    def retranslate(self) -> None:
        self.table.setHorizontalHeaderLabels([tr("seg_start"), tr("seg_width")])
        self.btn_add.setText(tr("seg_add"))
        self.btn_del.setText(tr("seg_del"))

    def set_total_frames(self, n: int) -> None:
        self._total_frames = n
        self._update_plot()

    def set_cursor(self, frame: int) -> None:
        self.cursor.setValue(frame)

    # ------------------------------------------------------------------ data
    def set_segments(self, segs: List[Tuple[int, float]]) -> None:
        self._busy = True
        try:
            self.table.setRowCount(0)
            for s, w in segs:
                self._append(s, w)
        finally:
            self._busy = False
        self._validate()
        self._update_plot()
        self.segmentsChanged.emit()

    def _append(self, s, w) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(int(s))))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(w):g}"))

    def add_row(self) -> None:
        segs = self.segments(strict=False)
        if segs:
            last_s, last_w = segs[-1]
            s = min(self._total_frames - 1, last_s + max(1, self._total_frames // 10)) if self._total_frames else last_s + 100
            w = last_w
        else:
            s, w = 0, 5.0
        self._busy = True
        self._append(s, w)
        self._busy = False
        self._validate()
        self._update_plot()
        self.segmentsChanged.emit()

    def remove_rows(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self._validate()
        self._update_plot()
        self.segmentsChanged.emit()

    def segments(self, strict: bool = True) -> List[Tuple[int, float]]:
        out = []
        for r in range(self.table.rowCount()):
            try:
                s = int(float(self.table.item(r, 0).text()))
                w = float(self.table.item(r, 1).text())
            except (AttributeError, ValueError):
                if strict:
                    raise ValueError(f"row {r + 1}: invalid number")
                continue
            out.append((s, w))
        out.sort(key=lambda t: t[0])
        return out

    def width_at(self, frame: int) -> Optional[float]:
        segs = self.segments(strict=False)
        if not segs:
            return None
        w = segs[0][1]
        for s, ww in segs:
            if s <= frame:
                w = ww
        return w

    # ------------------------------------------------------------------ events
    def _on_item_changed(self, item) -> None:
        if self._busy:
            return
        self._validate()
        self._update_plot()
        self.segmentsChanged.emit()

    def _on_cell_clicked(self, row, col) -> None:
        it = self.table.item(row, 0)
        try:
            self.jumpToFrame.emit(int(float(it.text())))
        except (AttributeError, ValueError):
            pass

    def _validate(self) -> None:
        self._busy = True
        try:
            for r in range(self.table.rowCount()):
                for c in range(2):
                    it = self.table.item(r, c)
                    if it is None:
                        continue
                    try:
                        v = float(it.text())
                        bad = (c == 0 and v < 0) or (c == 1 and v == 0)
                    except ValueError:
                        bad = True
                    it.setBackground(QtGui.QColor(255, 80, 80, 90) if bad else QtGui.QBrush())
        finally:
            self._busy = False

    def _update_plot(self) -> None:
        segs = self.segments(strict=False)
        if not segs:
            self.curve.setData([], [])
            return
        xs = [s for s, _ in segs]
        ys = [abs(w) for _, w in segs]
        end = max(self._total_frames, xs[-1] + 1)
        self.curve.setData(np.array(xs + [end], float), np.array(ys, float))
