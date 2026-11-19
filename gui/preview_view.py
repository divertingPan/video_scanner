"""pyqtgraph based preview with editable overlays (position / mask / width)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

POS_COLOR = (0, 200, 255)
MASK_COLOR = (255, 170, 0)
WIDTH_COLOR = (80, 255, 120)


class MaskROI(pg.ROI):
    """Rectangle ROI with 8 handles and a translucent fill."""

    def __init__(self, pos, size, bounds: QtCore.QRectF):
        super().__init__(pos, size, pen=pg.mkPen(MASK_COLOR, width=2), maxBounds=bounds,
                         snapSize=1.0, translateSnap=True, scaleSnap=True, rotatable=False,
                         movable=True, removable=False)
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0, 1], [1, 0])
        self.addScaleHandle([1, 0], [0, 1])
        self.addScaleHandle([0.5, 0], [0.5, 1])
        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([0, 0.5], [1, 0.5])
        self.addScaleHandle([1, 0.5], [0, 0.5])
        self._fill_alpha = 60

    def set_editable(self, on: bool) -> None:
        self.translatable = on
        self.resizable = on
        for h in self.getHandles():
            h.setVisible(on)
        self._fill_alpha = 60 if on else 35
        self.setPen(pg.mkPen(MASK_COLOR, width=2 if on else 1,
                             style=QtCore.Qt.PenStyle.SolidLine if on else QtCore.Qt.PenStyle.DashLine))
        self.update()

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        p.fillRect(r, QtGui.QColor(*MASK_COLOR, self._fill_alpha))
        p.setPen(self.currentPen)
        p.drawRect(r)

    def mask(self) -> Tuple[int, int, int, int]:
        x, y = self.pos()
        w, h = self.size()
        x0, x1 = sorted((x, x + w))
        y0, y1 = sorted((y, y + h))
        return int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))


class PreviewView(pg.GraphicsLayoutWidget):
    """Frame preview with toolbar modes. Coordinates are always original pixels."""

    positionChanged = QtCore.Signal(object)   # int | None
    maskChanged = QtCore.Signal(object)       # (yt, yb, xl, xr) | None
    widthChanged = QtCore.Signal(object)      # float | None
    hoverInfo = QtCore.Signal(str)
    modeChanged = QtCore.Signal(str)

    MODES = ("pan", "position", "mask", "width")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vb: pg.ViewBox = self.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.vb.setMouseMode(pg.ViewBox.PanMode)
        self.img_item = pg.ImageItem()
        self.vb.addItem(self.img_item)
        self.frame_w, self.frame_h = 0, 0
        self._img_full: Optional[np.ndarray] = None   # RGB array currently displayed
        self._img_scale = 1.0
        self.mode = "pan"
        self.adaptive = False
        self._adaptive_width: Optional[float] = None
        self._saved_range = None
        self._guard = False

        self.position: Optional[int] = None
        self.width: Optional[float] = None
        self.pos_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(POS_COLOR, width=2),
                                        hoverPen=pg.mkPen(POS_COLOR, width=4))
        self.pos_line.setZValue(20)
        self.pos_line.sigPositionChanged.connect(self._on_line_moved)
        self.pos_line.hide()
        self.vb.addItem(self.pos_line)

        self.band = pg.LinearRegionItem(values=(0, 5), orientation="vertical", movable=False,
                                        brush=pg.mkBrush(*WIDTH_COLOR, 70),
                                        pen=pg.mkPen(WIDTH_COLOR, width=1))
        self.band.setZValue(15)
        for ln in self.band.lines:
            ln.setMovable(False)
        self.band.lines[1].setPen(pg.mkPen(WIDTH_COLOR, width=2))
        self.band.lines[1].setHoverPen(pg.mkPen(WIDTH_COLOR, width=4))
        self.band.lines[1].sigPositionChanged.connect(self._on_band_edge_moved)
        self.band.hide()
        self.vb.addItem(self.band)

        self.roi: Optional[MaskROI] = None

        self.live_text = pg.TextItem("", color=(255, 60, 60), anchor=(0, 0))
        self.live_text.setZValue(30)
        self.live_text.hide()
        self.vb.addItem(self.live_text)

        self.scene().sigMouseClicked.connect(self._on_click)
        self.scene().sigMouseMoved.connect(self._on_move)

    # ------------------------------------------------------------------ image
    def set_frame_size(self, w: int, h: int) -> None:
        self.frame_w, self.frame_h = int(w), int(h)
        self.pos_line.setBounds([0, max(0, w - 1)])
        if self.position is not None and w > 0 and self.position >= w:
            self.set_position(w - 1, emit=True)
        if self.roi is not None:
            self.roi.maxBounds = QtCore.QRectF(0, 0, w, h)
            yt, yb, xl, xr = self.roi.mask()
            if w > 0 and h > 0 and (yb > h or xr > w or yt < 0 or xl < 0):
                yt = max(0, min(yt, h - 2))
                yb = max(yt + 1, min(yb, h))
                xl = max(0, min(xl, w - 2))
                xr = max(xl + 1, min(xr, w))
                self.set_mask((yt, yb, xl, xr), emit=True)
        self._update_band()

    def set_image(self, bgr: np.ndarray, reset_view: bool = False) -> None:
        """Show a BGR image; it is stretched to the original frame size so all
        overlay coordinates stay in original pixels even for downscaled images."""
        if bgr is None:
            return
        rgb = np.ascontiguousarray(bgr[..., ::-1]) if bgr.ndim == 3 else bgr
        self._img_full = rgb
        self.img_item.setImage(rgb, autoLevels=False, levels=(0, 255))
        fw = self.frame_w or rgb.shape[1]
        fh = self.frame_h or rgb.shape[0]
        self._img_scale = rgb.shape[1] / float(fw)
        self.img_item.setRect(QtCore.QRectF(0, 0, fw, fh))
        if reset_view:
            self.reset_view()

    def set_live_label(self, text: str) -> None:
        if text:
            self.live_text.setText(text)
            self.live_text.setPos(0, 0)
            self.live_text.show()
        else:
            self.live_text.hide()

    # ------------------------------------------------------------------ view actions
    def reset_view(self) -> None:
        if self.frame_w and self.frame_h:
            self.vb.setRange(QtCore.QRectF(0, 0, self.frame_w, self.frame_h), padding=0.02)

    def zoom(self, factor: float) -> None:
        self.vb.scaleBy((1.0 / factor, 1.0 / factor))

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            return
        if mode == "width" and self.adaptive:
            mode = "pan"
        prev = self.mode
        self.mode = mode
        if prev == "width" and mode != "width" and self._saved_range is not None:
            xr, yr = self._saved_range
            self.vb.setRange(xRange=xr, yRange=yr, padding=0)
            self._saved_range = None
        self.pos_line.setMovable(mode == "position")
        if mode == "mask" and self.roi is None and self.frame_w:
            w, h = self.frame_w, self.frame_h
            self.set_mask((h // 3, 2 * h // 3, w // 3, 2 * w // 3), emit=True)
        if self.roi is not None:
            self.roi.set_editable(mode == "mask")
        if mode == "width":
            if self.position is None and self.frame_w:
                self.set_position(self.frame_w // 2, emit=True)
            if self.width is None:
                self.set_width(5.0, emit=True)
            self._saved_range = self.vb.viewRange()
            p = self.position or 0
            self.vb.setXRange(p - 100, p + (self.width or 5) + 100, padding=0)
        self.band.lines[1].setMovable(mode == "width")
        self.modeChanged.emit(mode)

    def delete_current(self) -> bool:
        if self.mode == "position":
            self.set_position(None, emit=True)
        elif self.mode == "mask":
            self.set_mask(None, emit=True)
        elif self.mode == "width":
            self.set_width(None, emit=True)
        else:
            return False
        return True

    # ------------------------------------------------------------------ overlays (setters)
    def set_position(self, x: Optional[int], emit: bool = False) -> None:
        self.position = None if x is None else int(round(x))
        self._guard = True
        try:
            if self.position is None:
                self.pos_line.hide()
            else:
                self.pos_line.setValue(self.position)
                self.pos_line.show()
        finally:
            self._guard = False
        self._update_band()
        if emit:
            self.positionChanged.emit(self.position)

    def set_width(self, w: Optional[float], emit: bool = False) -> None:
        self.width = None if w is None else max(0.01, round(float(w), 2))
        self._update_band()
        if emit:
            self.widthChanged.emit(self.width)

    def set_adaptive(self, adaptive: bool, current_width: Optional[float] = None) -> None:
        self.adaptive = adaptive
        self._adaptive_width = current_width
        if adaptive and self.mode == "width":
            self.set_mode("pan")
        self._update_band()

    def _update_band(self) -> None:
        w = (abs(self._adaptive_width) if self._adaptive_width else None) if self.adaptive else self.width
        if self.position is None or not w:
            self.band.hide()
            return
        self._guard = True
        try:
            self.band.setRegion((self.position, self.position + w))
            self.band.lines[1].setBounds([self.position + 0.01, max(self.position + 0.01, self.frame_w or 1e9)])
        finally:
            self._guard = False
        self.band.show()

    def set_mask(self, m: Optional[Tuple[int, int, int, int]], emit: bool = False) -> None:
        if m is None:
            if self.roi is not None:
                self.vb.removeItem(self.roi)
                self.roi = None
        else:
            yt, yb, xl, xr = m
            if self.roi is None:
                bounds = QtCore.QRectF(0, 0, self.frame_w or 1e6, self.frame_h or 1e6)
                self.roi = MaskROI((xl, yt), (xr - xl, yb - yt), bounds)
                self.roi.setZValue(10)
                self.roi.sigRegionChanged.connect(self._on_roi_changed)
                self.vb.addItem(self.roi)
                self.roi.set_editable(self.mode == "mask")
            else:
                self._guard = True
                try:
                    self.roi.setPos((xl, yt), update=False)
                    self.roi.setSize((xr - xl, yb - yt))
                finally:
                    self._guard = False
        if emit:
            self.maskChanged.emit(None if self.roi is None else self.roi.mask())

    # ------------------------------------------------------------------ overlay events
    def _on_line_moved(self) -> None:
        if self._guard:
            return
        v = int(round(self.pos_line.value()))
        self.position = v
        self._update_band()
        self.positionChanged.emit(v)

    def _on_band_edge_moved(self) -> None:
        if self._guard or self.position is None or self.adaptive:
            return
        x = self.band.lines[1].value()
        self.width = max(0.01, round(x - self.position, 2))
        self.widthChanged.emit(self.width)

    def _on_roi_changed(self) -> None:
        if self._guard or self.roi is None:
            return
        self.maskChanged.emit(self.roi.mask())

    def _on_click(self, ev) -> None:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if ev.double():
            self.reset_view()
            return
        if self.mode == "position" and self.frame_w:
            p = self.vb.mapSceneToView(ev.scenePos())
            x = int(round(p.x()))
            if 0 <= x < self.frame_w:
                self.set_position(x, emit=True)

    def _on_move(self, scene_pos) -> None:
        if self._img_full is None or not self.frame_w:
            return
        p = self.vb.mapSceneToView(scene_pos)
        x, y = int(p.x()), int(p.y())
        if 0 <= x < self.frame_w and 0 <= y < self.frame_h:
            ix = min(self._img_full.shape[1] - 1, int(x * self._img_scale))
            iy = min(self._img_full.shape[0] - 1, int(y * self._img_scale))
            px = self._img_full[iy, ix]
            r, g, b = (int(px[0]), int(px[1]), int(px[2])) if np.ndim(px) else (int(px),) * 3
            self.hoverInfo.emit(f"{x},{y},{r},{g},{b}")
        else:
            self.hoverInfo.emit("")


class RibbonView(QtWidgets.QWidget):
    """Live panorama thumbnail; follows the newest end while processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg = QtGui.QColor(28, 30, 34)
        self._rgb_buf: Optional[np.ndarray] = None
        self._qimage: Optional[QtGui.QImage] = None
        self._shape: Tuple[int, int] = (0, 0)
        self.follow: bool = True
        self.newest_left: bool = False
        self._x0: float = 0.0
        self._drag_last_x: Optional[float] = None
        self._empty_text: str = ""
        self.setMinimumHeight(84)
        self.setMaximumHeight(140)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def setBackground(self, color) -> None:
        if isinstance(color, QtGui.QColor):
            self._bg = color
        elif isinstance(color, (tuple, list)) and len(color) >= 3:
            self._bg = QtGui.QColor(int(color[0]), int(color[1]), int(color[2]))
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    def clear(self) -> None:
        self._rgb_buf = None
        self._qimage = None
        self._shape = (0, 0)
        self._x0 = 0.0
        self.follow = True
        self.update()

    def _visible_span(self) -> float:
        h, _w = self._shape
        if h <= 0:
            return 1.0
        return h * max(1.0, float(self.width() - 2)) / max(1.0, float(self.height() - 2))

    def _clamp_x0(self, x0: float) -> float:
        _h, w = self._shape
        return max(0.0, min(max(0.0, float(w) - self._visible_span()), x0))

    def _follow(self) -> None:
        h, w = self._shape
        if not w or not h:
            self._x0 = 0.0
            return
        self._x0 = 0.0 if self.newest_left else max(0.0, float(w) - self._visible_span())

    def set_ribbon(self, bgr: Optional[np.ndarray], newest_left: bool) -> None:
        if bgr is None or bgr.size == 0:
            return
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        h, w = rgb.shape[:2]
        self._rgb_buf = rgb
        self._qimage = QtGui.QImage(
            self._rgb_buf.data, w, h, int(self._rgb_buf.strides[0]),
            QtGui.QImage.Format.Format_RGB888,
        )
        self.newest_left = newest_left
        self._shape = (h, w)
        if self.follow:
            self._follow()
        else:
            self._x0 = self._clamp_x0(self._x0)
        self.update()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._qimage is not None:
            if self.follow:
                self._follow()
            else:
                self._x0 = self._clamp_x0(self._x0)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._qimage is not None:
            self._drag_last_x = float(ev.position().x())
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drag_last_x is not None and self._qimage is not None:
            cur_x = float(ev.position().x())
            dx_px = cur_x - self._drag_last_x
            self._drag_last_x = cur_x
            scale = max(1.0, float(self.height() - 2)) / max(1.0, float(self._shape[0]))
            self.follow = False
            self._x0 = self._clamp_x0(self._x0 - dx_px / scale)
            self.update()
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._drag_last_x is not None:
            self._drag_last_x = None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.follow = True
            self._follow()
            self.update()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if self._qimage is not None:
            delta = ev.angleDelta().x() or ev.angleDelta().y()
            if delta:
                self.follow = False
                self._x0 = self._clamp_x0(self._x0 - (delta / 120.0) * (self._visible_span() * 0.15))
                self.update()
                ev.accept()
                return
        super().wheelEvent(ev)

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self.rect()
        p.fillRect(rect, self._bg)
        inner = rect.adjusted(1, 1, -1, -1)
        if self._qimage is None or self._shape[1] <= 0 or inner.width() <= 0 or inner.height() <= 0:
            if self._empty_text:
                p.setPen(QtGui.QColor(145, 152, 165))
                f = p.font()
                f.setPointSize(9)
                p.setFont(f)
                p.drawText(inner, QtCore.Qt.AlignmentFlag.AlignCenter, self._empty_text)
        else:
            img_h, img_w = self._shape
            vh = float(inner.height())
            scale = vh / float(img_h)
            span = float(inner.width()) / scale
            src_x0 = max(0.0, min(float(img_w), self._x0))
            src_w = max(0.0, min(float(img_w) - src_x0, span))
            if src_w > 0:
                dst_w = src_w * scale
                src_rect = QtCore.QRectF(src_x0, 0.0, src_w, float(img_h))
                dst_rect = QtCore.QRectF(float(inner.left()), float(inner.top()), dst_w, vh)
                p.drawImage(dst_rect, self._qimage, src_rect)
                edge_x = float(inner.left()) if (self.newest_left and src_x0 <= 0.5) else (
                    float(inner.left()) + dst_w - 1.0 if (not self.newest_left and src_x0 + src_w >= img_w - 0.5) else -1.0
                )
                if edge_x >= 0.0:
                    p.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255, 200), 2.0))
                    p.drawLine(QtCore.QPointF(edge_x, float(inner.top())),
                               QtCore.QPointF(edge_x, float(inner.bottom())))
        p.setPen(QtGui.QPen(QtGui.QColor(90, 98, 112), 1.0))
        p.drawRect(rect.adjusted(0, 0, -1, -1))
        p.end()
