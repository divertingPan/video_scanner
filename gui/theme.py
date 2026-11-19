"""Light / dark / auto theme on top of Qt's Fusion style."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def system_is_dark() -> bool:
    try:
        return QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
    except AttributeError:
        pal = QtWidgets.QApplication.palette()
        return pal.color(QtGui.QPalette.ColorRole.Window).lightness() < 128


def _dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    C = QtGui.QColor
    R = QtGui.QPalette.ColorRole
    G = QtGui.QPalette.ColorGroup
    p.setColor(R.Window, C(37, 39, 43))
    p.setColor(R.WindowText, C(225, 227, 230))
    p.setColor(R.Base, C(28, 30, 33))
    p.setColor(R.AlternateBase, C(45, 47, 52))
    p.setColor(R.ToolTipBase, C(50, 52, 58))
    p.setColor(R.ToolTipText, C(230, 230, 230))
    p.setColor(R.Text, C(225, 227, 230))
    p.setColor(R.Button, C(50, 53, 58))
    p.setColor(R.ButtonText, C(225, 227, 230))
    p.setColor(R.Midlight, C(62, 66, 74))
    p.setColor(R.Mid, C(92, 98, 110))
    p.setColor(R.Dark, C(22, 24, 27))
    p.setColor(R.BrightText, C(255, 90, 90))
    p.setColor(R.Highlight, C(64, 140, 230))
    p.setColor(R.HighlightedText, C(255, 255, 255))
    p.setColor(R.Link, C(100, 170, 255))
    p.setColor(R.PlaceholderText, C(140, 140, 140))
    for role in (R.Text, R.ButtonText, R.WindowText):
        p.setColor(G.Disabled, role, C(120, 120, 120))
    return p


def _light_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    C = QtGui.QColor
    R = QtGui.QPalette.ColorRole
    G = QtGui.QPalette.ColorGroup
    p.setColor(R.Window, C(243, 244, 246))
    p.setColor(R.WindowText, C(25, 25, 28))
    p.setColor(R.Base, C(255, 255, 255))
    p.setColor(R.AlternateBase, C(236, 238, 241))
    p.setColor(R.ToolTipBase, C(255, 255, 230))
    p.setColor(R.ToolTipText, C(20, 20, 20))
    p.setColor(R.Text, C(25, 25, 28))
    p.setColor(R.Button, C(250, 250, 252))
    p.setColor(R.ButtonText, C(25, 25, 28))
    p.setColor(R.BrightText, C(200, 0, 0))
    p.setColor(R.Highlight, C(48, 120, 220))
    p.setColor(R.HighlightedText, C(255, 255, 255))
    p.setColor(R.Link, C(30, 100, 210))
    p.setColor(R.PlaceholderText, C(130, 130, 130))
    for role in (R.Text, R.ButtonText, R.WindowText):
        p.setColor(G.Disabled, role, C(150, 150, 150))
    return p


def _build_style(dark: bool) -> str:
    if dark:
        btn_bg = "#363a42"
        btn_border = "#636b78"
        btn_text = "#e4e7ec"
        btn_hover_bg = "#434954"
        btn_hover_border = "#58a6ff"
        btn_press_bg = "#2b2f36"
        btn_dis_bg = "#292b30"
        btn_dis_border = "#3b3f47"
        btn_dis_text = "#6e7480"
        act_bg = "#2563eb"
        act_border = "#60a5fa"
        act_hover_bg = "#1d4ed8"
        accent_bg = "#1f3556"
        accent_border = "#4c8bf5"
        accent_hover_bg = "#284570"
        prim_bg = "#238636"
        prim_border = "#3fb950"
        prim_hover_bg = "#2ea043"
    else:
        btn_bg = "#ffffff"
        btn_border = "#a8b2c0"
        btn_text = "#1c1e24"
        btn_hover_bg = "#eef4ff"
        btn_hover_border = "#3b82f6"
        btn_press_bg = "#dbeafe"
        btn_dis_bg = "#ebedf0"
        btn_dis_border = "#d2d6dc"
        btn_dis_text = "#9ca3af"
        act_bg = "#2563eb"
        act_border = "#1d4ed8"
        act_hover_bg = "#1e40af"
        accent_bg = "#edf4ff"
        accent_border = "#3b82f6"
        accent_hover_bg = "#dbeafe"
        prim_bg = "#16a34a"
        prim_border = "#15803d"
        prim_hover_bg = "#15803d"

    return f"""
QGroupBox {{
    font-weight: 600;
    border: 1px solid palette(mid);
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QPushButton, QToolButton {{
    background-color: {btn_bg};
    color: {btn_text};
    border: 1px solid {btn_border};
    border-radius: 5px;
}}
QPushButton {{
    padding: 4px 11px;
}}
QToolButton {{
    padding: 4px 8px;
}}
QPushButton:hover:!disabled, QToolButton:hover:!disabled {{
    background-color: {btn_hover_bg};
    border: 1px solid {btn_hover_border};
}}
QPushButton:pressed:!disabled, QToolButton:pressed:!disabled {{
    background-color: {btn_press_bg};
}}
QToolButton:checked, QPushButton:checked {{
    background-color: {act_bg};
    color: #ffffff;
    border: 1px solid {act_border};
    font-weight: 600;
}}
QToolButton:checked:hover:!disabled, QPushButton:checked:hover:!disabled {{
    background-color: {act_hover_bg};
    border: 1px solid {act_border};
}}
QPushButton#accent {{
    background-color: {accent_bg};
    border: 1px solid {accent_border};
    font-weight: 600;
    padding: 5px 12px;
}}
QPushButton#accent:hover:!disabled {{
    background-color: {accent_hover_bg};
    border: 1px solid {btn_hover_border};
}}
QPushButton#primary {{
    background-color: {prim_bg};
    color: #ffffff;
    border: 1px solid {prim_border};
    font-weight: 600;
    padding: 6px 18px;
}}
QPushButton#primary:hover:!disabled {{
    background-color: {prim_hover_bg};
}}
QPushButton:disabled, QToolButton:disabled, QPushButton#accent:disabled, QPushButton#primary:disabled {{
    background-color: {btn_dis_bg};
    color: {btn_dis_text};
    border: 1px solid {btn_dis_border};
}}
QProgressBar {{
    border: 1px solid palette(mid);
    border-radius: 4px;
    text-align: center;
    min-height: 18px;
}}
"""


def apply_theme(app: QtWidgets.QApplication, mode: str) -> bool:
    """mode: auto | light | dark. Returns True if the resulting theme is dark."""
    app.setStyle("Fusion")
    dark = system_is_dark() if mode == "auto" else (mode == "dark")
    app.setPalette(_dark_palette() if dark else _light_palette())
    app.setStyleSheet(_build_style(dark))
    return dark
