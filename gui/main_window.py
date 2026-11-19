"""Main application window for video_scanner v0.4.0 (PySide6 + pyqtgraph)."""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from core import __version__
from core.config import Mask, ScanConfig, StabilizerConfig
from core.motion import MotionResult, estimate_segments
from core.pipeline import BackgroundTask, ScanJob
from core.tile_writer import find_existing_outputs
from core.video_io import VideoInfo, read_info
from .decoder_panel import DecoderPanel
from .i18n import language, set_language, tr
from .preview_view import PreviewView, RibbonView
from .preview_worker import PreviewRequest, PreviewWorker
from .segment_table import SegmentTable
from .theme import apply_theme

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv"}
VIDEO_FILTER = "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.mts *.m2ts *.mpg *.mpeg *.wmv);;All files (*)"


def _fmt_eta(sec: float) -> str:
    if sec < 0 or sec > 360000:
        return "--:--"
    s = int(round(sec))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _NoWheelFilter(QtCore.QObject):
    """Prevent mouse-wheel scrolling from changing spinbox/combo values in scrollable panels."""

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if ev.type() == QtCore.QEvent.Type.Wheel:
            ev.ignore()
            return True
        return super().eventFilter(obj, ev)


def _detect_system_lang() -> str:
    loc = QtCore.QLocale.system()
    if loc.language() == QtCore.QLocale.Language.Chinese or loc.name().lower().startswith("zh"):
        return "zh"
    return "en"


def _config_file_path() -> str:
    """Return path to portable config.ini stored alongside the executable / main.py."""
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(app_dir, "config.ini")
    if os.path.exists(candidate):
        if os.access(candidate, os.W_OK):
            return candidate
    elif os.access(app_dir, os.W_OK):
        return candidate
    fallback_dir = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppConfigLocation
    )
    os.makedirs(fallback_dir, exist_ok=True)
    return os.path.join(fallback_dir, "config.ini")


def _default_app_icon() -> QtGui.QIcon:
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(meipass)
    roots.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for root in roots:
        for rel in (("assets", "icon.ico"), ("assets", "icon.png")):
            p = os.path.join(root, *rel)
            if os.path.isfile(p):
                ico = QtGui.QIcon(p)
                if not ico.isNull():
                    return ico
    return QtGui.QIcon()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app: QtWidgets.QApplication):
        super().__init__()
        self.app = app
        if app.windowIcon().isNull():
            ico = _default_app_icon()
            if not ico.isNull():
                app.setWindowIcon(ico)
        if not app.windowIcon().isNull():
            self.setWindowIcon(app.windowIcon())
        self._config_path = _config_file_path()
        self.settings = QtCore.QSettings(self._config_path, QtCore.QSettings.Format.IniFormat)
        self._video_info: Optional[VideoInfo] = None
        self._save_root: str = ""
        self._last_out_dir: str = ""
        self._syncing = False
        self._need_reset_view = False
        self._last_live_frame = -1
        self._job: Optional[ScanJob] = None
        self._job_direction: int = 1
        self._motion_task: Optional[BackgroundTask] = None
        self._preview_worker = PreviewWorker()

        saved_lang = self.settings.value("ui/lang", None)
        set_language(str(saved_lang) if saved_lang in ("en", "zh") else _detect_system_lang())
        self._theme_mode = str(self.settings.value("ui/theme", "auto"))
        if self._theme_mode not in ("auto", "light", "dark"):
            self._theme_mode = "auto"

        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()
        self._load_settings()
        self._apply_current_theme()
        if hasattr(self.app.styleHints(), "colorSchemeChanged"):
            self.app.styleHints().colorSchemeChanged.connect(self._on_system_scheme_changed)
        self.retranslate()
        self._update_mode_widgets()
        self._update_ui_state()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(120)
        self._poll_timer.timeout.connect(self._on_poll)
        self._poll_timer.start()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        self.setAcceptDrops(True)
        self.resize(1260, 820)
        self.setMinimumSize(940, 620)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        self.btn_open = QtWidgets.QPushButton()
        self.btn_save = QtWidgets.QPushButton()
        self.btn_open_folder = QtWidgets.QPushButton()
        self.lbl_video = QtWidgets.QLabel()
        self.lbl_video.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_theme = QtWidgets.QLabel()
        self.cmb_theme = QtWidgets.QComboBox()
        for key in ("auto", "light", "dark"):
            self.cmb_theme.addItem(key, key)
        idx_t = self.cmb_theme.findData(self._theme_mode)
        if idx_t >= 0:
            self.cmb_theme.setCurrentIndex(idx_t)
        self.btn_lang = QtWidgets.QPushButton()

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_open_folder)
        top.addSpacing(8)
        top.addWidget(self.lbl_video, 1)
        top.addWidget(self.lbl_theme)
        top.addWidget(self.cmb_theme)
        top.addWidget(self.btn_lang)
        root.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        self._build_left_pane(splitter)
        self._build_right_pane(splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _build_left_pane(self, splitter: QtWidgets.QSplitter) -> None:
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 4, 0)
        left_lay.setSpacing(5)

        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(4)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)

        def make_tb(checkable: bool = False) -> QtWidgets.QToolButton:
            b = QtWidgets.QToolButton()
            b.setCheckable(checkable)
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            return b

        self.tb_pan = make_tb(True)
        self.tb_pan.setChecked(True)
        self.tb_zoom_in = make_tb(False)
        self.tb_zoom_out = make_tb(False)
        self.tb_reset = make_tb(False)
        self.tb_position = make_tb(True)
        self.tb_mask = make_tb(True)
        self.tb_width = make_tb(True)
        self.tb_delete = make_tb(False)
        self.tb_info = make_tb(False)

        for b in (self.tb_pan, self.tb_position, self.tb_mask, self.tb_width):
            self.mode_group.addButton(b)

        def vsep() -> QtWidgets.QFrame:
            f = QtWidgets.QFrame()
            f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            return f

        tb.addWidget(self.tb_pan)
        tb.addWidget(self.tb_zoom_in)
        tb.addWidget(self.tb_zoom_out)
        tb.addWidget(self.tb_reset)
        tb.addWidget(vsep())
        tb.addWidget(self.tb_position)
        tb.addWidget(self.tb_mask)
        tb.addWidget(self.tb_width)
        tb.addWidget(self.tb_delete)
        tb.addStretch(1)
        tb.addWidget(self.tb_info)
        left_lay.addLayout(tb)

        self.preview = PreviewView()
        left_lay.addWidget(self.preview, 1)

        tl = QtWidgets.QHBoxLayout()
        tl.setSpacing(6)
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setAutoRepeat(True)
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setAutoRepeat(True)
        self.lbl_frame = QtWidgets.QLabel()
        self.spin_frame = QtWidgets.QSpinBox()
        self.spin_frame.setRange(0, 0)
        tl.addWidget(self.btn_prev)
        tl.addWidget(self.slider_frame, 1)
        tl.addWidget(self.btn_next)
        tl.addWidget(self.lbl_frame)
        tl.addWidget(self.spin_frame)
        left_lay.addLayout(tl)

        self.ribbon = RibbonView()
        left_lay.addWidget(self.ribbon)

        st_row = QtWidgets.QHBoxLayout()
        self.lbl_hover = QtWidgets.QLabel()
        self.lbl_hover.setMinimumWidth(240)
        self.lbl_hover.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        st_row.addWidget(self.lbl_hover)
        st_row.addWidget(self.lbl_status, 1)
        left_lay.addLayout(st_row)
        splitter.addWidget(left)

    def _build_right_pane(self, splitter: QtWidgets.QSplitter) -> None:
        right = QtWidgets.QWidget()
        right.setMinimumWidth(380)
        right.setMaximumWidth(480)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(4, 0, 0, 0)
        right_lay.setSpacing(6)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.param_panel = QtWidgets.QWidget()
        p_lay = QtWidgets.QVBoxLayout(self.param_panel)
        p_lay.setContentsMargins(0, 0, 4, 0)
        p_lay.setSpacing(6)

        self.decoder_panel = DecoderPanel()
        p_lay.addWidget(self.decoder_panel)

        self.grp_scan = QtWidgets.QGroupBox()
        scan_lay = QtWidgets.QVBoxLayout(self.grp_scan)
        scan_lay.setSpacing(6)
        scan_top = QtWidgets.QFormLayout()
        scan_top.setSpacing(5)
        self.lbl_pos = QtWidgets.QLabel()
        self.spin_pos = QtWidgets.QSpinBox()
        self.spin_pos.setRange(-1, 99999)
        self.spin_pos.setValue(-1)
        scan_top.addRow(self.lbl_pos, self.spin_pos)

        self.lbl_mode = QtWidgets.QLabel()
        mode_row = QtWidgets.QHBoxLayout()
        self.rad_manual = QtWidgets.QRadioButton()
        self.rad_adaptive = QtWidgets.QRadioButton()
        self.rad_manual.setChecked(True)
        mode_row.addWidget(self.rad_manual)
        mode_row.addWidget(self.rad_adaptive)
        mode_row.addStretch(1)
        scan_top.addRow(self.lbl_mode, mode_row)
        scan_lay.addLayout(scan_top)

        self.box_manual = QtWidgets.QWidget()
        man_form = QtWidgets.QFormLayout(self.box_manual)
        man_form.setContentsMargins(0, 0, 0, 0)
        man_form.setSpacing(5)
        self.lbl_width = QtWidgets.QLabel()
        self.spin_width = QtWidgets.QDoubleSpinBox()
        self.spin_width.setDecimals(2)
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setRange(0.0, 99999.0)
        self.spin_width.setValue(5.0)
        man_form.addRow(self.lbl_width, self.spin_width)

        self.lbl_dir = QtWidgets.QLabel()
        self.cmb_dir = QtWidgets.QComboBox()
        self.cmb_dir.addItem("", 1)
        self.cmb_dir.addItem("", 0)
        man_form.addRow(self.lbl_dir, self.cmb_dir)
        scan_lay.addWidget(self.box_manual)

        self.box_adaptive = QtWidgets.QWidget()
        adp_form = QtWidgets.QFormLayout(self.box_adaptive)
        adp_form.setContentsMargins(0, 0, 0, 0)
        adp_form.setSpacing(5)
        self.lbl_start_frame = QtWidgets.QLabel()
        self.spin_start_frame = QtWidgets.QSpinBox()
        self.spin_start_frame.setRange(0, 9999999)
        self.spin_start_frame.setMinimumWidth(68)
        self.btn_use_cur_frame = QtWidgets.QPushButton()
        start_row = QtWidgets.QHBoxLayout()
        start_row.setContentsMargins(0, 0, 0, 0)
        start_row.setSpacing(4)
        start_row.addWidget(self.spin_start_frame, 1)
        start_row.addWidget(self.btn_use_cur_frame)
        adp_form.addRow(self.lbl_start_frame, start_row)
        self.lbl_sens = QtWidgets.QLabel()
        self.spin_sens = QtWidgets.QSpinBox()
        self.spin_sens.setRange(1, 1000)
        self.spin_sens.setValue(10)
        adp_form.addRow(self.lbl_sens, self.spin_sens)
        self.btn_calc_move = QtWidgets.QPushButton()
        self.btn_calc_move.setObjectName("accent")
        adp_form.addRow(self.btn_calc_move)
        scan_lay.addWidget(self.box_adaptive)

        mask_hdr = QtWidgets.QHBoxLayout()
        self.chk_mask = QtWidgets.QCheckBox()
        self.lbl_mask_none = QtWidgets.QLabel()
        self.lbl_mask_none.setStyleSheet("color: gray; font-size: 11px;")
        mask_hdr.addWidget(self.chk_mask)
        mask_hdr.addStretch(1)
        mask_hdr.addWidget(self.lbl_mask_none)
        scan_lay.addLayout(mask_hdr)

        self.box_mask_coords = QtWidgets.QWidget()
        mg = QtWidgets.QGridLayout(self.box_mask_coords)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setSpacing(4)
        self.spin_yt = QtWidgets.QSpinBox()
        self.spin_yb = QtWidgets.QSpinBox()
        self.spin_xl = QtWidgets.QSpinBox()
        self.spin_xr = QtWidgets.QSpinBox()
        for sp, pfx in ((self.spin_yt, "y1: "), (self.spin_yb, "y2: "),
                        (self.spin_xl, "x1: "), (self.spin_xr, "x2: ")):
            sp.setRange(0, 99999)
            sp.setPrefix(pfx)
        mg.addWidget(self.spin_yt, 0, 0)
        mg.addWidget(self.spin_yb, 0, 1)
        mg.addWidget(self.spin_xl, 1, 0)
        mg.addWidget(self.spin_xr, 1, 1)
        self.box_mask_coords.setVisible(False)
        scan_lay.addWidget(self.box_mask_coords)
        p_lay.addWidget(self.grp_scan)

        self._build_extra_groups(p_lay)
        self._no_wheel_filter = _NoWheelFilter(self)
        for cls in (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox):
            for w in self.param_panel.findChildren(cls):
                w.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                w.installEventFilter(self._no_wheel_filter)
        p_lay.addStretch(1)
        scroll.setWidget(self.param_panel)
        right_lay.addWidget(scroll, 1)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        right_lay.addWidget(self.progress_bar)

        self.lbl_prog = QtWidgets.QLabel()
        self.lbl_prog.setWordWrap(True)
        right_lay.addWidget(self.lbl_prog)

        act_row = QtWidgets.QHBoxLayout()
        act_row.setSpacing(6)
        self.btn_start = QtWidgets.QPushButton()
        self.btn_start.setObjectName("primary")
        self.btn_stop = QtWidgets.QPushButton()
        act_row.addWidget(self.btn_start, 2)
        act_row.addWidget(self.btn_stop, 1)
        right_lay.addLayout(act_row)
        splitter.addWidget(right)

    def _build_extra_groups(self, p_lay: QtWidgets.QVBoxLayout) -> None:
        self.grp_segments = QtWidgets.QGroupBox()
        seg_lay = QtWidgets.QVBoxLayout(self.grp_segments)
        seg_lay.setContentsMargins(6, 6, 6, 6)
        self.seg_table = SegmentTable()
        seg_lay.addWidget(self.seg_table)
        p_lay.addWidget(self.grp_segments)

        self.grp_stab = QtWidgets.QGroupBox()
        stab_form = QtWidgets.QFormLayout(self.grp_stab)
        stab_form.setSpacing(5)
        self.chk_stab = QtWidgets.QCheckBox()
        stab_form.addRow(self.chk_stab)

        self.lbl_stab_hint = QtWidgets.QLabel()
        self.lbl_stab_hint.setWordWrap(True)
        self.lbl_stab_hint.setStyleSheet("color: #7d8590; font-size: 11px;")
        stab_form.addRow(self.lbl_stab_hint)

        self.lbl_det = QtWidgets.QLabel()
        self.cmb_det = QtWidgets.QComboBox()
        self.cmb_det.addItem("ORB", "orb")
        self.cmb_det.addItem("SIFT", "sift")
        stab_form.addRow(self.lbl_det, self.cmb_det)

        self.lbl_model = QtWidgets.QLabel()
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItem("", "translation")
        self.cmb_model.addItem("", "similarity")
        self.cmb_model.addItem("", "homography")
        self.cmb_model.setCurrentIndex(1)
        stab_form.addRow(self.lbl_model, self.cmb_model)

        self.lbl_workers = QtWidgets.QLabel()
        self.spin_workers = QtWidgets.QSpinBox()
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setValue(0)
        stab_form.addRow(self.lbl_workers, self.spin_workers)
        p_lay.addWidget(self.grp_stab)

        self.grp_output = QtWidgets.QGroupBox()
        out_form = QtWidgets.QFormLayout(self.grp_output)
        out_form.setSpacing(5)
        self.lbl_split = QtWidgets.QLabel()
        self.spin_split = QtWidgets.QSpinBox()
        self.spin_split.setRange(100, 65000)
        self.spin_split.setSingleStep(1000)
        self.spin_split.setValue(20000)
        out_form.addRow(self.lbl_split, self.spin_split)

        self.lbl_interp = QtWidgets.QLabel()
        self.cmb_interp = QtWidgets.QComboBox()
        self.cmb_interp.addItem("", "cubic")
        self.cmb_interp.addItem("", "lanczos")
        self.cmb_interp.addItem("", "linear")
        out_form.addRow(self.lbl_interp, self.cmb_interp)

        self.lbl_feather = QtWidgets.QLabel()
        self.spin_feather = QtWidgets.QSpinBox()
        self.spin_feather.setRange(0, 128)
        self.spin_feather.setValue(0)
        out_form.addRow(self.lbl_feather, self.spin_feather)
        p_lay.addWidget(self.grp_output)

    # ------------------------------------------------------------------ signals & shortcuts
    def _wire_signals(self) -> None:
        self.btn_open.clicked.connect(self._on_open_clicked)
        self.btn_save.clicked.connect(self._on_save_dir_clicked)
        self.btn_open_folder.clicked.connect(self._on_open_folder_clicked)
        self.cmb_theme.currentIndexChanged.connect(self._on_theme_changed)
        self.btn_lang.clicked.connect(self._on_lang_toggle)

        self.tb_pan.clicked.connect(lambda: self.preview.set_mode("pan"))
        self.tb_position.clicked.connect(lambda: self.preview.set_mode("position"))
        self.tb_mask.clicked.connect(lambda: self.preview.set_mode("mask"))
        self.tb_width.clicked.connect(lambda: self.preview.set_mode("width"))
        self.tb_zoom_in.clicked.connect(lambda: self.preview.zoom(1.25))
        self.tb_zoom_out.clicked.connect(lambda: self.preview.zoom(1.0 / 1.25))
        self.tb_reset.clicked.connect(self.preview.reset_view)
        self.tb_delete.clicked.connect(self._on_delete_clicked)
        self.tb_info.clicked.connect(self._show_info_dialog)

        self.preview.modeChanged.connect(self._on_preview_mode_changed)
        self.preview.positionChanged.connect(self._on_preview_pos_changed)
        self.preview.widthChanged.connect(self._on_preview_width_changed)
        self.preview.maskChanged.connect(self._on_preview_mask_changed)
        self.preview.hoverInfo.connect(self._on_hover_info)

        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_next.clicked.connect(lambda: self._step_frame(1))
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        self.spin_frame.valueChanged.connect(self._on_frame_changed)

        self.decoder_panel.decoderChanged.connect(self._on_decoder_changed)
        self.spin_pos.valueChanged.connect(self._on_spin_pos_changed)
        self.rad_manual.toggled.connect(self._on_mode_toggled)
        self.spin_width.valueChanged.connect(self._on_spin_width_changed)
        self.chk_mask.toggled.connect(self._on_chk_mask_toggled)
        for sp in (self.spin_yt, self.spin_yb, self.spin_xl, self.spin_xr):
            sp.valueChanged.connect(self._on_spin_mask_changed)

        self.chk_stab.toggled.connect(self._on_stab_changed)
        self.cmb_det.currentIndexChanged.connect(self._on_stab_changed)
        self.cmb_model.currentIndexChanged.connect(self._on_stab_changed)

        self.btn_use_cur_frame.clicked.connect(self._on_use_cur_frame)
        self.btn_calc_move.clicked.connect(self._on_calc_move)
        self.seg_table.segmentsChanged.connect(self._on_segments_changed)
        self.seg_table.jumpToFrame.connect(self._set_frame)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)


    def _setup_shortcuts(self) -> None:
        sc_open = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        sc_open.activated.connect(self._on_open_clicked)
        self.app.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if ev.type() == QtCore.QEvent.Type.KeyPress and self.isActiveWindow() and not self._is_busy():
            ke = ev  # type: QtGui.QKeyEvent
            if ke.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
                fw = self.focusWidget()
                if not isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
                    step = 10 if (ke.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier) else 1
                    self._step_frame(-step if ke.key() == QtCore.Qt.Key.Key_Left else step)
                    return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ theme & i18n
    def _apply_current_theme(self) -> None:
        dark = apply_theme(self.app, self._theme_mode)
        canvas_bg = (28, 30, 34) if dark else (38, 40, 46)
        plot_bg = (28, 30, 33) if dark else (255, 255, 255)
        self.preview.setBackground(canvas_bg)
        self.ribbon.setBackground(canvas_bg)
        self.seg_table.plot.setBackground(plot_bg)

    def _on_system_scheme_changed(self, _scheme=None) -> None:
        if self._theme_mode == "auto":
            self._apply_current_theme()

    def _on_theme_changed(self, _idx: int) -> None:
        mode = self.cmb_theme.currentData()
        if mode in ("auto", "light", "dark"):
            self._theme_mode = mode
            self.settings.setValue("ui/theme", mode)
            self.settings.sync()
            self._apply_current_theme()

    def _on_lang_toggle(self) -> None:
        new_lang = "zh" if language() == "en" else "en"
        set_language(new_lang)
        self.settings.setValue("ui/lang", new_lang)
        self.settings.sync()
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("app_title", v=__version__))
        self.btn_open.setText(tr("open_video"))
        self.btn_save.setText(tr("save_dir"))
        self.btn_open_folder.setText(tr("open_folder"))
        self._update_save_tooltip()
        self._update_video_label()

        self.lbl_theme.setText(tr("theme"))
        self.cmb_theme.blockSignals(True)
        for i, key in enumerate(("theme_auto", "theme_light", "theme_dark")):
            self.cmb_theme.setItemText(i, tr(key))
        self.cmb_theme.blockSignals(False)
        self.btn_lang.setText(tr("lang_btn"))

        self.tb_pan.setText(f"✋ {tr('tb_pan')}")
        self.tb_zoom_in.setText(f"➕ {tr('tb_zoom_in')}")
        self.tb_zoom_out.setText(f"➖ {tr('tb_zoom_out')}")
        self.tb_reset.setText(f"🔄 {tr('tb_reset')}")
        self.tb_position.setText(f"📍 {tr('tb_position')}")
        self.tb_mask.setText(f"🔲 {tr('tb_mask')}")
        self.tb_width.setText(f"📏 {tr('tb_width')}")
        self.tb_delete.setText(f"🗑 {tr('tb_delete')}")
        self.tb_info.setText(f"ℹ {tr('tb_info')}")

        self.tb_pan.setToolTip(tr("tip_pan"))
        self.tb_zoom_in.setToolTip(tr("tip_zoom_in"))
        self.tb_zoom_out.setToolTip(tr("tip_zoom_out"))
        self.tb_reset.setToolTip(tr("tip_reset"))
        self.tb_position.setToolTip(tr("tip_position"))
        self.tb_mask.setToolTip(tr("tip_mask"))
        self.tb_width.setToolTip(tr("tip_width"))
        self.tb_delete.setToolTip(tr("tip_delete"))
        self.tb_info.setToolTip(tr("tip_info"))

        self.lbl_frame.setText(tr("frame"))
        self.ribbon.setToolTip(tr("ribbon_tip"))
        self.ribbon.set_empty_text(tr("ribbon_empty"))

        self.decoder_panel.retranslate()
        self.grp_scan.setTitle(tr("grp_scan"))
        self.lbl_pos.setText(tr("position"))
        self.spin_pos.setSpecialValueText(tr("not_set"))
        self.lbl_mode.setText(tr("mode_lbl"))
        self.rad_manual.setText(tr("manual"))
        self.rad_adaptive.setText(tr("adaptive"))
        self.lbl_width.setText(tr("width"))
        self.spin_width.setSpecialValueText(tr("not_set"))
        self.lbl_dir.setText(tr("direction"))
        self.cmb_dir.setItemText(0, tr("dir_l2r"))
        self.cmb_dir.setItemText(1, tr("dir_r2l"))
        self.lbl_start_frame.setText(tr("start_frame"))
        self.btn_use_cur_frame.setText(tr("use_cur_frame"))
        self.btn_use_cur_frame.setToolTip(tr("use_cur_frame_tip"))
        self.lbl_sens.setText(tr("sensitivity"))
        self.btn_calc_move.setText(tr("calc_move"))
        self.chk_mask.setText(tr("mask_lbl"))
        self.chk_mask.setToolTip(tr("mask_tip"))
        self.box_mask_coords.setToolTip(tr("mask_tip"))
        self.lbl_mask_none.setText("" if self.chk_mask.isChecked() else tr("mask_none"))

        self.grp_stab.setTitle(tr("grp_stab"))
        self.chk_stab.setText(tr("stab_enable"))
        self.lbl_stab_hint.setText(tr("stab_mask_hint"))
        self.lbl_det.setText(tr("detector"))
        self.lbl_model.setText(tr("model"))
        self.cmb_model.setItemText(0, tr("model_translation"))
        self.cmb_model.setItemText(1, tr("model_similarity"))
        self.cmb_model.setItemText(2, tr("model_homography"))
        self.lbl_workers.setText(tr("workers"))

        self.grp_segments.setTitle(tr("grp_segments"))
        self.seg_table.retranslate()

        self.grp_output.setTitle(tr("grp_output"))
        self.lbl_split.setText(tr("split_width"))
        self.lbl_interp.setText(tr("interp"))
        self.cmb_interp.setItemText(0, tr("interp_cubic"))
        self.cmb_interp.setItemText(1, tr("interp_lanczos"))
        self.cmb_interp.setItemText(2, tr("interp_linear"))
        self.lbl_feather.setText(tr("feather"))

        self.btn_start.setText(tr("start"))
        self.btn_stop.setText(tr("stop"))
        if not self._is_busy() and not self.lbl_prog.text():
            self.lbl_prog.setText(tr("st_ready"))

    def _update_video_label(self) -> None:
        if self._video_info is None:
            self.lbl_video.setText(tr("no_video"))
            return
        vi = self._video_info
        name = os.path.basename(vi.path)
        info_str = tr("video_info", w=vi.width, h=vi.height, fps=vi.fps, n=vi.total_frames,
                      codec=vi.fourcc or "?")
        self.lbl_video.setText(f"{name}  ({info_str})")
        self.lbl_video.setToolTip(vi.path)

    def _update_save_tooltip(self) -> None:
        p = self._save_root or "-"
        self.btn_save.setToolTip(tr("save_dir_tip", p=p))

    def _show_info_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("info_title"))
        dlg.resize(640, 420)
        lay = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        style = (
            "<style>"
            "body { font-size: 13px; line-height: 1.45; }"
            "table { border-collapse: collapse; width: 100%; }"
            "th, td { padding: 6px 8px; text-align: left; }"
            "li { margin-bottom: 4px; }"
            "</style>"
        )
        browser.setHtml(style + tr("info_html"))
        lay.addWidget(browser, 1)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    def _on_delete_clicked(self) -> None:
        if self.preview.delete_current():
            return
        menu = QtWidgets.QMenu(self)
        act_pos = menu.addAction(tr("del_position"))
        act_pos.setEnabled(self.preview.position is not None)
        act_mask = menu.addAction(tr("del_mask"))
        act_mask.setEnabled(self.preview.roi is not None)
        act_w = menu.addAction(tr("del_width"))
        act_w.setEnabled(not self._is_adaptive() and self.preview.width is not None)
        chosen = menu.exec(self.tb_delete.mapToGlobal(QtCore.QPoint(0, self.tb_delete.height())))
        if chosen is act_pos:
            self.preview.set_position(None, emit=True)
        elif chosen is act_mask:
            self.preview.set_mask(None, emit=True)
        elif chosen is act_w:
            self.preview.set_width(None, emit=True)

    # ------------------------------------------------------------------ drag & drop / file open
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if not self._is_busy() and ev.mimeData().hasUrls():
            for u in ev.mimeData().urls():
                if u.isLocalFile() and os.path.splitext(u.toLocalFile())[1].lower() in VIDEO_EXTS:
                    ev.acceptProposedAction()
                    return
        ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        if self._is_busy():
            return
        for u in ev.mimeData().urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if os.path.isfile(p):
                    self.load_video(p)
                    break

    def _on_open_clicked(self) -> None:
        if self._is_busy():
            return
        start_dir = str(self.settings.value("paths/last_open_dir", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("dlg_open"), start_dir, VIDEO_FILTER)
        if path:
            self.load_video(path)

    def _on_save_dir_clicked(self) -> None:
        start_dir = self._save_root or str(self.settings.value("paths/last_open_dir", ""))
        d = QtWidgets.QFileDialog.getExistingDirectory(self, tr("dlg_save"), start_dir)
        if d:
            self._save_root = d
            self._update_save_tooltip()
            self._update_ui_state()

    def _on_open_folder_clicked(self) -> None:
        target = self._last_out_dir
        if not target or not os.path.isdir(target):
            if self._video_info and self._save_root:
                stem = os.path.splitext(os.path.basename(self._video_info.path))[0]
                cand = os.path.join(self._save_root, stem)
                target = cand if os.path.isdir(cand) else self._save_root
            else:
                target = self._save_root
        if target and os.path.isdir(target):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))

    def load_video(self, path: str) -> None:
        try:
            info = read_info(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, tr("dlg_error"), str(e))
            return
        self._video_info = info
        self._save_root = os.path.dirname(os.path.abspath(path))
        self._last_out_dir = ""
        self.settings.setValue("paths/last_open_dir", self._save_root)
        self._update_save_tooltip()
        self._update_video_label()

        max_f = max(0, info.total_frames - 1)
        self._syncing = True
        try:
            self.slider_frame.setRange(0, max_f)
            self.spin_frame.setRange(0, max_f)
            self.slider_frame.setValue(0)
            self.spin_frame.setValue(0)
            self.spin_pos.setRange(-1, max(0, info.width - 1))
            self.spin_width.setRange(0.0, float(max(1, info.width)))
            self.spin_start_frame.setRange(0, max(0, info.total_frames - 3))
            self.spin_start_frame.setValue(0)
            self.spin_yt.setRange(0, max(0, info.height - 1))
            self.spin_yb.setRange(1, max(1, info.height))
            self.spin_xl.setRange(0, max(0, info.width - 1))
            self.spin_xr.setRange(1, max(1, info.width))
        finally:
            self._syncing = False

        self.preview.set_frame_size(info.width, info.height)
        if self.preview.position is None:
            self.preview.set_position(info.width // 2, emit=True)
        self.seg_table.set_total_frames(info.total_frames)
        self.seg_table.set_segments([])
        self.seg_table.set_cursor(0)
        self.preview.set_adaptive(self._is_adaptive(), None)
        self.ribbon.clear()
        self._preview_worker.invalidate()
        self.decoder_panel.set_video(path)
        self._need_reset_view = True
        self._request_preview(0)
        self.lbl_prog.setText(tr("st_ready"))
        self.progress_bar.setValue(0)
        self._update_ui_state()

    # ------------------------------------------------------------------ timeline & preview
    def _step_frame(self, delta: int) -> None:
        if self._video_info is None or self._is_busy():
            return
        target = max(0, min(self.slider_frame.maximum(), self.slider_frame.value() + delta))
        self._set_frame(target)

    def _set_frame(self, frame: int) -> None:
        if self._video_info is None or self._is_busy():
            return
        frame = max(0, min(self.slider_frame.maximum(), int(frame)))
        self._on_frame_changed(frame)

    def _on_frame_changed(self, frame: int) -> None:
        if self._syncing or self._video_info is None:
            return
        self._syncing = True
        try:
            self.slider_frame.setValue(frame)
            self.spin_frame.setValue(frame)
        finally:
            self._syncing = False
        self.seg_table.set_cursor(frame)
        if self._is_adaptive():
            self.preview.set_adaptive(True, self.seg_table.width_at(frame))
        if not self._is_busy():
            self._request_preview(frame)

    def _current_stab_cfg(self) -> StabilizerConfig:
        return StabilizerConfig(
            enabled=self.chk_stab.isChecked(),
            detector=str(self.cmb_det.currentData() or "orb"),
            model=str(self.cmb_model.currentData() or "similarity"),
            workers=int(self.spin_workers.value()),
        )

    def _current_mask(self) -> Optional[Mask]:
        if not self.chk_mask.isChecked():
            return None
        return (int(self.spin_yt.value()), int(self.spin_yb.value()),
                int(self.spin_xl.value()), int(self.spin_xr.value()))

    def _request_preview(self, frame: Optional[int] = None) -> None:
        if self._video_info is None or self._is_busy():
            return
        idx = self.slider_frame.value() if frame is None else int(frame)
        stab = self._current_stab_cfg() if self.chk_stab.isChecked() else None
        self._preview_worker.request(PreviewRequest(
            path=self._video_info.path,
            frame=idx,
            decoder=self.decoder_panel.decoder(),
            stab=stab,
            mask=self._current_mask(),
        ))

    def _on_decoder_changed(self, _dec) -> None:
        self._preview_worker.invalidate()
        self._request_preview()

    def _on_stab_changed(self, *_args) -> None:
        self._request_preview()



    # ------------------------------------------------------------------ two-way overlay sync
    def _is_adaptive(self) -> bool:
        return self.rad_adaptive.isChecked()

    def _on_mode_toggled(self, *_args) -> None:
        self._update_mode_widgets()
        self._update_ui_state()

    def _update_mode_widgets(self) -> None:
        adp = self._is_adaptive()
        self.box_manual.setVisible(not adp)
        self.box_adaptive.setVisible(adp)
        self.grp_segments.setVisible(adp)
        self.tb_width.setEnabled(not adp and not self._is_busy())
        cur_w = self.seg_table.width_at(self.slider_frame.value()) if adp else None
        self.preview.set_adaptive(adp, cur_w)

    def _on_preview_mode_changed(self, mode: str) -> None:
        mapping = {
            "pan": self.tb_pan,
            "position": self.tb_position,
            "mask": self.tb_mask,
            "width": self.tb_width,
        }
        btn = mapping.get(mode)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def _on_preview_pos_changed(self, pos: Optional[int]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin_pos.setValue(-1 if pos is None else int(pos))
        finally:
            self._syncing = False
        self._update_ui_state()

    def _on_spin_pos_changed(self, val: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.preview.set_position(None if val < 0 else val, emit=False)
        finally:
            self._syncing = False
        self._update_ui_state()

    def _on_preview_width_changed(self, w: Optional[float]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin_width.setValue(0.0 if w is None else float(w))
        finally:
            self._syncing = False
        self._update_ui_state()

    def _on_spin_width_changed(self, val: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.preview.set_width(None if val <= 0.0 else val, emit=False)
        finally:
            self._syncing = False
        self._update_ui_state()

    def _on_preview_mask_changed(self, mask: Optional[Mask]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if mask is None:
                self.chk_mask.setChecked(False)
                self.box_mask_coords.setVisible(False)
                self.lbl_mask_none.setText(tr("mask_none"))
            else:
                yt, yb, xl, xr = mask
                self.chk_mask.setChecked(True)
                self.box_mask_coords.setVisible(True)
                self.lbl_mask_none.setText("")
                self.spin_yt.setValue(yt)
                self.spin_yb.setValue(yb)
                self.spin_xl.setValue(xl)
                self.spin_xr.setValue(xr)
        finally:
            self._syncing = False
        if self.chk_stab.isChecked():
            self._request_preview()

    def _on_chk_mask_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        self.box_mask_coords.setVisible(checked)
        self.lbl_mask_none.setText("" if checked else tr("mask_none"))
        if not checked:
            self._syncing = True
            try:
                self.preview.set_mask(None, emit=False)
            finally:
                self._syncing = False
        else:
            w = self._video_info.width if self._video_info else 900
            h = self._video_info.height if self._video_info else 600
            m = (h // 3, 2 * h // 3, w // 3, 2 * w // 3)
            self._syncing = True
            try:
                self.spin_yt.setValue(m[0])
                self.spin_yb.setValue(m[1])
                self.spin_xl.setValue(m[2])
                self.spin_xr.setValue(m[3])
                self.preview.set_mask(m, emit=False)
            finally:
                self._syncing = False
        if self.chk_stab.isChecked():
            self._request_preview()

    def _on_spin_mask_changed(self, _val: int) -> None:
        if self._syncing or not self.chk_mask.isChecked():
            return
        yt, yb = self.spin_yt.value(), self.spin_yb.value()
        xl, xr = self.spin_xl.value(), self.spin_xr.value()
        max_h = self._video_info.height if self._video_info else 99999
        max_w = self._video_info.width if self._video_info else 99999
        self._syncing = True
        try:
            if yt >= yb:
                yb = min(max_h, yt + 1)
                yt = min(yt, yb - 1)
                self.spin_yt.setValue(yt)
                self.spin_yb.setValue(yb)
            if xl >= xr:
                xr = min(max_w, xl + 1)
                xl = min(xl, xr - 1)
                self.spin_xl.setValue(xl)
                self.spin_xr.setValue(xr)
            self.preview.set_mask((yt, yb, xl, xr), emit=False)
        finally:
            self._syncing = False
        if self.chk_stab.isChecked():
            self._request_preview()

    def _on_segments_changed(self) -> None:
        if self._is_adaptive():
            self.preview.set_adaptive(True, self.seg_table.width_at(self.slider_frame.value()))
        self._update_ui_state()

    def _on_hover_info(self, info: str) -> None:
        if not info:
            self.lbl_hover.setText("")
            return
        parts = info.split(",")
        if len(parts) == 5:
            self.lbl_hover.setText(tr("hover", x=parts[0], y=parts[1], r=parts[2], g=parts[3], b=parts[4]))

    # ------------------------------------------------------------------ state & actions
    def _is_busy(self) -> bool:
        return self._job is not None or self._motion_task is not None

    def _update_ui_state(self) -> None:
        busy = self._is_busy()
        has_vid = self._video_info is not None
        adp = self._is_adaptive()

        self.btn_open.setEnabled(not busy)
        self.btn_save.setEnabled(not busy)
        self.btn_open_folder.setEnabled(bool(self._save_root or self._last_out_dir))
        self.param_panel.setEnabled(not busy)

        self.tb_position.setEnabled(not busy and has_vid)
        self.tb_mask.setEnabled(not busy and has_vid)
        self.tb_width.setEnabled(not busy and has_vid and not adp)
        self.tb_delete.setEnabled(not busy and has_vid)
        if busy and self.preview.mode != "pan":
            self.preview.set_mode("pan")

        self.btn_prev.setEnabled(not busy and has_vid)
        self.btn_next.setEnabled(not busy and has_vid)
        self.slider_frame.setEnabled(not busy and has_vid)
        self.spin_frame.setEnabled(not busy and has_vid)
        self.btn_use_cur_frame.setEnabled(not busy and has_vid)
        self.btn_calc_move.setEnabled(not busy and has_vid)

        can_start = (
            not busy
            and has_vid
            and self.preview.position is not None
            and (
                (adp and len(self.seg_table.segments(strict=False)) > 0)
                or (not adp and self.preview.width is not None and self.preview.width > 0)
            )
        )
        self.btn_start.setEnabled(can_start)
        self.btn_stop.setEnabled(busy)

    def _on_use_cur_frame(self) -> None:
        if self._video_info is None or self._is_busy():
            return
        cur = int(self.slider_frame.value())
        clamped = max(self.spin_start_frame.minimum(), min(self.spin_start_frame.maximum(), cur))
        self.spin_start_frame.setValue(clamped)

    def _on_calc_move(self) -> None:
        if self._video_info is None or self._is_busy():
            return
        vi = self._video_info
        start_f = int(self.spin_start_frame.value())
        samples = int(self.spin_sens.value())
        mask = self._current_mask()
        stab = self._current_stab_cfg()
        dec = self.decoder_panel.decoder()

        def run_motion(cancel_ev, prog_cb):
            return estimate_segments(
                vi.path, vi.total_frames, start_f, samples,
                mask=mask, stab_cfg=stab, decoder=dec,
                cancel=cancel_ev, progress=prog_cb,
            )

        self.progress_bar.setValue(0)
        self.lbl_prog.setText(tr("st_motion", i=0, n=samples))
        self._motion_task = BackgroundTask(run_motion).start()
        self._update_ui_state()

    def _resolve_output_conflict(self, cfg: ScanConfig) -> bool:
        out_dir = cfg.output_dir()
        existing = find_existing_outputs(out_dir, cfg.video_stem)
        if not existing:
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle(tr("dlg_conflict_title"))
        box.setText(tr("dlg_conflict", n=len(existing), dir=out_dir))
        btn_del = box.addButton(tr("dlg_conflict_delete"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        btn_new = box.addButton(tr("dlg_conflict_new"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(tr("dlg_cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_new)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_del:
            for f in existing:
                try:
                    os.remove(f)
                except OSError:
                    pass
            return True
        if clicked is btn_new:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cfg.out_dir = os.path.join(cfg.save_root, f"{cfg.video_stem}_{ts}")
            return True
        return False

    def _on_start(self) -> None:
        if self._is_busy():
            return
        if self._video_info is None:
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), tr("err_no_video"))
            return
        if self.preview.position is None:
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), tr("err_no_position"))
            return
        adp = self._is_adaptive()
        if not adp and (self.preview.width is None or self.preview.width <= 0):
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), tr("err_no_width"))
            return
        try:
            segs = self.seg_table.segments(strict=True) if adp else []
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), str(e))
            return
        if adp and not segs:
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), tr("err_no_segments"))
            return

        save_root = self._save_root or os.path.dirname(os.path.abspath(self._video_info.path))
        cfg = ScanConfig(
            video_path=self._video_info.path,
            save_root=save_root,
            position=int(self.preview.position),
            mode="adaptive" if adp else "manual",
            width=float(self.preview.width or 5.0),
            direction=int(self.cmb_dir.currentData()),
            segments=segs,
            split_width=int(self.spin_split.value()),
            mask=self._current_mask(),
            stabilizer=self._current_stab_cfg(),
            decoder=self.decoder_panel.decoder(),
            interpolation=str(self.cmb_interp.currentData() or "cubic"),
            feather=int(self.spin_feather.value()),
        )
        try:
            cfg.validate(self._video_info.width, self._video_info.height)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, tr("dlg_error"), str(e))
            return

        if not self._resolve_output_conflict(cfg):
            return

        self._save_settings()
        self._last_out_dir = cfg.output_dir()
        self._job_direction = cfg.effective_direction()
        self._last_live_frame = -1
        self.ribbon.clear()
        self.progress_bar.setValue(0)
        self.lbl_prog.setText(tr("st_preparing"))
        self._job = ScanJob(cfg, self._video_info).start()
        self._update_ui_state()

    def _on_stop(self) -> None:
        if self._job is not None:
            self._job.cancel()
        if self._motion_task is not None:
            self._motion_task.cancel()

    # ------------------------------------------------------------------ pull polling (~8 Hz)
    def _on_poll(self) -> None:
        if not self._is_busy():
            res = self._preview_worker.take_result()
            if res is not None:
                if res.image is not None:
                    self.preview.set_image(res.image, reset_view=self._need_reset_view)
                    self._need_reset_view = False
                if res.error:
                    self.lbl_status.setText(tr("err_stab_anchor", msg=res.error))
                else:
                    self.lbl_status.setText("")

        if self._motion_task is not None:
            mt = self._motion_task
            done_i, total_n = mt.progress
            if total_n > 0:
                self.progress_bar.setValue(int(round(1000.0 * done_i / total_n)))
                self.lbl_prog.setText(tr("st_motion", i=done_i, n=total_n))
            if mt.done:
                self._motion_task = None
                if mt.error:
                    self.lbl_prog.setText(tr("st_error"))
                    QtWidgets.QMessageBox.critical(self, tr("dlg_error"), mt.error)
                elif isinstance(mt.result, MotionResult):
                    if mt.result.cancelled:
                        self.progress_bar.setValue(0)
                        self.lbl_prog.setText(tr("st_ready"))
                    else:
                        self.seg_table.set_segments(mt.result.segments)
                        warn = tr("st_motion_warn", n=len(mt.result.failed_samples)) if mt.result.failed_samples else ""
                        self.lbl_prog.setText(tr("st_motion_done", warn=warn))
                        self.progress_bar.setValue(1000)
                self._update_ui_state()

        if self._job is not None:
            self._job.state.request_preview()
            snap = self._job.state.snapshot(with_images=True)
            if snap.preview is not None and snap.preview_frame != self._last_live_frame:
                self._last_live_frame = snap.preview_frame
                self.preview.set_image(snap.preview, reset_view=False)
                self.preview.set_live_label(tr("live", f=snap.preview_frame))
            if snap.ribbon is not None and snap.ribbon.size > 0:
                self.ribbon.set_ribbon(snap.ribbon, newest_left=(self._job_direction == 1))

            if snap.stage == "preparing":
                self.lbl_prog.setText(tr("st_preparing"))
            elif snap.stage == "processing":
                pct = int(round(1000.0 * snap.frames_done / snap.total_est)) if snap.total_est > 0 else 0
                self.progress_bar.setValue(max(0, min(1000, pct)))
                self.lbl_prog.setText(tr(
                    "st_processing",
                    done=snap.frames_done,
                    total=snap.total_est or "?",
                    fps=snap.fps,
                    eta=_fmt_eta(snap.eta),
                    tiles=snap.tiles_saved,
                ))
            elif snap.stage == "finalizing":
                if snap.total_est > 0:
                    pct = int(round(1000.0 * snap.frames_done / snap.total_est))
                    self.progress_bar.setValue(max(self.progress_bar.value(), min(999, pct)))
                self.lbl_prog.setText(tr("st_finalizing"))

            if snap.finished:
                self._job.join(timeout=2.0)
                self._job = None
                self.preview.set_live_label("")
                if snap.stage == "done":
                    self.progress_bar.setValue(1000)
                    self.lbl_prog.setText(tr(
                        "st_done", tiles=len(snap.files), cols=snap.columns,
                        sec=snap.elapsed, fps=snap.fps,
                    ))
                elif snap.stage == "stopped":
                    self.lbl_prog.setText(tr("st_stopped", tiles=len(snap.files), cols=snap.columns))
                elif snap.stage == "error":
                    self.lbl_prog.setText(tr("st_error"))
                    QtWidgets.QMessageBox.critical(self, tr("dlg_error"), snap.error)
                self._update_ui_state()
                self._request_preview()

    # ------------------------------------------------------------------ QSettings & exit
    def _load_settings(self) -> None:
        s = self.settings
        geom = s.value("ui/geometry")
        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)

        kind = str(s.value("dec/kind", "software"))
        dev = int(s.value("dec/device", -1))
        self.decoder_panel.set_by_key(kind, dev)

        self._syncing = True
        try:
            mode = str(s.value("scan/mode", "manual"))
            self.rad_adaptive.setChecked(mode == "adaptive")
            self.rad_manual.setChecked(mode != "adaptive")
            w = float(s.value("scan/width", 5.0))
            self.spin_width.setValue(max(0.0, w))
            self.preview.set_width(w if w > 0 else None, emit=False)

            d = int(s.value("scan/direction", 1))
            idx_d = self.cmb_dir.findData(1 if d else 0)
            if idx_d >= 0:
                self.cmb_dir.setCurrentIndex(idx_d)
            self.spin_sens.setValue(max(1, int(s.value("scan/samples", 10))))

            self.chk_stab.setChecked(str(s.value("stab/enabled", "false")).lower() in ("true", "1"))
            idx_det = self.cmb_det.findData(str(s.value("stab/detector", "orb")))
            if idx_det >= 0:
                self.cmb_det.setCurrentIndex(idx_det)
            idx_mod = self.cmb_model.findData(str(s.value("stab/model", "similarity")))
            if idx_mod >= 0:
                self.cmb_model.setCurrentIndex(idx_mod)
            self.spin_workers.setValue(max(0, int(s.value("stab/workers", 0))))

            self.spin_split.setValue(max(100, min(65000, int(s.value("out/split_width", 20000)))))
            idx_int = self.cmb_interp.findData(str(s.value("out/interp", "cubic")))
            if idx_int >= 0:
                self.cmb_interp.setCurrentIndex(idx_int)
            self.spin_feather.setValue(max(0, int(s.value("out/feather", 0))))
        finally:
            self._syncing = False

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("ui/geometry", self.saveGeometry())
        dec = self.decoder_panel.decoder()
        s.setValue("dec/kind", dec.kind)
        s.setValue("dec/device", dec.device)
        s.setValue("scan/mode", "adaptive" if self._is_adaptive() else "manual")
        s.setValue("scan/width", float(self.spin_width.value()))
        s.setValue("scan/direction", int(self.cmb_dir.currentData()))
        s.setValue("scan/samples", int(self.spin_sens.value()))
        s.setValue("stab/enabled", self.chk_stab.isChecked())
        s.setValue("stab/detector", str(self.cmb_det.currentData() or "orb"))
        s.setValue("stab/model", str(self.cmb_model.currentData() or "similarity"))
        s.setValue("stab/workers", int(self.spin_workers.value()))
        s.setValue("out/split_width", int(self.spin_split.value()))
        s.setValue("out/interp", str(self.cmb_interp.currentData() or "cubic"))
        s.setValue("out/feather", int(self.spin_feather.value()))
        s.sync()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self._poll_timer.stop()
        if self._job is not None:
            self._job.cancel()
            self._job.join(timeout=3.0)
            self._job = None
        if self._motion_task is not None:
            self._motion_task.cancel()
            self._motion_task = None
        self._preview_worker.close()
        self._save_settings()
        super().closeEvent(ev)



