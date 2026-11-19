"""Entry point for video_scanner v0.4.0."""
from __future__ import annotations

from contextlib import suppress
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _resource_candidates(*rel_parts: str) -> list[str]:
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(meipass)
    roots.append(HERE)
    roots.append(os.path.dirname(HERE))
    return [os.path.join(r, *rel_parts) for r in roots]


def find_asset(*rel_parts: str) -> str | None:
    for p in _resource_candidates(*rel_parts):
        if os.path.isfile(p):
            return p
    return None


def _set_windows_app_id() -> None:
    if sys.platform.startswith("win"):
        with suppress(Exception):
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "divertingPan.VideoScanner.v040"
            )


def _has_pyi_splash() -> bool:
    with suppress(ModuleNotFoundError):
        import pyi_splash  # type: ignore  # noqa: F401
        return True
    return False


def _close_pyi_splash() -> None:
    with suppress(ModuleNotFoundError, Exception):
        import pyi_splash  # type: ignore

        pyi_splash.update_text("loaded...")
        pyi_splash.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    _set_windows_app_id()

    from PySide6 import QtCore, QtGui, QtWidgets

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("VideoScanner")
    app.setOrganizationName("divertingPan")

    icon_path = find_asset("assets", "icon.ico") or find_asset("assets", "icon.png")
    app_icon = QtGui.QIcon(icon_path) if icon_path else QtGui.QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    qt_splash: QtWidgets.QSplashScreen | None = None
    if not _has_pyi_splash() and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        splash_path = find_asset("assets", "startup.png") or find_asset("startup.png")
        if splash_path:
            pm = QtGui.QPixmap(splash_path)
            if not pm.isNull():
                if pm.width() > 560:
                    pm = pm.scaledToWidth(
                        560, QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                qt_splash = QtWidgets.QSplashScreen(
                    pm, QtCore.Qt.WindowType.WindowStaysOnTopHint
                )
                qt_splash.show()
                app.processEvents()

    from gui.main_window import MainWindow

    win = MainWindow(app)
    if not app_icon.isNull():
        win.setWindowIcon(app_icon)
    win.show()

    if qt_splash is not None:
        qt_splash.finish(win)
    _close_pyi_splash()

    if len(argv) > 1 and os.path.isfile(argv[1]):
        QtCore.QTimer.singleShot(0, lambda: win.load_video(argv[1]))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
