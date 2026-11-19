# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for VideoScanner v0.4.0 (self-contained inside v040/).

Usage (run inside the v040 directory):
    pyinstaller build_exe.spec
"""
import os
import sys

HERE = os.path.abspath(SPECPATH)
ASSETS_DIR = os.path.join(HERE, "assets")
ICON_FILE = os.path.join(ASSETS_DIR, "icon.ico")
SPLASH_FILE = os.path.join(ASSETS_DIR, "startup.png")

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

a = Analysis(
    [os.path.join(HERE, "main.py")],
    pathex=[HERE],
    binaries=[],
    datas=[
        (ICON_FILE, "assets"),
        (SPLASH_FILE, "assets"),
    ],
    hiddenimports=["pyqtgraph"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure)

# PyInstaller's native bootloader Splash is supported on Windows (on macOS/Linux
# main.py automatically falls back to Qt's QSplashScreen using assets/startup.png).
splash_items = []
if IS_WIN and os.path.isfile(SPLASH_FILE):
    splash = Splash(
        SPLASH_FILE,
        binaries=a.binaries,
        datas=a.datas,
        text_pos=(20, 460),
        text_size=10,
        text_color="#4a5568",
        max_img_size=(800, 480),
        minify_script=True,
        always_on_top=True,
    )
    splash_items = [splash, splash.binaries]

if IS_MAC:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="VideoScanner_v0.4.0",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=ICON_FILE if os.path.isfile(ICON_FILE) else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="VideoScanner_v0.4.0",
    )
    app = BUNDLE(
        coll,
        name="VideoScanner_v0.4.0.app",
        icon=ICON_FILE if os.path.isfile(ICON_FILE) else None,
        bundle_identifier="com.divertingpan.videoscanner",
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        *splash_items,
        [],
        name="VideoScanner_v0.4.0",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        icon=ICON_FILE if os.path.isfile(ICON_FILE) else None,
    )

