"""Hardware decoder enumeration, opening and probing.

OpenCV exposes hardware decoding through the FFmpeg backend via
``CAP_PROP_HW_ACCELERATION`` / ``CAP_PROP_HW_DEVICE`` (OpenCV >= 4.5.2).
OpenCV silently falls back to software decoding if the requested
acceleration is not available, so after opening we always read back the
property to know what is really in use.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .config import DecoderConfig

_HW_KIND = {
    "any": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
    "d3d11": getattr(cv2, "VIDEO_ACCELERATION_D3D11", 2),
    "vaapi": getattr(cv2, "VIDEO_ACCELERATION_VAAPI", 3),
}
_HAS_HW_API = hasattr(cv2, "CAP_PROP_HW_ACCELERATION")


# --------------------------------------------------------------------------- enumeration
def _enum_dxgi_adapters() -> List[Tuple[int, str]]:
    """Return [(adapter_index, description)] of hardware DXGI adapters (Windows)."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DXGI_ADAPTER_DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128),
                    ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                    ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                    ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t),
                    ("SharedSystemMemory", ctypes.c_size_t),
                    ("AdapterLuid", LUID), ("Flags", ctypes.c_uint)]

    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA,
               (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory))
    if hr != 0 or not factory:
        return []

    HRESULT = ctypes.c_long

    def vtbl(obj):
        return ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]

    release_t = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
    enum1_t = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
    desc1_t = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1))

    out = []
    try:
        # IDXGIFactory1::EnumAdapters1 is vtable slot 12
        enum1 = enum1_t(vtbl(factory)[12])
        i = 0
        while i < 16:
            adapter = ctypes.c_void_p()
            hr = enum1(factory, i, ctypes.byref(adapter))
            if hr != 0 or not adapter:
                break
            try:
                desc = DXGI_ADAPTER_DESC1()
                # IDXGIAdapter1::GetDesc1 is vtable slot 10
                if desc1_t(vtbl(adapter)[10])(adapter, ctypes.byref(desc)) == 0:
                    if not (desc.Flags & 2):  # DXGI_ADAPTER_FLAG_SOFTWARE
                        out.append((i, desc.Description.strip()))
            finally:
                release_t(vtbl(adapter)[2])(adapter)
            i += 1
    finally:
        release_t(vtbl(factory)[2])(factory)
    return out


def _enum_drm_nodes() -> List[Tuple[int, str]]:
    vendors = {"0x8086": "Intel", "0x1002": "AMD", "0x10de": "NVIDIA"}
    out = []
    for i, node in enumerate(sorted(glob.glob("/dev/dri/renderD*"))):
        name = os.path.basename(node)
        vendor = ""
        try:
            with open(f"/sys/class/drm/{name}/device/vendor") as f:
                v = f.read().strip()
                vendor = vendors.get(v, v)
        except OSError:
            pass
        out.append((i, f"{vendor} {name}".strip()))
    return out


def list_decoders() -> List[DecoderConfig]:
    """All decoder choices for the current platform (software first)."""
    opts = [DecoderConfig("software", -1, "Software (CPU)")]
    if sys.platform == "darwin":
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            opts.append(DecoderConfig("avfoundation", -1, "AVFoundation (system decoder)"))
        return opts
    if not _HAS_HW_API:
        return opts
    if sys.platform.startswith("win"):
        try:
            for idx, name in _enum_dxgi_adapters():
                opts.append(DecoderConfig("d3d11", idx, f"D3D11 · {name}"))
        except Exception:
            opts.append(DecoderConfig("any", -1, "Hardware (auto)"))
    else:
        nodes = _enum_drm_nodes()
        for idx, name in nodes:
            opts.append(DecoderConfig("vaapi", idx, f"VAAPI · {name}"))
        if not nodes:
            opts.append(DecoderConfig("any", -1, "Hardware (auto)"))
    return opts


# --------------------------------------------------------------------------- opening
@dataclass
class DecodeInfo:
    requested: str
    hw_active: bool
    fell_back: bool
    message: str = ""


def open_capture(path: str, decoder: Optional[DecoderConfig] = None) -> Tuple[cv2.VideoCapture, DecodeInfo]:
    """Open ``path`` with the requested decoder; fall back to software on failure."""
    decoder = decoder or DecoderConfig()
    if decoder.kind == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap, DecodeInfo(decoder.label, True, False, "AVFoundation")
        cap.release()
    elif decoder.kind in _HW_KIND and _HAS_HW_API:
        params = [cv2.CAP_PROP_HW_ACCELERATION, _HW_KIND[decoder.kind]]
        if decoder.device >= 0:
            params += [cv2.CAP_PROP_HW_DEVICE, int(decoder.device)]
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
        except cv2.error:
            cap = None
        if cap is not None and cap.isOpened():
            accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
            if accel != 0:
                return cap, DecodeInfo(decoder.label, True, False, "hardware decoding active")
            return cap, DecodeInfo(decoder.label, False, True,
                                   "hardware decoder unavailable for this video, using software")
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(path)
        return cap, DecodeInfo(decoder.label, False, True, "hardware open failed, using software")

    cap = cv2.VideoCapture(path)
    return cap, DecodeInfo(decoder.label, False, decoder.kind != "software", "software decoding")


@dataclass
class ProbeResult:
    ok: bool
    hw_active: bool
    fell_back: bool
    fps: float
    message: str


def probe_decoder(path: str, decoder: DecoderConfig, n_frames: int = 40) -> ProbeResult:
    """Open the video with ``decoder``, decode a few frames and measure speed."""
    try:
        cap, info = open_capture(path, decoder)
    except Exception as e:  # pragma: no cover - defensive
        return ProbeResult(False, False, True, 0.0, str(e))
    try:
        if not cap.isOpened():
            return ProbeResult(False, False, True, 0.0, "cannot open video")
        ok, _ = cap.read()  # first frame includes decoder warm-up
        if not ok:
            return ProbeResult(False, info.hw_active, info.fell_back, 0.0, "cannot read frames")
        t0 = time.perf_counter()
        n = 0
        for _ in range(n_frames):
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        dt = time.perf_counter() - t0
        fps = n / dt if dt > 0 and n else 0.0
        return ProbeResult(True, info.hw_active, info.fell_back, fps, info.message)
    finally:
        cap.release()
