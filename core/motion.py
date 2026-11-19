"""Movement (strip width) estimation -- the 'calculate move' function."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .config import DecoderConfig, Mask, StabilizerConfig
from .hwaccel import open_capture
from .stabilizer import Stabilizer
from .video_io import read_frame_at

_FEATURES = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
_LK = dict(winSize=(21, 21), maxLevel=5,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def horizontal_motion(frame_1: np.ndarray, frame_2: np.ndarray, mask: Optional[Mask] = None,
                      max_dy: float = 2.0) -> Optional[float]:
    """Median horizontal displacement (px) of features inside ``mask``.

    Positive = the object moves to the right. ``None`` if nothing trackable.
    Uses a forward-backward consistency check to reject bad tracks.
    """
    g1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY) if frame_1.ndim == 3 else frame_1
    g2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY) if frame_2.ndim == 3 else frame_2
    m = None
    if mask is not None:
        h, w = g1.shape
        yt, yb, xl, xr = mask
        m = np.zeros((h, w), np.uint8)
        m[max(0, yt):max(0, yb), max(0, xl):max(0, xr)] = 255
        if not m.any():
            m = None
    p0 = cv2.goodFeaturesToTrack(g1, mask=m, **_FEATURES)
    if p0 is None or len(p0) == 0:
        return None
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, **_LK)
    if p1 is None:
        return None
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, p1, None, **_LK)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1)
    fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    d = (p1 - p0).reshape(-1, 2)
    ok &= (fb < 1.0) & (np.abs(d[:, 1]) < max_dy)
    if ok.sum() < 3:
        return None
    return float(np.median(d[ok, 0]))


@dataclass
class MotionResult:
    segments: List[Tuple[int, float]] = field(default_factory=list)
    failed_samples: List[int] = field(default_factory=list)
    cancelled: bool = False


def estimate_segments(video_path: str, total_frames: int, start: int, samples: int,
                      mask: Optional[Mask] = None,
                      stab_cfg: Optional[StabilizerConfig] = None,
                      decoder: Optional[DecoderConfig] = None,
                      cancel: Optional[threading.Event] = None,
                      progress: Optional[Callable[[int, int], None]] = None) -> MotionResult:
    """Sample ``samples`` positions from ``start`` and estimate the strip width.

    Returns segments ``[(start_frame, width), ...]``; the first start is
    always 0 (same convention as v0.3.5). Widths are signed (see config).
    """
    samples = max(1, int(samples))
    start = max(0, int(start))
    cancel = cancel or threading.Event()
    res = MotionResult()
    cap, _ = open_capture(video_path, decoder)
    try:
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        stab = None
        if stab_cfg is not None and stab_cfg.enabled:
            anchor = read_frame_at(cap, stab_cfg.anchor_frame)
            if anchor is None:
                raise IOError("cannot read the stabilizer anchor frame")
            stab = Stabilizer(stab_cfg, anchor, mask)
        length = max(3, total_frames - start)
        step = max(1, (length - 3) // samples)
        last_w: Optional[float] = None
        for i in range(samples):
            if cancel.is_set():
                res.cancelled = True
                break
            fidx = start + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            frames = []
            for _ in range(3):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            if len(frames) < 2:
                break
            if stab is not None:
                frames = [stab.warp_full(f, stab.estimate(f)[0]) for f in frames]
            ws = [horizontal_motion(frames[j], frames[j + 1], mask) for j in range(len(frames) - 1)]
            ws = [w for w in ws if w is not None]
            if ws:
                w = round(float(np.mean(ws)), 2)
                last_w = w
            else:
                res.failed_samples.append(fidx)
                w = last_w if last_w is not None else 0.0
            res.segments.append((0 if i == 0 else fidx, w))
            if progress:
                progress(i + 1, samples)
        # samples that failed before any success inherit the first valid width
        valid = [w for _, w in res.segments if w != 0]
        if valid:
            first = valid[0]
            res.segments = [(s, first if w == 0 else w) for s, w in res.segments]
        return res
    finally:
        cap.release()
