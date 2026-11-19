"""Parameter containers and validation.

Conventions (kept from v0.3.5):
- ``direction == 1``: the object moves from left to right. Each new strip is
  placed to the LEFT of the previous one (the first frame ends up at the
  right end of the panorama).
- ``direction == 0``: the object moves from right to left. Each new strip is
  appended to the RIGHT.
- In adaptive mode the sign of the first non-zero segment width decides the
  direction (positive optical-flow dx == object moves right == direction 1).
- ``mask`` is ``(y_top, y_bottom, x_left, x_right)`` in original pixel
  coordinates. It marks the region of the moving object: motion estimation
  searches features INSIDE it, the stabilizer searches features OUTSIDE it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

Mask = Tuple[int, int, int, int]

INTERPOLATIONS = ("cubic", "lanczos", "linear")
DETECTORS = ("orb", "sift")
MODELS = ("translation", "similarity", "homography")


@dataclass
class StabilizerConfig:
    enabled: bool = False
    detector: str = "orb"            # orb | sift
    model: str = "similarity"        # translation | similarity | homography
    detect_width: int = 960          # features are detected on a frame resized to this width
    n_features: int = 2000
    ratio: float = 0.8               # Lowe ratio test
    ransac_thresh: float = 3.0       # in detection-resolution pixels
    min_inliers: int = 12
    use_clahe: bool = True
    anchor_frame: int = 0
    workers: int = 0                 # 0 = auto

    def validate(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"unknown detector: {self.detector}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model: {self.model}")
        if self.detect_width < 64:
            raise ValueError("detect_width too small")

    def resolved_workers(self) -> int:
        if self.workers and self.workers > 0:
            return self.workers
        n = os.cpu_count() or 4
        return max(1, min(8, n - 2))


@dataclass
class DecoderConfig:
    """Which backend / hardware acceleration to use for decoding."""
    kind: str = "software"   # software | d3d11 | vaapi | any | avfoundation
    device: int = -1         # adapter index for hardware decoders
    label: str = "Software (CPU)"


@dataclass
class ScanConfig:
    video_path: str
    save_root: str                       # outputs go to <save_root>/<video_stem>/ unless out_dir given
    position: int
    mode: str = "manual"                 # manual | adaptive
    width: float = 5.0                   # manual mode strip width (may be fractional)
    direction: int = 1                   # manual mode direction (see module doc)
    segments: List[Tuple[int, float]] = field(default_factory=list)  # adaptive: (start_frame, width)
    split_width: int = 20000
    mask: Optional[Mask] = None
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    interpolation: str = "cubic"
    feather: int = 0                     # seam feathering overlap in pixels (0 = off)
    jpeg_quality: int = 100
    out_dir: Optional[str] = None        # explicit output directory (overrides save_root/stem)

    # ------------------------------------------------------------------ helpers
    @property
    def video_stem(self) -> str:
        return os.path.splitext(os.path.basename(self.video_path))[0]

    def output_dir(self) -> str:
        if self.out_dir:
            return self.out_dir
        return os.path.join(self.save_root, self.video_stem)

    def effective_direction(self) -> int:
        if self.mode == "adaptive":
            for _, w in self.segments:
                if w != 0:
                    return 1 if w > 0 else 0
            return 1
        return 1 if self.direction else 0

    def width_schedule(self) -> "WidthSchedule":
        if self.mode == "adaptive":
            return WidthSchedule([(int(s), abs(float(w))) for s, w in self.segments])
        return WidthSchedule([(0, abs(float(self.width)))])

    def max_width(self) -> float:
        return self.width_schedule().max_width()

    def validate(self, frame_width: int, frame_height: int) -> None:
        if not self.video_path or not os.path.isfile(self.video_path):
            raise ValueError("video file not found")
        if self.mode not in ("manual", "adaptive"):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.mode == "adaptive" and not self.segments:
            raise ValueError("adaptive mode needs at least one segment (run 'calculate move' first)")
        sched = self.width_schedule()
        if sched.max_width() <= 0:
            raise ValueError("strip width must be > 0")
        if not (0 <= self.position < frame_width):
            raise ValueError(f"position must be within [0, {frame_width})")
        if self.position + sched.max_width() + self.feather > frame_width:
            raise ValueError("position + width (+ feather) exceeds the frame width")
        if not (1 <= self.split_width <= 65000):
            raise ValueError("tile width must be within [1, 65000] (JPEG limit 65535)")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"unknown interpolation: {self.interpolation}")
        if self.feather < 0:
            raise ValueError("feather must be >= 0")
        if self.mask is not None:
            yt, yb, xl, xr = self.mask
            if not (yt < yb and xl < xr):
                raise ValueError("mask must satisfy y_top < y_bottom and x_left < x_right")
        self.stabilizer.validate()

    def to_dict(self) -> dict:
        return asdict(self)


class WidthSchedule:
    """Piecewise-constant strip width as a function of the frame index."""

    def __init__(self, segments: List[Tuple[int, float]]):
        if not segments:
            raise ValueError("empty width schedule")
        segs = sorted(segments, key=lambda s: s[0])
        self.starts = [s for s, _ in segs]
        self.widths = [w for _, w in segs]

    def width_at(self, frame_idx: int) -> float:
        # same rule as v0.3.5: the last segment whose start <= frame_idx
        # (frames before the first start use the first width)
        w = self.widths[0]
        for s, ww in zip(self.starts, self.widths):
            if s <= frame_idx:
                w = ww
            else:
                break
        return w

    def max_width(self) -> float:
        return max(self.widths)
