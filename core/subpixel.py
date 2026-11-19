"""Sub-pixel strip planning, strip sampling and the streaming stitcher.

Geometry (plan section 8.3, method 2)
------------------------------------
Treat the panorama as a continuous axis ``U`` that grows with time. Frame
``k`` (width ``w_k``) covers ``[U_k, U_k + w_k)``. Every *integer* panorama
column is sampled exactly once from the frame responsible for it, at a
(possibly fractional) source position. Integer widths therefore reduce to a
plain pixel copy (identical to v0.3.5), fractional widths are resampled with
interpolation -- no gaps, no duplicates, no accumulated error.

For each frame :class:`StripPlanner` returns ``(phase, n)``: the chunk has
``n`` output columns sampled at source x = ``pos + phase + [0, 1, ..., n-1]``
(left-to-right order in the frame).

The :class:`Stitcher` works in *growth order* (new columns always appended
at the end). For direction 1 (strips grow to the left) chunks are flipped
before being added and tiles are flipped back when written.
"""
from __future__ import annotations

import math
import threading
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .config import WidthSchedule

INTERP_FLAGS = {
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "linear": cv2.INTER_LINEAR,
}


class StripPlanner:
    def __init__(self, schedule: WidthSchedule, direction: int):
        self.schedule = schedule
        self.direction = 1 if direction else 0
        self.U = 0.0

    def next(self, frame_idx: int) -> Tuple[float, int, float]:
        """Return ``(phase, n, width)`` for ``frame_idx`` and advance."""
        w = float(self.schedule.width_at(frame_idx))
        u0 = self.U
        u1 = round(u0 + w, 9)
        if self.direction == 0:
            c0, c1 = math.ceil(u0 - 1e-9), math.ceil(u1 - 1e-9)
            phase = c0 - u0
            n = c1 - c0
        else:
            f0, f1 = math.floor(u0 + 1e-9), math.floor(u1 + 1e-9)
            phase = u1 - f1
            n = f1 - f0
        phase = round(phase, 6)
        if phase >= 1.0 or phase <= 1e-6:
            phase = 0.0
        self.U = u1
        return phase, int(n), w


def _translation(x: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def sample_strip(img: np.ndarray, x_start: float, count: int, interp: int,
                 M: Optional[np.ndarray] = None) -> np.ndarray:
    """Sample ``count`` columns starting at (fractional) source x ``x_start``.

    ``M`` is an optional anchor->current 3x3 matrix (stabilizer); the strip is
    then taken from the *stabilized* frame without warping the whole frame.
    """
    h, w = img.shape[:2]
    if count <= 0:
        return np.empty((h, 0, img.shape[2]), img.dtype)
    xi = int(round(x_start))
    if M is None and abs(x_start - xi) < 1e-6 and 0 <= xi and xi + count <= w:
        return img[:, xi:xi + count].copy()
    T = _translation(x_start)
    if M is None:
        A = T
    else:
        A = M @ T
    flags = interp | cv2.WARP_INVERSE_MAP
    if abs(A[2, 0]) < 1e-12 and abs(A[2, 1]) < 1e-12 and abs(A[2, 2] - 1.0) < 1e-12:
        return cv2.warpAffine(img, A[:2], (count, h), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
    return cv2.warpPerspective(img, A, (count, h), flags=flags, borderMode=cv2.BORDER_REFLECT_101)


def chunk_source_start(position: float, phase: float, direction: int, feather: int) -> float:
    """Source x of the left-most sampled column (including feather columns)."""
    if direction == 0:
        return position + phase
    return position + phase - feather


class Stitcher:
    """Streams strips into fixed-width tiles (memory = one tile)."""

    def __init__(self, height: int, split_width: int, direction: int, feather: int,
                 on_tile: Callable[[int, np.ndarray], None], thumb_height: int = 120,
                 channels: int = 3):
        self.h = height
        self.split = int(split_width)
        self.direction = 1 if direction else 0
        self.feather = int(feather)
        self.on_tile = on_tile
        self.ch = channels
        self.tile_index = 0
        self.tile = np.empty((self.h, self.split, channels), np.uint8)
        self.fill = 0
        self.total_columns = 0
        self.pending: Optional[np.ndarray] = None    # feather tail of previous chunk

        # thumbnail (growth order) for the live ribbon preview
        self.thumb_h = max(8, int(thumb_height))
        self.thumb_scale = self.thumb_h / float(self.h)
        self.batch_w = max(4, int(round(2.0 / self.thumb_scale)))
        self.batch = np.empty((self.h, self.batch_w, channels), np.uint8)
        self.batch_fill = 0
        self._thumb = np.zeros((self.thumb_h, 1024, channels), np.uint8)
        self.thumb_cols = 0
        self._thumb_scaled_cols = 0.0
        self._thumb_lock = threading.Lock()

    # ------------------------------------------------------------------ input
    def add(self, chunk_growth: np.ndarray, n: int) -> None:
        """Add a chunk in growth order: ``n`` own columns + ``feather`` tail columns."""
        f = self.feather
        if f <= 0:
            self._commit(chunk_growth[:, :n])
            return
        if self.pending is None:
            self._commit(chunk_growth[:, :n])
        else:
            k = min(f, n, self.pending.shape[1])
            a = (np.arange(1, k + 1, dtype=np.float32) / (k + 1))[None, :, None]
            blended = self.pending[:, :k].astype(np.float32) * (1 - a) + chunk_growth[:, :k].astype(np.float32) * a
            self._commit(np.clip(blended + 0.5, 0, 255).astype(np.uint8))
            if n > k:
                self._commit(chunk_growth[:, k:n])
        self.pending = chunk_growth[:, n:n + f].copy()

    def _commit(self, cols: np.ndarray) -> None:
        m = cols.shape[1]
        off = 0
        while off < m:
            take = min(m - off, self.split - self.fill)
            self.tile[:, self.fill:self.fill + take] = cols[:, off:off + take]
            self.fill += take
            self._to_thumb(cols[:, off:off + take])
            off += take
            if self.fill == self.split:
                self._emit(self.tile)
                self.tile = np.empty((self.h, self.split, self.ch), np.uint8)
                self.fill = 0
        self.total_columns += m

    def _emit(self, tile_growth: np.ndarray) -> None:
        img = tile_growth[:, ::-1] if self.direction == 1 else tile_growth
        self.on_tile(self.tile_index, img)
        self.tile_index += 1

    # ------------------------------------------------------------------ thumbnail
    def _to_thumb(self, cols: np.ndarray) -> None:
        m = cols.shape[1]
        off = 0
        with self._thumb_lock:
            while off < m:
                take = min(m - off, self.batch_w - self.batch_fill)
                self.batch[:, self.batch_fill:self.batch_fill + take] = cols[:, off:off + take]
                self.batch_fill += take
                off += take
                if self.batch_fill == self.batch_w:
                    self._flush_batch_locked()

    def _flush_batch(self) -> None:
        with self._thumb_lock:
            self._flush_batch_locked()

    def _flush_batch_locked(self) -> None:
        bf = self.batch_fill
        if bf == 0:
            return
        target = self._thumb_scaled_cols + bf * self.thumb_scale
        out_w = int(round(target)) - self.thumb_cols
        self._thumb_scaled_cols = target
        self.batch_fill = 0
        if out_w <= 0:
            return
        small = cv2.resize(self.batch[:, :bf], (out_w, self.thumb_h), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = small[:, :, None]
        need = self.thumb_cols + out_w
        if need > self._thumb.shape[1]:
            cap = max(need, self._thumb.shape[1] * 2)
            new = np.zeros((self.thumb_h, cap, self.ch), np.uint8)
            new[:, :self.thumb_cols] = self._thumb[:, :self.thumb_cols]
            self._thumb = new
        self._thumb[:, self.thumb_cols:need] = small
        self.thumb_cols = need

    def thumbnail(self) -> np.ndarray:
        """Current ribbon thumbnail in panorama (display) orientation (a copy)."""
        with self._thumb_lock:
            tc = self.thumb_cols
            bf = self.batch_fill
            extra = None
            if bf > 0:
                rem = max(1 if tc == 0 else 0,
                          int(round(self._thumb_scaled_cols + bf * self.thumb_scale)) - tc)
                if rem > 0:
                    extra = cv2.resize(self.batch[:, :bf], (rem, self.thumb_h), interpolation=cv2.INTER_AREA)
                    if extra.ndim == 2:
                        extra = extra[:, :, None]
            if tc == 0 and extra is None:
                return np.empty((self.thumb_h, 0, self.ch), np.uint8)
            if extra is not None:
                base = np.concatenate([self._thumb[:, :tc], extra], axis=1) if tc > 0 else extra
            else:
                base = self._thumb[:, :tc].copy()
        return np.ascontiguousarray(base[:, ::-1] if self.direction == 1 else base)

    # ------------------------------------------------------------------ end
    def finish(self) -> int:
        """Flush the partially filled tile (cropped). Returns total columns."""
        self._flush_batch()
        if self.fill > 0:
            self._emit(self.tile[:, :self.fill])
            self.tile = np.empty((self.h, self.split, self.ch), np.uint8)
            self.fill = 0
        return self.total_columns
