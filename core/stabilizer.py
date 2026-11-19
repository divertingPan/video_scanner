"""Anchor based video stabilizer (rewritten for v0.4.0).

Every frame is aligned to ONE anchor frame (frame 0 by default), so frames
are independent of each other and can be processed in parallel.

Speed-ups compared to v0.3.5 (see plan section 2.1):
  * features are detected on a downscaled grey image (``detect_width``);
  * the anchor features are computed once and the matcher is trained once
    per worker thread;
  * the caller only warps the strip it needs (``strip_matrix``) instead of
    the whole frame.

Matrices returned by :meth:`Stabilizer.estimate` are 3x3 and map ANCHOR
coordinates to CURRENT-frame coordinates in full resolution, i.e. they are
meant to be used with ``cv2.WARP_INVERSE_MAP``:
``stabilized(p) = frame(M @ p)``.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Mask, StabilizerConfig


@dataclass
class StabDiag:
    matches: int = 0
    inliers: int = 0
    ok: bool = False
    dx: float = 0.0
    dy: float = 0.0
    angle: float = 0.0
    scale: float = 1.0
    fallback: bool = False

    @staticmethod
    def header() -> str:
        return "frame,matches,inliers,ok,fallback,dx,dy,angle_deg,scale"

    def row(self, frame_idx: int) -> str:
        return (f"{frame_idx},{self.matches},{self.inliers},{int(self.ok)},{int(self.fallback)},"
                f"{self.dx:.3f},{self.dy:.3f},{self.angle:.4f},{self.scale:.5f}")


class Stabilizer:
    def __init__(self, cfg: StabilizerConfig, anchor_frame: np.ndarray, mask: Optional[Mask] = None):
        cfg.validate()
        self.cfg = cfg
        self.h, self.w = anchor_frame.shape[:2]
        self.scale = min(1.0, cfg.detect_width / float(self.w))
        self.small_size = (max(1, int(round(self.w * self.scale))), max(1, int(round(self.h * self.scale))))
        self._S = np.diag([self.scale, self.scale, 1.0])
        self._S_inv = np.diag([1.0 / self.scale, 1.0 / self.scale, 1.0])
        self._local = threading.local()

        # detection mask in small coordinates: exclude the moving object
        sw, sh = self.small_size
        self.det_mask = np.full((sh, sw), 255, np.uint8)
        if mask is not None:
            yt, yb, xl, xr = mask
            s = self.scale
            self.det_mask[max(0, int(yt * s)):max(0, int(math.ceil(yb * s))),
                          max(0, int(xl * s)):max(0, int(math.ceil(xr * s)))] = 0

        gray = self._prep(anchor_frame)
        kps, des = self._detector().detectAndCompute(gray, self.det_mask)
        if des is None or len(kps) < 8:
            raise RuntimeError("too few features in the anchor frame for stabilization "
                               "(scene too plain / too dark, or mask too large)")
        self.anchor_pts = np.float32([k.pt for k in kps])
        self.anchor_des = des

    # ------------------------------------------------------------------ per-thread objects
    def _detector(self):
        d = getattr(self._local, "detector", None)
        if d is None:
            if self.cfg.detector == "sift":
                d = cv2.SIFT_create(nfeatures=self.cfg.n_features)
            else:
                d = cv2.ORB_create(nfeatures=self.cfg.n_features, fastThreshold=10)
            self._local.detector = d
        return d

    def _matcher(self):
        m = getattr(self._local, "matcher", None)
        if m is None:
            norm = cv2.NORM_L2 if self.cfg.detector == "sift" else cv2.NORM_HAMMING
            m = cv2.BFMatcher(norm)
            m.add([self.anchor_des])
            m.train()
            self._local.matcher = m
        return m

    def _clahe(self):
        c = getattr(self._local, "clahe", None)
        if c is None:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self._local.clahe = c
        return c

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, self.small_size, interpolation=cv2.INTER_AREA) if self.scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if self.cfg.use_clahe:
            gray = self._clahe().apply(gray)
        return gray

    # ------------------------------------------------------------------ estimation
    def estimate(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], StabDiag]:
        """Return (M, diag); M is None when the estimation failed."""
        diag = StabDiag()
        gray = self._prep(frame)
        kps, des = self._detector().detectAndCompute(gray, self.det_mask)
        if des is None or len(kps) < 4:
            return None, diag
        try:
            knn = self._matcher().knnMatch(des, k=2)
        except cv2.error:
            return None, diag
        q_idx, t_idx = [], []
        ratio = self.cfg.ratio
        for pair in knn:
            if len(pair) == 2:
                m1, m2 = pair
                if m1.distance < ratio * m2.distance:
                    q_idx.append(m1.queryIdx)
                    t_idx.append(m1.trainIdx)
            elif len(pair) == 1:
                pass  # ambiguous -> skip
        diag.matches = len(q_idx)
        if diag.matches < max(4, self.cfg.min_inliers // 2):
            return None, diag
        cur = np.float32([kps[i].pt for i in q_idx])
        anc = self.anchor_pts[t_idx]

        M_small, inl = self._fit(anc, cur)
        if M_small is None:
            return None, diag
        diag.inliers = int(inl)
        M = self._S_inv @ M_small @ self._S
        self._fill_diag(diag, M)
        if diag.inliers < self.cfg.min_inliers or not self._sane(M):
            return None, diag
        diag.ok = True
        return M, diag

    def _fit(self, anc: np.ndarray, cur: np.ndarray):
        thr = self.cfg.ransac_thresh
        model = self.cfg.model
        if model == "translation":
            d = cur - anc
            med = np.median(d, axis=0)
            inl = np.linalg.norm(d - med, axis=1) < thr
            if inl.sum() < 2:
                return None, 0
            med = d[inl].mean(axis=0)
            M = np.array([[1, 0, med[0]], [0, 1, med[1]], [0, 0, 1]], np.float64)
            return M, int(inl.sum())
        if model == "similarity":
            A, inl = cv2.estimateAffinePartial2D(anc, cur, method=cv2.RANSAC,
                                                 ransacReprojThreshold=thr, maxIters=2000, confidence=0.995)
            if A is None:
                return None, 0
            return np.vstack([A, [0, 0, 1]]).astype(np.float64), int(inl.sum()) if inl is not None else 0
        H, inl = cv2.findHomography(anc, cur, cv2.RANSAC, thr)
        if H is None:
            return None, 0
        return H.astype(np.float64), int(inl.sum()) if inl is not None else 0

    def _fill_diag(self, diag: StabDiag, M: np.ndarray) -> None:
        diag.dx = float(M[0, 2])
        diag.dy = float(M[1, 2])
        a, b = M[0, 0], M[1, 0]
        diag.angle = float(math.degrees(math.atan2(b, a)))
        diag.scale = float(math.hypot(a, b))

    def _sane(self, M: np.ndarray) -> bool:
        if not np.all(np.isfinite(M)):
            return False
        if abs(M[0, 2]) > 0.3 * self.w or abs(M[1, 2]) > 0.3 * self.h:
            return False
        det = float(np.linalg.det(M[:2, :2]))
        if not (0.6 < det < 1.6):
            return False
        if abs(M[2, 0]) > 1e-3 or abs(M[2, 1]) > 1e-3:
            return False
        return True

    # ------------------------------------------------------------------ warping helpers
    @staticmethod
    def is_affine(M: np.ndarray) -> bool:
        return abs(M[2, 0]) < 1e-12 and abs(M[2, 1]) < 1e-12 and abs(M[2, 2] - 1) < 1e-12

    @staticmethod
    def warp(frame: np.ndarray, M: np.ndarray, size: Tuple[int, int], interp: int,
             border=cv2.BORDER_REFLECT_101) -> np.ndarray:
        """Warp with an anchor->current matrix (WARP_INVERSE_MAP)."""
        flags = interp | cv2.WARP_INVERSE_MAP
        if Stabilizer.is_affine(M):
            return cv2.warpAffine(frame, M[:2], size, flags=flags, borderMode=border)
        return cv2.warpPerspective(frame, M, size, flags=flags, borderMode=border)

    def warp_full(self, frame: np.ndarray, M: Optional[np.ndarray], interp: int = cv2.INTER_LINEAR) -> np.ndarray:
        if M is None:
            return frame
        return self.warp(frame, M, (self.w, self.h), interp)

    def warp_preview(self, frame: np.ndarray, M: Optional[np.ndarray], max_width: int = 960) -> np.ndarray:
        """Downscale first, then warp with the rescaled matrix (cheap preview)."""
        h, w = frame.shape[:2]
        s = min(1.0, max_width / float(w))
        size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        if M is None:
            return small.copy() if small is frame else small
        S = np.diag([s, s, 1.0])
        Ms = S @ M @ np.diag([1.0 / s, 1.0 / s, 1.0])
        return self.warp(small, Ms, size, cv2.INTER_LINEAR)
