"""Asynchronous frame preview: latest request wins, GUI polls the result."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.config import DecoderConfig, Mask, StabilizerConfig
from core.hwaccel import open_capture
from core.stabilizer import Stabilizer


@dataclass
class PreviewRequest:
    path: str
    frame: int
    decoder: DecoderConfig
    stab: Optional[StabilizerConfig]   # None = no stabilization
    mask: Optional[Mask]


@dataclass
class PreviewResult:
    frame: int
    image: Optional[np.ndarray]        # BGR overlay of frame n and n+1
    error: str = ""


class PreviewWorker:
    def __init__(self):
        self._cv = threading.Condition()
        self._req: Optional[PreviewRequest] = None
        self._res: Optional[PreviewResult] = None
        self._stop = False
        self._cap = None
        self._cap_key = None
        self._stab: Optional[Stabilizer] = None
        self._stab_key = None
        self._stab_err = ""
        self._cache: Dict[int, np.ndarray] = {}
        self._next_pos = -1
        self._t = threading.Thread(target=self._run, name="PreviewWorker", daemon=True)
        self._t.start()

    def request(self, req: PreviewRequest) -> None:
        with self._cv:
            self._req = req
            self._cv.notify()

    def take_result(self) -> Optional[PreviewResult]:
        with self._cv:
            r, self._res = self._res, None
            return r

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(2)
        if self._cap is not None:
            self._cap.release()

    def invalidate(self) -> None:
        """Forget the cached capture/stabilizer (new video or decoder)."""
        with self._cv:
            self._cap_key = None
            self._stab_key = None

    # ------------------------------------------------------------------ worker
    def _read(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        if idx != self._next_pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self._cap.read()
        self._next_pos = idx + 1 if ok else -1
        if not ok:
            return None
        if len(self._cache) > 4:
            for k in sorted(self._cache)[:-3]:
                del self._cache[k]
        self._cache[idx] = f
        return f

    def _ensure(self, req: PreviewRequest) -> None:
        key = (req.path, req.decoder.kind, req.decoder.device)
        if key != self._cap_key:
            if self._cap is not None:
                self._cap.release()
            self._cap, _ = open_capture(req.path, req.decoder)
            self._cap_key = key
            self._cache.clear()
            self._next_pos = -1
            self._stab_key = None
        skey = None if req.stab is None else (req.path, req.stab.detector, req.stab.model,
                                              req.stab.anchor_frame, req.mask)
        if skey != self._stab_key:
            self._stab, self._stab_err = None, ""
            if skey is not None:
                try:
                    anchor = self._read(req.stab.anchor_frame)
                    if anchor is None:
                        raise RuntimeError("cannot read the anchor frame")
                    self._stab = Stabilizer(req.stab, anchor, req.mask)
                except Exception as e:
                    self._stab_err = str(e)
            self._stab_key = skey

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                req, self._req = self._req, None
            try:
                self._ensure(req)
                f1 = self._read(req.frame)
                f2 = self._read(req.frame + 1)
                if f1 is None:
                    res = PreviewResult(req.frame, None, "cannot read frame")
                else:
                    if f2 is None:
                        f2 = f1
                    if self._stab is not None:
                        f1 = self._stab.warp_full(f1, self._stab.estimate(f1)[0])
                        f2 = self._stab.warp_full(f2, self._stab.estimate(f2)[0])
                    img = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
                    res = PreviewResult(req.frame, img, self._stab_err)
            except Exception as e:
                res = PreviewResult(req.frame, None, f"{type(e).__name__}: {e}")
            with self._cv:
                self._res = res
