"""Streaming tile writer (background JPEG encoding) and output-dir helpers."""
from __future__ import annotations

import json
import os
import queue
import re
import threading
from typing import List, Optional

import cv2
import numpy as np

_STOP = object()


def tile_name(stem: str, index: int) -> str:
    return f"{stem}_{index}.jpg"


def part_name(stem: str, t: int) -> str:
    return f"{stem}_part_{t:04d}.jpg"


def find_existing_outputs(out_dir: str, stem: str) -> List[str]:
    """Files from a previous run that would conflict with a new run."""
    if not os.path.isdir(out_dir):
        return []
    pat = re.compile(r"^" + re.escape(stem) + r"_(part_)?\d+\.jpg(\.tmp)?$", re.IGNORECASE)
    extra = {"manifest.json", "stabilizer_log.csv"}
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if pat.match(f) or f in extra)


def write_jpeg(path: str, img: np.ndarray, quality: int = 100) -> None:
    """Unicode-path-safe, atomic JPEG write."""
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError(f"JPEG encoding failed for {path}")
    tmp = path + ".tmp"
    buf.tofile(tmp)
    os.replace(tmp, path)


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class TileWriter:
    """Encodes and writes tiles in a background thread.

    Tiles arrive in time (growth) order ``t = 0, 1, 2 ...``.
      * direction 0: time order == left-to-right order -> final names directly;
      * direction 1: time order == right-to-left order -> temporary
        ``<stem>_part_XXXX.jpg`` names, renamed to the final left-to-right
        numbering in :meth:`close` (also after a stop).
    """

    def __init__(self, out_dir: str, stem: str, direction: int, quality: int = 100,
                 max_pending: int = 2, on_written=None):
        self.out_dir = out_dir
        self.stem = stem
        self.direction = 1 if direction else 0
        self.quality = quality
        self.on_written = on_written
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, max_pending))
        self.written: List[str] = []          # in time order
        self.widths: List[int] = []
        self._error: Optional[BaseException] = None
        os.makedirs(out_dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="TileWriter", daemon=True)
        self._thread.start()

    def submit(self, t: int, img: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self.q.put((t, img))

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is _STOP:
                return
            if self._error is not None:
                continue  # drain
            t, img = item
            try:
                name = tile_name(self.stem, t) if self.direction == 0 else part_name(self.stem, t)
                write_jpeg(os.path.join(self.out_dir, name), img, self.quality)
                self.written.append(name)
                self.widths.append(int(img.shape[1]))
                if self.on_written:
                    self.on_written(len(self.written))
            except BaseException as e:
                self._error = e

    def close(self) -> List[str]:
        """Wait for pending tiles, finalize names. Returns final names left->right."""
        self.q.put(_STOP)
        self._thread.join()
        if self._error is not None:
            raise self._error
        if self.direction == 0:
            return list(self.written)
        n = len(self.written)
        final = []
        for t, name in enumerate(self.written):
            dst = tile_name(self.stem, n - 1 - t)
            os.replace(os.path.join(self.out_dir, name), os.path.join(self.out_dir, dst))
            final.append(dst)
        self.widths = self.widths[::-1]
        return final[::-1]
