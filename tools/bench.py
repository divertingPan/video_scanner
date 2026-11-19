"""Headless benchmark / regression tool.

Examples (run from the v040 folder):
    python -m tools.bench ../test.mp4 --pos 200 --width 5 --dir 1 --split 2000 --out ../_bench/new
    python -m tools.bench ../test.mp4 --pos 200 --width 5 --dir 1 --split 2000 --out ../_bench/old --legacy
    python -m tools.bench --compare ../_bench/old/test ../_bench/new/test
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V040 = os.path.dirname(HERE)
ROOT = os.path.dirname(V040)
if V040 not in sys.path:
    sys.path.insert(0, V040)

from core.config import DecoderConfig, ScanConfig, StabilizerConfig  # noqa: E402
from core.pipeline import ScanJob  # noqa: E402
from core.video_io import read_info  # noqa: E402


# --------------------------------------------------------------------------- memory
class PeakMemory:
    def __init__(self, interval=0.05):
        import psutil
        self.p = psutil.Process()
        self.peak = self.p.memory_info().rss
        self.base = self.peak
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.interval = interval

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        mi = self.p.memory_info()
        self.peak = max(self.peak, getattr(mi, "peak_wset", 0), mi.rss)


# --------------------------------------------------------------------------- legacy (v0.3.5 logic)
def run_legacy(video, pos, width, direction, split, out_root, stable=False):
    """Faithful copy of v0.3.5 manual-mode merge_video_slides + save_img (no GUI)."""
    import math
    vc = cv2.VideoCapture(video)
    total = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vc.release()
    img_length = int(total * width)
    img = np.zeros((H, img_length, 3), dtype="uint8")
    width_list = np.array([math.ceil(width)])
    adj = np.array([0])
    stab = None
    if stable:
        sys.path.insert(0, ROOT)
        from utils.stabilizer import Stabilizer as OldStab
        stab = OldStab(border_type="reflect_101", stream=cv2.VideoCapture(video))
    cap = cv2.VideoCapture(video)
    pixel_start = img_length * (2 ** direction - 1)
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"legacy: stream ended at frame {i} < reported {total} (v0.3.5 would hang)")
        if stab is not None:
            frame = stab.stabilize(frame)
        w = abs(width_list[((adj - i) <= 0).sum() - 1])
        if direction:
            pixel_start -= w
            img[:, pixel_start:pixel_start + w, :] = frame[:, pos:pos + w, :]
        else:
            img[:, pixel_start:pixel_start + w, :] = frame[:, pos:pos + w, :]
            pixel_start += w
    cap.release()
    stem = os.path.splitext(os.path.basename(video))[0]
    save_dir = os.path.join(out_root, stem)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(img.shape[1] // split + 1):
        a = i * split
        b = img.shape[1] if i == img.shape[1] // split else a + split
        cv2.imencode(".jpg", img[:, a:b, :], [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tofile(
            os.path.join(save_dir, f"{stem}_{i}.jpg"))
    return img.shape[1]


# --------------------------------------------------------------------------- compare
def load_panorama(d):
    files = glob.glob(os.path.join(d, "*.jpg"))
    rx = re.compile(r"_(\d+)\.jpg$")
    files = sorted((f for f in files if rx.search(f) and "_part_" not in f),
                   key=lambda f: int(rx.search(f).group(1)))
    imgs = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in files]
    imgs = [im for im in imgs if im is not None and im.size]
    return np.hstack(imgs) if imgs else None


def compare(a, b):
    A, B = load_panorama(a), load_panorama(b)
    print(f"A: {None if A is None else A.shape}   B: {None if B is None else B.shape}")
    if A is None or B is None:
        return
    w = min(A.shape[1], B.shape[1])
    d = np.abs(A[:, :w].astype(np.int16) - B[:, :w].astype(np.int16))
    print(f"compared width {w}: max abs diff {d.max()}, mean {d.mean():.4f}, "
          f"pixels differing {(d.max(axis=2) > 0).mean() * 100:.3f}%")


# --------------------------------------------------------------------------- main
def parse_decoder(s):
    if s == "software":
        return DecoderConfig()
    kind, _, dev = s.partition(":")
    return DecoderConfig(kind, int(dev) if dev else -1, s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--pos", type=int, default=200)
    ap.add_argument("--width", type=float, default=5)
    ap.add_argument("--dir", type=int, default=1)
    ap.add_argument("--split", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(V040, "_bench", "new"))
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--stab", action="store_true")
    ap.add_argument("--detector", default="orb")
    ap.add_argument("--model", default="similarity")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--interp", default="cubic")
    ap.add_argument("--feather", type=int, default=0)
    ap.add_argument("--decoder", default="software")
    ap.add_argument("--mask", type=int, nargs=4, default=None, metavar=("YT", "YB", "XL", "XR"))
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--cvthreads", type=int, default=-1)
    a = ap.parse_args(argv)
    if a.cvthreads >= 0:
        cv2.setNumThreads(a.cvthreads)

    if a.compare:
        compare(*a.compare)
        return

    info = read_info(a.video)
    print(f"video {a.video}: {info.width}x{info.height} {info.fps:.2f}fps "
          f"{info.total_frames} frames ({info.fourcc})")
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        if a.legacy:
            cols = run_legacy(a.video, a.pos, a.width, a.dir, a.split, a.out, a.stab)
            frames = info.total_frames
            extra = ""
        else:
            cfg = ScanConfig(video_path=a.video, save_root=a.out, position=a.pos, mode="manual",
                             width=a.width, direction=a.dir, split_width=a.split,
                             mask=tuple(a.mask) if a.mask else None,
                             stabilizer=StabilizerConfig(enabled=a.stab, detector=a.detector,
                                                         model=a.model, workers=a.workers),
                             decoder=parse_decoder(a.decoder), interpolation=a.interp, feather=a.feather)
            snap = ScanJob(cfg, info).run()
            if snap.stage == "error":
                print(snap.error)
                sys.exit(1)
            cols, frames = snap.columns, snap.frames_done
            extra = f" | stab fallbacks {snap.stab_failures} | decoder: {snap.decoder_msg}"
    dt = time.perf_counter() - t0
    print(f"{'legacy' if a.legacy else 'v0.4.0'}: {frames} frames, panorama width {cols}px, "
          f"{dt:.2f}s ({frames / dt:.1f} fps), peak RSS {mem.peak / 2**20:.0f} MB "
          f"(base {mem.base / 2**20:.0f} MB){extra}")


if __name__ == "__main__":
    main()
