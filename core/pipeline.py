"""Scan pipeline: reader -> (parallel stabilizer) -> stitcher -> tile writer.

The whole job runs in background threads. The GUI never gets pushed data;
it polls :class:`ProgressState` (pull model, plan section 3).
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import __version__
from .config import ScanConfig
from .hwaccel import open_capture
from .stabilizer import StabDiag, Stabilizer
from .subpixel import INTERP_FLAGS, Stitcher, StripPlanner, chunk_source_start, sample_strip
from .tile_writer import TileWriter, write_json
from .video_io import FrameReader, VideoInfo, read_frame_at, read_info


@dataclass
class Snapshot:
    stage: str = "idle"           # idle | preparing | processing | finalizing | done | stopped | error
    frames_done: int = 0
    total_est: int = 0
    fps: float = 0.0
    eta: float = -1.0
    elapsed: float = 0.0
    tiles_saved: int = 0
    columns: int = 0
    stab_failures: int = 0
    decoder_msg: str = ""
    hw_active: bool = False
    error: str = ""
    out_dir: str = ""
    files: List[str] = field(default_factory=list)
    preview: Optional[np.ndarray] = None      # downscaled current frame (BGR)
    preview_frame: int = -1
    ribbon: Optional[np.ndarray] = None       # panorama thumbnail (BGR, display orientation)

    @property
    def finished(self) -> bool:
        return self.stage in ("done", "stopped", "error")


class ProgressState:
    """Thread-safe progress holder shared between a job and the GUI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._s = Snapshot()
        self._want_preview = False
        self._ribbon_src: Optional[Callable[[], np.ndarray]] = None

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._s, k, v)

    def request_preview(self) -> None:
        self._want_preview = True

    def take_preview_request(self) -> bool:
        if self._want_preview:
            self._want_preview = False
            return True
        return False

    def set_ribbon_source(self, fn: Optional[Callable[[], np.ndarray]]) -> None:
        self._ribbon_src = fn

    def snapshot(self, with_images: bool = True) -> Snapshot:
        with self._lock:
            s = Snapshot(**{k: getattr(self._s, k) for k in self._s.__dataclass_fields__})
            s.files = list(self._s.files)
        if with_images and self._ribbon_src is not None:
            try:
                s.ribbon = self._ribbon_src()
            except Exception:
                s.ribbon = None
        if not with_images:
            s.preview = None
            s.ribbon = None
        return s


def _fmt_matrix(M: Optional[np.ndarray]):
    return None if M is None else [[float(v) for v in row] for row in M]


class ScanJob:
    def __init__(self, cfg: ScanConfig, info: Optional[VideoInfo] = None,
                 state: Optional[ProgressState] = None):
        self.cfg = cfg
        self.info = info or read_info(cfg.video_path)
        self.state = state or ProgressState()
        self.cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ control
    def start(self) -> "ScanJob":
        self._thread = threading.Thread(target=self._run_safe, name="ScanJob", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self.cancel_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> Snapshot:
        """Run synchronously (CLI / tests)."""
        self._run_safe()
        return self.state.snapshot(with_images=False)

    # ------------------------------------------------------------------ main
    def _run_safe(self) -> None:
        try:
            self._run()
        except BaseException as e:
            self.state.update(stage="error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _run(self) -> None:
        cfg, info, st = self.cfg, self.info, self.state
        cfg.validate(info.width, info.height)
        H, W = info.height, info.width
        direction = cfg.effective_direction()
        sched = cfg.width_schedule()
        planner = StripPlanner(sched, direction)
        interp = INTERP_FLAGS[cfg.interpolation]
        feather = int(cfg.feather)
        if feather > 0:
            min_w = min(sched.widths)
            if feather > math.floor(min_w):
                raise ValueError(f"feather ({feather}) must be <= the smallest strip width ({min_w})")
        out_dir = cfg.output_dir()
        os.makedirs(out_dir, exist_ok=True)
        t_start = time.perf_counter()
        st.update(stage="preparing", total_est=info.total_frames, out_dir=out_dir, error="",
                  frames_done=0, tiles_saved=0, columns=0, stab_failures=0, files=[])

        manifest = {
            "app": "video_scanner", "version": __version__,
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video": {"path": cfg.video_path, "fps": info.fps, "frames_reported": info.total_frames,
                      "width": W, "height": H, "fourcc": info.fourcc},
            "params": cfg.to_dict(),
            "direction_effective": direction,
            "tile_order": "left_to_right",
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        write_json(manifest_path, manifest)

        writer = TileWriter(out_dir, cfg.video_stem, direction, cfg.jpeg_quality,
                            on_written=lambda n: st.update(tiles_saved=n))
        stitcher = Stitcher(H, cfg.split_width, direction, feather, writer.submit)
        st.set_ribbon_source(stitcher.thumbnail)

        stab_log = None
        stab_fail = 0
        frames_done = 0
        fps_t0, fps_n0, fps = time.perf_counter(), 0, 0.0
        status = "done"
        decode_info = None

        def progress(i_done: int):
            nonlocal fps_t0, fps_n0, fps
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                inst = (i_done - fps_n0) / (now - fps_t0)
                fps = inst if fps == 0 else 0.7 * fps + 0.3 * inst
                fps_t0, fps_n0 = now, i_done
                remaining = max(0, info.total_frames - i_done)
                st.update(fps=fps, eta=(remaining / fps) if fps > 0 else -1.0,
                          elapsed=now - t_start, columns=stitcher.total_columns, stab_failures=stab_fail)
            st.update(frames_done=i_done)

        def add_chunk(chunk: np.ndarray, n: int):
            if chunk.shape[1] == 0:
                return
            stitcher.add(chunk[:, ::-1] if direction == 1 else chunk, n)

        try:
            if cfg.stabilizer.enabled:
                # ---------------------------------------------------- stabilized path
                cap, _ = open_capture(cfg.video_path, cfg.decoder)
                try:
                    anchor = read_frame_at(cap, cfg.stabilizer.anchor_frame)
                finally:
                    cap.release()
                if anchor is None:
                    raise IOError("cannot read the stabilizer anchor frame")
                stab = Stabilizer(cfg.stabilizer, anchor, cfg.mask)
                del anchor
                stab_log = open(os.path.join(out_dir, "stabilizer_log.csv"), "w", encoding="utf-8")
                stab_log.write(StabDiag.header() + "\n")
                workers = cfg.stabilizer.resolved_workers()
                reader = FrameReader(cfg.video_path, cfg.decoder, self.cancel_event,
                                     queue_size=max(2, workers // 2)).start()
                decode_info = reader.decode_info
                if decode_info:
                    st.update(decoder_msg=decode_info.message, hw_active=decode_info.hw_active)
                st.update(stage="processing")

                def task(frame, x0, count, want_prev):
                    M, diag = stab.estimate(frame)
                    chunk = sample_strip(frame, x0, count, interp, M) if M is not None else None
                    prev = stab.warp_preview(frame, M) if want_prev else None
                    return M, diag, chunk, (frame if chunk is None else None), prev

                last_good: Optional[np.ndarray] = None
                pending: deque = deque()
                max_inflight = workers + 2

                def consume():
                    nonlocal last_good, stab_fail, frames_done
                    idx, n, fut = pending.popleft()
                    M, diag, chunk, frame, prev = fut.result()
                    if chunk is None:
                        diag.fallback = True
                        stab_fail += 1
                        x0, count = frame_meta[idx]
                        chunk = sample_strip(frame, x0, count, interp, last_good)
                    else:
                        last_good = M
                    stab_log.write(diag.row(idx) + "\n")
                    add_chunk(chunk, n)
                    if prev is not None:
                        st.update(preview=prev, preview_frame=idx)
                    frames_done += 1
                    progress(frames_done)

                frame_meta = {}
                try:
                    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stab") as ex:
                        for idx, frame, _ in reader:
                            if self.cancel_event.is_set():
                                break
                            phase, n, _w = planner.next(idx)
                            count = n + feather
                            x0 = chunk_source_start(cfg.position, phase, direction, feather)
                            frame_meta[idx] = (x0, count)
                            fut = ex.submit(task, frame, x0, count, st.take_preview_request())
                            pending.append((idx, n, fut))
                            del frame
                            while len(pending) >= max_inflight:
                                i0 = pending[0][0]
                                consume()
                                frame_meta.pop(i0, None)
                        while pending and not self.cancel_event.is_set():
                            i0 = pending[0][0]
                            consume()
                            frame_meta.pop(i0, None)
                        for _, _, fut in pending:
                            fut.cancel()
                        pending.clear()
                finally:
                    reader.stop()
                    reader.join(2)
            else:
                # ---------------------------------------------------- plain path (strip crop in reader)
                margin = 4 + feather
                cx0 = max(0, int(math.floor(cfg.position)) - margin)
                cx1 = min(W, int(math.ceil(cfg.position + sched.max_width())) + margin + 1)
                reader = FrameReader(cfg.video_path, cfg.decoder, self.cancel_event, queue_size=64,
                                     crop=(cx0, cx1), want_preview=st.take_preview_request).start()
                decode_info = reader.decode_info
                if decode_info:
                    st.update(decoder_msg=decode_info.message, hw_active=decode_info.hw_active)
                st.update(stage="processing")
                try:
                    for idx, strip, prev in reader:
                        if self.cancel_event.is_set():
                            break
                        phase, n, _w = planner.next(idx)
                        count = n + feather
                        x0 = chunk_source_start(cfg.position, phase, direction, feather) - cx0
                        add_chunk(sample_strip(strip, x0, count, interp), n)
                        if prev is not None:
                            st.update(preview=prev, preview_frame=idx)
                        frames_done += 1
                        progress(frames_done)
                finally:
                    reader.stop()
                    reader.join(2)
            if self.cancel_event.is_set():
                status = "stopped"
        except BaseException:
            status = "error"
            raise
        finally:
            st.update(stage="finalizing")
            files: List[str] = []
            err = None
            try:
                total_cols = stitcher.finish()
                files = writer.close()
            except BaseException as e:  # keep the original error if any
                err = e
                total_cols = stitcher.total_columns
            if stab_log is not None:
                stab_log.close()
            elapsed = time.perf_counter() - t_start
            manifest.update({
                "status": status if err is None else "error",
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_sec": round(elapsed, 3),
                "frames_processed": frames_done,
                "panorama_width": total_cols,
                "panorama_height": H,
                "tiles": [{"file": f, "width": w} for f, w in zip(files, writer.widths)],
                "decoder": None if decode_info is None else decode_info.__dict__,
                "stabilizer_fallback_frames": stab_fail if cfg.stabilizer.enabled else None,
            })
            try:
                write_json(manifest_path, manifest)
            except OSError:
                pass
            st.update(frames_done=frames_done, columns=total_cols, files=files, elapsed=elapsed,
                      stab_failures=stab_fail, fps=(frames_done / elapsed if elapsed > 0 else 0.0), eta=0.0)
            if status != "error" and err is None:
                st.update(stage=status)
            if err is not None:
                raise err


class BackgroundTask:
    """Run ``fn(cancel_event, progress_cb)`` in a thread; poll from the GUI."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.cancel_event = threading.Event()
        self.progress: Tuple[int, int] = (0, 0)
        self.result = None
        self.error: Optional[str] = None
        self.done = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "BackgroundTask":
        self._thread.start()
        return self

    def _set_progress(self, i: int, n: int) -> None:
        self.progress = (i, n)

    def _run(self) -> None:
        try:
            self.result = self.fn(self.cancel_event, self._set_progress)
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            self.done = True

    def cancel(self) -> None:
        self.cancel_event.set()
