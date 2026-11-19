"""Video information and a threaded, cancellable, bounded frame reader."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import cv2
import numpy as np

from .config import DecoderConfig
from .hwaccel import DecodeInfo, open_capture

_EOF = object()


@dataclass
class VideoInfo:
    path: str
    fps: float
    total_frames: int     # as reported by the container -- may be inaccurate!
    width: int
    height: int
    fourcc: str


def read_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_i >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        return VideoInfo(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fourcc=fourcc,
        )
    finally:
        cap.release()


def make_preview(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame.copy()
    s = max_width / w
    return cv2.resize(frame, (max_width, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


class FrameReader:
    """Decode frames in a background thread into a bounded queue.

    Each item is ``(frame_index, data, preview)``:
      * ``data`` is the full frame, or only columns ``crop[0]:crop[1]`` when
        ``crop`` is given (then the full frame is dropped immediately, which
        keeps the memory footprint tiny);
      * ``preview`` is a downscaled copy of the full frame, produced only when
        ``want_preview()`` returns True (pull model, see plan section 3).

    Iteration ends at the real end of the stream (the container frame count
    is never trusted). Errors in the reader thread are re-raised in the
    consumer.
    """

    def __init__(self, path: str, decoder: Optional[DecoderConfig] = None,
                 cancel: Optional[threading.Event] = None, queue_size: int = 8,
                 crop: Optional[Tuple[int, int]] = None,
                 want_preview: Optional[Callable[[], bool]] = None,
                 start_frame: int = 0):
        self.path = path
        self.decoder = decoder
        self.cancel = cancel or threading.Event()
        self._stop_ev = threading.Event()
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self.crop = crop
        self.want_preview = want_preview
        self.start_frame = start_frame
        self.decode_info: Optional[DecodeInfo] = None
        self._error: Optional[BaseException] = None
        self._opened = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FrameReader", daemon=True)

    # ------------------------------------------------------------------ thread
    def start(self) -> "FrameReader":
        self._thread.start()
        self._opened.wait(timeout=30)
        return self

    def _stopped(self) -> bool:
        return self._stop_ev.is_set() or self.cancel.is_set()

    def _put(self, item) -> bool:
        while not self._stopped():
            try:
                self.q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        cap = None
        try:
            cap, self.decode_info = open_capture(self.path, self.decoder)
            self._opened.set()
            if not cap.isOpened():
                raise IOError(f"cannot open video: {self.path}")
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            idx = self.start_frame
            while not self._stopped():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                preview = None
                if self.want_preview is not None and self.want_preview():
                    preview = make_preview(frame)
                if self.crop is not None:
                    data = np.ascontiguousarray(frame[:, self.crop[0]:self.crop[1]])
                    del frame
                else:
                    data = frame
                if not self._put((idx, data, preview)):
                    break
                idx += 1
        except BaseException as e:  # forwarded to the consumer
            self._error = e
        finally:
            self._opened.set()
            if cap is not None:
                cap.release()
            # the sentinel must get through unless we were cancelled
            while True:
                try:
                    self.q.put(_EOF, timeout=0.1)
                    break
                except queue.Full:
                    if self._stopped():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass

    # ------------------------------------------------------------------ consumer
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        while True:
            item = self.q.get()
            if item is _EOF:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def stop(self) -> None:
        self._stop_ev.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)


def read_frame_at(cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = cap.read()
    return frame if ok else None
