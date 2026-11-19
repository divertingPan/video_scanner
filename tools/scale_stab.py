import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.config import StabilizerConfig
from core.stabilizer import Stabilizer

video = sys.argv[1]
cvt = int(sys.argv[2]) if len(sys.argv) > 2 else -1
if cvt >= 0:
    cv2.setNumThreads(cvt)
cap = cv2.VideoCapture(video)
frames = []
for i in range(100):
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
st = Stabilizer(StabilizerConfig(enabled=True), frames[0])
for n in (1, 2, 4, 8, 12):
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(st.estimate, frames[:n * 2]))  # warm-up per-thread objects
        t = time.perf_counter()
        list(ex.map(st.estimate, frames))
        dt = time.perf_counter() - t
    print(f"cv threads={cv2.getNumThreads()} workers={n}: {len(frames) / dt:.1f} fps")
