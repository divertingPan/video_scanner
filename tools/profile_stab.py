import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from core.config import StabilizerConfig
from core.stabilizer import Stabilizer

video = sys.argv[1]
nthreads = int(sys.argv[2]) if len(sys.argv) > 2 else -1
if nthreads >= 0:
    cv2.setNumThreads(nthreads)
cap = cv2.VideoCapture(video)
frames = []
t = time.perf_counter()
for i in range(120):
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
dec = time.perf_counter() - t
print(f"decode {len(frames)} frames: {len(frames)/dec:.1f} fps, cv threads={cv2.getNumThreads()}")
for det in ("orb", "sift"):
    cfg = StabilizerConfig(enabled=True, detector=det)
    st = Stabilizer(cfg, frames[0])
    T = {k: 0.0 for k in ("prep", "detect", "match", "fit", "total")}
    d = st._detector(); m = st._matcher()
    for f in frames[1:]:
        t0 = time.perf_counter(); g = st._prep(f); t1 = time.perf_counter()
        kps, des = d.detectAndCompute(g, st.det_mask); t2 = time.perf_counter()
        knn = m.knnMatch(des, k=2); t3 = time.perf_counter()
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        cur = np.float32([kps[x.queryIdx].pt for x in good]); anc = st.anchor_pts[[x.trainIdx for x in good]]
        st._fit(anc, cur); t4 = time.perf_counter()
        T["prep"] += t1 - t0; T["detect"] += t2 - t1; T["match"] += t3 - t2; T["fit"] += t4 - t3
    n = len(frames) - 1
    t0 = time.perf_counter()
    for f in frames[1:]:
        st.estimate(f)
    T["total"] = time.perf_counter() - t0
    print(det, {k: f"{v / n * 1000:.1f}ms" for k, v in T.items()}, f"kps~{len(kps)} good~{len(good)}")
