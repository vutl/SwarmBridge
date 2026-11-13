"""POST-HOC: RANSAC homography on ALL common ids of a frame (no hold-out), inlier share true vs permuted, at two
tolerances; plus the same with a frame offset between the drones (is a time shift hiding the geometry?).
Usage: python -I m3ot_id_diag2.py M3OT_FILES_DIR"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m3ot_id_check as C


def share(p1, p2, tol):
    import cv2
    H, m = cv2.findHomography(p1, p2, cv2.RANSAC, tol)
    return 0.0 if H is None else float(m.mean())


def main():
    R, rng = Path(sys.argv[1]), np.random.default_rng(0)
    for d1 in sorted((R / "1" / "rgb").glob("*/1-*")):
        split, n = d1.parent.name, d1.name.split("-")[1]
        g1, g2 = C.load(d1 / "gt" / "gt.txt"), C.load(R / "2" / "rgb" / split / f"2-{n}" / "gt" / "gt.txt")
        frames = sorted(set(g1) & set(g2))
        samp = [frames[k] for k in np.unique(np.linspace(0, len(frames) - 1, 60).astype(int))]
        out = {}
        for tol in (C.TOL, 40.0):
            for off in (0, 10, 50):
                t, c = [], []
                for f in samp:
                    a, b = g1.get(f, {}), g2.get(f + off, {})
                    com = sorted(set(a) & set(b))
                    if len(com) >= 10:
                        p1 = np.array([C.foot(a[i]) for i in com], np.float32); p2 = np.array([C.foot(b[i]) for i in com], np.float32)
                        t.append(share(p1, p2, tol)); c.append(share(p1, p2[rng.permutation(len(p2))], tol))
                out[(round(tol), off)] = (round(float(np.median(t)), 2) if t else None, round(float(np.median(c)), 2) if c else None, len(t))
        print(n, split, out, flush=True)


if __name__ == "__main__":
    main()
