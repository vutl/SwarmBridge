"""POST-HOC diagnostics for the M3OT cross-drone id screening (labels only). Not part of the pre-fixed rule.
 (1) presence: for each id, IoU of its presence intervals (set of frames) in drone 1 and drone 2, vs the same for the
     id paired with a random other id (geometry-free; works with few objects).
 (2) affine held-out: like the homography test but a 6-dof affine model (3-point minimal set), min 8 common ids.
 (3) motion: per id present in both drones over >= 20 common frames, correlation of its speed profile (|displacement|
     over 5 frames) between the drones, vs a random other id.
Usage: python -I m3ot_id_diag.py M3OT_FILES_DIR"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m3ot_id_check as C

TOL = C.TOL


def affine_heldout(p1, p2, rng):
    import cv2
    n = len(p1)
    perm = rng.permutation(n)
    fit, test = perm[: n // 2], perm[n // 2:]
    A, _ = cv2.estimateAffine2D(p1[fit], p2[fit], method=cv2.RANSAC, ransacReprojThreshold=TOL)
    if A is None:
        return 0.0
    q = p1[test] @ A[:, :2].T + A[:, 2]
    return float((np.linalg.norm(q - p2[test], axis=1) <= TOL).mean())


def speed(g, i, frames, k=5):
    c = {f: np.array(C.foot(g[f][i])) for f in frames}
    return np.array([np.linalg.norm(c[f + k] - c[f]) for f in frames if f + k in c])


def main():
    R, rng = Path(sys.argv[1]), np.random.default_rng(0)
    for d1 in sorted((R / "1" / "rgb").glob("*/1-*")):
        split, n = d1.parent.name, d1.name.split("-")[1]
        g1, g2 = C.load(d1 / "gt" / "gt.txt"), C.load(R / "2" / "rgb" / split / f"2-{n}" / "gt" / "gt.txt")
        pres1, pres2 = {}, {}
        for g, pres in ((g1, pres1), (g2, pres2)):
            for f in g:
                for i in g[f]:
                    pres.setdefault(i, set()).add(f)
        ids = sorted(set(pres1) & set(pres2))
        iou = lambda a, b: len(a & b) / max(len(a | b), 1)
        same = [iou(pres1[i], pres2[i]) for i in ids]
        perm = rng.permutation(len(ids))
        other = [iou(pres1[i], pres2[ids[j]]) for i, j in zip(ids, perm) if ids[j] != i]
        frames = sorted(set(g1) & set(g2))
        samp = [frames[k] for k in np.unique(np.linspace(0, len(frames) - 1, 60).astype(int))]
        inl, ctl = [], []
        for f in samp:
            com = sorted(set(g1[f]) & set(g2[f]))
            if len(com) >= 8:
                p1 = np.array([C.foot(g1[f][i]) for i in com], np.float64); p2 = np.array([C.foot(g2[f][i]) for i in com], np.float64)
                inl.append(affine_heldout(p1, p2, rng)); ctl.append(affine_heldout(p1, p2[rng.permutation(len(p2))], rng))
        cs, cc = [], []
        for a, i in enumerate(ids):
            fr = sorted(pres1[i] & pres2[i])
            j = ids[perm[a]]
            if len(fr) >= 20:
                s1, s2 = speed(g1, i, fr), speed(g2, i, fr)
                if len(s1) >= 10 and s1.std() > 0 and s2.std() > 0:
                    cs.append(np.corrcoef(s1, s2)[0, 1])
            fr = sorted(pres1[i] & pres2[j]) if j != i else []
            if len(fr) >= 20:
                s1, s2 = speed(g1, i, fr), speed(g2, j, fr)
                if len(s1) >= 10 and s1.std() > 0 and s2.std() > 0:
                    cc.append(np.corrcoef(s1, s2)[0, 1])
        med = lambda x: round(float(np.median(x)), 3) if len(x) else None
        print(n, split, "presence IoU same/other", med(same), med(other), "| affine heldout", med(inl), med(ctl), len(inl),
              "| speed corr same/other", med(cs), med(cc), len(cs), len(cc), flush=True)


if __name__ == "__main__":
    main()
