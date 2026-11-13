"""M3OT cross-drone id screening on LABELS ONLY (gt.txt, MOT format). Same method as scripts/uav_semcom/b1_id_check.py:
per sampled frame with >= MIN_COMMON ids present in both drones, fit a homography drone1 -> drone2 on half of the common
ids (bottom-centre of the boxes, RANSAC) and count held-out ids within TOL px; control = view-2 points permuted.
TOL = 40 px * 640 / 1920 = 13.3 px (the MDMT tolerance scaled to the 640 x 512 frames; fixed before running).
Usage: python -I m3ot_id_check.py M3OT_FILES_DIR OUT_JSON"""
import configparser
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TOL, MIN_COMMON, MAX_EPOCHS = 40.0 * 640 / 1920, 12, 60


def load(p):
    g = defaultdict(dict)
    for line in open(p):
        x = line.strip().split(",")
        if len(x) < 6:
            continue
        f, i = int(float(x[0])), int(float(x[1]))
        l, t, w, h = map(float, x[2:6])
        g[f][i] = (l, t, l + w, t + h)
    return g


def heldout(p1, p2, rng):
    import cv2
    n = len(p1)
    perm = rng.permutation(n)
    fit, test = perm[: n // 2], perm[n // 2:]
    H, _ = cv2.findHomography(p1[fit], p2[fit], cv2.RANSAC, TOL)
    if H is None:
        return 0.0
    q = cv2.perspectiveTransform(p1[test].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float((np.linalg.norm(q - p2[test], axis=1) <= TOL).mean())


def foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def seqinfo(d):
    c = configparser.ConfigParser(); c.read(d / "seqinfo.ini")
    return {k: c["Sequence"][k] for k in c["Sequence"]}


def main():
    R, rng, out = Path(sys.argv[1]), np.random.default_rng(0), {}
    for d1 in sorted((R / "1" / "rgb").glob("*/1-*")):
        split, n = d1.parent.name, d1.name.split("-")[1]
        d2 = R / "2" / "rgb" / split / f"2-{n}"
        g1, g2 = load(d1 / "gt" / "gt.txt"), load(d2 / "gt" / "gt.txt")
        frames = sorted(set(g1) | set(g2))
        samp = [frames[k] for k in np.unique(np.linspace(0, len(frames) - 1, MAX_EPOCHS).astype(int))]
        inl, ctl, ncom, n1, n2 = [], [], [], [], []
        for f in samp:
            a, b = g1.get(f, {}), g2.get(f, {})
            com = sorted(set(a) & set(b))
            ncom.append(len(com)); n1.append(len(a)); n2.append(len(b))
            if len(com) >= MIN_COMMON:
                p1 = np.array([foot(a[i]) for i in com], np.float32); p2 = np.array([foot(b[i]) for i in com], np.float32)
                inl.append(heldout(p1, p2, rng)); ctl.append(heldout(p1, p2[rng.permutation(len(p2))], rng))
        ids1 = {i for f in g1 for i in g1[f]}; ids2 = {i for f in g2 for i in g2[f]}
        wh = np.array([(b[2] - b[0], b[3] - b[1]) for f in samp for b in g1.get(f, {}).values()] or [(0, 0)])
        # RGB vs IR of the same drone: same ids? box offset for common ids at the same frame
        ir = d1.parents[2] / "ir" / split / f"1-{n}T" / "gt" / "gt.txt"
        gi = load(ir) if ir.exists() else {}
        idsi = {i for f in gi for i in gi[f]}
        off = [np.hypot(*(np.array(foot(g1[f][i])) - np.array(foot(gi[f][i]))))
               for f in samp for i in set(g1.get(f, {})) & set(gi.get(f, {}))]
        s1, s2 = seqinfo(d1), seqinfo(d2)
        r = dict(split=split, len1=int(s1["seqlength"]), len2=int(s2["seqlength"]), size=f'{s1["imwidth"]}x{s1["imheight"]}',
                 fps=s1["framerate"], frames_gt1=len(g1), frames_gt2=len(g2), ids1=len(ids1), ids2=len(ids2),
                 ids_both=len(ids1 & ids2), obj_per_frame=(float(np.mean(n1)), float(np.mean(n2))),
                 common_per_frame=float(np.mean(ncom)), epochs_geom=len(inl),
                 inlier_median=float(np.median(inl)) if inl else None, control_median=float(np.median(ctl)) if ctl else None,
                 box_wh_median=[float(x) for x in np.median(wh, axis=0)],
                 ir_ids=len(idsi), ir_ids_shared_with_rgb=len(idsi & ids1),
                 rgb_ir_foot_offset_px_median=float(np.median(off)) if off else None)
        r["pass"] = bool(inl and r["inlier_median"] >= 0.70 and r["inlier_median"] - r["control_median"] >= 0.40)
        out[n] = r
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(dict(tol_px=TOL, min_common=MIN_COMMON, sequences=out, n_pass=sum(v["pass"] for v in out.values())),
              open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
