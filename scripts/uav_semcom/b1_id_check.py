"""B1 condition 1 (reports/swarmbridge-a8-b1-preregistration.md): are GT track ids consistent across the two UAV
views? Labels only (no image, no detector, no scorer). Two metadata checks per sequence, each with a shuffled control:
  class    share of ids present in both views (same epoch) that carry the same class in both
  geometry per epoch with >= 12 common ids: split the common ids at random in two halves, fit a homography view1 ->
           view2 on one half (bottom-centre of the boxes, RANSAC), and count the held-out ids whose reprojection error
           is <= TOL px. Control: the same after permuting the view-2 points among the ids.
Pass rule (pre-registered): class agreement >= 0.95, median held-out inlier share >= 0.70 and >= 0.40 above control.
TOL = 40 px (1920 x 1080 frames; fixed before running; the ground-plane model is only approximate).
Usage: python -I b1_id_check.py MDMT_ZIP CODE_DIR OUT_JSON
"""
import json
import sys

import numpy as np

TOL, MIN_COMMON, STEP, MAX_EPOCHS = 40.0, 12, 10, 60


def foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def heldout_share(p1, p2, rng):
    import cv2
    n = len(p1)
    perm = rng.permutation(n)
    fit, test = perm[: n // 2], perm[n // 2:]
    H, _ = cv2.findHomography(p1[fit], p2[fit], cv2.RANSAC, TOL)
    if H is None:
        return 0.0
    q = cv2.perspectiveTransform(p1[test].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float((np.linalg.norm(q - p2[test], axis=1) <= TOL).mean())


def main():
    sys.path.insert(0, sys.argv[2])
    import mdmt_adapter as M
    ds = M.MDMT(sys.argv[1])                                   # labels only; no rights note -> no pixels
    rng = np.random.default_rng(0)
    out = {}
    for s in ds.pilot_order("test"):
        g1, g2 = ds.labels(s, 1), ds.labels(s, 2)
        same = tot = 0
        inl, ctl, n_common, n_ep = [], [], [], 0
        for f in ds.epochs(s, STEP, MAX_EPOCHS):
            a, b = g1.get(f, {}), g2.get(f, {})
            common = sorted(set(a) & set(b))
            n_common.append(len(common))
            same += sum(a[i][4] == b[i][4] for i in common); tot += len(common)
            if len(common) >= MIN_COMMON:
                n_ep += 1
                p1 = np.array([foot(a[i]) for i in common], np.float32)
                p2 = np.array([foot(b[i]) for i in common], np.float32)
                inl.append(heldout_share(p1, p2, rng))
                ctl.append(heldout_share(p1, p2[rng.permutation(len(p2))], rng))
        ids1 = {i for f in g1 for i in g1[f]}; ids2 = {i for f in g2 for i in g2[f]}
        r = dict(class_agree=same / max(tot, 1), n_common_pairs=tot, mean_common_per_epoch=float(np.mean(n_common)),
                 ids_view1=len(ids1), ids_view2=len(ids2), ids_both=len(ids1 & ids2), epochs_geom=n_ep,
                 inlier_median=float(np.median(inl)) if inl else None,
                 inlier_p10=float(np.percentile(inl, 10)) if inl else None,
                 control_median=float(np.median(ctl)) if ctl else None)
        r["pass"] = bool(inl and r["class_agree"] >= 0.95 and r["inlier_median"] >= 0.70
                         and r["inlier_median"] - r["control_median"] >= 0.40)
        out[str(s)] = r
        print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = dict(tol_px=TOL, min_common=MIN_COMMON, sequences=out, n_pass=sum(v["pass"] for v in out.values()))
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    print("PASS", res["n_pass"], "/", len(out))


if __name__ == "__main__":
    main()
