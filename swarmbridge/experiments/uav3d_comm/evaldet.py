"""UAV3D detection metric, re-implemented to match the official UAV3D evaluator
(collaborative_perception/bevfusion/projects/{loaders,evaluate}.py + nuScenes calc_ap):

- GT = every annotation whose centre is inside the +-102.4 m square around UAV0 (CAMERA_BOTTOM_id_0 position, z=0).
  No visibility filter, so occluded or unseen cars count as GT.
- Predictions are also restricted to that square (official: ego_dist < 150, a superset that never binds for preds
  produced inside the BEV grid).
- Greedy matching by xy centre distance, predictions sorted by score; AP for thresholds 0.5/1/2/4 m with
  min_recall = min_precision = 0.1 (nuScenes calc_ap). mAP = mean over the 4 thresholds (single class 'car').
"""
import json
from pathlib import Path
import numpy as np

HALF = 102.4
DIST_THS = (0.5, 1.0, 2.0, 4.0)


class MiniDB:
    """GT and UAV0 positions from the v1.0-mini JSON tables."""

    def __init__(self, root):
        meta = Path(root) / "v1.0-mini"
        load = lambda n: json.loads((meta / f"{n}.json").read_text())
        self.scenes = {s["token"]: s for s in load("scene")}
        self.samples = {s["token"]: s for s in load("sample")}
        cals = {c["token"]: c for c in load("calibrated_sensor")}
        self.uav0 = {}
        for d in load("sample_data"):
            if "CAMERA_BOTTOM_id_0/" in d["filename"]:
                self.uav0[d["sample_token"]] = np.array(cals[d["calibrated_sensor_token"]]["translation"][:2])
        self.gt = {t: [] for t in self.samples}
        for a in load("sample_annotation"):
            self.gt[a["sample_token"]].append(a)
        self.scene_name = {t: self.scenes[s["scene_token"]]["name"] for t, s in self.samples.items()}

    def tokens(self, scene_names):
        return [t for t, n in self.scene_name.items() if n in set(scene_names)]

    def gt_xy(self, token):
        xy = np.array([a["translation"][:2] for a in self.gt[token]]).reshape(-1, 2)
        return xy[in_square(xy, self.uav0[token])]


def in_square(xy, centre):
    d = np.abs(np.asarray(xy).reshape(-1, 2) - centre)
    return (d[:, 0] < HALF) & (d[:, 1] < HALF)


def calc_ap(tp_sorted, n_gt, min_recall=0.1, min_precision=0.1):
    if n_gt == 0 or len(tp_sorted) == 0:
        return 0.0
    tp = np.cumsum(tp_sorted).astype(float)
    fp = np.cumsum(1 - tp_sorted).astype(float)
    prec = tp / (tp + fp)
    rec = tp / n_gt
    rec_interp = np.linspace(0, 1, 101)
    prec_i = np.interp(rec_interp, rec, prec, right=0)
    # nuScenes: precision envelope is taken on the interpolated curve after accumulate()
    prec_i = np.maximum.accumulate(prec_i[::-1])[::-1]
    prec_i = prec_i[round(100 * min_recall) + 1:]
    prec_i = np.clip(prec_i - min_precision, 0, None)
    return float(np.mean(prec_i) / (1.0 - min_precision))


def evaluate(db, preds, tokens):
    """preds: {token: (xy (N,2) global, score (N,))}. Returns dict with AP per threshold and mAP."""
    out = {}
    n_gt = sum(len(db.gt_xy(t)) for t in tokens)
    for th in DIST_THS:
        rows = []  # (score, tp)
        for t in tokens:
            gxy = db.gt_xy(t)
            pxy, ps = preds.get(t, (np.zeros((0, 2)), np.zeros(0)))
            keep = in_square(pxy, db.uav0[t])
            pxy, ps = np.asarray(pxy)[keep], np.asarray(ps)[keep]
            order = np.argsort(-ps)
            taken = np.zeros(len(gxy), bool)
            for i in order:
                if len(gxy):
                    d = np.linalg.norm(gxy - pxy[i], axis=1)
                    d[taken] = np.inf
                    j = int(np.argmin(d))
                    hit = d[j] < th
                    if hit:
                        taken[j] = True
                else:
                    hit = False
                rows.append((ps[i], hit))
        rows.sort(key=lambda r: -r[0])
        out[f"ap_{th}"] = calc_ap(np.array([r[1] for r in rows], float), n_gt)
    out["mAP"] = float(np.mean([out[f"ap_{th}"] for th in DIST_THS]))
    out["n_gt"] = n_gt
    return out


def load_nusc_results(path, tokens=None):
    R = json.loads(Path(path).read_text())["results"]
    out = {}
    for t, boxes in R.items():
        if tokens is not None and t not in tokens:
            continue
        out[t] = (np.array([b["translation"][:2] for b in boxes]).reshape(-1, 2),
                  np.array([b["detection_score"] for b in boxes]))
    return out


if __name__ == "__main__":
    import sys
    db = MiniDB(sys.argv[1])
    held = db.tokens(["scene_11", "scene_13", "scene_15"])
    for f in sys.argv[2:]:
        print(Path(f).name, evaluate(db, load_nusc_results(f, set(held)), held))
