"""Songdo Traffic coverage / concurrency / cross-drone overlap audit (v2, review #25 R5).

ACCESS LABEL: the Oct 7 PM5 session was inspected for feasibility on 2026-11-15 and belongs to the proposed test day ->
its statistics are AUDITED-TEST and must not be used to choose designs or thresholds.

Per intersection CSV: rows, vehicles (ids unique within a CSV only), drones, coverage seconds, gaps.
Per pair of (intersection, drone) streams with a different drone:
  span_intersection_s          length of the intersection of the two [min t, max t] extents (NOT concurrent observation)
  concurrent_observed_seconds  seconds in which BOTH streams have at least one row
  cross-drone vehicle matches  on concurrent seconds only, at 1 Hz: one-to-one mutual nearest neighbours of the same
                               class within MATCH_M metres in the common georeferenced frame; a vehicle pair is accepted
                               if matched in >= MIN_SECONDS distinct seconds. Distance statistics use ACCEPTED pairs only.
                               collision_rate = share of candidate points whose nearest neighbour is not mutual.
No cross-drone identity is provided by the dataset; matches are geometric estimates, not validated identities.
Usage: python audit_overlap.py CSV_ROOT OUT_JSON [LABEL]"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_M, MIN_SECONDS, GAP_S = 2.0, 3, 10
COLS = ["Vehicle_ID", "Local_Time", "Drone_ID", "Local_X", "Local_Y", "Vehicle_Class", "Vehicle_Speed"]


def load(f):
    d = pd.read_csv(f, usecols=COLS)
    h, m, s = d["Local_Time"].str.split(":", expand=True).T.values
    d["t"] = h.astype(float) * 3600 + m.astype(float) * 60 + s.astype(float)
    d["sec"] = np.floor(d["t"]).astype(int)
    return d.drop(columns=["Local_Time"])


def coverage(secs):
    secs = np.sort(np.unique(secs))
    g = np.diff(secs)
    return dict(seconds=int(len(secs)), span_s=int(secs[-1] - secs[0] + 1) if len(secs) else 0,
                gaps_over_10s=int((g > GAP_S).sum()), longest_gap_s=int(g.max()) if len(g) else 0)


def match(a, b, common):
    a = a[a["sec"].isin(common)].drop_duplicates(["Vehicle_ID", "sec"])
    b = b[b["sec"].isin(common)].drop_duplicates(["Vehicle_ID", "sec"])
    bg = dict(tuple(b.groupby("sec")))
    hits, dist, cand, coll = {}, {}, 0, 0
    for s, ga in a.groupby("sec"):
        gb = bg.get(s)
        if gb is None:
            continue
        P, Q = ga[["Local_X", "Local_Y"]].values, gb[["Local_X", "Local_Y"]].values
        D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        D[ga["Vehicle_Class"].values[:, None] != gb["Vehicle_Class"].values[None, :]] = np.inf
        r, c = D.argmin(1), D.argmin(0)
        for i in range(len(P)):
            if D[i, r[i]] > MATCH_M:
                continue
            cand += 1
            if c[r[i]] != i:
                coll += 1
                continue
            key = (int(ga["Vehicle_ID"].values[i]), int(gb["Vehicle_ID"].values[r[i]]))
            hits[key] = hits.get(key, 0) + 1
            dist.setdefault(key, []).append(float(D[i, r[i]]))
    acc = [k for k, n in hits.items() if n >= MIN_SECONDS]
    dd = [x for k in acc for x in dist[k]]
    return dict(accepted_pairs=len(acc), candidate_points=cand, collision_rate=coll / cand if cand else None,
                accepted_median_dist_m=float(np.median(dd)) if dd else None,
                accepted_p90_dist_m=float(np.percentile(dd, 90)) if dd else None,
                share_a_vehicles_matched=len({k[0] for k in acc}) / max(a["Vehicle_ID"].nunique(), 1))


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "UNLABELLED"
    per, streams = {}, {}
    for f in sorted(glob.glob(str(root / "*" / "files" / "*.csv"))):
        X = Path(f).stem.split("_")[1]
        d = load(f)
        per[X] = dict(rows=len(d), vehicles=int(d["Vehicle_ID"].nunique()),
                      drones={int(k): int(v) for k, v in d["Drone_ID"].value_counts().items()},
                      coverage=coverage(d["sec"].values),
                      concurrent_multi_drone_seconds=int((d.groupby("sec")["Drone_ID"].nunique() >= 2).sum()),
                      t0=float(d["t"].min()), t1=float(d["t"].max()))
        for k, g in d.groupby("Drone_ID"):
            streams[(X, int(k))] = g
        print(X, per[X], flush=True)
    keys = sorted(streams)
    pairs = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if k1[1] == k2[1]:
                continue
            a, b = streams[k1], streams[k2]
            span = max(0.0, min(a["t"].max(), b["t"].max()) - max(a["t"].min(), b["t"].min()))
            common = np.intersect1d(a["sec"].unique(), b["sec"].unique())
            if span <= 0 or not len(common):
                continue
            r = dict(span_intersection_s=float(span), concurrent_observed_seconds=int(len(common)), **match(a, b, common))
            pairs[f"{k1[0]}-D{k1[1]}|{k2[0]}-D{k2[1]}"] = r
            print(k1, k2, r, flush=True)
    json.dump(dict(access_label=label, params=dict(match_m=MATCH_M, min_seconds=MIN_SECONDS, gap_s=GAP_S),
                   per_intersection=per, cross_drone_pairs_with_concurrent_seconds=pairs), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
