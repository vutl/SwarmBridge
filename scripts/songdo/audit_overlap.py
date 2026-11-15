"""Songdo Traffic feasibility audit on the Oct 7 PM5 session of all intersections (trajectory CSVs only).
Per CSV: rows, vehicles, Drone_IDs, time span, spatial extent. Across drones: time overlap, spatial overlap and
cross-drone same-vehicle evidence from the common georeferenced frame (no GT identity exists across CSVs/drones;
matching is purely by position, time and class, so this is an estimate).
Usage: python songdo_audit.py CSV_ROOT OUT_JSON"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), sys.argv[2]
COLS = ["Vehicle_ID", "Local_Time", "Drone_ID", "Local_X", "Local_Y", "Vehicle_Class", "Vehicle_Speed", "Visibility"]
MATCH_M, MIN_SECONDS, SAMPLE_HZ = 2.0, 3.0, 1.0


def tsec(s):
    h, m, x = s.str.split(":", expand=True).T.values
    return h.astype(float) * 3600 + m.astype(float) * 60 + x.astype(float)


def load(f):
    d = pd.read_csv(f, usecols=COLS)
    d["t"] = tsec(d["Local_Time"])
    return d.drop(columns=["Local_Time"])


tracks, per = {}, {}
for f in sorted(glob.glob(str(root / "*" / "files" / "*PM5.csv"))):
    X = Path(f).stem.split("_")[1]
    d = load(f)
    tracks[X] = d
    xs, ys = d["Local_X"].values, d["Local_Y"].values
    per[X] = dict(rows=len(d), vehicles=int(d["Vehicle_ID"].nunique()),
                  drones={int(k): int(v) for k, v in d["Drone_ID"].value_counts().items()},
                  t0=float(d["t"].min()), t1=float(d["t"].max()),
                  bbox=[float(np.percentile(xs, 1)), float(np.percentile(ys, 1)), float(np.percentile(xs, 99)), float(np.percentile(ys, 99))],
                  classes={int(k): int(v) for k, v in d.groupby("Vehicle_ID")["Vehicle_Class"].first().value_counts().items()},
                  median_speed_kmh=float(d["Vehicle_Speed"].median()))
    print(X, {k: v for k, v in per[X].items() if k not in ("bbox",)}, flush=True)

# per (intersection, drone) streams
streams = {}
for X, d in tracks.items():
    for dr, g in d.groupby("Drone_ID"):
        streams[(X, int(dr))] = g


def overlap(b1, b2):
    w = min(b1[2], b2[2]) - max(b1[0], b2[0]); h = min(b1[3], b2[3]) - max(b1[1], b2[1])
    return max(w, 0) * max(h, 0)


def match_streams(a, b):
    """same-vehicle evidence between two streams: at 1 Hz samples, nearest same-class point within MATCH_M metres;
    a vehicle pair counts if it is matched in >= MIN_SECONDS distinct seconds."""
    t0, t1 = max(a["t"].min(), b["t"].min()), min(a["t"].max(), b["t"].max())
    if t1 - t0 < MIN_SECONDS:
        return dict(time_overlap_s=max(t1 - t0, 0.0), pairs=0)
    a = a[(a["t"] >= t0) & (a["t"] <= t1)].copy(); b = b[(b["t"] >= t0) & (b["t"] <= t1)].copy()
    a["s"] = np.floor(a["t"] * SAMPLE_HZ); b["s"] = np.floor(b["t"] * SAMPLE_HZ)
    a = a.drop_duplicates(["Vehicle_ID", "s"]); b = b.drop_duplicates(["Vehicle_ID", "s"])
    hits, dists = {}, []
    bg = dict(tuple(b.groupby("s")))
    for s, ga in a.groupby("s"):
        gb = bg.get(s)
        if gb is None:
            continue
        P, Q = ga[["Local_X", "Local_Y"]].values, gb[["Local_X", "Local_Y"]].values
        D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        D[ga["Vehicle_Class"].values[:, None] != gb["Vehicle_Class"].values[None, :]] = np.inf
        j = D.argmin(1)
        ok = D[np.arange(len(P)), j] <= MATCH_M
        for i in np.where(ok)[0]:
            key = (int(ga["Vehicle_ID"].values[i]), int(gb["Vehicle_ID"].values[j[i]]))
            hits[key] = hits.get(key, 0) + 1
            dists.append(float(D[i, j[i]]))
    pairs = [k for k, n in hits.items() if n >= MIN_SECONDS]
    va = set(k[0] for k in pairs)
    return dict(time_overlap_s=float(t1 - t0), pairs=len(pairs), vehicles_a=int(a["Vehicle_ID"].nunique()),
                vehicles_b=int(b["Vehicle_ID"].nunique()), share_a_matched=len(va) / max(a["Vehicle_ID"].nunique(), 1),
                median_dist_m=float(np.median(dists)) if dists else None)


keys = sorted(streams)
bb = {k: [np.percentile(streams[k]["Local_X"], 1), np.percentile(streams[k]["Local_Y"], 1),
          np.percentile(streams[k]["Local_X"], 99), np.percentile(streams[k]["Local_Y"], 99)] for k in keys}
pairs = {}
for i, k1 in enumerate(keys):
    for k2 in keys[i + 1:]:
        if k1[1] == k2[1]:
            continue                                   # same drone in two intersection files: not cross-drone
        if overlap(bb[k1], bb[k2]) <= 0:
            continue
        r = match_streams(streams[k1], streams[k2])
        if r["pairs"]:
            pairs[f"{k1[0]}-D{k1[1]}|{k2[0]}-D{k2[1]}"] = r
            print(k1, k2, r, flush=True)
json.dump(dict(per_csv=per, streams=[f"{k[0]}-D{k[1]}" for k in keys], cross_drone=pairs,
               params=dict(match_m=MATCH_M, min_seconds=MIN_SECONDS, sample_hz=SAMPLE_HZ)), open(out, "w"), indent=1)
print("streams", len(keys), "cross-drone stream pairs with shared vehicles", len(pairs))
