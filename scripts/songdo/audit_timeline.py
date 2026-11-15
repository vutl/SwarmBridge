import glob, sys, json
from pathlib import Path
import numpy as np, pandas as pd
segs = []
cover = {}
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*" / "files" / "*PM5.csv"))):
    X = Path(f).stem.split("_")[1]
    d = pd.read_csv(f, usecols=["Local_Time", "Drone_ID"])
    h, m, s = d["Local_Time"].str.split(":", expand=True).T.values
    d["sec"] = np.floor(h.astype(float) * 3600 + m.astype(float) * 60 + s.astype(float)).astype(int)
    secs = np.sort(d["sec"].unique())
    gaps = np.diff(secs); cover[X] = dict(seconds=int(len(secs)), span=int(secs[-1] - secs[0] + 1),
                                          gaps_over_10s=int((gaps > 10).sum()), longest_gap_s=int(gaps.max()) if len(gaps) else 0)
    for k, g in d.groupby("Drone_ID"):
        u = np.sort(g["sec"].unique()); br = np.where(np.diff(u) > 10)[0]
        starts = np.r_[u[0], u[br + 1]]; ends = np.r_[u[br], u[-1]]
        for a, b in zip(starts, ends):
            segs.append((int(k), X, int(a), int(b)))
print("per intersection coverage:", json.dumps(cover))
t0 = min(s[2] for s in segs)
for k in range(1, 11):
    row = sorted((a, b, X) for kk, X, a, b in segs if kk == k)
    print("D%-2d" % k, " ".join("%s[%d-%d]" % (X, (a - t0) // 60, (b - t0) // 60) for a, b, X in row))
