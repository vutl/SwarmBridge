"""Per (intersection, drone) stream: time span, spatial extent; for multi-drone CSVs, time overlap between drones
and whether the same Vehicle_ID is reported by both drones (Vehicle_ID is unique within a CSV).
Usage: python songdo_spans.py CSV_ROOT"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*" / "files" / "*PM5.csv"))):
    X = Path(f).stem.split("_")[1]
    d = pd.read_csv(f, usecols=["Vehicle_ID", "Local_Time", "Drone_ID", "Local_X", "Local_Y"])
    if d["Drone_ID"].nunique() < 2:
        continue
    h, m, s = d["Local_Time"].str.split(":", expand=True).T.values
    d["t"] = h.astype(float) * 3600 + m.astype(float) * 60 + s.astype(float)
    spans = {int(k): (round(g["t"].min() / 60, 1), round(g["t"].max() / 60, 1), int(g["Vehicle_ID"].nunique()),
                      [round(float(np.percentile(g[c], p))) for c in ("Local_X", "Local_Y") for p in (5, 95)])
             for k, g in d.groupby("Drone_ID")}
    ids = d.groupby("Vehicle_ID")["Drone_ID"].nunique()
    both = int((ids > 1).sum())
    # simultaneous rows of different drones at the same instant for one vehicle id
    dup = d.groupby(["Vehicle_ID", "t"])["Drone_ID"].nunique()
    print(X, "spans(min) / vehicles / x5,x95,y5,y95:", spans, "| ids reported by >1 drone:", both,
          "| (id, t) rows with >1 drone:", int((dup > 1).sum()), flush=True)
