import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*" / "files" / "*PM5.csv"))):
    X = Path(f).stem.split("_")[1]
    d = pd.read_csv(f, usecols=["Local_Time", "Drone_ID"])
    if d["Drone_ID"].nunique() < 2:
        continue
    h, m, s = d["Local_Time"].str.split(":", expand=True).T.values
    sec = np.floor(h.astype(float) * 3600 + m.astype(float) * 60 + s.astype(float)).astype(int)
    d["sec"] = sec
    per = d.groupby("sec")["Drone_ID"].nunique()
    occ = {int(k): int(g["sec"].nunique()) for k, g in d.groupby("Drone_ID")}
    print(X, "seconds per drone", occ, "| seconds with >=2 drones", int((per >= 2).sum()), "of", len(per), flush=True)
