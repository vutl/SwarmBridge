"""Draw projected GT boxes on a few camera images to check the geometry by eye."""
import json, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
from uavgeo import project, box_corners

root = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
meta = root / "v1.0-mini"
load = lambda n: json.loads((meta / f"{n}.json").read_text())
samples, sds, cals, anns = load("sample"), load("sample_data"), load("calibrated_sensor"), load("sample_annotation")
cal_of = {c["token"]: c for c in cals}
s = samples[0]
A = [a for a in anns if a["sample_token"] == s["token"]]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
for d in sds:
    if d["sample_token"] != s["token"] or not any(k in d["filename"] for k in ("FRONT_id_0", "BOTTOM_id_0", "LEFT_id_2")):
        continue
    cal = cal_of[d["calibrated_sensor_token"]]
    img = Image.open(root / "sweeps" / Path(d["filename"]).parent.name / Path(d["filename"]).name).convert("RGB")
    dr = ImageDraw.Draw(img)
    for a in A:
        for zb, col in ((False, (255, 0, 0)), (True, (0, 255, 0))):
            uv, z = project(box_corners(a, zb), cal)
            if (z < 1).any() or (uv[:, 0].max() < 0) or uv[:, 0].min() > 800 or uv[:, 1].max() < 0 or uv[:, 1].min() > 450:
                continue
            for i, j in EDGES:
                dr.line([tuple(uv[i]), tuple(uv[j])], fill=col, width=1)
    img.save(out / f"{Path(d['filename']).parent.name}.png")
    print("saved", Path(d["filename"]).parent.name)
