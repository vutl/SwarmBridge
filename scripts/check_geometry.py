"""Verify the CoPerception-UAV geometry module on the town3 sample.

1. box recall : fraction of vehicle boxes projected inside a BOTTOM image that land on
                vehicle pixels of its seg image (per UAV).
2. alignment  : road IoU between overlapping BOTTOM views of two UAVs on a 0.5 m global grid,
                at the given poses and at the best shift (residual pose error).

    python scripts/check_geometry.py data/coperception_uav/town3
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmbridge.data.coperception_uav import H, W, ground_to_pixel, pixel_to_ground  # noqa: E402

VEHICLE = (95, 90, 229)
ROAD = [(235, 124, 133), (139, 242, 254), (215, 252, 228), (155, 154, 125), (197, 177, 129)]
RES = 0.5

root = sys.argv[1] + "/airsim_camera_seg/"
meta = lambda n: json.load(open(root + "v1.0-mini/" + n + ".json"))
cams = {c["token"]: c for c in meta("calibrated_sensor")}
boxes = defaultdict(list)
for a in meta("sample_annotation"):
    boxes[a["sample_token"]].append(a["translation"][:2])
boxes = {k: np.array(v) for k, v in boxes.items()}
by_sample = defaultdict(dict)
for s in meta("sample_data"):
    if "CAMERA_BOTTOM" in s["filename"]:
        by_sample[s["sample_token"]][s["filename"].split("/")[1][-1]] = s
seg_of = lambda sd: np.array(Image.open(root + sd["filename"].replace(".png", "_seg.png")).convert("RGB"))

# 1. box recall
rec = defaultdict(lambda: [0, 0])
for tok, sds in list(by_sample.items())[::3]:
    for uav, sd in sds.items():
        veh = ndimage.binary_dilation((seg_of(sd) == VEHICLE).all(-1), iterations=4)
        b = boxes[tok]
        u, v, front = ground_to_pixel(b[:, 0], b[:, 1], cams[sd["calibrated_sensor_token"]])
        ins = front & (u > 10) & (u < W - 10) & (v > 10) & (v < H - 10)
        rec[uav][0] += veh[v[ins].astype(int), u[ins].astype(int)].sum()
        rec[uav][1] += ins.sum()
print("1. box -> vehicle-pixel recall per UAV:",
      "  ".join(f"{k}: {a / max(1, n):.1%} (n={n})" for k, (a, n) in sorted(rec.items())))

# 2. cross-UAV alignment
vv, uu = np.mgrid[0:H, 0:W]


def ground_mask(sd):
    seg = seg_of(sd)
    road = np.zeros((H, W), bool)
    for c in ROAD:
        road |= (seg == c).all(-1)
    gx, gy = pixel_to_ground(uu, vv, cams[sd["calibrated_sensor_token"]])
    ok = np.isfinite(gx) & (np.abs(gx) < 1e4) & (np.abs(gy) < 1e4)
    return np.round(gx[ok] / RES).astype(int), np.round(gy[ok] / RES).astype(int), road[ok]


rows = []
for tok, sds in list(by_sample.items())[::4]:
    m = {k: ground_mask(sd) for k, sd in sds.items()}
    for a, b in combinations(sorted(m), 2):
        (xa, ya, ra), (xb, yb, rb) = m[a], m[b]
        x0, y0 = min(xa.min(), xb.min()) - 40, min(ya.min(), yb.min()) - 40
        shape = (max(xa.max(), xb.max()) - x0 + 41, max(ya.max(), yb.max()) - y0 + 41)
        grids = []
        for x, y, r in ((xa, ya, ra), (xb, yb, rb)):
            o, g = np.zeros(shape, bool), np.zeros(shape, bool)
            o[x - x0, y - y0] = True
            g[x[r] - x0, y[r] - y0] = True
            grids.append((o, g))
        (oa, ma), (ob, mb) = grids
        if (oa & ob).sum() < 2000:
            continue
        best, iou0 = (-1, 0, 0), None
        for dx in range(-20, 21, 2):
            for dy in range(-20, 21, 2):
                so, sb = np.roll(ob, (dx, dy), (0, 1)), np.roll(mb, (dx, dy), (0, 1))
                ov = oa & so
                iou = (ma & sb & ov).sum() / max(1, ((ma | sb) & ov).sum())
                best = max(best, (iou, dx, dy))
                if dx == dy == 0:
                    iou0 = iou
        rows.append((iou0, best[0], np.hypot(best[1], best[2]) * RES))

r = np.array(rows)
print(f"2. {len(r)} overlapping UAV pairs: median road IoU at given poses {np.median(r[:, 0]):.2f}, "
      f"at best shift {np.median(r[:, 1]):.2f}; median residual shift {np.median(r[:, 2]):.1f} m; "
      f"pairs with residual <= 1 m: {(r[:, 2] <= 1).mean():.0%}")
