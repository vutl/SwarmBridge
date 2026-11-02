"""Inspect label problems in the CoPerception-UAV town3 sample with the verified geometry.

For every BOTTOM image: box recall (boxes landing on vehicle pixels) and the share of seg
vehicle blobs with no box. Prints per-UAV stats + recall vs distance from the image centre
(pose/yaw errors grow with distance, occlusion does not), and renders the worst images:
red = projected box, green = seg vehicle outline, blue fill = seg vehicle blob with no box.

    python scripts/inspect_labels.py data/coperception_uav/town3 out.png [--flat]

--flat projects boxes onto z = 0 (the old, wrong assumption) for comparison.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarmbridge.data.coperception_uav import H, W, ground_to_pixel  # noqa: E402

VEHICLE = (95, 90, 229)
ELEV = 0.0 if "--flat" in sys.argv else 1.0
root = sys.argv[1] + "/airsim_camera_seg/"
meta = lambda n: json.load(open(root + "v1.0-mini/" + n + ".json"))
cams = {c["token"]: c for c in meta("calibrated_sensor")}
anns = defaultdict(list)
for a in meta("sample_annotation"):
    anns[a["sample_token"]].append(a)
bottoms = [s for s in meta("sample_data") if "CAMERA_BOTTOM" in s["filename"]]


def box_polys(sd):
    cam = cams[sd["calibrated_sensor_token"]]
    polys = []
    for a in anns[sd["sample_token"]]:
        w, l = a["size"][0], a["size"][1]
        q = a["rotation"]
        th = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = np.array(a["translation"][:2])
        corners = c + (R @ np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).T / 2).T
        z = ELEV * a["translation"][2]
        u, v, f = ground_to_pixel(np.append(corners[:, 0], c[0]), np.append(corners[:, 1], c[1]), cam, z)
        if f.all() and -20 < u[4] < W + 20 and -20 < v[4] < H + 20:
            polys.append((np.stack([u[:4], v[:4]], 1), (u[4], v[4])))
    return polys


rows, dist_hits = [], defaultdict(lambda: np.zeros((6, 2)))
for sd in bottoms:
    uav = sd["filename"].split("/")[1][-1]
    seg = np.array(Image.open(root + sd["filename"].replace(".png", "_seg.png")).convert("RGB"))
    veh = (seg == VEHICLE).all(-1)
    near = ndimage.binary_dilation(veh, iterations=4)
    polys = box_polys(sd)
    boxmask = Image.new("L", (W, H), 0)
    for p, _ in polys:
        ImageDraw.Draw(boxmask).polygon([tuple(x) for x in p], fill=1)
    boxmask = ndimage.binary_dilation(np.array(boxmask, bool), iterations=4)
    hits = []
    for p, (u, v) in polys:
        if 10 < u < W - 10 and 10 < v < H - 10:
            hit = near[int(v), int(u)]
            hits.append(hit)
            b = min(5, int(np.hypot(u - W / 2, v - H / 2) / 80))
            dist_hits[uav][b] += (hit, 1)
    lab, n = ndimage.label(veh)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    nobox = [k + 1 for k in range(n) if sizes[k] > 60 and not (boxmask & (lab == k + 1)).any()]
    rows.append(dict(sd=sd, uav=uav, recall=np.mean(hits) if hits else np.nan, n_box=len(hits),
                     n_blob=int((sizes > 60).sum()), n_nobox=len(nobox), nobox=nobox, lab=lab))

print("UAV  box recall   blobs without box")
for u in "01234":
    r = [x for x in rows if x["uav"] == u]
    rec = np.nansum([x["recall"] * x["n_box"] for x in r]) / max(1, sum(x["n_box"] for x in r))
    nb = sum(x["n_nobox"] for x in r) / max(1, sum(x["n_blob"] for x in r))
    print(f"  {u}   {rec:7.1%}      {nb:7.1%}")
print("\nbox recall vs distance from image centre (80 px bins):")
for u in "01234":
    h = dist_hits[u]
    print(f"  UAV {u}: " + "  ".join(f"{h[i, 0] / h[i, 1]:.0%}({int(h[i, 1])})" for i in range(6) if h[i, 1] >= 10))

worst = sorted([x for x in rows if x["n_box"] >= 3], key=lambda x: x["recall"])[:3] + \
        sorted(rows, key=lambda x: -x["n_nobox"])[:3]
tiles = []
for x in worst:
    sd = x["sd"]
    img = np.array(Image.open(root + sd["filename"]).convert("RGB"))
    seg = np.array(Image.open(root + sd["filename"].replace(".png", "_seg.png")).convert("RGB"))
    veh = (seg == VEHICLE).all(-1)
    for k in x["nobox"]:
        m = x["lab"] == k
        img[m] = (0.3 * img[m] + 0.7 * np.array([0, 80, 255])).astype(np.uint8)
    img[veh & ~ndimage.binary_erosion(veh, iterations=2)] = (0, 255, 0)
    t = Image.fromarray(img)
    d = ImageDraw.Draw(t)
    for p, _ in box_polys(sd):
        d.polygon([tuple(q) for q in p], outline=(255, 0, 0), width=3)
    d.rectangle((0, 0, 520, 18), fill=(0, 0, 0))
    d.text((3, 3), f"UAV {x['uav']}  recall {x['recall']:.0%} of {x['n_box']}  no-box blobs {x['n_nobox']}/{x['n_blob']}",
           fill=(255, 255, 255))
    tiles.append(t.resize((560, 315)))
canvas = Image.new("RGB", (1120, 945))
for i, t in enumerate(tiles):
    canvas.paste(t, (560 * (i % 2), 315 * (i // 2)))
canvas.save(sys.argv[2])
