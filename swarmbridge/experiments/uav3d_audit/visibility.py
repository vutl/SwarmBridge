"""UAV3D v1.0-mini collaboration-signal audit (seg-based occlusion), all 400 samples x 25 cameras.

An object counts as SEEN by a camera when its projected box (bounding rect of 8 corners, mostly inside the image,
>= MIN_PIX pixels) is covered by >= MIN_VIS vehicle-colour pixels in that camera's _seg.png.
A UAV sees an object if any of its 5 cameras does. Reports, per object-sample:
  - how many UAVs have it in FOV vs actually see it,
  - for each UAV: share of team-visible objects it misses (= what teammates could add),
  - same restricted to the UAV's own neighbourhood (objects within R m of the UAV ground point),
  - distance profile of 'seen by me' vs 'seen only by others'.
Writes per-object rows to rows.npz for later analysis (sample idx, scene idx, ann idx, fov mask, vis mask, xy).
"""
import json, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from uavgeo import project, box_corners

root = Path(sys.argv[1]); out = Path(sys.argv[2])
MIN_VIS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
MIN_PIX = 20
VEH = np.array([95, 90, 229])
meta = root / "v1.0-mini"
load = lambda n: json.loads((meta / f"{n}.json").read_text())
scenes, samples, sds, cals, anns = (load(n) for n in ("scene", "sample", "sample_data", "calibrated_sensor", "sample_annotation"))
cal_of = {c["token"]: c for c in cals}
scene_idx = {s["token"]: i for i, s in enumerate(scenes)}
sd_by = collections.defaultdict(list)
for d in sds: sd_by[d["sample_token"]].append(d)
ann_by = collections.defaultdict(list)
for a in anns: ann_by[a["sample_token"]].append(a)
inst_idx = {}

rows = []  # (sample i, scene i, instance i, fov mask(5 bits), vis mask, x, y, min dist to uav[5])
for si, s in enumerate(samples):
    A = ann_by[s["token"]]
    corners = [box_corners(a) for a in A]
    fov = np.zeros((len(A), 5), bool); vis = np.zeros((len(A), 5), bool); uav_xy = np.zeros((5, 2))
    for d in sd_by[s["token"]]:
        cal = cal_of[d["calibrated_sensor_token"]]
        cam = Path(d["filename"]).parent.name
        u = int(cam.split("_id_")[1])
        if "BOTTOM" in cam: uav_xy[u] = cal["translation"][:2]
        seg = np.asarray(Image.open(root / "sweeps" / cam / (Path(d["filename"]).stem + "_seg.png")).convert("RGB"))
        mask = (seg == VEH).all(-1)
        W, H = d["width"], d["height"]
        for k, C in enumerate(corners):
            uv, z = project(C, cal)
            if (z < 1).any():
                continue
            u0, v0, u1, v1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
            iu0, iv0, iu1, iv1 = max(int(u0), 0), max(int(v0), 0), min(int(u1) + 1, W), min(int(v1) + 1, H)
            if iu1 <= iu0 or iv1 <= iv0:
                continue
            area = (iu1 - iu0) * (iv1 - iv0)
            if area < 0.5 * (u1 - u0) * (v1 - v0):
                continue
            fov[k, u] = True
            if area >= MIN_PIX and mask[iv0:iv1, iu0:iu1].mean() >= MIN_VIS:
                vis[k, u] = True
    for k, a in enumerate(A):
        xy = np.array(a["translation"][:2])
        ii = inst_idx.setdefault(a["instance_token"], len(inst_idx))
        rows.append((si, scene_idx[s["scene_token"]], ii, *fov[k], *vis[k], *xy, *np.linalg.norm(uav_xy - xy, axis=1)))
    if si % 50 == 0:
        print(f"sample {si}/{len(samples)}", flush=True)

R = np.array(rows, dtype=float)
np.savez(out / f"rows_{MIN_VIS}.npz", rows=R)
fov, vis, dist = R[:, 3:8].astype(bool), R[:, 8:13].astype(bool), R[:, 15:20]
n = len(R)
print(f"\nobject-samples {n}; seg threshold {MIN_VIS}")
print("UAVs with object in FOV :", {k: f"{(fov.sum(1) == k).mean():.1%}" for k in range(6)})
print("UAVs that SEE the object:", {k: f"{(vis.sum(1) == k).mean():.1%}" for k in range(6)})
team = vis.any(1)
print(f"seen by at least one UAV: {team.mean():.1%}")
for u in range(5):
    own = vis[team, u].mean()
    in_fov_occl = (fov[team, u] & ~vis[team, u]).mean()
    print(f"UAV {u}: sees {own:.1%} of team-visible; teammates add {1 - own:.1%} "
          f"(of which in my FOV but occluded {in_fov_occl:.1%})")
for rad in (30, 60, 100):
    m_own, m_add = [], []
    for u in range(5):
        near = team & (dist[:, u] <= rad)
        if near.sum():
            m_own.append(vis[near, u].mean())
    print(f"objects within {rad} m of a UAV (team-visible): that UAV sees {np.mean(m_own):.1%}, teammates add {1 - np.mean(m_own):.1%}")
single = vis.sum(1) == 1
print(f"objects seen by exactly ONE UAV: {single.mean():.1%} of all, {single[team].mean():.1%} of team-visible")
