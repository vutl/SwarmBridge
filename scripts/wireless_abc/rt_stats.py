"""Milestone 0: LoS / reflection-only / no-path statistics for BS->UAV links in the Sionna RT Munich scene (CPU).

Classes per receiver point (per solver config):
  los      : at least one valid path with no interaction (direct path)
  nlos     : no direct path, but >= 1 valid path (reflection / diffraction / diffuse)
  nopath   : no valid path found with this solver config
Points under a roof (a downward ray from 1 km hits geometry before reaching the point) are flagged `inside` and
excluded from the fractions (they are inside/under buildings, not valid UAV positions).
Configs: base (LoS + specular, depth 3), +diffraction, +diffuse, depth 5. Output: OUT/rt_stats.json
Usage: python rt_stats.py OUT [--n 12] [--half 200]
"""
import argparse, json, time
from pathlib import Path
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import drjit as dr
import numpy as np
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--half", type=float, default=200)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

BS = [8.5, 21.0, 27.0]
HEIGHTS = [20.0, 50.0, 100.0]
CONFIGS = {"base": dict(max_depth=3), "diffraction": dict(max_depth=3, diffraction=True),
           "diffuse": dict(max_depth=3, diffuse_reflection=True), "depth5": dict(max_depth=5)}

scene = load_scene(rt.scene.munich)
scene.frequency = 28e9
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(Transmitter(name="bs", position=BS))
bb = scene.mi_scene.bbox()
xs = np.linspace(BS[0] - args.half, BS[0] + args.half, args.n)
ys = np.linspace(BS[1] - args.half, BS[1] + args.half, args.n)
pts = np.array([[x, y, h] for h in HEIGHTS for x in xs for y in ys])

# under-roof test: ray straight down from z = 1000 m
o = mi.Point3f(pts[:, 0], pts[:, 1], np.full(len(pts), 1000.0))
d = mi.Vector3f(np.zeros(len(pts)), np.zeros(len(pts)), -np.ones(len(pts)))
si = scene.mi_scene.ray_intersect(mi.Ray3f(o, d))
t_hit = np.array(si.t)
inside = np.isfinite(t_hit) & (t_hit < 1000.0 - pts[:, 2] - 0.05)
for i, p in enumerate(pts):
    scene.add(Receiver(name=f"u{i}", position=[float(v) for v in p]))

res = {"bs": BS, "heights": HEIGHTS, "grid_n": args.n, "half_m": args.half, "freq": 28e9,
       "scene_bbox": [list(map(float, bb.min)), list(map(float, bb.max))], "n_points": len(pts),
       "inside": inside.tolist(), "configs": {}}
for name, cfg in CONFIGS.items():
    t0 = time.time()
    paths = PathSolver()(scene, **cfg)
    dt = time.time() - t0
    valid = np.array(paths.valid.numpy() if hasattr(paths.valid, "numpy") else paths.valid).astype(bool)  # [rx, tx, path]
    inter = np.array(paths.interactions.numpy() if hasattr(paths.interactions, "numpy") else paths.interactions)  # [depth, rx, tx, path]
    a_re, a_im = paths.a
    a = np.array(a_re.numpy() if hasattr(a_re, "numpy") else a_re) + 1j * np.array(a_im.numpy() if hasattr(a_im, "numpy") else a_im)
    direct = (inter == int(rt.InteractionType.NONE)).all(axis=0)                     # [rx, tx, path]
    has_any = valid[:, 0].any(-1)
    has_los = (valid & direct)[:, 0].any(-1)
    pw = (np.abs(a) ** 2).reshape(a.shape[0], -1).sum(-1)
    cls = np.where(has_los, "los", np.where(has_any, "nlos", "nopath"))
    rows = {}
    for k, h in enumerate(HEIGHTS):
        m = (pts[:, 2] == h) & ~inside
        c = cls[m]
        rows[str(int(h))] = {"n_valid_points": int(m.sum()), "n_inside": int(((pts[:, 2] == h) & inside).sum()),
                             "frac_los": round(float((c == "los").mean()), 3), "frac_nlos": round(float((c == "nlos").mean()), 3),
                             "frac_nopath": round(float((c == "nopath").mean()), 3),
                             "nlos_power_db_median": (round(float(np.median(10 * np.log10(pw[m][c == "nlos"]))), 1)
                                                      if (c == "nlos").any() else None),
                             "los_power_db_median": (round(float(np.median(10 * np.log10(pw[m][c == "los"]))), 1)
                                                     if (c == "los").any() else None)}
    res["configs"][name] = {"cfg": cfg, "solve_s": round(dt, 1), "by_height": rows, "class": cls.tolist()}
    print(name, f"{dt:.1f}s", json.dumps(rows), flush=True)
(out / "rt_stats.json").write_text(json.dumps(res))
