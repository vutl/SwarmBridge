"""Milestone 1: BS -> low-altitude UAV channel traces in the Sionna RT Munich scene (CPU / LLVM).

Geometry: BS at (8.5, 21, 27) m with a 1x16 ULA (half wavelength, iso, V-pol), 28 GHz; each UAV has one iso antenna.
Episodes of DUR seconds sampled every DT_RT (ray tracing grid). Each episode has U UAVs at 15-25 m altitude flying
random walks in free space (segments checked against the scene geometry, never under a roof), inside +-HALF m of BS.
Two episode kinds (alternating by episode index, fixed by seed):
  convoy : UAV 0 leads; UAV u follows the same path delayed by LAG*u seconds with a lateral offset (spatially
           correlated channels, the setting where sharing measurements could help)
  indep  : every UAV flies its own random walk (control: little shared information)
The scene is static (no moving blockers): Milestone-0 statistics showed links to UAVs at 15-25 m are blocked by
buildings, which ground vehicles (2-4 m) cannot affect; this is stated as a scope limit.
Per receiver sample we keep the K strongest paths: a (16 complex, per BS antenna), Doppler (Hz, receiver velocity set),
delay, LoS flag. The narrowband channel at time offset dt inside a step is h(dt) = sum_p a_p exp(j 2 pi f_p dt).
Output: OUT/ep{e:04d}.npz + OUT/meta.json.
Usage: python trace_gen.py OUT [--episodes 60] [--seed 0] [--scene munich|etoile|...] [--bs x,y,z]
"""
import argparse
import json
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import numpy as np
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

BS = np.array([8.5, 21.0, 27.0])
N_ANT, FREQ = 16, 28e9
DUR, DT_RT, U, K = 20.0, 0.1, 4, 24
HALF, LAG, ALT = 200.0, 2.0, (15.0, 25.0)
SOLVER = dict(max_depth=5, diffraction=True)


def build_scene(name="munich"):
    scene = load_scene(getattr(rt.scene, name))
    scene.frequency = FREQ
    scene.tx_array = PlanarArray(num_rows=1, num_cols=N_ANT, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="bs", position=[float(v) for v in BS]))
    return scene


class Free:
    """Free-space checks against the mitsuba scene."""

    def __init__(self, scene):
        self.s = scene.mi_scene

    def under_roof(self, p):
        si = self.s.ray_intersect(mi.Ray3f(mi.Point3f(float(p[0]), float(p[1]), 1000.0), mi.Vector3f(0, 0, -1)))
        t = float(np.array(si.t).ravel()[0])
        return np.isfinite(t) and t < 1000.0 - p[2] - 0.5

    def seg_clear(self, p, q, margin=2.0):
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return True
        si = self.s.ray_intersect(mi.Ray3f(mi.Point3f(*map(float, p)), mi.Vector3f(*map(float, d / L))))
        t = float(np.array(si.t).ravel()[0])
        return not (np.isfinite(t) and t < L + margin)

    def ok(self, p, q):
        return (abs(q[0] - BS[0]) < HALF and abs(q[1] - BS[1]) < HALF and not self.under_roof(q)
                and self.seg_clear(p, q))


def random_walk(free, rng, T, dt, start=None):
    """Positions (T, 3) of a random walk with speed 5-12 m/s and heading changes every ~2 s."""
    for _ in range(200):
        p = start if start is not None else np.array([*(BS[:2] + rng.uniform(-HALF * 0.9, HALF * 0.9, 2)),
                                                      rng.uniform(*ALT)])
        if not free.under_roof(p):
            break
    v = rng.uniform(5, 12)
    head = rng.uniform(0, 2 * np.pi)
    pos = [p]
    for t in range(1, T):
        if t % int(round(2.0 / dt)) == 0:
            head += rng.normal(0, 0.6)
        for k in range(16):
            hd = head + (0 if k == 0 else rng.uniform(-np.pi, np.pi))
            q = pos[-1] + np.array([np.cos(hd), np.sin(hd), 0.0]) * v * dt
            if free.ok(pos[-1], q):
                head = hd
                break
        else:
            q = pos[-1].copy()                    # hover if boxed in
        pos.append(q)
    return np.array(pos)


def episode_tracks(free, rng, kind, T, dt):
    if kind == "indep":
        return np.stack([random_walk(free, rng, T, dt) for _ in range(U)])
    lag = int(round(LAG / dt))
    lead = random_walk(free, rng, T + lag * (U - 1), dt)
    tracks = [lead[lag * (U - 1):]]
    for u in range(1, U):
        base = lead[lag * (U - 1 - u): lag * (U - 1 - u) + T].copy()
        off = np.array([*rng.normal(0, 3.0, 2), rng.normal(0, 1.5)])
        cand = base + off
        if any(free.under_roof(c) for c in cand[:: max(1, T // 20)]):
            cand = base                            # fall back to the exact path if the offset hits a building
        tracks.append(cand)
    return np.stack(tracks)


def solve(scene, tracks, dt):
    """Batch all (uav, t) positions of one episode as receivers (static scene)."""
    Uu, T, _ = tracks.shape
    vel = np.gradient(tracks, dt, axis=1)
    names = []
    for u in range(Uu):
        for t in range(T):
            nm = f"r{u}_{t}"
            scene.add(Receiver(name=nm, position=[float(v) for v in tracks[u, t]],
                               velocity=[float(v) for v in vel[u, t]]))
            names.append(nm)
    paths = PathSolver()(scene, **SOLVER)
    a = np.array(paths.a[0]) + 1j * np.array(paths.a[1])          # [rx, 1, 1, N_ANT, P]
    a = a[:, 0, 0]                                                 # [rx, N_ANT, P]
    valid = np.array(paths.valid)[:, 0].astype(bool)               # [rx, P]
    dop = np.array(paths.doppler)[:, 0]
    tau = np.array(paths.tau)[:, 0]
    inter = np.array(paths.interactions)[:, :, 0]                  # [depth, rx, P]
    los = (inter == int(rt.InteractionType.NONE)).all(0) & valid
    P = a.shape[-1]
    if P < K:                                                      # pad the path axis to K
        pad = K - P
        a = np.pad(a, ((0, 0), (0, 0), (0, pad)))
        valid, los = np.pad(valid, ((0, 0), (0, pad))), np.pad(los, ((0, 0), (0, pad)))
        dop, tau = np.pad(dop, ((0, 0), (0, pad))), np.pad(tau, ((0, 0), (0, pad)))
    for nm in names:
        scene.remove(nm)
    pw = np.where(valid, (np.abs(a) ** 2).sum(1), 0.0)
    idx = np.argsort(-pw, axis=1)[:, :K]
    take = lambda x: np.take_along_axis(x, idx, axis=1)
    aK = np.take_along_axis(a, idx[:, None, :], axis=2).transpose(0, 2, 1)   # [rx, K, N_ANT]
    vK = take(valid)
    aK = np.where(vK[:, :, None], aK, 0)
    out = dict(a=aK.reshape(Uu, T, K, N_ANT).astype(np.complex64),
               dop=np.where(vK, take(dop), 0).reshape(Uu, T, K).astype(np.float32),
               tau=np.where(vK, take(tau), 0).reshape(Uu, T, K).astype(np.float32),
               los=(take(los) & vK).reshape(Uu, T, K), valid=vK.reshape(Uu, T, K),
               n_paths=valid.sum(1).reshape(Uu, T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="munich"); ap.add_argument("--bs", default=None, help="x,y,z of the BS")
    args = ap.parse_args()
    global BS
    if args.bs:
        BS = np.array([float(v) for v in args.bs.split(",")])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.scene)
    free = Free(scene)
    assert not free.under_roof(BS), "BS position is under a roof / inside a building"
    T = int(round(DUR / DT_RT)) + 1
    meta = dict(scene=args.scene, bs=BS.tolist(), n_ant=N_ANT, freq=FREQ, dur=DUR, dt_rt=DT_RT, uavs=U, k_paths=K, half=HALF, lag=LAG,
                alt=ALT, solver=SOLVER, seed=args.seed, episodes=[])
    for e in range(args.episodes):
        f = out / f"ep{e:04d}.npz"
        kind = "convoy" if e % 2 == 0 else "indep"
        if f.exists():
            meta["episodes"].append(dict(ep=e, kind=kind, file=f.name, cached=True))
            continue
        rng = np.random.default_rng([args.seed, e])
        t0 = time.time()
        tracks = episode_tracks(free, rng, kind, T, DT_RT)
        res = solve(scene, tracks, DT_RT)
        np.savez_compressed(f, pos=tracks.astype(np.float32), kind=kind, **res)
        los_frac = float(res["los"].any(-1).mean())
        nopath = float((res["n_paths"] == 0).mean())
        meta["episodes"].append(dict(ep=e, kind=kind, file=f.name, los_frac=round(los_frac, 3),
                                     nopath_frac=round(nopath, 3), secs=round(time.time() - t0, 1)))
        print(f"ep {e:3d} {kind:6s} los {los_frac:.2f} nopath {nopath:.2f} {time.time() - t0:.1f}s", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
