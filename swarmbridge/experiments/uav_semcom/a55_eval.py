"""A5.5 (reports/uav-semcom-a5.5.md, section 0): self-calibrating heuristic + 3-level uncertainty.

  tune  --regimes R1,R2..    the SC family (a55_heur.family('sc'), 56 members) on the TRAIN pool, seeds 0..29 (as A5);
                             -> runs/uav_semcom/a55/tune/<regime>.json: rows, T:SC, and H* = the better of {A5 T:best,
                             T:SC} by train return (A5's tune rows: same pool, seeds, env)
  eval  OUT --regimes .. --pool test|testdev --seeds 8000:8100 --policies SPEC,..
  boot  OUT --regimes .. --pool test|testdev --policies SPEC,.. [--B 50 --n 20]
                             video cluster bootstrap with re-simulation: replicate b resamples the pool's video
                             sequences with replacement (pool = all frames of the drawn sequences, multiplicity kept),
                             env seeds 20000 + 20 b + i, i < n; every policy on the same (replicate, seed) pairs.
Policy specs: H:NAME (A5 names or a55 'sc:...' names), T5:FAMILY (A5 tuned member: base / budget / reuse / fixed /
best), T:SC, T:HSTAR, P:LABEL=PATH:PART:HEUR:COMBO (as a5_eval; HEUR 'T~best' = A5 T:best of the regime).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import a55_heur as H
import env_ev as V
import env_survey as SV

OUT = C.A5.parent / "a55"
CONF_SEEDS = list(range(8000, 8100))
KEEP = AE.KEEP
BOOT_SEED0 = 20000


def tune_path(regime):
    return OUT / "tune" / f"{regime}.json"


def resolve(regime, spec):
    """Policy spec -> heuristic name (heuristic specs only)."""
    if spec.startswith("H:"):
        return spec[2:]
    if spec.startswith("T5:"):
        return AE.tuned(regime, spec[3:])
    d = json.loads(tune_path(regime).read_text())
    return {"T:SC": d["tuned"]["sc"], "T:HSTAR": d["hstar"]}[spec]


def boot_pools(pools, pool, B):
    """{b: image-index array} for the video cluster bootstrap of a pool (deterministic)."""
    man = C.manifest()["images"]
    tab = AE.G["table"]
    seq = {e["name"]: e["seq"] for e in man}
    idx = np.asarray(pools[pool])
    by = {}
    for i in idx:
        by.setdefault(seq[tab.images[i]], []).append(int(i))
    seqs = sorted(by)
    out = {}
    for b in range(B):
        rng = np.random.default_rng([BOOT_SEED0, b, 0 if pool == "test" else 1])
        draw = rng.choice(len(seqs), size=len(seqs), replace=True)
        out[b] = np.array([i for j in draw for i in by[seqs[j]]])
    return out


def init(testdev, boot_pool, B):
    AE.init(testdev)
    if boot_pool:
        for b, arr in boot_pools(AE.G["pools"], boot_pool, B).items():
            AE.G["pools"][f"{boot_pool}#{b}"] = arr


def job(j):
    regime, pool, spec, seed = j
    if spec.startswith("P:"):
        label, rest = spec[2:].split("=", 1)
        path, part, heur, combo = rest.split(":")
        heur = AE.tuned(regime, heur[2:]) if heur.startswith("T~") else heur
        m, _ = AE.rl_episode(regime, pool, path, part, heur, combo, seed)
    else:
        name = resolve(regime, spec)
        sp, mp = H.from_name(name, C.MODES)
        core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"][pool], C.regime_cfg(regime))
        m = V.run_episode_ev(core, sp, mp, seed)
    return (regime, pool, spec, seed), {k: float(m[k]) for k in KEEP}


def run_pool(jobs, procs, testdev, boot_pool=None, B=0, chunk=2):
    res, t0 = [], time.time()
    with Pool(procs, initializer=init, initargs=(testdev, boot_pool, B)) as p:
        for i, x in enumerate(p.imap(job, jobs, chunksize=chunk)):
            res.append(x)
            if (i + 1) % max(len(jobs) // 20, 1) == 0:
                el = time.time() - t0
                print(f"progress {i + 1}/{len(jobs)} {el:.0f}s eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s", flush=True)
    return res


def run_tune(args):
    regimes = args.regimes.split(",")
    names = H.family("sc")
    jobs = [(r, "train", "H:" + n, s) for r in regimes for n in names for s in C.TRAIN_SEEDS]
    res = run_pool(jobs, args.procs, False, chunk=8)
    for r in regimes:
        rows = {}
        for (rg, _, spec, _), m in res:
            if rg == r:
                rows.setdefault(spec[2:], []).append(m)
        summ = {n: {k: float(np.mean([e[k] for e in eps])) for k in ("return", "ok_rate", "mean_psnr", "mission_s",
                                                                    "energy_kj")} for n, eps in rows.items()}
        sc = max(names, key=lambda n: summ[n]["return"])
        a5 = json.loads(AE.tune_path(r).read_text())
        a5_best = AE.tuned(r, "best")
        hstar = sc if summ[sc]["return"] > a5["rows"][a5_best]["return"] else a5_best
        tune_path(r).parent.mkdir(parents=True, exist_ok=True)
        tune_path(r).write_text(json.dumps(dict(regime=r, cfg=C.regime_cfg(r), seeds=C.TRAIN_SEEDS, pool="train",
                                                tuned=dict(sc=sc), a5_best=a5_best,
                                                a5_best_train_return=a5["rows"][a5_best]["return"],
                                                hstar=hstar, rows=summ), indent=1))
        print(f"{r:14s} T:SC={sc} ({summ[sc]['return']:+.2f})  A5 T:best={a5_best} "
              f"({a5['rows'][a5_best]['return']:+.2f})  H*={hstar}", flush=True)


def run_eval(args):
    regimes = args.regimes.split(",")
    a, b = map(int, args.seeds.split(":"))
    seeds = list(range(a, b))
    assert set(seeds) <= set(CONF_SEEDS) or args.exploratory
    specs = args.policies.split(",")
    jobs = [(r, args.pool, sp, s) for r in regimes for sp in specs for s in seeds]
    res = run_pool(jobs, args.procs, args.pool == "testdev", chunk=args.chunk)
    out = {}
    for (r, _, spec, s), m in res:
        out.setdefault(r, {}).setdefault(spec, []).append(m)
    names = {r: {sp: (resolve(r, sp) if not sp.startswith("P:") else sp) for sp in specs} for r in regimes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pool=args.pool, seeds=seeds, resolved=names, results=out), open(args.out, "w"))
    for r in regimes:
        for sp in specs:
            e = out[r][sp]
            print(f"{r:14s} {sp[:50]:50s} ({names[r][sp][:30]:30s}) return {np.mean([x['return'] for x in e]):+8.2f}"
                  f"  ok {np.mean([x['ok_rate'] for x in e]):.3f}", flush=True)


def run_boot(args):
    regimes = args.regimes.split(",")
    specs = args.policies.split(",")
    jobs = [(r, f"{args.pool}#{b}", sp, BOOT_SEED0 + args.n * b + i)
            for r in regimes for b in range(args.B) for sp in specs for i in range(args.n)]
    res = run_pool(jobs, args.procs, args.pool == "testdev", boot_pool=args.pool, B=args.B, chunk=args.chunk)
    out = {}
    for (r, pool, spec, s), m in res:
        b = int(pool.split("#")[1])
        out.setdefault(r, {}).setdefault(str(b), {}).setdefault(spec, []).append(m["return"])
    names = {r: {sp: (resolve(r, sp) if not sp.startswith("P:") else sp) for sp in specs} for r in regimes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pool=args.pool, B=args.B, n=args.n, seed0=BOOT_SEED0, resolved=names, results=out),
              open(args.out, "w"))
    print("BOOT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--regimes", required=True)
    t.add_argument("--procs", type=int, default=6)
    for name in ("eval", "boot"):
        e = sub.add_parser(name)
        e.add_argument("out")
        e.add_argument("--regimes", required=True)
        e.add_argument("--pool", required=True, choices=["test", "testdev", "val"])
        e.add_argument("--policies", required=True)
        e.add_argument("--procs", type=int, default=6)
        e.add_argument("--chunk", type=int, default=2)
        if name == "eval":
            e.add_argument("--seeds", default="8000:8100")
            e.add_argument("--exploratory", action="store_true")
        else:
            e.add_argument("--B", type=int, default=50)
            e.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    dict(tune=run_tune, eval=run_eval, boot=run_boot)[args.cmd](args)


if __name__ == "__main__":
    main()
