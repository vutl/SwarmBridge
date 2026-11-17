"""Review #41 (A-X6): consistent block accounting, scheduler-aware link adaptation, interference assumptions, and the
scheduling-opportunity check. No RL training.

  invariance OUT : the A-X5 policies (incl. the trained C-only checkpoints, evaluation only) on K5_la, K5_la_good,
                   K5_la_rand, paired seeds, twice: la_atomic False (must reproduce the stored A-X5 rows) and True.
  tune VARIANT   : the joint-assignment family (ja_sched.family, causal and oracle information) on the TRAIN pool,
                   seeds 0..29 -> runs/uav_semcom/ax6/tune_VARIANT.json (best member per (info, reuse)).
  eval OUT       : variants x policies on the TEST pool, fresh seeds (default 8800..8899), paired, matched workload:
                   every policy uses movement / power / modes of T:best@K5_la (fixed_avif_20); only channels differ.
Policies:
  NR        T:best@K5_la channel rule (the M least-slack UAVs, no reuse)
  RU        + every other UAV with traffic reuses the least-loaded channel (mode_heur.reuse)
  RUG       RU + NACK gate (theta / cool of gate_K5_la.json)
  JA-NR-c   joint assignment, causal statistical channel model (positions + LoS / fading statistics), no reuse (tuned)
  JA-R-c    the same, reuse allowed (tuned)
  JA-NR-o   oracle information (realised slot fading), no reuse      } UPPER-BOUND DIAGNOSTICS,
  JA-R-o    oracle information, reuse allowed                         } not deployable controllers
Usage: python ax6_eval.py invariance runs/uav_semcom/ax6/invariance.json.gz --procs 6
       python ax6_eval.py tune nom_sched_persist --procs 6
       python ax6_eval.py eval runs/uav_semcom/ax6/eval.json --variants ... --procs 6
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import ax5_eval as X5
import env_ev as V
import env_survey as SV
import ja_sched as JA
import mode_heur as MH

OUT = C.REPO / "runs" / "uav_semcom" / "ax6"
STORED_AX5 = C.REPO / "runs" / "uav_semcom" / "ax5" / "eval.json.gz"

VARIANTS = {}
for cond, base in (("nom", "K5_la"), ("good", "K5_la_good"), ("slow", "K5_la_slow")):
    for mode in ("blind", "sched", "oracle"):
        for itf, iv in (("bursty", "bursty"), ("persist", "persistent")):
            VARIANTS[f"{cond}_{mode}_{itf}"] = (base, dict(la_mode=mode, la_interf=iv))
for itf, iv in (("bursty", "bursty"), ("persist", "persistent")):
    VARIANTS[f"blockfade_sched_{itf}"] = ("K5_la", dict(la_mode="sched", la_interf=iv, la_fade="block"))
VARIANTS["mcsg"] = ("K5_mcsg", {})       # A-X2b's genie-LA env: does its +11.3 come from reuse or from selection?
MAIN = [f"nom_{m}_{i}" for m in ("blind", "sched", "oracle") for i in ("bursty", "persist")] + \
       [f"{c}_sched_{i}" for c in ("good", "slow", "blockfade") for i in ("bursty", "persist")]
POLICIES = ("NR", "RU", "RUG", "JA-NR-c", "JA-R-c", "JA-NR-o", "JA-R-o")
AX5_SPECS = ("T:best@K5_la_rand,T:best@K5_la,T:reuse@K5_la_rand,T:reuse@K5_la_rand+gate,C:K5_mcsg,C:K5_mcsg+gate,"
             "C:K5_la_rand,C:K5_la_rand+gate,C:K5_la").split(",")


def cfg_of(variant):
    base, extra = VARIANTS[variant]
    return C.regime_cfg(base, **extra)


def base_name(variant=None):
    return AE.tuned("K5_mcsg" if variant == "mcsg" else "K5_la", "best")


def policy(name, variant):
    """-> (slot policy, mode policy, member name)"""
    sp, mp = MH.from_name(base_name(variant), C.MODES)
    if name == "NR":
        return sp, mp, base_name(variant)
    if name in ("RU", "RUG"):
        n = base_name(variant) + "+reuse"
        if name == "RUG":
            th, cool = X5.gate_params("K5_la")
            n += f"+gate{th:g}c{cool}"
        sp, mp = MH.from_name(n, C.MODES)
        return sp, mp, n
    _, r, info = name.split("-")
    tuned = json.loads((OUT / f"tune_{variant}.json").read_text())["selected"]
    member = tuned[f"{'oracle' if info == 'o' else 'stat'}:{r}"]
    return JA.from_name(member, sp), mp, member


def run_one(cfg, sp, mp, seed, pool):
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"][pool], cfg)
    r = V.run_episode_ev(core, sp, mp, seed)
    if not hasattr(core, "la_metrics"):            # non-LA env (variant mcsg): no link-adaptation metrics
        core.la_metrics = dict
    m = X5._metrics(core, r["return"], None)
    log = core.__dict__.get("_ja_log", [])
    contended = [x for x in log if x[3] > core.c["M"]]
    m.update(ja_slots=len(log), ja_contended=len(contended),
             ja_reuse_frac=float(np.mean([x[2] for x in contended])) if contended else float("nan"),
             ja_pred_gain=float(np.mean([(x[0] - x[1]) / max(x[1], 1e-9) for x in contended])) if contended
             else float("nan"))
    return m


# ---------- invariance ----------
def _inv_job(j):
    import torch
    torch.set_num_threads(1)
    return X5.job(j)[0]


def invariance(a):
    envs = ("K5_la", "K5_la_good", "K5_la_rand")
    seeds = list(range(8800, 8900))
    chunks = [seeds[x:x + 25] for x in range(0, 100, 25)]
    jobs = [(sp, e, i, ch, "K5_la_rand", dict(la_atomic=at)) for at in (False, True) for e in envs for sp in AX5_SPECS
            for i in (range(3) if sp.startswith("C:") else [-1]) for ch in chunks]
    t0 = time.time()
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(_inv_job, jobs, chunksize=1)
    out = {}
    for j, r in zip(jobs, res):
        key = f"{r['env_regime']}|{r['spec']}|{r['ckpt']}|{'atomic' if j[5]['la_atomic'] else 'old'}"
        out.setdefault(key, dict(seeds=[], rows=[]))
        out[key]["seeds"] += r["seeds"]
        out[key]["rows"] += r["rows"]
    stored = json.load(gzip.open(STORED_AX5, "rt"))
    st = {f"{r['env_regime']}|{r['spec']}|{r['ckpt']}": dict(zip(r["seeds"], r["rows"])) for r in stored["results"]}
    maxdiff = 0.0
    for key, v in out.items():
        if key.endswith("|old"):
            ref = st[key[:-4]]
            for s, row in zip(v["seeds"], v["rows"]):
                maxdiff = max(maxdiff, abs(row["return"] - ref[s]["return"]))
    json.dump(dict(wall_s=time.time() - t0, repro_max_abs_return_diff=maxdiff, results=out),
              gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w"))
    print(f"reproduction of stored A-X5 rows with la_atomic False: max |return diff| = {maxdiff:.3g}")


# ---------- tuning ----------
def _tune_job(j):
    variant, member, seed = j
    sp, mp = MH.from_name(base_name(variant), C.MODES)
    return member, run_one(cfg_of(variant), JA.from_name(member, sp), mp, seed, "train")["return"]


def tune(a):
    members = JA.family("stat") + JA.family("oracle")
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(_tune_job, [(a.variant, m, s) for m in members for s in C.TRAIN_SEEDS], chunksize=4)
    ret = {m: float(np.mean([r for mm, r in res if mm == m])) for m in members}
    sel = {}
    for info in ("stat", "oracle"):
        for r in ("NR", "R"):
            fam = [m for m in members if m.startswith(f"JA:{info}:{r}:")]
            sel[f"{info}:{r}"] = max(fam, key=lambda m: ret[m])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tune_{a.variant}.json").write_text(json.dumps(dict(variant=a.variant, seeds=C.TRAIN_SEEDS, returns=ret,
                                                                selected=sel), indent=1))
    for k, m in sel.items():
        print(f"{a.variant:24s} {k:10s} {m:24s} {ret[m]:8.2f}")


# ---------- evaluation ----------
def _eval_job(j):
    variant, name, seeds = j
    sp, mp, member = policy(name, variant)
    cfg = cfg_of(variant)
    return dict(variant=variant, policy=name, member=member, seeds=seeds,
                rows=[run_one(cfg, sp, mp, s, "test") for s in seeds])


def run_eval(a):
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    chunks = [seeds[x:x + 25] for x in range(0, len(seeds), 25)]
    jobs = [(v, p, ch) for v in a.variants.split(",") for p in a.policies.split(",") for ch in chunks]
    t0 = time.time()
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(_eval_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["variant"], r["policy"]), dict(variant=r["variant"], policy=r["policy"],
                                                                 member=r["member"], seeds=[], rows=[]))
        m["seeds"] += r["seeds"]
        m["rows"] += r["rows"]
    out = sorted(merged.values(), key=lambda x: (x["variant"], x["policy"]))
    for o in out:
        f = lambda k: np.nanmean([x.get(k, np.nan) for x in o["rows"]])
        print(f"{o['variant']:24s} {o['policy']:8s} ret {f('return'):8.2f} ok {f('ok_rate'):.3f} bler {f('bler'):.3f} "
              f"shbler {f('shared_bler'):.3f} cochan {f('cochannel_air_frac'):.3f} goodput {f('goodput_kbps'):6.1f}",
              flush=True)
    json.dump(dict(wall_s=time.time() - t0, base={v: base_name(v) for v in a.variants.split(",")}, variants={v: VARIANTS[v] for v in a.variants.split(",")},
                   results=out), gzip.open(a.out, "wt"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("invariance")
    i.add_argument("out")
    i.add_argument("--procs", type=int, default=6)
    t = sub.add_parser("tune")
    t.add_argument("variant")
    t.add_argument("--procs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("out")
    e.add_argument("--variants", default=",".join(MAIN))
    e.add_argument("--policies", default=",".join(POLICIES))
    e.add_argument("--seeds", default="8800:8900")
    e.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"invariance": invariance, "tune": tune, "eval": run_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
