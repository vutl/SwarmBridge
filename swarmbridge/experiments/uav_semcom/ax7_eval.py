"""Review #42 (A-X7): decompose the information opportunity, spatially correlated channels with causal filtering,
JA latency / Monte Carlo sensitivity, and the A-X2 re-check. No RL training.

Information levels of the joint-assignment scheduler (ja_sched), suffix of the policy names:
  c = stat (causal statistical model), f = filt (causal Bayesian LoS filter, ch_filter), l = los (LoS-only ORACLE),
  o = oracle (full-channel ORACLE). The two oracles are NON-CAUSAL diagnostics specific to this policy class.
Policies: NR (tuned least-slack rule, no reuse), JA-{NR,R}-{c,f,l,o}, and JA-{NR,R}-c-n{4,8,128} (Monte Carlo count).
Variants (all: matched workload = movement / power / modes of the regime's T:best, only channels differ):
  nom|good|slow_sched_persist, nom_sched_bursty      : A-X6 PHY (i.i.d. LoS per slot), information decomposition
  c{10,30,100}_sched_persist                          : spatially correlated LoS, correlation distance in metres
  c{0,10,30,100}_srs_sched_persist                    : the same with a sounding probe every slot
  c30sh4_sched_persist                                : + correlated 4 dB shadowing (changes the marginal; sensitivity)
  K5, ref                                             : A-X2 regimes (Shannon PHY), for the re-check against A-X2 RL
Usage: python ax7_eval.py tune VARIANT [--infos stat,los,oracle,filt] --procs 6
       python ax7_eval.py eval OUT --variants ... --policies ... --procs 6
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

OUT = C.REPO / "runs" / "uav_semcom" / "ax7"
INFO = {"c": "stat", "f": "filt", "l": "los", "o": "oracle"}
SP = dict(la_mode="sched", la_interf="persistent")
VARIANTS = {
    "nom_sched_persist": ("K5_la", dict(SP)),
    "nom_sched_bursty": ("K5_la", dict(la_mode="sched", la_interf="bursty")),
    "good_sched_persist": ("K5_la_good", dict(SP)),
    "slow_sched_persist": ("K5_la_slow", dict(SP)),
    "c30sh4_sched_persist": ("K5_la", dict(SP, ch_corr_m=30.0, ch_shadow_db=4.0)),
    "K5": ("K5", {}),
    "ref": ("ref", {}),
}
for d in (10, 30, 100):
    VARIANTS[f"c{d}_sched_persist"] = ("K5_la", dict(SP, ch_corr_m=float(d)))
for d in (0, 10, 30, 100):
    VARIANTS[f"c{d}_srs_sched_persist"] = ("K5_la", dict(SP, ch_corr_m=float(d), la_srs_period=1))
BASE_REG = {"K5": "K5", "ref": "ref"}


def cfg_of(variant):
    base, extra = VARIANTS[variant]
    return C.regime_cfg(base, **extra)


def base_name(variant):
    return AE.tuned(BASE_REG.get(variant, "K5_la"), "best")


def resolve(name, variant):
    """policy name -> (slot policy, mode policy, member)"""
    sp, mp = MH.from_name(base_name(variant), C.MODES)
    if name == "NR":
        return sp, mp, base_name(variant)
    f = name.split("-")                      # JA-NR-c[-n8]
    sel = json.loads((OUT / f"tune_{variant}.json").read_text())["selected"]
    member = sel[f"{INFO[f[2]]}:{f[1]}"]
    if len(f) > 3:
        member += ":" + f[3]
    return JA.from_name(member, sp), mp, member


def run_one(cfg, sp, mp, seed, pool):
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"][pool], cfg)
    r = V.run_episode_ev(core, sp, mp, seed)
    if not hasattr(core, "la_metrics"):
        core.la_metrics = dict
    m = X5._metrics(core, r["return"], None)
    t = np.asarray(core.__dict__.get("_ja_t", [])) * 1e3
    if len(t):
        m.update(ja_ms_mean=float(t.mean()), ja_ms_p50=float(np.percentile(t, 50)), ja_ms_p99=float(np.percentile(t, 99)))
    dg = core.__dict__.get("_ja_diag", [])
    if dg:
        pr, us, lt = (np.asarray(x, float) for x in zip(*dg))
        m.update(diag_n=int(len(lt)), los_frac=float(lt.mean()), prior_mean=float(pr.mean()),
                 brier_prior=float(((pr - lt) ** 2).mean()), brier_used=float(((us - lt) ** 2).mean()),
                 surprise_frac=float(((pr > 0.5) != (lt > 0.5)).mean()),
                 ambiguous_frac=float(((pr > 0.2) & (pr < 0.8)).mean()))
    log = core.__dict__.get("_ja_log", [])
    cont = [x for x in log if x[3] > core.c["M"]]
    m.update(ja_reuse_frac=float(np.mean([x[2] for x in cont])) if cont else float("nan"))
    if hasattr(core, "la_st"):
        m["srs"] = core.la_st.get("srs", 0)
    return m


def _tune_job(j):
    variant, member, seed = j
    sp, mp = MH.from_name(base_name(variant), C.MODES)
    return member, run_one(cfg_of(variant), JA.from_name(member, sp), mp, seed, "train")["return"]


def tune(a):
    infos = a.infos.split(",")
    members = [m for i in infos for m in JA.family(i)]
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(_tune_job, [(a.variant, m, s) for m in members for s in C.TRAIN_SEEDS], chunksize=4)
    ret = {m: float(np.mean([r for mm, r in res if mm == m])) for m in members}
    path = OUT / f"tune_{a.variant}.json"
    old = json.loads(path.read_text()) if path.exists() else dict(returns={}, selected={})
    old["returns"].update(ret)
    for info in infos:
        for r in ("NR", "R"):
            fam = [m for m in members if m.startswith(f"JA:{info}:{r}:")]
            old["selected"][f"{info}:{r}"] = max(fam, key=lambda m: ret[m])
    old.update(variant=a.variant, seeds=C.TRAIN_SEEDS)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=1))
    for k, m in sorted(old["selected"].items()):
        print(f"{a.variant:24s} {k:10s} {m:24s} {old['returns'][m]:8.2f}")


def _eval_job(j):
    variant, name, seeds = j
    sp, mp, member = resolve(name, variant)
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
        print(f"{o['variant']:24s} {o['policy']:12s} ret {f('return'):8.2f} ok {f('ok_rate'):.3f} "
              f"cochan {f('cochannel_air_frac'):.3f} ms p99 {f('ja_ms_p99'):.1f}", flush=True)
    json.dump(dict(wall_s=time.time() - t0, variants={v: VARIANTS[v] for v in a.variants.split(",")},
                   base={v: base_name(v) for v in a.variants.split(",")}, results=out), gzip.open(a.out, "wt"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("variant")
    t.add_argument("--infos", default="stat,los,oracle")
    t.add_argument("--procs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("out")
    e.add_argument("--variants", required=True)
    e.add_argument("--policies", required=True)
    e.add_argument("--seeds", default="8800:8900")
    e.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"tune": tune, "eval": run_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
