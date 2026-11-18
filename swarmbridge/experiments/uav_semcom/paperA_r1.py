"""Paper A, R1 (review #49): one table, the same FROZEN policies evaluated on the Shannon PHY and on the realistic LA PHY,
same workload and fresh paired seeds. No training; the only tuning is the heuristic tuned on each PHY's TRAIN seeds.
Plan: reports/phases/2026-11-18_paperA-writeup/paperA-outline.md (R1).

PHYs:  S = regime K5 (Shannon rate, A5 reference PHY)
       L = regime K5_la + scheduler-aware LA, persistent co-channel interference (A-X9 environment)
Block A, "literature-style evaluation" (movement / power / channels of the A5 K5 T:best = pacedB2+budget1+reuse;
       the untuned literature rules use their own mission flight, max power, no reuse):
       untuned literature rules (std_rand, std_rr, std_maxrate, untuned greedy), T:best (tuned on S in A5),
       PPO-S mode-only (A5 K5 checkpoints, trained on S)
Block B, "final stack" (scheduler / movement frozen by A-X9 stage 1: JA:stat:R:a1:m3 @ pacedB3):
       H*_P = best mode rule of the A-X9 family tuned on PHY P's TRAIN seeds 0..29, H*_P + calibrated selective MPC,
       H*_P + the same MPC uncalibrated, PPO-S (+ calibrated MPC), PPO-LA (A-X9 checkpoints, LA observation; L only:
       its observation needs the LA features), and the other PHY's H* (zero-shot heuristic transfer)
Policies trained on S and run on L are ZERO-SHOT transfers; PPO-LA on L is trained on the target PHY.
Seeds: TEST pool, 9800..9899 (never used before). Output: runs/uav_semcom/paperA_r1/
Usage: python paperA_r1.py tune|test|smoke --procs 6
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import ax2_eval as X2
import ax9_run as A9

OUT = C.REPO / "runs" / "uav_semcom" / "paperA_r1"
PHY = {"S": ("K5", {}), "L": ("K5_la", dict(la_mode="sched", la_interf="persistent"))}
LIT = ("std_rand", "std_rr", "std_maxrate", "greedy")
SEEDS = "9800:9900"


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def _init():
    AE.init(False)
    X2.DELIV_EXT = True


def lit_episode(env_regime, extra, name, seed, pool):
    """an untuned literature rule (whole slot + mode policy) in the given PHY, through the A5 event harness"""
    cfg = C.regime_cfg(env_regime, **dict(extra, obs_la=False))
    sp, mp = AE.MH.from_name(name, C.MODES)
    core = AE.SV.UavSurveyEnv(AE.G["table"], AE.G["pools"][pool], cfg)
    m = AE.V.run_episode_ev(core, sp, mp, seed)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))}, AE.deliveries(core)


def _job(j):
    phy, combo, i, seeds, pool = j
    import torch
    torch.set_num_threads(1)
    env_regime, extra = PHY[phy]
    rows, dl = [], []
    for s in seeds:
        if combo.startswith("lit:"):
            m, d = lit_episode(env_regime, extra, combo[4:], s, pool)
        else:
            ex = dict(extra, obs_la=("ppola" in combo.split("|")[1])) if phy == "L" else None
            m, d = X2.episode("K5", env_regime, combo, i, s, ex, None, pool)
        rows.append(m)
        dl.append(d)
    return dict(phy=phy, combo=combo, ckpt=i, seeds=list(seeds), rows=rows, deliv=dl)


def run(items, seeds, pool, procs, chunk):
    chunks = [seeds[x:x + chunk] for x in range(0, len(seeds), chunk)]
    jobs = [(p, c, i, ch, pool) for p, c in items
            for i in (range(3) if not c.startswith("lit:") and X2.needs_ckpt(c) else [-1]) for ch in chunks]
    with Pool(procs, initializer=_init) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["phy"], r["combo"], r["ckpt"]),
                              dict(phy=r["phy"], combo=r["combo"], ckpt=r["ckpt"], seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    return list(merged.values())


def tune_stage(a):
    """H*_S: the A-X9 heuristic family on the Shannon PHY, frozen scheduler / movement, TRAIN seeds 0..29. H*_L is read
    from the A-X9 tune (runs/uav_semcom/ax9/heur.json) and re-checked here with the same family."""
    OUT.mkdir(parents=True, exist_ok=True)
    S = sched()
    t0 = time.time()
    items = [(p, f"{S}|h={h}") for p in ("S", "L") for h in A9.H_FAMILY]
    res = run(items, list(C.TRAIN_SEEDS), "train", a.procs, 10)
    ret = {p: {} for p in ("S", "L")}
    for r in res:
        ret[r["phy"]][r["combo"].split("|h=")[1]] = float(np.mean([x["return"] for x in r["rows"]]))
    best = {p: max(ret[p], key=ret[p].get) for p in ret}
    ax9 = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "heur.json").read_text())
    out = dict(wall_s=time.time() - t0, sched=S, train_return=ret, H_star=best,
               ax9_H_star=ax9.get("best_H") or ax9.get("heuristic_best"))
    json.dump(out, open(OUT / "tune.json", "w"), indent=1)
    for p in ret:
        top = sorted(ret[p], key=lambda k: -ret[p][k])[:5]
        print(p, "H* =", best[p], {k: round(ret[p][k], 2) for k in top})
    print("A-X9 H* (LA):", out["ax9_H_star"])


def items_test():
    tu = json.loads((OUT / "tune.json").read_text())
    S, H = sched(), tu["H_star"]
    items = []
    for p in ("S", "L"):
        items += [(p, f"lit:{n}") for n in LIT] + [(p, "tbest|heur"), (p, "tbest|ppo")]
        hp, ho = H[p], H["L" if p == "S" else "S"]
        items += [(p, f"{S}|h={hp}"), (p, f"{S}|h={hp}+mpc"), (p, f"{S}|h={hp}+mpcnc"),
                  (p, f"{S}|ppo"), (p, f"{S}|ppo+mpc")]
        if ho != hp:
            items.append((p, f"{S}|h={ho}"))
        if p == "L":
            items += [(p, f"{S}|ppola"), (p, f"{S}|ppola+mpc")]
    return items


def test_stage(a):
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    t0 = time.time()
    res = run(items_test(), seeds, "test", a.procs, 25 if not a.smoke else 1)
    name = "smoke" if a.smoke else "test"
    json.dump(dict(wall_s=time.time() - t0, sched=sched(), seeds=seeds, phy={k: [v[0], v[1]] for k, v in PHY.items()},
                   tune=json.loads((OUT / "tune.json").read_text())["H_star"],
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]), open(OUT / f"{name}.json", "w"))
    json.dump({f"{r['phy']}|{r['combo']}|{r['ckpt']}": r["deliv"] for r in res},
              gzip.open(OUT / f"{name}.deliv.json.gz", "wt"))
    for r in sorted(res, key=lambda r: (r["phy"], r["combo"], r["ckpt"])):
        print(f"{r['phy']} {r['combo']:60s} ckpt {r['ckpt']:2d} ret {np.mean([x['return'] for x in r['rows']]):8.2f}",
              flush=True)
    print(f"{name} done {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "test", "smoke"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", default=SEEDS)
    a = ap.parse_args()
    a.smoke = a.cmd == "smoke"
    if a.smoke:
        a.seeds = "9800:9801"
    tune_stage(a) if a.cmd == "tune" else test_stage(a)


if __name__ == "__main__":
    main()
