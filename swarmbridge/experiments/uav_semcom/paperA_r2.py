"""Paper A, robustness round (2026-11-19, after the novelty reviews): no training, frozen policies from R1.
  E1  PHY variants of the LA PHY (test pool, fresh seeds 9900..9999):
        L        = R1 LA PHY (scheduler-aware LA, persistent interference, slot fading)
        L_fade   = small-scale fading redrawn every 0.1 s block (A-X7 showed slot-frozen fading matters)
        L_bursty = co-channel UAVs interfere only while sending data
        L_badfb  = degraded CQI feedback: 3 dB report error, 10-block (1 s) report delay
      plus S (Shannon) on the same seeds.
  E2  RobustMPC-style baseline (Yin et al. 2015): the same MPC with a history-only robust harmonic-mean rate
      predictor (a9_mpc.HarmRates, mode suffix +mpcrh), next to the SNR-binned calibrated (+mpc) and the nominal
      (+mpcnc) MPC, in every E1 / E3 configuration.
  E3  Held-out content: the TEST-DEV pool (VisDrone2019-DET-test-dev, 49 videos), S and L, fresh seeds 12000..12099.
Policies (frozen, as R1): T:best (A5 K5), PPO-S (A5 K5 mode-only), PPO-LA (A-X9), H*_S = fixed AVIF-50 and
H*_L = fixed mbt2018-mean q3 (tuned in R1 on the train seeds of S and L; NOT re-tuned per variant).
Output: runs/uav_semcom/paperA_r2/{e12,e3}.json + .deliv.json.gz
Usage: python paperA_r2.py e12|e3|smoke --procs 6
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

OUT = C.REPO / "runs" / "uav_semcom" / "paperA_r2"
LA = dict(la_mode="sched", la_interf="persistent")
PHY = {"S": ("K5", {}), "L": ("K5_la", LA),
       "L_fade": ("K5_la", dict(LA, la_fade="block")),
       "L_bursty": ("K5_la", dict(LA, la_interf="bursty")),
       "L_badfb": ("K5_la", dict(LA, la_cqi_sigma=3.0, la_fb_delay=10))}
H = {"S": "fixed_avif_50", "L": "fixed_mbt2018-mean_q3"}


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def arms(phy):
    S = sched()
    native = H["S"] if phy == "S" else H["L"]
    other = H["L"] if phy == "S" else H["S"]
    a = ["tbest|heur", "tbest|ppo",
         f"{S}|h={native}", f"{S}|h={native}+mpc", f"{S}|h={native}+mpcnc", f"{S}|h={native}+mpcrh",
         f"{S}|h={other}", f"{S}|ppo", f"{S}|ppo+mpc"]
    if phy != "S":
        a += [f"{S}|ppola", f"{S}|ppola+mpc", f"{S}|ppola+mpcrh"]
    return a


def _init(testdev):
    AE.init(testdev)
    X2.DELIV_EXT = True


def _job(j):
    phy, combo, i, seeds, pool = j
    import torch
    torch.set_num_threads(1)
    env_regime, extra = PHY[phy]
    rows, dl = [], []
    for s in seeds:
        ex = dict(extra, obs_la=("ppola" in combo.split("|")[1])) if phy != "S" else None
        m, d = X2.episode("K5", env_regime, combo, i, s, ex, None, pool)
        rows.append(m)
        dl.append(d)
    return dict(phy=phy, combo=combo, ckpt=i, seeds=list(seeds), rows=rows, deliv=dl)


def run(items, seeds, pool, procs, chunk, testdev):
    chunks = [seeds[x:x + chunk] for x in range(0, len(seeds), chunk)]
    jobs = [(p, c, i, ch, pool) for p, c in items for i in (range(3) if X2.needs_ckpt(c) else [-1]) for ch in chunks]
    with Pool(procs, initializer=_init, initargs=(testdev,)) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["phy"], r["combo"], r["ckpt"]),
                              dict(phy=r["phy"], combo=r["combo"], ckpt=r["ckpt"], seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    return list(merged.values())


def stage(name, phys, seeds, pool, testdev, procs):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = [(p, c) for p in phys for c in arms(p)]
    res = run(items, seeds, pool, procs, 25 if len(seeds) > 5 else 1, testdev)
    json.dump(dict(wall_s=time.time() - t0, sched=sched(), seeds=seeds, pool=pool, H=H,
                   phy={p: [PHY[p][0], PHY[p][1]] for p in phys},
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]), open(OUT / f"{name}.json", "w"))
    json.dump({f"{r['phy']}|{r['combo']}|{r['ckpt']}": r["deliv"] for r in res}, gzip.open(OUT / f"{name}.deliv.json.gz", "wt"))
    for r in sorted(res, key=lambda r: (r["phy"], r["combo"], r["ckpt"])):
        print(f"{r['phy']:8s} {r['combo']:58s} ckpt {r['ckpt']:2d} ret {np.mean([x['return'] for x in r['rows']]):8.2f}",
              flush=True)
    print(f"{name} done {time.time() - t0:.0f}s", flush=True)


VARS = ("L_fade", "L_bursty", "L_badfb")


def retune(procs):
    """E1b fairness check: H* re-tuned on the TRAIN seeds 0..29 of each LA variant (same 22-rule family as R1),
    then H*_V and H*_V + calibrated MPC evaluated on the E1 seeds 9900..9999 (paired with e12; PPO-LA is NOT
    retrained, so this comparison favours the heuristic)."""
    import ax9_run as A9
    S = sched()
    t0 = time.time()
    tr = run([(v, f"{S}|h={h}") for v in VARS for h in A9.H_FAMILY], list(C.TRAIN_SEEDS), "train", procs, 10, False)
    tune = {v: {} for v in VARS}
    for r in tr:
        tune[r["phy"]][r["combo"].split("|h=")[1]] = float(np.mean([x["return"] for x in r["rows"]]))
    best = {v: max(tune[v], key=tune[v].get) for v in VARS}
    print("re-tuned H*:", best, flush=True)
    items = [(v, f"{S}|h={best[v]}") for v in VARS] + [(v, f"{S}|h={best[v]}+mpc") for v in VARS]
    items = list(dict.fromkeys(items))
    res = run(items, list(range(9900, 10000)), "test", procs, 25, False)
    json.dump(dict(wall_s=time.time() - t0, sched=S, seeds=list(range(9900, 10000)), pool="test", H_retuned=best,
                   train_return=tune, results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]),
              open(OUT / "e1b.json", "w"))
    json.dump({f"{r['phy']}|{r['combo']}|{r['ckpt']}": r["deliv"] for r in res}, gzip.open(OUT / "e1b.deliv.json.gz", "wt"))
    for r in res:
        print(f"{r['phy']:8s} {r['combo']:58s} ret {np.mean([x['return'] for x in r['rows']]):8.2f}", flush=True)
    print(f"e1b done {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("e12", "e3", "e1b", "smoke"))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "e1b":
        retune(a.procs)
        return
    if a.cmd == "smoke":
        stage("smoke", ["L_fade"], [9900], "test", False, a.procs)
    elif a.cmd == "e12":
        stage("e12", ["S", "L", "L_fade", "L_bursty", "L_badfb"], list(range(9900, 10000)), "test", False, a.procs)
    else:
        stage("e3", ["S", "L"], list(range(12000, 12100)), "testdev", True, a.procs)


if __name__ == "__main__":
    main()
