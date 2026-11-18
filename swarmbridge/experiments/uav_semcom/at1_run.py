"""Review #45 (A-T1): task-utility headroom study. Protocol (pre-registered before the test stage):
reports/phases/2026-11-17_paperA-scheduling/swarmbridge-at1-protocol.md.

Environment as A-X9 (K5_la, scheduler-aware LA, persistent interference, no SRS, obs K5), scheduler / movement frozen
by A-X9 stage 1 (runs/uav_semcom/ax9/sched.json). Metric: full-workload on-time recall = sum of TP of frames delivered
by their deadline (any PSNR) / sum of GT objects of every captured frame (late and dropped frames count 0).
  tune : TRAIN pool, seeds 0..29: 14 fixed modes, rv:{mean,gt}:LAM (7 values), tc=tb:{mean,gt}:TAU (5 values);
         plus the task MPC (mean / gt) seeded with the best fixed mode -> runs/uav_semcom/at1/tune.json
  test : TEST pool, fresh seeds 9200..9299 -> runs/uav_semcom/at1/test.json + test.deliv.json.gz
Usage: python at1_run.py tune|test --procs 6
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import at1_task as AT
import ax2_eval as X2

OUT = C.REPO / "runs" / "uav_semcom" / "at1"
EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=False)
LAMS = (0, 2, 5, 10, 20, 40, 80)
TAUS = (0.5, 1.0, 1.5, 2.0, 3.0)
TEST_SEEDS = "9200:9300"
Q3 = "h=fixed_mbt2018-mean_q3"


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def _init():
    AE.init(False)
    X2.DELIV_EXT = True


def _job(j):
    combo, seeds, pool = j
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, -1, s, EXTRA, None, pool)
        rows.append(m)
        dl.append(d)
    return dict(combo=combo, seeds=list(seeds), rows=rows, deliv=dl)


def run(combos, seeds, pool, procs, chunk):
    chunks = [seeds[x:x + chunk] for x in range(0, len(seeds), chunk)]
    jobs = [(c, ch, pool) for c in combos for ch in chunks]
    with Pool(procs, initializer=_init) as p:
        res = p.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault(r["combo"], dict(combo=r["combo"], seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    return merged


class Scorer:
    def __init__(self):
        AE.init(False)
        self.t = AT.tables(AE.G["table"], AE.G["pools"])
        self.rows = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}

    def per_ep(self, deliv, src="gt", ontime=True):
        return np.array([AT.score([ep], self.t[src], self.t["ngt"], self.rows, ontime) for ep in deliv])


def tune_stage(a):
    OUT.mkdir(parents=True, exist_ok=True)
    S = sched()
    t0 = time.time()
    fixed = [f"h=fixed_{m}" for m in C.MODES]
    rules = [f"t=rv:{s}:{l:g}" for s in ("mean", "gt") for l in LAMS] + \
            [f"tc=tb:{s}:{t:g}" for s in ("mean", "gt") for t in TAUS]
    res = run([f"{S}|{c}" for c in fixed + rules], list(C.TRAIN_SEEDS), "train", a.procs, 10)
    sc = Scorer()
    rec = {c.split("|")[1]: float(sc.per_ep(v["deliv"]).mean()) for c, v in res.items()}
    best_fixed = max(fixed, key=rec.get)
    best = {f"{k}:{s}": max((c for c in rules if c.split("=")[1].startswith(f"{k}:{s}:")), key=rec.get)
            for k in ("rv", "tb") for s in ("mean", "gt")}
    res2 = run([f"{S}|{best_fixed}+tmpc={s}" for s in ("mean", "gt")] + [f"{S}|{Q3}+mpc"], list(C.TRAIN_SEEDS),
               "train", a.procs, 10)
    for c, v in res2.items():
        rec[c.split("|")[1]] = float(sc.per_ep(v["deliv"]).mean())
    ret = {c.split("|")[1]: float(np.mean([x["return"] for x in v["rows"]])) for c, v in {**res, **res2}.items()}
    out = dict(wall_s=time.time() - t0, sched=S, seeds=list(C.TRAIN_SEEDS), recall_ontime=rec, a5_return=ret,
               best_fixed=best_fixed, best_rules=best)
    json.dump(out, open(OUT / "tune.json", "w"), indent=1)
    print(f"{S}: best fixed by train on-time recall {best_fixed} ({rec[best_fixed]:.4f}); q3 {rec[Q3]:.4f}")
    for k, v in best.items():
        print(f"  best {k:8s} {v:22s} {rec[v]:.4f}")
    for c in sorted(rec, key=lambda c: -rec[c])[:12]:
        print(f"  {c:40s} recall {rec[c]:.4f}  A5 return {ret[c]:8.2f}")


def arms():
    tu = json.loads((OUT / "tune.json").read_text())
    F, B = tu["best_fixed"], tu["best_rules"]
    flip = lambda c: c.replace(":gt:", ":flip:")
    return {
        # causal (deployable)
        "q3 (H*)": Q3, "F_task": F, "q3+MPC (I*+MPC, A5 objective)": f"{Q3}+mpc",
        "RV-mean": B["rv:mean"], "TB-mean": B["tb:mean"], "TMPC-mean": f"{F}+tmpc=mean",
        # non-deployable task-utility oracles (ground truth of the current image; channel still causal)
        "RV-gt": B["rv:gt"], "TB-gt": B["tb:gt"], "TMPC-gt": f"{F}+tmpc=gt",
        # cross-check: the same oracles selecting with the flipped-image measurement
        "RV-flip": flip(B["rv:gt"]), "TB-flip": flip(B["tb:gt"]), "TMPC-flip": f"{F}+tmpc=flip"}


def test_stage(a):
    S = sched()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    A = arms()
    combos = list(dict.fromkeys(A.values()))
    t0 = time.time()
    res = run([f"{S}|{c}" for c in combos], seeds, "test", a.procs, 25)
    json.dump(dict(wall_s=time.time() - t0, sched=S, seeds=seeds, arms=A,
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res.values()]),
              open(OUT / "test.json", "w"))
    json.dump({c: r["deliv"] for c, r in res.items()}, gzip.open(OUT / "test.deliv.json.gz", "wt"))
    print(f"test done {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "test"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", default=TEST_SEEDS)
    a = ap.parse_args()
    dict(tune=tune_stage, test=test_stage)[a.cmd](a)


if __name__ == "__main__":
    main()
