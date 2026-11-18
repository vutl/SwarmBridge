"""Review #46 step C (C0 + C1). Protocol (pre-registered): reports/scheduling/swarmbridge-atc-protocol.md.

Environment as A-T1 (K5_la, scheduler-aware LA, persistent interference, no SRS), scheduler / movement frozen by A-X9
stage 1, env table WITH the test-dev profiles (a5_eval.init(True)).
  dev  : D2 tune on the TRAIN pool, seeds 0..29 (14 fixed, q3+MPC, rv:SRC:LAM, rvq:SRC:LAM:DELTA for SRC in
         mean/pA/pB/pC/count/gt); D3 predictor selection on the VAL pool, seeds 3000..3019
         -> runs/uav_semcom/atc/dev.json (frozen choices)
  test : TEST-DEV pool, fresh seeds 9600..9699, every arm once -> test.json + test.deliv.json.gz
Predictor compute is charged as env info_extra_s (pB 10 ms, pC 35 ms) in every arm that uses it.
Usage: python atc_run.py dev|test --procs 6
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

OUT = C.REPO / "runs" / "uav_semcom" / "atc"
EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=False)
LAMS = (0, 2, 5, 10, 20, 40, 80)
DELTAS = (0, 1, 2)
SRCS = ("mean", "pA", "pB", "pC", "count", "gt")
PRED = ("pA", "pB", "pC")
COST_S = {"pA": 0.0, "pB": 0.010, "pC": 0.035}
Q3 = "h=fixed_mbt2018-mean_q3"
VAL_SEEDS = list(range(3000, 3020))
TEST_SEEDS = "9600:9700"


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def cost(combo):
    for p, c in COST_S.items():
        if f":{p}:" in combo:
            return c
    return 0.0


def _init():
    AE.init(True)
    X2.DELIV_EXT = True


def _job(j):
    combo, seeds, pool = j
    extra = dict(EXTRA, info_extra_s=cost(combo))
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, -1, s, extra, None, pool)
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
    """Per-episode metrics from delivery tuples (manifest index, mode, ok, ontime); env table with test-dev rows."""

    def __init__(self):
        AE.init(True)
        self.t = AT.tables(AE.G["table"], AE.G["pools"])
        self.rows = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}

    def ep(self, ep, what="recall"):
        tp = gt = fp = 0.0
        per = []
        for mi, mode, ok, ot in ep:
            r = self.rows[mi]
            n = self.t["ngt"][r]
            hit = mode >= 0 and (ok if what == "recall_qmin" else ot)
            x = self.t["gt"][r, mode] if hit else 0.0
            tp += x
            gt += n
            fp += self.t["fp"][r, mode] if hit else 0.0
            per.append(x / n)
        return dict(recall=tp / gt, recall_qmin=tp / gt, macro=float(np.mean(per)), fp=fp / len(ep))[what]

    def per_ep(self, deliv, what="recall"):
        return np.array([self.ep(e, what) for e in deliv])


def dev_stage(a):
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / "pred.json").exists(), "run atc_predict.py first (D1)"
    S = sched()
    t0 = time.time()
    fixed = [f"h=fixed_{m}" for m in C.MODES]
    rv = [f"t=rv:{s}:{l:g}" for s in SRCS for l in LAMS]
    rvq = [f"t=rvq:{s}:{l:g}:{d:g}" for s in SRCS for l in LAMS for d in DELTAS]
    combos = fixed + [f"{Q3}+mpc"] + rv + rvq
    res = run([f"{S}|{c}" for c in combos], list(C.TRAIN_SEEDS), "train", a.procs, 10)
    sc = Scorer()
    rec = {c.split("|")[1]: float(sc.per_ep(v["deliv"]).mean()) for c, v in res.items()}
    recq = {c.split("|")[1]: float(sc.per_ep(v["deliv"], "recall_qmin").mean()) for c, v in res.items()}
    best_fixed = max(fixed, key=rec.get)
    lam = {s: max((c for c in rv if c.startswith(f"t=rv:{s}:")), key=rec.get) for s in SRCS}
    lamq = {s: max((c for c in rvq if c.startswith(f"t=rvq:{s}:")), key=recq.get) for s in SRCS}
    cands = fixed + [f"{Q3}+mpc", lam["mean"]]
    c_star = max(cands, key=rec.get)
    qc_star = max(fixed + [f"{Q3}+mpc", lamq["mean"]], key=recq.get)
    print(f"D2 ({time.time() - t0:.0f}s): best fixed {best_fixed} {rec[best_fixed]:.4f}; C* {c_star} {rec[c_star]:.4f}; "
          f"Q-C* {qc_star} {recq[qc_star]:.4f}")
    for s in SRCS:
        print(f"   rv  {s:6s} {lam[s]:26s} train recall {rec[lam[s]]:.4f}   rvq {lamq[s]:28s} train Qmin-recall "
              f"{recq[lamq[s]]:.4f}")
    # ---- D3: predictor selection on the val pool ----
    vc = [c_star] + [lam[s] for s in PRED + ("count",)]
    vres = run([f"{S}|{c}" for c in dict.fromkeys(vc)], VAL_SEEDS, "val", a.procs, 5)
    vrec = {c.split("|")[1]: float(sc.per_ep(v["deliv"]).mean()) for c, v in vres.items()}
    off = json.loads((OUT / "pred.json").read_text())["val"]
    hv = vrec[lam["count"]] - vrec[c_star]
    log = []
    p_star = "pA"
    if vrec[lam["pB"]] >= vrec[lam[p_star]] + 0.003 and off["pB"]["rmse_log1p"] <= 0.9 * off["pA"]["rmse_log1p"]:
        p_star = "pB"
    p_b = p_star
    log.append(f"after pB check: {p_star}")
    rec_frac = (vrec[lam[p_star]] - vrec[c_star]) / hv if hv > 0 else float("nan")
    consider_c = hv >= 0.010 and not rec_frac >= 0.5
    if consider_c and vrec[lam["pC"]] >= vrec[lam[p_star]] + 0.003 and \
            off["pC"]["rmse_log1p"] <= 0.9 * off[p_star]["rmse_log1p"]:
        p_star = "pC"
    log.append(f"val count headroom {100 * hv:+.2f} pts; recovery of {p_b} {rec_frac:.2f}; pC considered "
               f"{consider_c}; P* = {p_star}")
    print("D3 val recall:", {k: round(100 * v, 2) for k, v in vrec.items()})
    print("D3", "; ".join(log))
    out = dict(sched=S, wall_s=time.time() - t0, train_recall=rec, train_recall_qmin=recq, best_fixed=best_fixed,
               lam=lam, lamq=lamq, C_star=c_star, QC_star=qc_star, val_recall=vrec, val_offline=off,
               val_count_headroom=hv, val_recovery=rec_frac, consider_pC=consider_c, P_star=p_star, log=log,
               train_seeds=list(C.TRAIN_SEEDS), val_seeds=VAL_SEEDS,
               a5_return={c.split("|")[1]: float(np.mean([x["return"] for x in v["rows"]])) for c, v in res.items()})
    json.dump(out, open(OUT / "dev.json", "w"), indent=1)


def arms(dev):
    L, Q = dev["lam"], dev["lamq"]
    A = {"q3": Q3, "F_task": dev["best_fixed"], "q3+MPC": f"{Q3}+mpc", "C*": dev["C_star"]}
    A.update({f"RV-{s}": L[s] for s in SRCS})
    A["Q-C*"] = dev["QC_star"]
    A.update({f"RVQ-{s}": Q[s] for s in SRCS})
    return A


def test_stage(a):
    dev = json.loads((OUT / "dev.json").read_text())
    S = dev["sched"]
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    A = arms(dev)
    combos = list(dict.fromkeys(A.values()))
    t0 = time.time()
    res = run([f"{S}|{c}" for c in combos], seeds, "testdev", a.procs, 25)
    json.dump(dict(wall_s=time.time() - t0, sched=S, seeds=seeds, arms=A, P_star=dev["P_star"],
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res.values()]),
              open(OUT / "test.json", "w"))
    json.dump({c: r["deliv"] for c, r in res.items()}, gzip.open(OUT / "test.deliv.json.gz", "wt"))
    print(f"test done {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev", "test"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", default=TEST_SEEDS)
    a = ap.parse_args()
    dict(dev=dev_stage, test=test_stage)[a.cmd](a)


if __name__ == "__main__":
    main()
