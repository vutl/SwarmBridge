"""A2 v2.2 impact check (heuristics only, no learning): slotted mode decisions (env v2.1) vs event-driven mode
decisions (env_ev, two-timescale), and the decomposition of the oracle - predicted gap:
  oracle -> oracle_probe : cost of acquiring the information (probe encode time, decision waits for it)
  oracle_probe -> predicted : prediction error
Slot part (move / channel / power) from the v2 heuristics; mode part from the event-greedy mode policy.
Usage: python check_v22.py OUT_JSON --profiles ... --cpu-file ... --modes ...
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import env as E
import env_ev as V
import policies as P

G = {}
SLOT_POLS = ["greedy", "paced_q4", "pacedB0.5", "pacedB1", "stag_near"]


def init(profiles, modes, cpu_file):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["tr"], G["te"] = G["table"].split()
    G["modes"] = modes


def job(a):
    kind, info, wt, name = a
    cfg = dict(mode_info=info, w_time=wt)
    pol = P.from_name(name, G["modes"])
    out = []
    for s in range(1000, 1030):
        if kind == "slotted":
            out.append(P.run_episode(E.UavSemcomEnv(G["table"], G["te"], cfg), pol, s))
        else:
            out.append(V.run_episode_ev(V.UavSemcomEnvEv(G["table"], G["te"], cfg), pol, V.greedy_mode, s))
    return a, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    modes = args.modes.split(",")
    jobs = []
    for wt in (0.0, 0.3):
        for name in SLOT_POLS:
            for info in ("oracle", "predicted"):
                jobs.append(("slotted", info, wt, name))
            for info in ("oracle", "oracle_probe", "predicted", "mean"):
                jobs.append(("event", info, wt, name))
    with Pool(args.procs, initializer=init, initargs=(args.profiles, modes, args.cpu_file)) as pool:
        res = pool.map(job, jobs)
    table = {}
    for (kind, info, wt, name), eps in res:
        table.setdefault(f"wt{wt}", {}).setdefault(name, {})[f"{kind}:{info}"] = dict(
            ret=float(np.mean([e["return"] for e in eps])), util=float(np.mean([e["mean_util"] for e in eps])),
            delay=float(np.mean([e["mean_delay"] for e in eps])), psnr=float(np.mean([e["mean_psnr"] for e in eps])),
            ret_all=[e["return"] for e in eps])
    json.dump(table, open(args.out, "w"), indent=1)
    cols = ["slotted:oracle", "slotted:predicted", "event:oracle", "event:oracle_probe", "event:predicted", "event:mean"]
    for wt, rows in table.items():
        print(f"\n{wt}  return (mean delay s)")
        print(f"{'policy':12s} " + " ".join(f"{c:>20s}" for c in cols))
        for name, r in rows.items():
            print(f"{name:12s} " + " ".join(f"{r[c]['ret']:+8.2f} ({r[c]['delay']:.2f}s)".rjust(20) for c in cols))


if __name__ == "__main__":
    main()
