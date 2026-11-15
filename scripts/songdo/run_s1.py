"""Run Songdo S1 diagnostics (review #25) on the VALIDATION day: lane tables from one session (e.g. Oct 6 AM3),
simulation on another (e.g. Oct 6 PM1), all intersections. CPU only.
Outputs per-intersection results, fleet offered load (sum over intersections per second), and capacity checks.
Usage: python run_s1.py LANE_ROOT EVAL_ROOT CODE_DIR OUT_JSON [PROCS]"""
import glob
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EPS = (0.5, 1.0, 2.0, 4.0)
KINDS = ("hold", "cv", "lane")
LOSS = (0.1, 0.3)


def csv_of(root, X):
    f = glob.glob(str(Path(root) / X / "files" / "*.csv"))
    return f[0] if f else None


def one(args):
    X, lane_root, eval_root, code = args
    sys.path.insert(0, code)
    import s1 as S
    fe, fl = csv_of(eval_root, X), csv_of(lane_root, X)
    if fe is None or fl is None:
        return X, None
    d, lt = S.load_session(fe), S.lane_table(S.load_session(fl))
    runs = []
    for kind in KINDS:
        for eps in EPS:
            runs.append(S.simulate(d, kind, eps, lt))
    for kind in ("cv", "lane"):
        for p in LOSS:
            runs.append(S.simulate(d, kind, 1.0, lt, p_loss=p, seed=0))
    return X, dict(lane_keys=len(lt["keys"]) - 1, m0=S.m0_bits(d), full=S.full_state_bits(d), runs=runs)


def main():
    lane_root, eval_root, code, out = sys.argv[1:5]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    Xs = sorted(p.name for p in Path(eval_root).iterdir() if p.is_dir())
    with Pool(procs) as pool:
        res = dict(pool.map(one, [(X, lane_root, eval_root, code) for X in Xs]))
    res = {X: v for X, v in res.items() if v}
    # fleet offered load per second for each configuration (sum over intersections observed at that second)
    fleet = {}
    n_cfg = len(next(iter(res.values()))["runs"])
    for i in range(n_cfg):
        load = {}
        for v in res.values():
            for s, b in v["runs"][i]["ul_ip_bits_by_sec"].items():
                load[int(s)] = load.get(int(s), 0) + b
        r0 = next(iter(res.values()))["runs"][i]
        arr = np.array(list(load.values()), float)
        fleet[f'{r0["kind"]}/eps{r0["eps"]}/loss{r0["p_loss"]}'] = dict(
            seconds=len(arr), p50_kbps=float(np.percentile(arr, 50) / 1e3), p95_kbps=float(np.percentile(arr, 95) / 1e3),
            max_kbps=float(arr.max() / 1e3))
    for v in res.values():
        for r in v["runs"]:
            r.pop("ul_ip_bits_by_sec", None)
    shannon = {f"1MHz_SNR{s}dB_Mbps": 1.0 * math.log2(1 + 10 ** (s / 10)) for s in (0, 10, 20)}
    caps = dict(source="Zeng, Wu, Zhang, Accessing from the sky (Proc. IEEE 2019), Tables I-II (3GPP; app requirements)",
                payload_ul_3gpp_Mbps=50.0, surveillance_ul_Mbps=10.0, surveillance_latency_ms=3000,
                cnpc_ul_kbps=(60, 100), cnpc_latency_ms=50, shared_1MHz_uplink=shannon)
    json.dump(dict(split="validation day (lane table and evaluation sessions given on the command line)",
                   lane_root=lane_root, eval_root=eval_root, eps=EPS, loss=LOSS, per_intersection=res,
                   fleet_offered_load=fleet, capacity_scenarios=caps), open(out, "w"), indent=1, default=str)
    print(json.dumps(fleet, indent=1))


if __name__ == "__main__":
    main()
