"""Sanity run of the heuristic baselines on env v2 (no learning): every fixed mode, per-image greedy, pacing and
staggering, under the three information levels (oracle / predicted / mean) and with / without mission-time cost,
on the 30 shared test seeds. Purpose: check the corrected environment behaves sensibly and see where the current
parameter regime leaves room for decisions -- NOT a method comparison.
Usage: python sanity_v2.py OUT_DIR --profiles ... --cpu-file ... --modes ...
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import env as E
import policies as P

G = {}


def init(profiles, modes, cpu_file):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["tr"], G["te"] = G["table"].split()
    G["modes"] = modes


def job(a):
    name, cfg = a
    pol = P.from_name(name, G["modes"])
    eps = [P.run_episode(E.UavSemcomEnv(G["table"], G["te"], cfg), pol, s) for s in range(1000, 1030)]
    return name, json.dumps(cfg, sort_keys=True), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    args = ap.parse_args()
    modes = args.modes.split(",")
    names = [f"fixed_{m}" for m in modes] + ["greedy", "greedy_detour", "paced_q0", "paced_q3", "stag_idx"]
    cfgs = [dict(mode_info=mi, w_time=w) for w in (0.0, 0.3) for mi in ("oracle", "predicted", "mean")]
    jobs = [(n, c) for c in cfgs for n in names if not (n.startswith("fixed_") and c["mode_info"] != "oracle")]
    with Pool(6, initializer=init, initargs=(args.profiles, modes, args.cpu_file)) as pool:
        res = pool.map(job, jobs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["return", "mean_util", "ok_rate", "drop_rate", "q_viol_rate", "mean_psnr", "mean_delay", "mission_s",
            "energy_kj", "airtime_s"]
    table = {}
    for name, cfg, eps in res:
        table.setdefault(cfg, {})[name] = {k: float(np.mean([e[k] for e in eps])) for k in keys}
        table[cfg][name]["return_all"] = [e["return"] for e in eps]
    json.dump(table, open(out / "sanity_v2.json", "w"), indent=1)
    lines = ["# env v2 heuristic sanity run (30 test seeds; fixed modes do not depend on mode_info)", ""]
    for cfg, rows in table.items():
        lines += [f"## cfg {cfg}", "", "| policy | return | util | ok | dropped | PSNR<Qmin | PSNR | delay (s) | "
                  "mission (s) | energy (kJ) | airtime (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in sorted(rows.items(), key=lambda x: -x[1]["return"]):
            lines.append(f"| {n} | {r['return']:+.2f} | {r['mean_util']:+.3f} | {r['ok_rate']:.3f} | "
                         f"{r['drop_rate']:.3f} | {r['q_viol_rate']:.3f} | {r['mean_psnr']:.1f} | "
                         f"{r['mean_delay']:.2f} | {r['mission_s']:.1f} | {r['energy_kj']:.1f} | {r['airtime_s']:.1f} |")
        lines.append("")
    (out / "sanity_v2.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
