"""A5: apply the pre-registered regime-selection rule (reports/archive/uav-semcom-a/uav-semcom-a5.md section 0.2) to the VALIDATION sweep.
Heuristics only (no PPO result exists or is read).
  R1 = ref, R2 = K5;
  R3 = among regimes not in {ref, K*, wt*} with non-trivial control (best tuned - greedy >= 2 on val), the one where the
       A4 rules help least: min [ T:reuse - max(T:base, T:budget) ]; if the runner-up is within 0.5, the one with the
       larger adaptivity value T:reuse - T:fixed.
Usage: python a5_select.py SWEEP_VAL_JSON OUT_JSON
"""
import json
import sys

import numpy as np


def main():
    d = json.load(open(sys.argv[1]))
    R = d["results"]
    mean = {r: {sp: float(np.mean([e["return"] for e in eps])) for sp, eps in v.items()} for r, v in R.items()}
    rows = []
    for r, m in mean.items():
        if r == "ref" or r.startswith("K") or r.startswith("wt"):
            continue
        best = max(m["T:base"], m["T:budget"], m["T:reuse"], m["T:fixed"])
        nontrivial = best - m["H:greedy"] >= 2.0
        rows.append(dict(regime=r, reuse_gain=m["T:reuse"] - max(m["T:base"], m["T:budget"]),
                         adapt=m["T:reuse"] - m["T:fixed"], control=best - m["H:greedy"], eligible=nontrivial))
    cand = sorted((x for x in rows if x["eligible"]), key=lambda x: x["reuse_gain"])
    pick = cand[0]
    if len(cand) > 1 and cand[1]["reuse_gain"] - cand[0]["reuse_gain"] < 0.5:
        pick = max(cand[:2], key=lambda x: x["adapt"])
    out = dict(rule="reports/archive/uav-semcom-a/uav-semcom-a5.md section 0.2", R1="ref", R2="K5", R3=pick["regime"],
               candidates=sorted(rows, key=lambda x: x["reuse_gain"]))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for x in out["candidates"]:
        print(f"{x['regime']:14s} reuse_gain {x['reuse_gain']:+6.2f} adapt {x['adapt']:+6.2f} "
              f"control {x['control']:+6.2f} {'eligible' if x['eligible'] else '-'}")
    print("SELECTED", out["R1"], out["R2"], out["R3"])


if __name__ == "__main__":
    main()
