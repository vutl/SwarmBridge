"""E1b summary: H* re-tuned per LA variant (runs/uav_semcom/paperA_r2/e1b.json) paired with e12 (same seeds).
Usage: python paperA_r2b_summary.py -> runs/uav_semcom/paperA_r2/summary_e1b.json"""
import json

import numpy as np

import paperA_r2 as R
from ax2_summary import ci, fmt


def load(name):
    te = json.loads((R.OUT / f"{name}.json").read_text())
    out = {}
    for r in te["results"]:
        out.setdefault((r["phy"], r["combo"]), []).append([x["return"] for x in r["rows"]])
    return {k: np.mean(v, 0) for k, v in out.items()}, te


def main():
    a, te12 = load("e12")
    b, te1b = load("e1b")
    assert te12["seeds"] == te1b["seeds"]
    S = te12["sched"]
    out = {}
    for v, h in te1b["H_retuned"].items():
        hv, hvm = b.get((v, f"{S}|h={h}"), a.get((v, f"{S}|h={h}"))), b.get((v, f"{S}|h={h}+mpc"), a.get((v, f"{S}|h={h}+mpc")))
        lam, la = a[(v, f"{S}|ppola+mpc")], a[(v, f"{S}|ppola")]
        hl_m = a[(v, f"{S}|h=fixed_mbt2018-mean_q3+mpc")]
        c = {"H*_V (re-tuned) + cal. MPC - H*_L (frozen) + cal. MPC": ci(hvm - hl_m),
             "PPO-LA + cal. MPC - H*_V (re-tuned) + cal. MPC": ci(lam - hvm),
             "PPO-LA - H*_V (re-tuned)": ci(la - hv),
             "H*_V + cal. MPC - H*_V": ci(hvm - hv)}
        out[v] = dict(H_retuned=h, returns=dict(H_V=float(hv.mean()), H_V_mpc=float(hvm.mean()),
                                                PPO_LA_mpc=float(lam.mean())),
                      contrasts={k: [float(x) for x in d] for k, d in c.items()})
        print(f"== {v}: re-tuned H* = {h}")
        for k, d in c.items():
            print(f"   {k:52s} {fmt(d)}")
    json.dump(out, open(R.OUT / "summary_e1b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
