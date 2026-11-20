"""Paper A (rewrite after R2): all numbers of Table 1 and Fig. 2 from the robustness round (no new runs).
Configurations: S, L, L_fade, L_bursty, L_badfb (test pool, seeds 9900..9999) and TD-S, TD-L (test-dev, 12000..12099).
Contrasts (paired over environment seeds, PPO arms averaged over 3 checkpoints per seed; 95% t-CI):
  A  PPO-S - T:best            (block A, same T:best slot rule)        return and Qmin-compliant recall
  B  PPO-LA + MPC - H* + MPC   (H* frozen from nominal L / train images) return and recall   [S: PPO-S + MPC]
  C  PPO-LA + MPC - H*_V + MPC (H* re-tuned on the variant's train seeds) return              [variants only]
  D  calibrated MPC - nominal MPC            (incumbent H*)               return
  E  calibrated MPC - RobustMPC-style MPC    (incumbent H*)               return
  F  SNR-binned - pooled calibration         (incumbent H*)               return
Output: runs/uav_semcom/paperA_r2/paper_numbers.json
"""
import collections
import gzip
import json

import numpy as np

import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci

CONF = [("S", "e12", "S"), ("L", "e12", "L"), ("L_fade", "e12", "L_fade"), ("L_bursty", "e12", "L_bursty"),
        ("L_badfb", "e12", "L_badfb"), ("TD-S", "e3", "S"), ("TD-L", "e3", "L")]
ABL = {"e12": "e2b_test", "e3": "e2b_testdev"}


def main():
    AE.init(True)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    cache = {}

    def load(name):
        if name not in cache:
            te = json.loads((R.OUT / f"{name}.json").read_text())
            dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
            by = collections.defaultdict(list)
            for r in te["results"]:
                by[(r["phy"], r["combo"])].append(r)
            cache[name] = (te, dl, by)
        return cache[name]

    def ret(name, phy, combo):
        te, dl, by = load(name)
        rs = by[(phy, combo)]
        assert rs, (name, phy, combo)
        return np.mean([[x["return"] for x in r["rows"]] for r in rs], 0)

    def rec(name, phy, combo):
        te, dl, by = load(name)
        out = []
        for r in by[(phy, combo)]:
            v = []
            for ep in dl[f"{phy}|{combo}|{r['ckpt']}"]:
                tp = gt = 0.0
                for mi, mode, ok, ot in ep:
                    rr = rows_of[mi]
                    gt += T["ngt"][rr]
                    if mode >= 0 and ok:
                        tp += T["gt"][rr, mode]
                v.append(100 * tp / gt)
            out.append(v)
        return np.mean(out, 0)

    e1b = json.loads((R.OUT / "e1b.json").read_text())
    res = {}
    for lab, name, phy in CONF:
        te, _, _ = load(name)
        S = te["sched"]
        h = te["H"]["S"] if phy == "S" else te["H"]["L"]
        row = {}
        row["A_return"] = ci(ret(name, phy, "tbest|ppo") - ret(name, phy, "tbest|heur"))
        row["A_recall"] = ci(rec(name, phy, "tbest|ppo") - rec(name, phy, "tbest|heur"))
        learned = f"{S}|ppo+mpc" if phy == "S" else f"{S}|ppola+mpc"
        row["B_return"] = ci(ret(name, phy, learned) - ret(name, phy, f"{S}|h={h}+mpc"))
        row["B_recall"] = ci(rec(name, phy, learned) - rec(name, phy, f"{S}|h={h}+mpc"))
        if phy in e1b["H_retuned"]:
            hv = e1b["H_retuned"][phy]
            src = "e1b" if hv != h else name
            if src == "e1b":
                load("e1b")
            row["C_return"] = ci(ret(name, phy, learned) - ret(src, phy, f"{S}|h={hv}+mpc"))
            row["C_H_retuned"] = hv
        hm = ret(name, phy, f"{S}|h={h}+mpc")
        row["D_return"] = ci(hm - ret(name, phy, f"{S}|h={h}+mpcnc"))
        row["E_return"] = ci(hm - ret(name, phy, f"{S}|h={h}+mpcrh"))
        row["F_return"] = ci(hm - ret(ABL[name], phy, f"{S}|h={h}+mpcpl"))
        row = {k: ([float(x) for x in v] if not isinstance(v, str) else v) for k, v in row.items()}
        res[lab] = row
        print(lab, {k: (round(v[0], 2) if isinstance(v, list) else v) for k, v in row.items()})
    json.dump(res, open(R.OUT / "paper_numbers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
