"""Paper A robustness round summary (E1 PHY variants, E2 RobustMPC-style baseline, E3 test-dev).
Per PHY: key paired contrasts [95% CI over environment seeds], PPO arms averaged over the 3 checkpoints per seed;
Qmin-compliant and on-time detector recall of the main arms. Usage: python paperA_r2_summary.py (WSL env)
-> runs/uav_semcom/paperA_r2/summary.json + printed tables
"""
import collections
import gzip
import json

import numpy as np

import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci, fmt


def main():
    AE.init(True)                      # table with test-dev rows (covers the test pool too)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    out = {}
    for name in ("e12", "e3"):
        te = json.loads((R.OUT / f"{name}.json").read_text())
        dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
        S, H = te["sched"], te["H"]
        by = collections.defaultdict(list)
        for r in te["results"]:
            by[(r["phy"], r["combo"])].append(r)

        def ret(phy, c):
            rs = by[(phy, c)]
            assert rs and all(r["seeds"] == te["seeds"] for r in rs), (phy, c)
            return np.mean([[x["return"] for x in r["rows"]] for r in rs], 0)

        def recall(phy, c, ontime):
            vals = []
            for r in by[(phy, c)]:
                v = []
                for ep in dl[f"{phy}|{c}|{r['ckpt']}"]:
                    tp = gt = 0.0
                    for mi, mode, ok, ot in ep:
                        rr = rows_of[mi]
                        gt += T["ngt"][rr]
                        if mode >= 0 and (ot if ontime else ok):
                            tp += T["gt"][rr, mode]
                    v.append(tp / gt)
                vals.append(v)
            return float(np.mean(vals))

        out[name] = {}
        print(f"\n######## {name}: pool {te['pool']}, seeds {te['seeds'][0]}..{te['seeds'][-1]}")
        for phy in te["phy"]:
            nat = H["S"] if phy == "S" else H["L"]
            oth = H["L"] if phy == "S" else H["S"]
            c = dict(tb=f"tbest|heur", tbp=f"tbest|ppo", h=f"{S}|h={nat}", hm=f"{S}|h={nat}+mpc",
                     hn=f"{S}|h={nat}+mpcnc", hr=f"{S}|h={nat}+mpcrh", ho=f"{S}|h={oth}", p=f"{S}|ppo",
                     pm=f"{S}|ppo+mpc", la=f"{S}|ppola", lam=f"{S}|ppola+mpc", lar=f"{S}|ppola+mpcrh")
            R_ = {k: ret(phy, v) for k, v in c.items() if (phy, v) in by}
            pairs = [("PPO-S - T:best (block A, same slot rule)", "tbp", "tb"),
                     ("H* + cal. MPC - H* + nominal MPC", "hm", "hn"),
                     ("H* + cal. MPC - H* + RobustMPC-style", "hm", "hr"),
                     ("H* + RobustMPC-style - H* + nominal MPC", "hr", "hn"),
                     ("H* + cal. MPC - H* (no MPC)", "hm", "h"),
                     ("H* tuned on other PHY - H* + cal. MPC", "ho", "hm"),
                     ("PPO-S (JA) - H* + cal. MPC", "p", "hm"),
                     ("PPO-S + cal. MPC - H* + cal. MPC", "pm", "hm"),
                     ("PPO-LA - H* + cal. MPC", "la", "hm"),
                     ("PPO-LA + cal. MPC - H* + cal. MPC", "lam", "hm"),
                     ("PPO-LA + cal. MPC - PPO-LA + RobustMPC-style", "lam", "lar")]
            res = {}
            print(f"== {phy}: returns " + ", ".join(f"{k} {v.mean():.1f}" for k, v in R_.items()))
            for lab, a, b in pairs:
                if a in R_ and b in R_:
                    d = [float(x) for x in ci(R_[a] - R_[b])]
                    res[lab] = d
                    print(f"   {lab:46s} {fmt(d)}")
            rec = {}
            for k in ("tb", "tbp", "h", "hm", "hr", "lam"):
                if k in R_:
                    rec[k] = dict(qmin=recall(phy, c[k], False), ontime=recall(phy, c[k], True))
            print("   recall (Qmin / on-time): " + ", ".join(f"{k} {100 * v['qmin']:.1f}/{100 * v['ontime']:.1f}"
                                                         for k, v in rec.items()))
            out[name][phy] = dict(returns={k: float(v.mean()) for k, v in R_.items()}, contrasts=res, recall=rec)
    json.dump(out, open(R.OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
