"""A-T1 EXPLORATORY diagnostic (added after the pre-registered test; not part of the gate): how much of the oracle
gain needs per-mode, per-image detector outcomes, and how much is just knowing how many objects a frame holds?
  rv:count:LAM  U_i(m) = n_gt_i / mean_train(n_gt) * mu_m   (object count only; same payload-aware rule as RV)
LAM tuned on TRAIN seeds 0..29 (same grid as A-T1), best member run on the A-T1 test seeds 9200..9299.
Output: runs/uav_semcom/at1/explore_count.json
"""
import gzip
import json

import numpy as np

import a5_common as C
import at1_run as R
from ax2_summary import ci, fmt


def main():
    S = R.sched()
    names = [f"t=rv:count:{l:g}" for l in R.LAMS]
    tr = R.run([f"{S}|{c}" for c in names], list(C.TRAIN_SEEDS), "train", 6, 10)
    sc = R.Scorer()
    trec = {c.split("|")[1]: float(sc.per_ep(v["deliv"]).mean()) for c, v in tr.items()}
    best = max(names, key=trec.get)
    te = R.run([f"{S}|{best}"], list(range(9200, 9300)), "test", 6, 25)[f"{S}|{best}"]
    dl = json.load(gzip.open(R.OUT / "test.deliv.json.gz", "rt"))
    arms = json.loads((R.OUT / "test.json").read_text())["arms"]
    r_cnt = sc.per_ep(te["deliv"])
    out = dict(train_recall=trec, best=best, test_recall=float(r_cnt.mean()),
               test_recall_qmin=float(sc.per_ep(te["deliv"], ontime=False).mean()),
               ontime=float(np.mean([np.mean([x[3] for x in ep]) for ep in te["deliv"]])),
               mbit=float(np.mean([x["delivered_mbit"] for x in te["rows"]])),
               energy_kj=float(np.mean([x["energy_kj"] for x in te["rows"]])), contrasts={})
    print("train:", {k: round(100 * v, 2) for k, v in trec.items()}, "best", best)
    print(f"test {best}: on-time recall {100 * out['test_recall']:.2f}  Qmin-recall {100 * out['test_recall_qmin']:.2f}"
          f"  on-time {out['ontime']:.3f}  Mbit {out['mbit']:.1f}  E {out['energy_kj']:.1f}")
    for lab in ("q3+MPC (I*+MPC, A5 objective)", "RV-mean", "RV-gt"):
        other = sc.per_ep(dl[f"{S}|{arms[lab]}"])
        c = [100 * x for x in ci(r_cnt - other)]
        out["contrasts"][lab] = c
        print(f"   count-RV - {lab:32s} {fmt(c)}")
    json.dump(out, open(R.OUT / "explore_count.json", "w"), indent=1)


if __name__ == "__main__":
    main()
