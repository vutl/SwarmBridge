"""Paper A E7 summary (pre-registration: reports/phases/2026-11-18_paperA-writeup/paperA-e7-rl-families-prereg.md).
Incumbents H* (fixed mbt2018-mean q3), PPO-LA (A-X9), QR-DQN, Discrete SAC; arms X, X + calibrated MPC (+mpc),
X + nominal MPC (+mpcnc). Pools: test (seeds 13000..13099) and test-dev (14000..14099, replication).
RQ-B (registered): B1 = X+cal - X, B2 = X+cal - X+nominal, learned arms averaged over 3 training seeds per environment
seed, paired 95% t-CI over 100 environment seeds; the claim needs all six test-pool CIs > 0.
RQ-A (descriptive): X - H*, X+cal - H*+cal, X - PPO-LA. Also: per-training-seed contrasts, Qmin-compliant recall,
decision time, validation learning curves, pairing check against E4 N5x2 (same test seeds).
Output: runs/uav_semcom/paperA_r2/summary_e7.json + printed tables."""
import gzip
import json

import numpy as np

import a5_common as C
import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci

INC = [("H*", "h=fixed_mbt2018-mean_q3"), ("PPO-LA", "ppola"), ("QR-DQN", "qrla"), ("Discrete SAC", "dsla")]
ARMS = [("X", ""), ("X+cal", "+mpc"), ("X+nom", "+mpcnc")]
POOLS = [("test", "e7_test", False, list(range(13000, 13100))), ("testdev", "e7_testdev", True, list(range(14000, 14100)))]
TRAIN = {"PPO-LA": [C.REPO / "runs/uav_semcom/ax9/train" / f"ppo_la_s{s}.json" for s in range(3)],
         "QR-DQN": [C.REPO / "runs/uav_semcom/paperA_rl/train" / f"qrdqn_{t}_s{s}.train.json"
                    for t, s in (("lr1e-4", 0), ("lr1e-4", 1), ("lr1e-4", 2), ("lr5e-4", 0))],
         "Discrete SAC": [C.REPO / "runs/uav_semcom/paperA_rl/train" / f"dsac_{t}_s{s}.train.json"
                          for t, s in (("ent0.3", 0), ("ent0.3", 1), ("ent0.3", 2), ("ent0.6", 0))]}
METRICS = ("ok_rate", "drop_rate", "q_viol_rate", "ms_mode_p99", "ms_int_max_p99", "int_over50")


def f(v):
    m, lo, hi = v
    return f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]{'*' if lo > 0 or hi < 0 else ' '}"


def L(v):
    return [float(x) for x in v]


def pool_summary(name, testdev, seeds):
    AE.init(testdev)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    te = json.loads((R.OUT / f"{name}.json").read_text())
    dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
    S = te["sched"]

    def recall(eps):                                   # Qmin-compliant recall (as E4 / E6)
        v = []
        for ep in eps:
            tp = gt = 0.0
            for mi, mode, ok, ot in ep:
                r = rows_of[mi]
                gt += T["ngt"][r]
                if mode >= 0 and ok:
                    tp += T["gt"][r, mode]
            v.append(100 * tp / gt)
        return np.array(v)

    D = {}                                             # (incumbent, arm) -> dict(ret [n_ckpt x 100], rec, rows)
    for r in te["results"]:
        assert r["seeds"] == seeds and len(r["rows"]) == 100
        D.setdefault(r["combo"], []).append((r["ckpt"], r))
    data = {}
    for lab, inc in INC:
        for arm, suf in ARMS:
            items = sorted(D[f"{S}|{inc}{suf}"], key=lambda x: x[0])
            assert len(items) == (1 if lab == "H*" else 3)
            data[lab, arm] = dict(
                ret=np.array([[x["return"] for x in r["rows"]] for _, r in items]),
                rec=np.array([recall(dl[f"{r['phy']}|{r['combo']}|{r['ckpt']}"]) for _, r in items]),
                met={k: float(np.mean([[x[k] for x in r["rows"]] for _, r in items])) for k in METRICS})

    def m(lab, arm, k="ret"):
        return data[lab, arm][k].mean(0)               # average over training seeds per environment seed

    out = {"levels": {}, "B": {}, "B_per_seed": {}, "A": {}, "H_ref": {}}
    print(f"\n######## {name} (seeds {seeds[0]}..{seeds[-1]})")
    print("levels (mean over env seeds; learned = mean of 3 training seeds; per-seed returns in brackets)")
    for lab, _ in INC:
        for arm, _ in ARMS:
            d = data[lab, arm]
            lv = dict(ret=float(d["ret"].mean()), recall=float(d["rec"].mean()),
                      ret_per_seed=[float(x) for x in d["ret"].mean(1)], **d["met"])
            out["levels"][f"{lab}|{arm}"] = lv
            ps = " ".join(f"{x:7.1f}" for x in lv["ret_per_seed"]) if lab != "H*" else ""
            print(f"  {lab:12s} {arm:6s} ret {lv['ret']:7.1f} [{ps:23s}] recall {lv['recall']:5.1f} ok {lv['ok_rate']:.3f} "
                  f"drop {lv['drop_rate']:.3f} mode p99 {lv['ms_mode_p99']:6.2f} ms int p99 {lv['ms_int_max_p99']:5.1f} ms "
                  f">50ms {lv['int_over50']:.3f}")
    print("RQ-B (registered): B1 = X+cal - X, B2 = X+cal - X+nominal; recall difference for B1")
    out["H_ref"] = dict(B1=L(ci(m("H*", "X+cal") - m("H*", "X"))), B2=L(ci(m("H*", "X+cal") - m("H*", "X+nom"))),
                        B1_recall=L(ci(m("H*", "X+cal", "rec") - m("H*", "X", "rec"))))
    print(f"  (ref) H*       B1 {f(out['H_ref']['B1'])}  B2 {f(out['H_ref']['B2'])}  recall B1 {f(out['H_ref']['B1_recall'])}")
    for lab, _ in INC[1:]:
        b = dict(B1=L(ci(m(lab, "X+cal") - m(lab, "X"))), B2=L(ci(m(lab, "X+cal") - m(lab, "X+nom"))),
                 B1_recall=L(ci(m(lab, "X+cal", "rec") - m(lab, "X", "rec"))))
        out["B"][lab] = b
        print(f"  {lab:12s}   B1 {f(b['B1'])}  B2 {f(b['B2'])}  recall B1 {f(b['B1_recall'])}")
        ps = []
        for s in range(3):
            g = lambda arm, k="ret": data[lab, arm][k][s]
            ps.append(dict(B1=L(ci(g("X+cal") - g("X"))), B2=L(ci(g("X+cal") - g("X+nom"))),
                           B1_recall=L(ci(g("X+cal", "rec") - g("X", "rec")))))
            print(f"      seed {s}     B1 {f(ps[-1]['B1'])}  B2 {f(ps[-1]['B2'])}  recall B1 {f(ps[-1]['B1_recall'])}")
        out["B_per_seed"][lab] = ps
    npass = sum(out["B"][lab][k][1] > 0 for lab, _ in INC[1:] for k in ("B1", "B2"))
    out["B_pass"] = dict(n=int(npass), of=6)
    print(f"  RQ-B CIs > 0: {npass}/6 (claim needs 6/6 on test)")
    print("RQ-A (descriptive): return and recall differences")
    for lab, _ in INC[1:]:
        a = {"X - H*": L(ci(m(lab, "X") - m("H*", "X"))),
             "X+cal - H*+cal": L(ci(m(lab, "X+cal") - m("H*", "X+cal"))),
             "X - PPO-LA": L(ci(m(lab, "X") - m("PPO-LA", "X"))) if lab != "PPO-LA" else None,
             "recall X - H*": L(ci(m(lab, "X", "rec") - m("H*", "X", "rec"))),
             "recall X+cal - H*+cal": L(ci(m(lab, "X+cal", "rec") - m("H*", "X+cal", "rec")))}
        a["per_seed X - H*"] = [L(ci(data[lab, "X"]["ret"][s] - m("H*", "X"))) for s in range(3)]
        a["per_seed X+cal - H*+cal"] = [L(ci(data[lab, "X+cal"]["ret"][s] - m("H*", "X+cal"))) for s in range(3)]
        out["A"][lab] = a
        print(f"  {lab:12s} X - H* {f(a['X - H*'])}  X+cal - H*+cal {f(a['X+cal - H*+cal'])}"
              + (f"  X - PPO-LA {f(a['X - PPO-LA'])}" if a["X - PPO-LA"] else ""))
        print(f"  {'':12s} recall X - H* {f(a['recall X - H*'])}  recall X+cal - H*+cal {f(a['recall X+cal - H*+cal'])}")
        print(f"  {'':12s} per seed X - H*: " + "  ".join(f(v) for v in a["per_seed X - H*"]))
        print(f"  {'':12s} per seed X+cal - H*+cal: " + "  ".join(f(v) for v in a["per_seed X+cal - H*+cal"]))
    return out, data


def main():
    out = {}
    for pool, name, td, seeds in POOLS:
        out[pool], data = pool_summary(name, td, seeds)
        if pool == "test":                             # pairing check: H* arms vs E4 N5x2 (same seeds, same PHY)
            e4 = json.loads((R.OUT / "e4.json").read_text())
            ref = {r["combo"]: np.array([x["return"] for x in r["rows"]]) for r in e4["results"] if r["phy"] == "N5x2"}
            chk = {}
            for arm, suf in ARMS:
                e4v = ref[f"{e4['sched']}|h={e4['H_net']['N5x2']}{suf}"]
                chk[arm] = float(np.abs(data["H*", arm]["ret"][0] - e4v).max())
            out["pairing_check_vs_E4_N5x2_max_abs_diff"] = chk
            print(f"\n[check] H* arms vs E4 N5x2 (max |diff| over 100 seeds): {chk}")
    curves = {}
    print("\nvalidation learning curves (val return per 200k decisions; * = selected checkpoint)")
    for lab, paths in TRAIN.items():
        for p in paths:
            t = json.loads(p.read_text())
            c = [(int(r["steps"]), float(r["val_return"])) for r in t["log"]]
            curves[p.name] = dict(best_steps=t["best_steps"], curve=c)
            print(f"  {p.name:28s} " + " ".join(f"{v:6.1f}{'*' if s == t['best_steps'] else ' '}" for s, v in c))
    out["val_curves"] = curves
    json.dump(out, open(R.OUT / "summary_e7.json", "w"), indent=1)


if __name__ == "__main__":
    main()
