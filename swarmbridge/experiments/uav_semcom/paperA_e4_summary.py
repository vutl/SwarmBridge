"""Paper A E4 / E5 summary (pre-registration: reports/phases/2026-11-18_paperA-writeup/paperA-e4-e5-prereg.md).
E4 per network configuration (H* re-tuned per configuration on the train seeds):
  H1 calibrated MPC - H*, H2 calibrated - RobustMPC-style, H3 calibrated - nominal (paired 95% t-CI, 100 seeds),
  plus pooled - calibrated, and secondary metrics (return per image, Qmin-compliant recall, drop / late / quality
  violation rates, decision times).
E5 on N5x2: calibrated MPC with delayed / lossy rate reports minus immediate reports, and minus nominal MPC.
Output: runs/uav_semcom/paperA_r2/summary_e4.json + printed tables."""
import gzip
import json

import numpy as np

import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci


def main():
    AE.init(False)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    te = json.loads((R.OUT / "e4.json").read_text())
    dl = json.load(gzip.open(R.OUT / "e4.deliv.json.gz", "rt"))
    by = {(r["phy"], r["combo"]): r for r in te["results"]}
    S, best = te["sched"], te["H_net"]

    def ret(n, suf):
        return np.array([x["return"] for x in by[(n, f"{S}|h={best[n]}{suf}")]["rows"]])

    def met(n, suf, key):
        return np.array([x[key] for x in by[(n, f"{S}|h={best[n]}{suf}")]["rows"]])

    def rec(n, suf):
        r = by[(n, f"{S}|h={best[n]}{suf}")]
        v = []
        for ep in dl[f"{n}|{r['combo']}|{r['ckpt']}"]:
            tp = gt = 0.0
            for mi, mode, ok, ot in ep:
                rr = rows_of[mi]
                gt += T["ngt"][rr]
                if mode >= 0 and ok:
                    tp += T["gt"][rr, mode]
            v.append(100 * tp / gt)
        return np.array(v)

    def nimg(n):
        return R.NET[n]["K"] * 52

    def f(v):
        m, lo, hi = v
        return f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]{'*' if lo > 0 or hi < 0 else ' '}"

    out = {"H_net": best, "train_return": te["train_return"], "net": {}, "fb": {}}
    arms = ("", "+mpc", "+mpcnc", "+mpcrh", "+mpcpl")
    pass_h1 = pass_h2 = pass_h3 = 0
    for n in R.NET:
        o = {"H": best[n], "n_img": nimg(n)}
        o["H1_cal_minus_H"] = [float(x) for x in ci(ret(n, "+mpc") - ret(n, ""))]
        o["H2_cal_minus_rh"] = [float(x) for x in ci(ret(n, "+mpc") - ret(n, "+mpcrh"))]
        o["H3_cal_minus_nom"] = [float(x) for x in ci(ret(n, "+mpc") - ret(n, "+mpcnc"))]
        o["pool_minus_cal"] = [float(x) for x in ci(ret(n, "+mpcpl") - ret(n, "+mpc"))]
        o["recall_cal_minus_H"] = [float(x) for x in ci(rec(n, "+mpc") - rec(n, ""))]
        o["recall_cal_minus_rh"] = [float(x) for x in ci(rec(n, "+mpc") - rec(n, "+mpcrh"))]
        o["level"] = {a or "H": dict(ret=float(ret(n, a).mean()), ret_per_img=float(ret(n, a).mean() / nimg(n)),
                                     recall=float(rec(n, a).mean()),
                                     **{k: float(met(n, a, k).mean()) for k in
                                        ("drop_rate", "late_rate", "q_viol_rate", "ok_rate", "ms_slot_p99",
                                         "ms_int_max_p99", "int_over50")}) for a in arms}
        pass_h1 += o["H1_cal_minus_H"][1] > 0
        pass_h2 += o["H2_cal_minus_rh"][1] > 0
        pass_h3 += o["H3_cal_minus_nom"][1] > 0
        out["net"][n] = o
        print(f"== {n} (K={R.NET[n]['K']}, M={R.NET[n]['M']}, H*={best[n]})")
        print(f"   H1 cal - H*      {f(o['H1_cal_minus_H'])}   recall {f(o['recall_cal_minus_H'])}")
        print(f"   H2 cal - RobMPC  {f(o['H2_cal_minus_rh'])}   recall {f(o['recall_cal_minus_rh'])}")
        print(f"   H3 cal - nominal {f(o['H3_cal_minus_nom'])}")
        print(f"   pooled - cal     {f(o['pool_minus_cal'])}")
        for a in arms:
            L = o["level"][a or "H"]
            print(f"     {a or 'H*':8s} ret {L['ret']:8.1f} ({L['ret_per_img']:+.3f}/img) recall {L['recall']:5.1f} "
                  f"ok {L['ok_rate']:.3f} drop {L['drop_rate']:.3f} late {L['late_rate']:.3f} qviol {L['q_viol_rate']:.3f} "
                  f"sched p99 {L['ms_slot_p99']:5.1f} ms  int p99 {L['ms_int_max_p99']:5.1f} ms  >50ms {L['int_over50']:.3f}")
    out["pass"] = dict(H1=int(pass_h1), H2=int(pass_h2), H3=int(pass_h3), of=len(R.NET))
    print(f"PASS counts (CI > 0): H1 {pass_h1}/5  H2 {pass_h2}/5  H3 {pass_h3}/5  "
          f"(rule: H1 and H2 in >= 4/5 -> 'persisted across the evaluated network sizes')")
    # E5: the +mpcfb arms of e4.json are superseded (tracker saw immediate reports); use e5.json (fixed)
    e5 = json.loads((R.OUT / "e5.json").read_text())
    by.update({(r["phy"], r["combo"]): r for r in e5["results"]})
    print("== E5 (N5x2): delayed / lossy rate reports (e5.json, calibrator AND tracker delayed)")
    for fb in te["fb"]:
        a = f"+mpc{fb}"
        o = {"minus_immediate": [float(x) for x in ci(ret("N5x2", a) - ret("N5x2", "+mpc"))],
             "minus_nominal": [float(x) for x in ci(ret("N5x2", a) - ret("N5x2", "+mpcnc"))],
             "minus_H": [float(x) for x in ci(ret("N5x2", a) - ret("N5x2", ""))]}
        out["fb"][fb] = o
        print(f"   {fb:7s} - immediate {f(o['minus_immediate'])}   - nominal {f(o['minus_nominal'])}   - H* {f(o['minus_H'])}")
    json.dump(out, open(R.OUT / "summary_e4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
