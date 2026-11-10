"""G-L2 addendum (exploratory, after the gate decision): how much of the RV baseline's strength comes from IDEAL
payload knowledge, and how far the hindsight baseline envelope BL is from single pre-specified policies.

  RV_true  : argmax_m mu_m - lam * b_i(m)       (true payload of every mode; the G-L2 setting)
  RV_probe : argmax_m mu_m - lam * bhat_i(m)    (bhat from the causal AVIF-5 probe, as the A5 UAV: per-mode log-linear
             fit on train, env.ModeTable.predict); the bits CHARGED are the true ones
  F, MX    : fixed mode / train-vertex fixed-mode mixtures (need no payload knowledge)
Reported on test and test-dev at the three G-L2 budgets (recall points at matched mean bits). Writes
runs/uav_semcom/g_l2/caveats.json and prints a markdown table.
"""
import json

import numpy as np

import g2_common as G
from g2_predict import Baselines, Fam

PT = 100.0


def main():
    d = G.Data(True)
    tab = d.tab
    train = np.asarray(d.pools["train"])
    Bs = G.budgets(d)
    menu = list(range(len(d.modes)))
    mu = d.tp[train].mean(0)
    pred_bits = np.array([tab.predict(i)[0] for i in range(len(d.images))])
    pv = json.loads((G.G2 / "predict.json").read_text())
    out = {}
    print("| pool | mức bit | F | MX | RV_probe (triển khai được) | RV_true (payload lý tưởng) | BL (bao hindsight) | "
          "predictor G-L2 | RV_true − RV_probe | predictor − RV_probe |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in ("test", "testdev"):
        I = np.asarray(d.pools[p])
        pos = np.arange(len(I))
        N = d.ngt[I].sum()
        base = Baselines(d, I, train, menu)
        # RV_probe: choose with predicted bits, charge the true bits
        ch = []
        Bh = pred_bits[I][:, menu]
        for lam in G.LAMS:
            j = np.argmin(Bh, 1) if np.isinf(lam) else np.argmax(mu[menu][None, :] - lam * Bh, 1)
            ch.append(np.array(menu)[j])
        rvp = Fam(ch, d, I, d.tp, d.fp, d.tps)
        out[p] = {}
        for m, B in Bs.items():
            r_f = base.F.fixed_at(B, pos, N)[0] * PT
            r_mx = base.MX.at(B, pos, N)[0] * PT
            r_rt = base.RV.at(B, pos, N)[0] * PT
            r_bl = base.at(B, pos)[0] * PT
            r_rp = rvp.at(B, pos, N)[0] * PT
            fr = pv["final"][m]["pools"][p]
            # predictor point is at its own realized bits; compare RV_probe at those bits
            r_rp_at = rvp.at(fr["mbit_per_frame"] * 1e6, pos, N)[0] * PT
            out[p][m] = dict(F=r_f, MX=r_mx, RV_probe=r_rp, RV_true=r_rt, BL=r_bl, pred=fr["recall"],
                             pred_bits=fr["mbit_per_frame"], RV_probe_at_pred_bits=r_rp_at)
            print(f"| {p} | {m} | {r_f:.2f} | {r_mx:.2f} | {r_rp:.2f} | {r_rt:.2f} | {r_bl:.2f} | "
                  f"{fr['recall']:.2f} (@{fr['mbit_per_frame']:.3f}) | {r_rt - r_rp:+.2f} | {fr['recall'] - r_rp_at:+.2f} |")
    (G.G2 / "caveats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
