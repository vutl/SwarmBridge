"""Pilot 0B EXPL reliability (review #13): paired CLUSTER bootstrap over the 11 sequences (not the 459 epochs) and
leave-one-sequence-out sensitivity, from the saved result JSON only (no detector, no image, no training).
Configurations (alpha, tau_decl) are the ones chosen on the same EXPL sequences -> NOT an independent validation.

Effects (per sequence, then averaged = macro):
  C3: per-view F1  T1(q, B) - base(B), base = T0 or T0+ as chosen in the pilot (T0+ == T0 here)
  C4: unique F1    T2(q, B, rho 0) - T1(q, B)   and   T2v - T1
  extra: per-view F1 F(frame level) - T0(frame level)
Usage: python h1_pilot0b_cluster_ci.py RESULT_JSON OUT_JSON [--reps 20000]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def stats(d, seqs, reps, rng):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (reps, len(d)))
    boot = d[idx].mean(1)
    loso = {s: float(np.delete(d, k).mean()) for k, s in enumerate(seqs)}
    worst = min(loso, key=loso.get); best = max(loso, key=loso.get)
    i59 = seqs.index("59")
    d59 = np.delete(d, i59)
    boot59 = d59[rng.integers(0, len(d59), (reps, len(d59)))].mean(1)
    share = d[i59] / d.sum() if d.sum() else float("nan")
    return dict(
        mean=float(d.mean()), median=float(np.median(d)), n_pos=int((d > 0).sum()), n=len(d),
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        p_boot_le0=float((boot <= 0).mean()),
        per_seq=dict(zip(seqs, d.round(4).tolist())),
        loso_mean=loso, loso_min=[worst, loso[worst]], loso_max=[best, loso[best]],
        loso_sign_flips=[s for s, v in loso.items() if np.sign(v) != np.sign(d.mean())],
        without_59=dict(mean=float(d59.mean()), median=float(np.median(d59)), n_pos=int((d59 > 0).sum()),
                        ci95=[float(np.percentile(boot59, 2.5)), float(np.percentile(boot59, 97.5))]),
        seq59_share_of_total_gain=float(share),
        max_single_seq_share=float(d.max() / d.sum()) if d.sum() > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result"); ap.add_argument("out"); ap.add_argument("--reps", type=int, default=20000)
    a = ap.parse_args()
    R = json.loads(Path(a.result).read_text())
    seqs = sorted(R["screening"]["C1"]["per_seq"], key=int)
    rows = R["table"]

    def get(cond, q, B, rho=0.0):
        return next(r for r in rows if r["cond"] == cond and r["q"] == q and r["B"] == B and r["rho"] == rho)
    rng = np.random.default_rng(20261113)
    out = dict(note="EXPLORATORY: cluster bootstrap over 11 sequences; alpha / tau_decl chosen on the same EXPL "
                    "sequences -> not an independent validation", reps=a.reps, seqs=seqs, effects={})
    for q in R["config"]["avif_q"]:
        for B in R["budget"]["levels"]:
            t0, t0p, t1 = get("T0", None, B), get("T0+", None, B), get("T1", q, B)
            base = t0 if t0["f1_view_macro"] >= t0p["f1_view_macro"] else t0p
            d = np.subtract(t1["f1_view_per_seq"], base["f1_view_per_seq"])
            out["effects"][f"C3_T1_minus_{base['cond']}_q{q}_B{B}"] = stats(d, seqs, a.reps, rng)
            for c in ("T2", "T2v"):
                t2 = get(c, q, B)
                d = np.subtract(t2["f1_unique_per_seq"], t1["f1_unique_per_seq"])
                out["effects"][f"C4_{c}_minus_T1_unique_q{q}_B{B}"] = stats(d, seqs, a.reps, rng)
    fl = R["budget"]["frame_level"]
    d = np.subtract(get("F", None, fl)["f1_view_per_seq"], get("T0", None, fl)["f1_view_per_seq"])
    out["effects"][f"extra_F_minus_T0_B{fl}"] = stats(d, seqs, a.reps, rng)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("%-38s %7s %7s %5s %17s %7s | %-14s %-14s | %7s %17s %4s" % (
        "effect", "mean", "median", "pos", "CI95", "P<=0", "LOSO min", "LOSO max", "w/o59", "CI95 w/o59", "pos"))
    for k, s in out["effects"].items():
        print("%-38s %+7.4f %+7.4f %2d/%2d [%+.4f,%+.4f] %7.4f | %-3s %+.4f   %-3s %+.4f   | %+7.4f [%+.4f,%+.4f] %2d/10 flips=%s 59share=%.2f" % (
            k, s["mean"], s["median"], s["n_pos"], s["n"], *s["ci95"], s["p_boot_le0"], s["loso_min"][0], s["loso_min"][1],
            s["loso_max"][0], s["loso_max"][1], s["without_59"]["mean"], *s["without_59"]["ci95"], s["without_59"]["n_pos"],
            s["loso_sign_flips"], s["seq59_share_of_total_gain"]))


if __name__ == "__main__":
    main()
