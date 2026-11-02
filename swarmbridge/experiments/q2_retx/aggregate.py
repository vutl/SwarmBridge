"""Aggregate Q2 retransmission runs over seeds.

    python -m swarmbridge.experiments.q2_retx.aggregate "runs/q2_retx_v2_s*.json"

Prints mIoU (mean±std) vs extra budget per policy and condition, and the bit saving of the
uncertainty policy: how many times more extra bits a baseline needs to reach the mIoU that
the uncertainty policy gets at each budget (linear interpolation on the baseline's curve;
"> max" when the baseline never gets there within the tested budgets).
"""
import glob
import json
import sys

import numpy as np

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
runs = [json.load(open(f))["results"] for f in files]
print(f"{len(runs)} seeds: {files}")


def curve(c, pol, key="miou"):
    return np.array([[pt[key] for pt in r[c]["curves"][pol]] for r in runs])  # (seeds, budgets)


def bits_needed(kb, miou_curve, target):
    """extra kbit a policy needs to reach `target` on its (monotone-ish) curve."""
    if target <= miou_curve[0]:
        return 0.0
    for k in range(1, len(kb)):
        if miou_curve[k] >= target:
            a, b = miou_curve[k - 1], miou_curve[k]
            return kb[k - 1] + (kb[k] - kb[k - 1]) * (target - a) / max(b - a, 1e-9)
    return np.inf


for c in runs[0]:
    pols = list(runs[0][c]["curves"])
    kb = curve(c, pols[0], "extra_kbit_per_uav").mean(0)
    fr = [pt["frac"] for pt in runs[0][c]["curves"][pols[0]]]
    base = np.array([r[c]["base_miou"] for r in runs])
    print(f"\n[{c}] after first transmission: {base.mean():.3f}±{base.std(ddof=1) if len(base) > 1 else 0:.3f}")
    print(f"{'budget':>15s}" + "".join(f"{p[:16]:>17s}" for p in pols))
    for k in range(len(fr)):
        cells = []
        for p in pols:
            m = curve(c, p)[:, k]
            cells.append(f"{m.mean():.3f}±{m.std(ddof=1) if len(m) > 1 else 0:.3f}")
        print(f"{fr[k]:4.0%} {kb[k]:6.1f} kb" + "".join(f"{x:>17s}" for x in cells))

    unc = curve(c, "uncertainty").mean(0)
    print("  bit saving of 'uncertainty' (x times fewer extra bits than the baseline for the same mIoU):")
    for p in ("random", "channel", "uncertainty_per_uav"):
        if p not in pols:
            continue
        other = curve(c, p).mean(0)
        row = []
        for k in range(1, len(fr)):
            need = bits_needed(kb, other, unc[k])
            row.append(f"{fr[k]:.0%}: " + ("> max" if np.isinf(need) else f"{need / kb[k]:.1f}x"))
        print(f"    vs {p:20s} " + "   ".join(row))
