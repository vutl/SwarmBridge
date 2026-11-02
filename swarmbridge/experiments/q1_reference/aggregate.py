"""Aggregate Q1 runs over seeds and check the agreed decision criteria.

    python -m swarmbridge.experiments.q1_reference.aggregate runs/q1_real_6k_s*.json

Criteria (README):
  SB worth keeping : S1 >= M1 and >= F1 on task mIoU (same NFE budget)
  Q1a supported    : S3/S4 beat BOTH S1 (CSI in network) and M3 (channel-shaped augmentation)
  Q1b supported    : bridge U|r clearly > 0.5 AND beats the heteroscedastic baseline H1
A difference counts only if it exceeds 2 standard errors of the paired seed differences.
"""
import glob
import json
import sys

import numpy as np

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
runs = [json.load(open(f))["results"] for f in files]
models = [m for m in runs[0] if all(m in r for r in runs)]
conds = [c for c in runs[0][models[0]] if isinstance(runs[0][models[0]][c], dict)]
nfes = sorted({int(k.split("@")[1]) for k in runs[0][models[0]][conds[0]] if k.startswith("miou@")})
print(f"{len(runs)} seeds: {files}\n")


def best_miou(r, m, c):
    """best deterministic-or-sampled inference mode: miou@NFE, and the ODE path for bridges"""
    return max(v for k, v in r[m][c].items() if k.startswith(("miou@", "miou_ode@")))


def vals(m, c, key=None):
    return np.array([best_miou(r, m, c) if key is None else r[m][c][key] for r in runs])


def ms(x):
    return f"{x.mean():.3f}±{x.std(ddof=1) if len(x) > 1 else 0:.3f}"


print("mIoU, best over NFE {" + ",".join(map(str, nfes)) + "} and the bridge ODE path (mean±std over seeds)")
print(f"{'':7s}" + "".join(f"{c:>15s}" for c in conds))
for ref in ("miou_oracle", "miou_received"):
    print(f"{ref[5:]:7s}" + "".join(f"{ms(vals(models[0], c, ref)):>15s}" for c in conds))
for m in models:
    print(f"{m:7s}" + "".join(f"{ms(vals(m, c)):>15s}" for c in conds))

unc = [m for m in models if "auroc_U" in runs[0][m][conds[-1]]]
if unc:
    print(f"\nQ1b on '{conds[-1]}': AUROC (mean±std)")
    print(f"{'':7s}{'U':>15s}{'1-r':>15s}{'U|r':>15s}")
    for m in unc:
        print(f"{m:7s}" + "".join(f"{ms(vals(m, conds[-1], k)):>15s}" for k in ("auroc_U", "auroc_1-r", "auroc_U|r")))


def compare(a, b, c, key=None):
    """paired difference a - b over seeds -> (mean, significant)"""
    d = vals(a, c, key) - vals(b, c, key)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    return d.mean(), abs(d.mean()) > 2 * se


print("\nDecision criteria (paired over seeds; * = beyond 2 standard errors)")
for c in conds[1:]:
    print(f"  [{c}]")
    for a, b, what in [("S1", "M1", "SB vs mapping"), ("S1", "F1", "SB vs flow matching"),
                       ("S3", "S1", "Q1a: R1 ref vs CSI-in-network"), ("S4", "S1", "Q1a: R2 ref vs CSI-in-network"),
                       ("S3", "M3", "Q1a: R1 ref vs channel-shaped augmentation"),
                       ("S4", "M3", "Q1a: R2 ref vs channel-shaped augmentation")]:
        if a in models and b in models:
            d, sig = compare(a, b, c)
            print(f"    {what:45s} {a}-{b}: {d:+.3f}{' *' if sig else ''}")
if unc and "H1" in models:
    print(f"  [Q1b, {conds[-1]}: U|r of bridge vs heteroscedastic mapping]")
    for m in unc:
        if m != "H1":
            d, sig = compare(m, "H1", conds[-1], "auroc_U|r")
            print(f"    {m}-H1: {d:+.3f}{' *' if sig else ''}")
