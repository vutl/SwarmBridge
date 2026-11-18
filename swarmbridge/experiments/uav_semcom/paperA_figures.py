"""Paper A figures from runs/uav_semcom/paperA_r1/summary.json (no experiment is re-run).
  fig_r1_forest.pdf  : paired return deltas [95% CI] of block B vs H*_P + calibrated MPC, Shannon and LA panels;
                       marker = whether the policy was designed (tuned / trained) on the evaluated PHY or on the other
  fig_r1_blockA.pdf  : literature-style block (A5 slot rule): untuned rules, T:best, PPO-S on Shannon and on LA
  fig_r1_recall.pdf  : LA, block B: A5 return vs Qmin-compliant detector recall
Palette: categorical slots 1-2 of the dataviz reference palette (blue #2a78d6, orange #eb6834), hatch as a second
encoding for print / colour-vision deficiency. Usage: python paperA_figures.py OUTDIR
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import a5_common as C  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
J = json.loads((C.REPO / "runs" / "uav_semcom" / "paperA_r1" / "summary.json").read_text())
A = J["arms"]


def get(phy, key_end):
    return next(v for k, v in A[phy].items() if k.endswith(key_end))


def forest(out):
    rows = [("H* (tuned on this PHY)", "H*_{p} = {h}", True), ("H* + MPC, uncalibrated", "H*_{p} + MPC (uncalibrated)", True),
            ("H* tuned on the other PHY", "(tuned on other PHY)", False), ("PPO-S", "B (frozen JA scheduler @ pacedB3)|PPO-S", None),
            ("PPO-S + MPC (calibrated)", "PPO-S + MPC (calibrated)", None), ("PPO-LA", "PPO-LA (trained on LA)", True),
            ("PPO-LA + MPC (calibrated)", "PPO-LA + MPC (calibrated)", True)]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, phy, title in zip(axs, ("S", "L"), ("(a) Shannon PHY", "(b) Realistic LA PHY")):
        ys, labels = [], []
        for i, (lab, key, native) in enumerate(rows):
            key = key.format(p=phy, h=J["H_star"][phy])
            try:
                v = next(v for k, v in A[phy].items() if k.split("|", 1)[0].startswith("B") and k.endswith(key))
            except StopIteration:
                ax.annotate("n/a (needs LA features)", (0, len(rows) - i), xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7, color=MUTED)
                continue
            if native is None:                       # PPO-S: native on S, zero-shot on L
                native = phy == "S"
            d, lo, hi = v["delta_vs_ref"]
            y = len(rows) - i
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o" if native else "^", ms=6, color=BLUE if native else ORANGE,
                        ecolor=BLUE if native else ORANGE, elinewidth=1.5, capsize=0)
            ax.annotate(f"{d:+.1f}", (hi, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=7)
            ys.append(y)
            labels.append(lab)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, fontsize=8, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axs[0].set_yticks([len(rows) - i for i in range(len(rows))])
    axs[0].set_yticklabels([r[0] for r in rows])
    fig.supxlabel("Return difference vs H* + calibrated MPC on the same PHY (95% CI)", fontsize=8, y=0.07)
    h1 = axs[1].scatter([], [], marker="o", color=BLUE, label="designed on the evaluated PHY")
    h2 = axs[1].scatter([], [], marker="^", color=ORANGE, label="designed on the other PHY (zero-shot)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_r1_forest.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_r1_forest.png", bbox_inches="tight", dpi=200)


def block_a(out):
    labs = [("Random", "Random (untuned)"), ("Round robin", "Round robin (untuned)"), ("Max rate", "Max rate (untuned)"),
            ("Greedy", "Greedy (untuned)"), ("T:best (tuned)", "T:best (tuned on Shannon)"), ("PPO-S", "PPO-S (trained on Shannon)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.38
    for j, (phy, col, hat, name) in enumerate((("S", BLUE, "", "Shannon PHY"), ("L", ORANGE, "////", "Realistic LA PHY"))):
        vals = [get(phy, k)["ret"] for _, k in labs]
        xs = [i + (j - 0.5) * w for i in range(len(labs))]
        ax.bar(xs, vals, width=w - 0.04, color=col, hatch=hat, edgecolor="white", linewidth=0.5, label=name)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([l for l, _ in labs], rotation=25, ha="right")
    ax.set_ylabel("Return (A5 utility)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_r1_blockA.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_r1_blockA.png", bbox_inches="tight", dpi=200)


def recall(out):
    pts = [("H*", "H*_L = fixed_mbt2018-mean_q3"), ("H* + MPC", "H*_L + MPC (calibrated)"),
           ("H* + MPC (uncal.)", "H*_L + MPC (uncalibrated)"), ("PPO-S", "B (frozen JA scheduler @ pacedB3)|PPO-S"),
           ("PPO-S + MPC", "PPO-S + MPC (calibrated)"), ("PPO-LA", "PPO-LA (trained on LA)"),
           ("PPO-LA + MPC", "PPO-LA + MPC (calibrated)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for lab, key in pts:
        v = next(v for k, v in A["L"].items() if k.endswith(key))
        ax.scatter(v["ret"], 100 * v["rec_qmin"], s=28, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(lab, (v["ret"], 100 * v["rec_qmin"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("Return (A5 utility), realistic LA PHY")
    ax.set_ylabel("Qmin-compliant detector recall (%)")
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig_r1_recall.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_r1_recall.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    forest(o)
    block_a(o)
    recall(o)
    print("figures ->", o)
