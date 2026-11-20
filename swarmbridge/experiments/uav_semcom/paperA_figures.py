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


def architecture(out):
    """Fig. 1 (draft): system and control loop. Independent image offloading; no inter-UAV exchange is drawn."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 46)
    ax.axis("off")
    LS = 2.3                                       # line spacing (data units)

    def box(x, ytop, w, title, lines=(), fc="#f4f6fa", ec=MUTED, bold=True, tsize=6.4):
        h = 4.2 + LS * len(lines) + (1.0 if lines else 0.0)
        ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec,
                                    lw=0.8))
        ax.text(x + w / 2, ytop - 1.4, title, ha="center", va="top", fontsize=tsize, color=INK,
                fontweight="bold" if bold else "normal")
        for i, s in enumerate(lines):
            ax.text(x + w / 2, ytop - 4.6 - LS * i, s, ha="center", va="top", fontsize=5.6, color=INK)
        return ytop - h

    def arrow(p0, p1, col=INK, ls="-", rad=0.0, lw=1.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    for lab, yt in (("UAV 1", 38.5), ("UAV 2", 32.5), ("$\\vdots$", 26.5), ("UAV 5", 20.5)):
        box(1, yt, 11, lab, (), fc="white", bold=False)
    ax.text(6.5, 14.5, "disjoint survey\nsectors;\nindependent\nimages", ha="center", va="top", fontsize=5.8,
            color=MUTED)
    ym = box(15.5, 41, 28.5, "Mode decision (per image)",
             ("incumbent: tuned rule $H^*$ or PPO", "$\\downarrow$ selective MPC (8 channel paths,",
              "4-image lookahead, 50 ms budget);", "overrides if gain > 1 std. error",
              "$\\rightarrow$ mode $m$ of 14 (AVIF, mbt2018,", "JPEG2000, thumbnail, generative)"))
    box(47.3, 37, 12.4, "Encoder +", ("FIFO queue", "payload, PSNR,", "enc. time", "(measured)"))
    yu = box(62.7, 41, 24, "Uplink: 2 shared channels",
             ("slot policy: movement, channel,", "power (JA or T:best)",
              "S: $(B/M)\\log_2(1+\\mathrm{SNR})$", "L: delayed CQI $\\rightarrow$ MCS, BLER,",
              "ACK/NACK + OLLA, ARQ,", "co-channel interference"))
    box(89.6, 36, 10.4, "Ground BS", ("decode;", "utility $u$:", "PSNR vs $Q^{\\min}$,", "delay vs $D^{\\max}$",
                                    "(drop: $-1$)"))
    yc = box(46, 12.5, 24, "Online rate calibration", ("cold start each episode",), fc="#fdf1ec", ec=ORANGE)
    for x0, x1 in ((12.3, 15.2), (44.3, 47.0), (60.0, 62.4), (87.0, 89.3)):
        arrow((x0, 30), (x1, 30), col=BLUE, lw=1.3)
    arrow((75, yu - 0.4), (70.4, 9.0), col=ORANGE, ls="--", rad=-0.25)
    ax.text(86.5, 13.0, "delivered (ACKed) bits per slot,\nread at the slot boundary;\nfeedback delay and\noverhead "
            "not modelled", ha="center", va="top", fontsize=5.6, color=ORANGE)
    arrow((45.6, 7.5), (32, ym - 0.4), col=ORANGE, ls="--", rad=-0.3)
    ax.text(33.5, 8.0, "updated rate model", ha="center", va="center", fontsize=5.8, color=ORANGE)
    ax.plot([], [], color=BLUE, lw=1.3, label="data path")
    ax.plot([], [], color=ORANGE, lw=1.0, ls="--", label="feedback / control")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=2, frameon=False, fontsize=6.3)
    fig.savefig(out / "fig_system.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_system.png", bbox_inches="tight", dpi=200)


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


def misalignment(out):
    """Fig. 2 (rewrite): per configuration, paired return difference (a) and Qmin-compliant recall difference (b)
    of two controller pairs; from runs/uav_semcom/paperA_r2/paper_numbers.json (robustness round)."""
    N = json.loads((C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "paper_numbers.json").read_text())
    confs = ["S", "L", "L_fade", "L_bursty", "L_badfb", "TD-S", "TD-L"]
    names = {"S": "Shannon", "L": "LA", "L_fade": "LA, block fading", "L_bursty": "LA, bursty interf.",
             "L_badfb": "LA, degraded CQI", "TD-S": "Shannon, test-dev", "TD-L": "LA, test-dev"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    series = (("A", BLUE, "o", -0.14, "PPO-S $-$ T:best (same slot rule)"),
              ("B", ORANGE, "^", 0.14, "learned + MPC $-$ $H^*$ + MPC ($H^*$ frozen)"))
    for ax, metric, title in zip(axs, ("return", "recall"), ("(a) Return difference", "(b) Qmin-compliant recall difference (points)")):
        for key, col, mk, off, lab in series:
            for i, c in enumerate(confs):
                d, lo, hi = N[c][f"{key}_{metric}"]
                y = len(confs) - 1 - i + off
                ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt=mk, ms=4.5, color=col, ecolor=col, elinewidth=1.2,
                            capsize=0, label=lab if i == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.tick_params(labelsize=7)
    axs[0].set_yticks(range(len(confs)))
    axs[0].set_yticklabels([names[c] for c in reversed(confs)], fontsize=7)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "fig_misalign.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_misalign.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    misalignment(o)
    architecture(o)
    forest(o)
    block_a(o)
    recall(o)
    print("figures ->", o)
