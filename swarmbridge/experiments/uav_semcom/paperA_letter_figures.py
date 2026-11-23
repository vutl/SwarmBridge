"""Paper A method letter figures (no experiment is re-run).
  fig_method.pdf   : system and method (incumbent mode policy, selective MPC, ACK-calibrated rate model, LA uplink)
  fig_recall.pdf   : return gain vs Qmin-compliant recall change of the calibrated MPC layer (Table 1 LA rows:
                     H*+cal - H*; E7: X+cal - X and X+cal - X+nominal), from runs/uav_semcom/paperA_r2/letter_numbers.json
Palette: dataviz reference categorical slots (blue #2a78d6, orange #eb6834, + muted ink); marker shape as a second
encoding. Usage: python paperA_letter_figures.py OUTDIR"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import a5_common as C  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
GREEN = "#2f9e6e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def method(out):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 44)
    ax.axis("off")
    LS = 2.35

    def box(x, ytop, w, title, lines=(), fc="#f4f6fa", ec=MUTED, bold=True, tsize=6.4, lsize=5.6):
        h = 4.2 + LS * len(lines) + (1.0 if lines else 0.0)
        ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec,
                                    lw=0.9))
        ax.text(x + w / 2, ytop - 1.4, title, ha="center", va="top", fontsize=tsize, color=INK,
                fontweight="bold" if bold else "normal")
        for i, s in enumerate(lines):
            ax.text(x + w / 2, ytop - 4.6 - LS * i, s, ha="center", va="top", fontsize=lsize, color=INK)
        return ytop - h

    def arrow(p0, p1, col=INK, ls="-", rad=0.0, lw=1.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    for lab, yt in (("UAV 1", 40), ("$\\vdots$", 34), ("UAV $K$", 28)):
        box(0.5, yt, 9.5, lab, (), fc="white", bold=False)
    ax.text(5.2, 22.5, "independent\nimages,\ndeadline,\nquality floor", ha="center", va="top", fontsize=5.6, color=MUTED)
    box(12, 42, 13.5, "Incumbent $X$", ("any mode policy:", "tuned rule $H^*$,", "PPO, QR-DQN,", "Discrete SAC",
                                        "$\\rightarrow m_{\\mathrm{inc}}$"))
    ym = box(28, 42, 32.5, "Selective MPC (per image)",
             ("gate: plan only if the error tracker", "has $\\geq$20 s of data, else keep $m_{\\mathrm{inc}}$",
              "screen 14 modes at nominal and low rate", "$\\rightarrow$ 2 candidates; plan $N{=}8$ rate paths,",
              "4-image lookahead; switch only if gain", "$>$1 s.e. and robust over $[\\ell, h]$;",
              "50 ms budget, else keep $m_{\\mathrm{inc}}$"), ec=BLUE)
    box(63, 38, 9.5, "Encoder", ("FIFO", "queue", "(measured", "codec", "profiles)"))
    yu = box(75, 42, 15.5, "LA uplink, $M$ ch.", ("JA scheduler", "delayed CQI $\\rightarrow$", "MCS + OLLA,", "BLER, ARQ,",
                                               "co-channel", "interference"))
    box(93.3, 37, 8, "BS", ("utility", "$u$;", "detector", "recall"))
    yc = box(30, 13.5, 44,"ACK-calibrated rate model (cold start each episode)",
             ("per (UAV, slot) delivered rate in 2 dB bins of expected SNR;",
              "MPC samples rate paths from the bin; error tracker gives",
              "bias $b$ and s.e. $\\sigma$ $\\rightarrow$ band $[\\ell, h]$ around $1{+}b$"), fc="#fdf1ec", ec=ORANGE,
             lsize=5.6)
    for x0, x1, y in ((10.4, 11.7, 33), (25.9, 27.7, 33), (60.9, 62.7, 33), (72.9, 74.7, 33), (90.9, 93.0, 33)):
        arrow((x0, y), (x1, y), col=BLUE, lw=1.3)
    arrow((83, yu - 0.4), (74.6, 8.5), col=ORANGE, ls="--", rad=-0.3)
    ax.text(89.5, 15.0, "ACKed bits of each\nslot, read at the slot\nboundary (E5: reports\ndelayed or lost)",
            ha="center", va="top", fontsize=5.6, color=ORANGE)
    arrow((29.6, 7.5), (38, ym - 0.4), col=ORANGE, ls="--", rad=-0.35)
    ax.text(20.0, 10.5, "rate samples,\ngate, band", ha="center", va="center", fontsize=5.8, color=ORANGE)
    ax.plot([], [], color=BLUE, lw=1.3, label="data / decision path")
    ax.plot([], [], color=ORANGE, lw=1.0, ls="--", label="feedback")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=1, frameon=False, fontsize=6.0)
    fig.savefig(out / "fig_method.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_method.png", bbox_inches="tight", dpi=200)


def recall(out):
    N = json.loads((C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "letter_numbers.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    pts = []
    for k, r in N["table1"].items():
        if k.endswith(":S"):
            continue
        pts.append(("H", r["cal_minus_H"], r["recall_cal_minus_H"]))
    for pool in ("test", "testdev"):
        for lab, b in N["E7"][pool]["B"].items():
            pts.append(("B1", b["B1"], b["B1_recall"]))
    for pool in ("test", "testdev"):
        lv = N["E7"][pool]["levels"]
        for lab in ("H*", "PPO-LA", "QR-DQN", "Discrete SAC"):
            dr = lv[f"{lab}|X+cal"]["ret"] - lv[f"{lab}|X+nom"]["ret"]
            dc = lv[f"{lab}|X+cal"]["recall"] - lv[f"{lab}|X+nom"]["recall"]
            pts.append(("B2", [dr, dr, dr], [dc, dc, dc]))
    style = {"H": (BLUE, "o", "$H^*$+cal $-$ $H^*$ (12 LA configs)"),
             "B1": (ORANGE, "^", "E7: $X$+cal $-$ $X$ (learned $X$)"),
             "B2": (GREEN, "s", "E7: $X$+cal $-$ $X$+nominal")}
    for key, (col, mk, lab) in style.items():
        sel = [p for p in pts if p[0] == key]
        for i, (_, r, c) in enumerate(sel):
            ax.errorbar(r[0], c[0], xerr=[[r[0] - r[1]], [r[2] - r[0]]] if key != "B2" else None,
                        yerr=[[c[0] - c[1]], [c[2] - c[0]]] if key != "B2" else None, fmt=mk, ms=4, color=col,
                        ecolor=col, elinewidth=0.8, capsize=0, label=lab if i == 0 else None, alpha=0.9)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("Return difference", fontsize=7.5)
    ax.set_ylabel("Recall difference (points)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_recall.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_recall.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    method(o)
    recall(o)
    print("figures ->", o)
