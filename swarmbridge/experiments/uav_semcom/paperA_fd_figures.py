"""Paper A (ICT Express rewrite, 25/11) figures from runs/uav_semcom/paperA_fd/paper/paper_numbers.json only
(paperA_fd_numbers.py; no experiment is re-run).
  fig_fd_system.pdf  : Fig. 1, system and controller (flow checked against ax2_eval.py / a9_mpc.py: cold-start gate on
                       the error tracker, screen at rate scales 1 and l one image ahead, full plan N 8 paths over current
                       + queued + H anticipated images, paired-gain test and robustness check at l and h, 50 ms budget)
  fig_fd_results.pdf : Fig. 2, (a) mechanism ablation return differences (TD / V / F, paired 95% CIs), (b) return vs
                       detection recall per arm (TD), (c) post-hoc return decomposition guarded - nominal MPC
Usage: python paperA_fd_figures.py OUTDIR"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import a5_common as C  # noqa: E402

for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.7, "axes.edgecolor": "black", "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": True, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.framealpha": 1.0, "legend.fontsize": 6.3, "pdf.fonttype": 42,
    "mathtext.default": "regular"})
NUMS = C.REPO / "runs" / "uav_semcom" / "paperA_fd" / "paper" / "paper_numbers.json"
POOL = {"TD": dict(color="#0072bd", marker="o", label="TD (test-dev)"),
        "V": dict(color="#d95319", marker="s", label="V (DET-val)"),
        "F": dict(color="#4d4d4d", marker="^", label="F (held-out)")}
INK, MUTED, BLUE, ORANGE = "#1a1a1a", "#666666", "#0072bd", "#d95319"


def system(out):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.6, 30)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, lines=(), ec=MUTED, fc="#f5f6f8", ts=6.4, ls=5.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=INK)
        for i, s in enumerate(lines):
            ax.text(x + w / 2, y + h - 3.6 - 2.05 * i, s, ha="center", va="top", fontsize=ls, color=INK)

    def arrow(p0, p1, col=INK, ls="-", rad=0.0, lw=0.9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=col, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    box(0.3, 17.5, 9.2, 12, "UAV $k$", ("52 images,", "deadline,", "quality", "floor"), fc="white")
    box(11.3, 17.5, 12.5, 12, "Incumbent $X$", ("tuned rule $H^*$,", "PPO, QR-DQN", "or Discrete SAC", "$\\rightarrow m_{inc}$"))
    box(25.6, 11.4, 34.5, 18.4, "Uncertainty-aware selective MPC",
        ("1. gate: plan only once the tracker has 20 s of data", "2. screen 14 modes (one image ahead) at rate",
         "scales 1 and $\\ell$; best mode per scale = candidate", "3. plan $N{=}8$ rate paths over current, queued",
         "and $H$ next images", "4. switch only if paired gain > 1 s.e. and still",
         "positive at scales $\\ell$ and $h$; 50 ms budget"),
        ec=BLUE, ls=5.2)
    box(62.0, 17.5, 8.8, 12, "Encoder", ("FIFO", "queue,", "14 modes"))
    box(72.6, 17.5, 15.4, 12, "LA uplink, $M$ ch.", ("JA scheduler,", "delayed CQI, MCS,", "OLLA, BLER, ARQ,",
                                                    "co-channel interf."))
    box(89.8, 17.5, 9.9, 12, "BS", ("return $J$;", "recall", "(evaluated,", "not optimised)"))
    box(25.6, -2.2, 16.5, 11.4, "Rate model", ("nominal: LA rate of", "statistical fading, or", "ACK-calibrated: slot",
                                             "rates in 2 dB SNR bins"), ec=ORANGE, fc="#fdf2ec", ls=5.3)
    box(44.0, -2.2, 16.1, 11.4, "Error tracker", ("realised / predicted rate", "per SNR bin $\\rightarrow$ bias $b$,",
                                                "s.e. $\\sigma$ $\\rightarrow$ band $[\\ell, h]$", "and gate"),
        ec=ORANGE, fc="#fdf2ec", ls=5.3)
    y = 23.5
    for x0, x1 in ((9.6, 11.2), (23.9, 25.5), (60.2, 61.9), (70.9, 72.5), (88.1, 89.7)):
        arrow((x0, y), (x1, y), col=BLUE, lw=1.1)
    arrow((80.3, 17.3), (60.4, 4.5), col=ORANGE, ls="--", rad=-0.25)
    ax.text(83.5, 9.0, "ACKed bits per UAV and slot", ha="center", va="center", fontsize=5.5, color=ORANGE)
    arrow((33.8, 9.6), (33.8, 11.2), col=ORANGE, ls="--")
    arrow((52.0, 9.6), (52.0, 11.2), col=ORANGE, ls="--")
    ax.text(31.7, 10.1, "rate paths", ha="right", va="center", fontsize=5.3, color=ORANGE)
    ax.text(54.0, 12.0, "gate, $[\\ell, h]$", ha="left", va="center", fontsize=5.3, color=ORANGE)
    ax.plot([], [], color=BLUE, lw=1.1, label="decision / data path")
    ax.plot([], [], color=ORANGE, lw=0.9, ls="--", label="feedback")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=5.6)
    fig.savefig(out / "fig_fd_system.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_system.png", bbox_inches="tight", dpi=200)


ABL = [("nc-H", "Nominal MPC $-$ $H^*$"), ("A8", "Guarded MPC $-$ $H^*$"), ("A2", "Band only $-$ nominal"),
       ("A1", "Gate only $-$ nominal"), ("A6", "Gate + band $-$ band"), ("A4", "$H{=}4$ $-$ $H{=}1$"),
       ("A9", "$H{=}1$ $-$ $H{=}0$"), ("A7", "Calibrated $-$ guarded")]
ARMS = [("H", "$H^*$", "s"), ("nc", "nominal", "o"), ("gate", "gate", "v"), ("band", "band", "D"),
        ("ng", "guarded", "*"), ("h1", "$H{=}1$", "<"), ("h0", "$H{=}0$", ">"), ("cal", "calibrated", "P")]
COMP = [("sce", "SCE terms"), ("viol", "Below-$Q^{min}$"), ("drop", "Drops"), ("time", "Mission time"),
        ("energy", "Energy"), ("ret", "Return")]


def results(out):
    N = json.loads(NUMS.read_text())
    fig = plt.figure(figsize=(7.0, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.1], wspace=0.42, left=0.15, right=0.995, bottom=0.2,
                          top=0.97)
    # (a) ablation contrasts
    ax = fig.add_subplot(gs[0])
    yy = np.arange(len(ABL))
    for j, (pool, st) in enumerate(POOL.items()):
        con = N["ablation"][pool]["contrasts"]
        for i, (k, _) in enumerate(ABL):
            m, lo, hi = con[k]["ci"]
            sig = lo > 0 or hi < 0
            ax.errorbar(m, i + (j - 1) * 0.24, xerr=[[m - lo], [hi - m]], fmt=st["marker"], color=st["color"],
                        mfc=st["color"] if sig else "white", ms=3.6, lw=0.8, capsize=1.5,
                        label=st["label"] if i == 0 else None)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([l for _, l in ABL], fontsize=6.3)
    ax.set_ylim(len(ABL) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$ return (paired 95% CI)")
    ax.legend(loc="lower left", fontsize=5.6, handletextpad=0.2, borderpad=0.3, labelspacing=0.25)
    ax.set_title("(a)", y=-0.3, fontsize=7)
    # (b) return vs recall, TD
    ax = fig.add_subplot(gs[1])
    lv = N["ablation"]["TD"]["levels"]
    for k, lab, mk in ARMS:
        x, y = lv[k]["ret"], lv[k]["recall"]
        ax.plot(x, y, marker=mk, ls="none", color=INK if k == "H" else BLUE, ms=6.5 if mk == "*" else 4.3,
                mfc="white" if k == "H" else (INK if k == "H" else BLUE))
        dx, dy, ha = {"H": (1.0, 0.0, "left"), "nc": (1.0, 0.0, "left"), "gate": (1.0, 0.0, "left"),
                      "band": (1.0, 0.0, "left"), "ng": (1.0, -0.17, "left"), "h1": (-1.0, 0.12, "right"),
                      "h0": (1.0, 0.0, "left"), "cal": (1.0, 0.0, "left")}[k]
        ax.text(x + dx, y + dy, lab, fontsize=5.6, ha=ha, va="center")
    ax.set_xlabel("Return per episode (TD)")
    ax.set_ylabel("Detection recall (%)")
    ax.set_xlim(-67, -29)
    ax.set_title("(b)", y=-0.3, fontsize=7)
    # (c) decomposition guarded - nominal
    ax = fig.add_subplot(gs[2])
    xs = np.arange(len(COMP))
    for j, (pool, st) in enumerate(POOL.items()):
        d = N["decomp"][f"ablation|{pool}|N5x2|ng-nc"]["diff"]
        ms_ = [d[k][0] for k, _ in COMP]
        err = [[d[k][0] - d[k][1] for k, _ in COMP], [d[k][2] - d[k][0] for k, _ in COMP]]
        ax.bar(xs + (j - 1) * 0.26, ms_, 0.26, color=st["color"], yerr=err, capsize=1.2,
               error_kw=dict(lw=0.6), label=pool)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in COMP], rotation=30, ha="right", fontsize=6.0)
    ax.set_ylabel("Guarded $-$ nominal MPC")
    ax.set_ylim(-31, 25)
    ax.legend(loc="lower right", fontsize=5.6, ncol=1, handlelength=1.0, borderpad=0.3, labelspacing=0.2)
    ax.set_title("(c)", y=-0.42, fontsize=7)
    fig.savefig(out / "fig_fd_results.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_results.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    system(o)
    results(o)
