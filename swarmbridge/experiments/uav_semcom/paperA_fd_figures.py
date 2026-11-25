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
         "and $H$ next images", "4. switch only if paired gain > 1 s.e., also",
         "at rate scales $\\ell$ and $h$; 50 ms budget"),
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
    fig.savefig(out / "fig_fd_system_v1.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


GREEN, RED = "#2ca02c", "#d62728"


def system2(out):
    """Fig. 1 (approved layout, 26/11): (a) scenario, (b) mode-control layer. Pure vector drawing.
    Flow as implemented: cold-start gate (20 s of observed airtime) -> screen 14 modes one image ahead at rate scales
    1 and l -> plan N 8 paths over queues + next H 4 images -> override only if paired gain > 1 s.e. at scales 1, l, h
    (a9_mpc._resolvable); rate model = nominal or ACK-calibrated; tracker fed by the model's prediction and ACKs."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_xlim(0, 200)
    ax.set_ylim(-6, 80)
    ax.axis("off")
    ax.grid(False)

    def rbox(x, y, w, h, ec=MUTED, fc="white", lw=0.8, ls="-", r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    linestyle=ls))

    def t(x, y, s, fs=5.6, ha="center", va="center", **kw):
        ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=kw.pop("color", INK), **kw)

    def arrow(p0, p1, col=INK, ls="-", rad=0.0, lw=0.8, ms=6):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def uav(x, y, s=1.0):
        for dx in (-3.2, 3.2):
            ax.plot([x, x + dx * s], [y, y + 0.9 * s], color=INK, lw=0.7)
            ax.add_patch(Circle((x + dx * s, y + 1.3 * s), 1.6 * s, fc="#e6e6e6", ec=INK, lw=0.5))
        ax.add_patch(Rectangle((x - 1.6 * s, y - 0.9 * s), 3.2 * s, 1.8 * s, fc="#9aa4ad", ec=INK, lw=0.5))
        ax.plot([x - 0.8 * s, x - 1.3 * s], [y - 0.9 * s, y - 1.9 * s], color=INK, lw=0.5)
        ax.plot([x + 0.8 * s, x + 1.3 * s], [y - 0.9 * s, y - 1.9 * s], color=INK, lw=0.5)

    # ---------------- (a) scenario ----------------
    rbox(1, 66, 48, 12, ec=MUTED, fc="white")
    t(2.5, 75.4, "LA uplink: $M$ channels, band $B$", fs=5.8, ha="left", fontweight="bold")
    t(2.5, 71.4, "delayed CQI $\\rightarrow$ MCS, OLLA", ha="left")
    t(2.5, 67.9, "BLER, ARQ, JA scheduler", ha="left")
    for i, (col, ls, lab) in enumerate(((BLUE, "-", "channel 1"), (GREEN, "-", "channel 2"),
                                         (ORANGE, "--", "feedback"))):
        yy = 76 - 4 * i
        ax.plot([53, 58], [yy, yy], color=col, lw=1.0, ls=ls)
        t(59.5, yy, lab, ha="left")
    # ground sectors
    for i, (x0, lab) in enumerate(((22, "sector 1"), (36, "sector 2"), (54, "sector $K$"))):
        ax.add_patch(Polygon([(x0, 6), (x0 + 12, 6), (x0 + 15, 14), (x0 + 3, 14)], closed=True, fc="#eef3e8",
                             ec="#9db38a", lw=0.6))
        t(x0 + 7.5, 2.5, lab, fs=5.4, color=MUTED, style="italic")
    t(48.5, 10, "$\\cdots$", fs=6, color=MUTED)
    # BS tower
    ax.plot([6, 9, 12], [8, 40, 8], color=INK, lw=0.9)
    ax.plot([7.2, 10.8], [20, 20], color=INK, lw=0.6)
    ax.plot([8.1, 9.9], [30, 30], color=INK, lw=0.6)
    for r in (2.2, 3.6):
        ax.add_patch(matplotlib.patches.Arc((9, 41), r * 2, r * 2, theta1=30, theta2=150, color=INK, lw=0.6))
    rbox(1.5, 1.5, 16, 4.5, ec=MUTED, fc="#f5f6f8", r=0.6)
    t(9.5, 3.75, "BS (detector)", fs=5.4, fontweight="bold")
    # UAVs and links
    U1, U2, UK = (26, 56), (42, 45), (59, 60)
    for (x, y), lab in ((U1, "UAV 1"), (U2, "UAV 2"), (UK, "UAV $K$")):
        uav(x, y)
        t(x, y + 5.0, lab, fs=5.8, fontweight="bold")
    t(50, 53, "$\\cdots$", fs=7, color=MUTED)
    arrow((U1[0] - 2, U1[1] - 2), (10.5, 37), col=BLUE, lw=1.0)
    arrow((UK[0] - 3, UK[1] - 2), (11.0, 35), col=BLUE, lw=1.0)
    arrow((U2[0] - 3, U2[1] - 1), (11.0, 32), col=GREEN, lw=1.0)
    # co-channel interference (UAV 1 and UAV K share channel 1)
    zx = np.linspace(23, 16.5, 9)
    zy = np.linspace(49, 41, 9) + np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]) * 0.9
    ax.plot(zx, zy, color=RED, lw=0.8)
    arrow((zx[-2], zy[-2]), (16.0, 40.2), col=RED, lw=0.8)
    t(15, 50, "co-channel\ninterference", fs=5.4, color=RED, ha="right")
    # feedback
    arrow((12, 27), (38, 41), col=ORANGE, ls="--", rad=0.25, lw=0.9)
    t(28, 25.5, "CQI (delayed), ACK", fs=5.4, color=ORANGE)
    # altitude
    ax.plot([67, 67], [14, 56], color=MUTED, lw=0.5, ls=(0, (2, 2)))
    t(68.6, 35, "100 m", fs=5.4, color=MUTED, rotation=90)
    t(35, -5.2, "(a) $K$ UAVs offload images over a link-adapted uplink", fs=6.2)

    # ---------------- (b) mode-control layer ----------------
    rbox(73, 1, 126, 78, ec=MUTED, fc="#fbfbfb", ls=(0, (3, 2)), r=2)
    # image
    ax.add_patch(Rectangle((76, 57), 7, 5.5, fc="#dfe7ee", ec=MUTED, lw=0.5))
    ax.add_patch(Rectangle((77.2, 58.2), 7, 5.5, fc="#eef2f6", ec=MUTED, lw=0.5))
    t(80.5, 54, "image $i$", fs=5.6)
    arrow((84.8, 61), (88, 61), col=INK)
    # incumbent
    rbox(88.5, 52, 21, 25, ec=MUTED, fc="white")
    t(99, 73.6, "Incumbent $X$", fs=6.4, fontweight="bold")
    t(99, 66.8, "$H^*$ or PPO,", fs=5.6)
    t(99, 63.0, "QR-DQN,", fs=5.6)
    t(99, 59.2, "Discrete SAC", fs=5.6)
    arrow((110, 64), (114, 64), col=INK)
    t(112, 66.5, "$m_{inc}$", fs=5.6)
    # selective MPC
    rbox(114.5, 33, 46, 44.5, ec=BLUE, fc="#f2f7fc", lw=1.0)
    t(130, 74.4, "Selective MPC", fs=6.6, fontweight="bold", color=BLUE)
    t(158.8, 74.4, "50 ms budget\nper decision", fs=5.0, ha="right", color=BLUE)
    steps = [("gate closed $\\Rightarrow$ keep $m_{inc}$", ""),
             ("screen 14 modes one image ahead", "at rate scales 1 and $\\ell$"),
             ("plan on $N{=}8$ rate paths: queues", "and the next $H{=}4$ images"),
             ("override only if gain > 1 s.e.", "at scales 1, $\\ell$ and $h$")]
    for i, (l1, l2) in enumerate(steps):
        y0 = 62.5 - 9.4 * i
        rbox(116.5, y0, 42, 7.0, ec="#9bb7d4", fc="white", r=0.8)
        ax.add_patch(Circle((119.6, y0 + 3.5), 2.0, fc=BLUE, ec=BLUE))
        t(119.6, y0 + 3.4, str(i + 1), fs=5.6, color="white", fontweight="bold")
        if l2:
            t(122.8, y0 + 5.1, l1, fs=5.4, ha="left")
            t(122.8, y0 + 1.8, l2, fs=5.4, ha="left")
        else:
            t(122.8, y0 + 3.5, l1, fs=5.4, ha="left")
    arrow((161, 68), (166, 68), col=INK)
    t(163.5, 70.5, "$m$", fs=5.8)
    # encoder and uplink
    rbox(166.5, 59, 30, 18, ec=MUTED, fc="white")
    t(181.5, 73.6, "Encoder", fs=6.4, fontweight="bold")
    for j, c in enumerate(("#0072bd", "#edb120", "#77ac30")):
        ax.add_patch(Rectangle((176 + 3.6 * j, 65), 3, 4, fc=c, ec="none"))
    t(181.5, 61.6, "14 modes, FIFO queue", fs=5.4)
    arrow((181.5, 58.6), (181.5, 54.4), col=INK)
    rbox(166.5, 33, 30, 21, ec=MUTED, fc="white")
    t(181.5, 50.6, "LA uplink", fs=6.4, fontweight="bold")
    t(181.5, 45.6, "MCS + OLLA,", fs=5.4)
    t(181.5, 42.2, "ARQ $\\rightarrow$ BS", fs=5.4)
    t(181.5, 37.4, "(delayed CQI)", fs=5.0, color=MUTED)
    # arms inset
    t(78, 47.5, "Arms", fs=5.6, ha="left", fontweight="bold")
    for xx, lab in ((95.5, "rates"), (103, "gate"), (109.5, "band")):
        t(xx, 47.5, lab, fs=5.2, color=MUTED)
    ax.plot([77, 112.5], [45.4, 45.4], color=MUTED, lw=0.4)
    for r_, (arm, rate, g, bd) in enumerate((("nominal", "nom.", 0, 0), ("guarded", "nom.", 1, 1),
                                             ("calibrated", "ACK", 1, 1))):
        yy = 42.3 - 4.0 * r_
        t(78, yy, arm, fs=5.4, ha="left")
        t(95.5, yy, rate, fs=5.4)
        for xx, on in ((103, g), (109.5, bd)):
            if on:
                ax.add_patch(Circle((xx, yy), 1.0, fc=BLUE, ec=BLUE))
            else:
                t(xx, yy, "--", fs=5.4, color=MUTED)
    # rate model
    rbox(76, 4, 54, 24, ec=MUTED, fc="white")
    t(78, 25, "Rate model", fs=6.2, ha="left", fontweight="bold")
    t(99.5, 25, "(one of)", fs=5.4, ha="left", color=MUTED)
    rbox(78.5, 6.5, 20.5, 14, ec=MUTED, fc="#f5f6f8", r=0.8)
    t(88.75, 17.4, "nominal", fs=5.8, fontweight="bold")
    t(88.75, 13.2, "LA map of", fs=5.2)
    t(88.75, 9.8, "fading samples", fs=5.2)
    t(102, 13.5, "or", fs=5.6, style="italic", color=MUTED)
    rbox(105, 6.5, 22.5, 14, ec=ORANGE, fc="#fdf2ec", ls=(0, (2, 1.5)), r=0.8)
    t(116.25, 17.4, "ACK-calibrated", fs=5.8, fontweight="bold")
    t(116.25, 13.2, "ACKed rates in", fs=5.2)
    t(116.25, 9.8, "2 dB SNR bins", fs=5.2)
    # tracker
    rbox(135, 4, 37, 24, ec=BLUE, fc="#f2f7fc", lw=1.0)
    t(153.5, 25, "Rate-error tracker", fs=6.2, fontweight="bold", color=BLUE)
    t(137, 19.6, "$r_{k,t}/\\hat r_{k,t}$ per SNR bin", fs=5.3, ha="left")
    t(137, 15.4, "$\\rightarrow$ bias $b$, standard error $\\sigma$", fs=5.3, ha="left")
    t(137, 11.2, "$\\rightarrow$ gate: 20 s of observed airtime", fs=5.3, ha="left")
    t(137, 7.0, "$\\rightarrow$ band $[\\ell, h]$", fs=5.3, ha="left")
    # model / tracker links
    arrow((124, 28.6), (124, 32.6), col=INK)
    t(122.5, 30.6, "rate paths", fs=5.2, ha="right")
    arrow((130.6, 16), (134.6, 16), col=INK)
    t(132.6, 18.6, "$\\hat r_{k,t}$", fs=5.4)
    arrow((153.5, 28.6), (153.5, 32.6), col=INK)
    t(155, 30.6, "gate, $[\\ell, h]$", fs=5.2, ha="left")
    # ACK feedback from the uplink to the tracker and to the calibrated model
    ax.plot([190, 190, 176.5], [32.5, 16, 16], color=ORANGE, lw=0.9, ls="--")
    arrow((176.5, 16), (172.6, 16), col=ORANGE, lw=0.9)
    ax.plot([190, 190, 116.25], [16, 1.8, 1.8], color=ORANGE, lw=0.9, ls="--")
    arrow((116.25, 1.8), (116.25, 6.0), col=ORANGE, lw=0.9)
    t(188.5, 23, "ACKs:\nrealised\nrate $r_{k,t}$", fs=5.2, ha="right", color=ORANGE)
    t(136, -5.2, "(b) Mode-control layer (rate statistics pooled over UAVs)", fs=6.2)
    fig.savefig(out / "fig_fd_system.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_system.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


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


SETS = [("ref", "$K{=}5$, $M{=}2$, 1 MHz$^a$"), ("N3x2", "$K{=}3$"), ("N7x2", "$K{=}7$"), ("N5x3", "$M{=}3$"),
        ("N5x4", "$M{=}4$"), ("B0.5", "$B{=}0.5$ MHz"), ("B2", "$B{=}2$ MHz")]
SPLIT = [("both", "Both\ncompliant"), ("only_b", "Only\nguarded"), ("only_a", "Only\nnominal")]


def _ablation_panel(ax, N):
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


def ablation_col(out):
    """Fig. 2 (final revision): single-column mechanism ablation (the per-setting panel duplicated Table II)."""
    N = json.loads(NUMS.read_text())
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    _ablation_panel(ax, N)
    ax.legend(loc="lower left", fontsize=5.4, handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    fig.savefig(out / "fig_fd_ablation.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_ablation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def mechanism(out):
    """Fig. 2: (a) ablation (reference, ablation seeds), (b) guarded - H* and calibrated - guarded per setting (TD)."""
    N = json.loads(NUMS.read_text())
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.55, left=0.13, right=0.995, bottom=0.2, top=0.97)
    ax = fig.add_subplot(gs[0])
    _ablation_panel(ax, N)
    ax.set_title("(a)", y=-0.3, fontsize=7)
    ax = fig.add_subplot(gs[1])
    ser = [("G1", "Guarded MPC $-$ $H^*$", BLUE, "o"), ("G2", "Calibrated $-$ guarded MPC", ORANGE, "s")]
    for j, (g, lab, col, mk) in enumerate(ser):
        for i, (cf, _) in enumerate(SETS):
            if cf == "ref":
                c = N["ablation"]["TD"]["contrasts"]["A8" if g == "G1" else "A7"]
                ci, rej = c["ci"], (c["ci"][1] > 0 or c["ci"][2] < 0)
            else:
                c = N["phase4"]["TD"][cf][g]
                ci, rej = c["ci"], bool(c["holm"]["rejected"])
            m, lo, hi = ci
            ax.errorbar(m, i + (j - 0.5) * 0.3, xerr=[[m - lo], [hi - m]], fmt=mk, color=col,
                        mfc=col if rej else "white", ms=3.8, lw=0.8, capsize=1.5, label=lab if i == 0 else None)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(SETS)))
    ax.set_yticklabels([l for _, l in SETS], fontsize=6.3)
    ax.set_ylim(len(SETS) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$ return on TD (paired 95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.6, handletextpad=0.2, borderpad=0.3,
              columnspacing=0.8)
    ax.set_title("(b)", y=-0.3, fontsize=7)
    fig.savefig(out / "fig_fd_mechanism.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_mechanism.png", bbox_inches="tight", dpi=200)


def tradeoff(out):
    """Fig. 3: (a) return vs recall per arm (TD), (b) post-hoc return decomposition guarded - nominal,
    (c) post-hoc recall decomposition guarded - nominal (TD / V / F)."""
    N = json.loads(NUMS.read_text())
    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.1, 0.95], wspace=0.45, left=0.06, right=0.995, bottom=0.2,
                          top=0.97)
    ax = fig.add_subplot(gs[0])
    lv = N["ablation"]["TD"]["levels"]
    for k, lab, mk in ARMS:
        x, y = lv[k]["ret"], lv[k]["recall"]
        ax.plot(x, y, marker=mk, ls="none", color=INK if k == "H" else BLUE, ms=6.5 if mk == "*" else 4.3,
                mfc="white" if k == "H" else BLUE)
        dx, dy, ha = {"H": (1.0, 0.0, "left"), "nc": (1.0, 0.0, "left"), "gate": (1.0, 0.0, "left"),
                      "band": (1.0, 0.0, "left"), "ng": (1.0, -0.17, "left"), "h1": (-1.0, 0.12, "right"),
                      "h0": (1.0, 0.0, "left"), "cal": (1.0, 0.0, "left")}[k]
        ax.text(x + dx, y + dy, lab, fontsize=5.6, ha=ha, va="center")
    ax.set_xlabel("Return per episode (TD)")
    ax.set_ylabel("Detection recall (%)")
    ax.set_xlim(-67, -29)
    ax.set_title("(a)", y=-0.3, fontsize=7)
    ax = fig.add_subplot(gs[1])
    xs = np.arange(len(COMP))
    for j, (pool, st) in enumerate(POOL.items()):
        d = N["decomp"][f"ablation|{pool}|N5x2|ng-nc"]["diff"]
        err = [[d[k][0] - d[k][1] for k, _ in COMP], [d[k][2] - d[k][0] for k, _ in COMP]]
        ax.bar(xs + (j - 1) * 0.26, [d[k][0] for k, _ in COMP], 0.26, color=st["color"], yerr=err, capsize=1.2,
               error_kw=dict(lw=0.6), label=pool)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in COMP], rotation=30, ha="right", fontsize=6.0)
    ax.set_ylabel("$\\Delta$ return, guarded $-$ nominal")
    ax.set_ylim(-31, 25)
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.0, borderpad=0.3, labelspacing=0.2)
    ax.set_title("(b)", y=-0.42, fontsize=7)
    ax = fig.add_subplot(gs[2])
    xs = np.arange(len(SPLIT))
    for j, (pool, st) in enumerate(POOL.items()):
        rs = N["decomp"][f"ablation|{pool}|N5x2|ng-nc"]["recall_split"]
        err = [[rs[k][0] - rs[k][1] for k, _ in SPLIT], [rs[k][2] - rs[k][0] for k, _ in SPLIT]]
        ax.bar(xs + (j - 1) * 0.26, [rs[k][0] for k, _ in SPLIT], 0.26, color=st["color"], yerr=err, capsize=1.2,
               error_kw=dict(lw=0.6), label=pool)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in SPLIT], fontsize=5.6)
    ax.set_ylabel("$\\Delta$ recall (points), guarded $-$ nominal")
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.0, borderpad=0.3, labelspacing=0.2)
    ax.set_title("(c)", y=-0.42, fontsize=7)
    fig.savefig(out / "fig_fd_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "fig_fd_tradeoff.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    system(o)
    system2(o)
    results(o)
    mechanism(o)
    ablation_col(o)
    tradeoff(o)
