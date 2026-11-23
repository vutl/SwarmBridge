"""Paper A figures in IEEE / MATLAB style (user request 23/11: like the plots of Huang JSAC 2025, SchedCP, SAC-UAV papers):
boxed axes, light grid, Arial, saturated MATLAB-like colours, line + marker, framed legends, one plot per (sub)panel.
No experiment is re-run; all data from runs/uav_semcom/paperA_r2/letter_numbers.json (paperA_letter_numbers.py) and
summary_e7.json (validation curves).
  fig_sweeps.pdf : (a) bandwidth, (b) number of UAVs, (c) number of channels, (d) report delay -> return per image
  fig_phy.pdf    : (a) return gain of calibrated MPC per PHY variant (grouped bars by comparator), (b) recall change
  fig_rl.pdf     : (a) validation learning curves of PPO-LA / QR-DQN / Discrete SAC, (b) B1 / B2 return gains,
                   (c) B1 / B2 recall changes
Usage: python paperA_ieee_figures.py OUTDIR"""
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
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black", "axes.grid": True,
    "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.color": "black", "ytick.color": "black", "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.fontsize": 7, "lines.linewidth": 1.2, "lines.markersize": 5, "pdf.fonttype": 42, "mathtext.default": "regular"})

NUMS = C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "letter_numbers.json"
# MATLAB default colour order (blue, orange, yellow, purple, green) + pure red for the proposed method
RED, BLUE, YELLOW, PURPLE, GREEN, BLACK = "#e00000", "#0072bd", "#edb120", "#7e2f8e", "#77ac30", "#000000"
ARM = {"+mpc": dict(label="Calibrated MPC (proposed)", color=RED, marker="x", mfc=RED, ms=6, lw=1.4),
       "+mpcnc": dict(label="Nominal MPC", color=BLUE, marker="o", mfc="none", ms=5, lw=1.2),
       "+mpcrh": dict(label="RobustMPC-style MPC", color=YELLOW, marker="^", mfc="none", ms=5, lw=1.2),
       "H": dict(label="$H^*$ without MPC", color=PURPLE, marker="s", mfc="none", ms=5, lw=1.2, ls="--")}


def _line(ax, xs, ys, arm):
    s = dict(ARM[arm])
    lab, ls = s.pop("label"), s.pop("ls", "-")
    ax.plot(xs, ys, linestyle=ls, label=lab, mew=1.1, **s)


def sweeps(out):
    N = json.loads(NUMS.read_text())
    T = N["table1"]
    pan = (("(a)", "Total bandwidth (MHz)", [("BW:B0.5", 0.5), ("NET:N5x2", 1.0), ("BW:B2", 2.0)], 5),
           ("(b)", "Number of UAVs $K$ ($M$ = 2)", [("NET:N3x2", 3), ("NET:N5x2", 5), ("NET:N7x2", 7)], None),
           ("(c)", "Number of channels $M$ ($K$ = 5)", [("NET:N5x2", 2), ("NET:N5x3", 3), ("NET:N5x4", 4)], 5))
    fig, axg = plt.subplots(2, 2, figsize=(7.0, 4.7), gridspec_kw=dict(wspace=0.28, hspace=0.42))
    axs = axg.ravel()
    for ax, (tag, xl, pts, kfix) in zip(axs, pan):
        for arm in ("+mpc", "+mpcnc", "+mpcrh", "H"):
            xs = [x for _, x in pts]
            ys = [T[k]["level"][arm]["ret"] / (52 * (kfix or x)) for k, x in pts]
            _line(ax, xs, ys, arm)
        ax.set_xticks([x for _, x in pts])
        ax.set_xlabel(xl)
        ax.set_ylabel("Average return per image")
        ax.set_title(tag, loc="center", y=-0.36, fontsize=8)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), fontsize=7)
    fig.subplots_adjust(top=0.9)
    # (d) report delay
    ax = axs[3]
    n5, E5 = T["NET:N5x2"]["level"], N["E5"]
    xs = [0, 1, 3, 10]
    ys = [n5["+mpc"]["ret"]] + [E5[f"fb{d}l0"]["level"][0] for d in (1, 3, 10)]
    _line(ax, xs, [y / 260 for y in ys], "+mpc")
    ax.plot([1, 3], [E5[f"fb{d}l30"]["level"][0] / 260 for d in (1, 3)], linestyle="none", marker="D", ms=5,
            mfc="none", mec=RED, mew=1.1, label="Calibrated MPC, 30% reports lost")
    for arm in ("+mpcnc", "H"):
        _line(ax, xs, [n5[arm]["ret"] / 260] * 4, arm)
        ax.lines[-1].set_marker("none")
    ax.set_xticks(xs)
    ax.set_xlabel("Rate-report delay (slots)")
    ax.set_ylabel("Average return per image")
    ax.set_title("(d)", loc="center", y=-0.36, fontsize=8)
    ax.legend(loc="center right", fontsize=6.5)
    fig.savefig(out / "fig_sweeps.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_sweeps.png", bbox_inches="tight", dpi=200)


def _bars(ax, groups, series, width=0.25):
    """series: list of (label, colour, [(mean, lo, hi) per group])."""
    n = len(series)
    for j, (lab, col, vals) in enumerate(series):
        x = np.arange(len(groups)) + (j - (n - 1) / 2) * width
        m = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.bar(x, m, width * 0.95, color=col, edgecolor="black", lw=0.4, label=lab)
        ax.errorbar(x, m, yerr=err, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(axis="x", visible=False)


def phy(out):
    T = json.loads(NUMS.read_text())["table1"]
    rows = [("PHY:L", "LA"), ("PHY:L_fade", "Block\nfading"), ("PHY:L_bursty", "Bursty\ninterf."),
            ("PHY:L_badfb", "Degr.\nCQI"), ("TD:L", "LA\n(TD)"), ("PHY:S", "Shan-\nnon"),
            ("TD:S", "Shan-\nnon (TD)")]
    g = [l for _, l in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.3), gridspec_kw=dict(hspace=0.55, height_ratios=[1.35, 1]))
    _bars(a1, g, [("vs. nominal MPC", BLUE, [T[k]["cal_minus_nom"] for k, _ in rows]),
                  ("vs. RobustMPC-style MPC", YELLOW, [T[k]["cal_minus_rh"] for k, _ in rows]),
                  ("vs. $H^*$ without MPC", PURPLE, [T[k]["cal_minus_H"] for k, _ in rows])])
    a1.set_ylabel("Return gain of calibrated MPC")
    a1.set_ylim(top=52)
    a1.legend(loc="upper right", fontsize=6.5)
    a1.set_title("(a)", y=-0.42, fontsize=8)
    _bars(a2, g, [("Recall change vs. $H^*$ without MPC", PURPLE, [T[k]["recall_cal_minus_H"] for k, _ in rows])],
          width=0.5)
    a2.set_ylabel("Recall change (points)")
    a2.set_title("(b)", y=-0.6, fontsize=8)
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=6.5)
    fig.savefig(out / "fig_phy.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_phy.png", bbox_inches="tight", dpi=200)


def rl(out):
    N = json.loads(NUMS.read_text())
    E = N["E7"]
    vc = E["val_curves"]
    fams = (("PPO-LA", [f"ppo_la_s{s}.json" for s in range(3)], BLUE, "o"),
            ("QR-DQN", [f"qrdqn_lr1e-4_s{s}.train.json" for s in range(3)], RED, "x"),
            ("Discrete SAC", [f"dsac_ent0.3_s{s}.train.json" for s in range(3)], BLACK, "s"))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.38, width_ratios=[1.05, 1, 1]))
    ax = axs[0]
    for lab, files, col, mk in fams:
        curves = np.array([[v for _, v in vc[f]["curve"]] for f in files])
        steps = np.array([s for s, _ in vc[files[0]]["curve"]]) / 1e6
        ax.fill_between(steps, curves.min(0), curves.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(steps, curves.mean(0), color=col, marker=mk, mfc="none" if mk != "x" else col, ms=4.5, mew=1.0,
                label=lab)
    ax.set_xlabel("Training decisions (×10$^6$)")
    ax.set_ylabel("Validation return")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title("(a)", y=-0.42, fontsize=8)
    inc = [("H*", "$H^*$"), ("PPO-LA", "PPO-\nLA"), ("QR-DQN", "QR-\nDQN"), ("Discrete SAC", "Disc.\nSAC")]

    def get(pool, key, b):
        return (E[pool]["H_ref"] if key == "H*" else E[pool]["B"][key])[b]

    LP, LB = "#c39bd3", "#7fb8e0"                     # lighter purple / blue for test-dev
    for ax, suf, yl, tag in ((axs[1], "", "Return gain", "(b)"), (axs[2], "_recall", "Recall change (points)", "(c)")):
        _bars(ax, [l for _, l in inc],
              [("vs. $X$ alone, test", PURPLE, [get("test", k, "B1" + suf) for k, _ in inc]),
               ("vs. $X$ alone, test-dev", LP, [get("testdev", k, "B1" + suf) for k, _ in inc]),
               ("vs. nominal MPC, test", BLUE, [get("test", k, "B2" + suf) for k, _ in inc]),
               ("vs. nominal MPC, test-dev", LB, [get("testdev", k, "B2" + suf) for k, _ in inc])], width=0.2)
        ax.set_ylabel(yl)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(tag, y=-0.42, fontsize=8)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.66, 1.07), fontsize=6.3, handlelength=1.2,
               columnspacing=0.8)
    fig.savefig(out / "fig_rl.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_rl.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    sweeps(o)
    phy(o)
    rl(o)
    print("figures ->", o)
