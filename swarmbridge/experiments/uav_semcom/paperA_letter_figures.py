"""Paper A method letter figures (no experiment is re-run).
  fig_method.pdf   : system and method (incumbent mode policy, selective MPC, ACK-calibrated rate model, LA uplink)
  fig_robust.pdf   : forest plot over the 14 configurations (incumbent H*): calibrated MPC minus nominal MPC, minus
                     RobustMPC-style MPC, minus H* alone (return) and recall change vs H* (95% CIs; hollow = CI incl. 0)
  fig_e7.pdf       : RL-family experiment: return and recall of X, X + calibrated MPC, X + nominal MPC per incumbent,
                     test and test-dev pools (per-training-seed returns as small dots)
  fig_recall.pdf   : return gain vs Qmin-compliant recall change of the calibrated MPC layer (not in the 4-page letter)
All data from runs/uav_semcom/paperA_r2/letter_numbers.json (paperA_letter_numbers.py).
Palette: dataviz reference categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a; documented to pass the
all-pairs CVD / normal-vision checks in light mode), marker shape as a second encoding, text in ink colours.
Usage: python paperA_letter_figures.py OUTDIR"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import a5_common as C  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
GREEN = "#1baf7a"                                   # reference slot 3 (aqua)
NUMS = C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "letter_numbers.json"
ROWLAB = {"PHY:S": "Shannon", "PHY:L": "LA (nominal)", "PHY:L_fade": "LA, block fading",
          "PHY:L_bursty": "LA, bursty interf.", "PHY:L_badfb": "LA, degraded CQI",
          "TD:S": "Shannon", "TD:L": "LA (nominal)",
          "NET:N3x2": "$K$=3, $M$=2", "NET:N5x2": "$K$=5, $M$=2", "NET:N7x2": "$K$=7, $M$=2",
          "NET:N5x3": "$K$=5, $M$=3", "NET:N5x4": "$K$=5, $M$=4", "BW:B0.5": "0.5 MHz", "BW:B2": "2 MHz"}
GRPLAB = {"PHY": "Test pool", "TD": "Test-dev pool", "NET": "Network size", "BW": "Bandwidth"}


def forest(out):
    """Fig. 2: one row per configuration (grouped), four panels; filled marker = CI excludes 0."""
    N = json.loads(NUMS.read_text())["table1"]
    keys = list(N)
    ys, y, ylab, gpos, last = [], 0.0, [], [], None
    for k in keys:
        g = k.split(":")[0]
        if g != last:
            if last is not None:
                y += 0.7
            gpos.append((y, GRPLAB[g]))
            last = g
        ys.append(y)
        ylab.append(ROWLAB[k])
        y += 1
    panels = (("cal_minus_nom", "(a) vs. nominal MPC", BLUE, "o"),
              ("cal_minus_rh", "(b) vs. RobustMPC-style", BLUE, "o"),
              ("cal_minus_H", "(c) vs. $H^*$ alone", BLUE, "o"),
              ("recall_cal_minus_H", "(d) recall vs. $H^*$ alone", ORANGE, "D"))
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.25, 1, 1, 0.8], wspace=0.08))
    for ax, (key, title, col, mk) in zip(axs, panels):
        for k, yy in zip(keys, ys):
            m, lo, hi = N[k][key]
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(m, yy, mk, ms=4.2, mfc=col if sig else "white", mec=col, mew=1.1)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_title(title, fontsize=7.2, color=INK, loc="left")
        ax.tick_params(labelsize=6.5, length=2)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels(ylab, fontsize=6.5, color=INK)
    for gy, glab in gpos:
        axs[0].text(-0.02, gy - 0.75, glab, transform=axs[0].get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.5, color=MUTED, style="italic")
    axs[0].set_ylim(ys[-1] + 0.7, -1.4)
    for ax in axs[:3]:
        ax.set_xlabel("return difference", fontsize=6.8, color=MUTED)
    axs[3].set_xlabel("recall points", fontsize=6.8, color=MUTED)
    fig.savefig(out / "fig_robust.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_robust.png", bbox_inches="tight", dpi=200)


def e7(out):
    """Fig. 3: per incumbent, X / X+cal / X+nominal on test and test-dev; return (with per-training-seed dots) and recall."""
    E = json.loads(NUMS.read_text())["E7"]
    inc = [("H*", "$H^*$"), ("PPO-LA", "PPO-LA"), ("QR-DQN", "QR-DQN"), ("Discrete SAC", "Discrete SAC")]
    arms = (("X", "$X$ alone", BLUE, "o", -0.22), ("X+cal", "$X$ + calibrated MPC", ORANGE, "s", 0.0),
            ("X+nom", "$X$ + nominal MPC", GREEN, "^", 0.22))
    fig, axg = plt.subplots(2, 2, figsize=(3.45, 2.35), sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.62))
    axs = [axg[0, 0], axg[0, 1], axg[1, 0], axg[1, 1]]
    pan = (("test", "ret", "(a) return, test"), ("testdev", "ret", "(b) return, test-dev"),
           ("test", "recall", "(c) recall (%), test"), ("testdev", "recall", "(d) recall (%), test-dev"))
    for ax, (pool, met, title) in zip(axs, pan):
        lv = E[pool]["levels"]
        for i, (key, _) in enumerate(inc):
            for a, lab, col, mk, off in arms:
                L = lv[f"{key}|{a}"]
                if met == "ret" and key != "H*":
                    ax.plot(L["ret_per_seed"], [i + off] * 3, "|", ms=4, color=col, alpha=0.6, mew=0.9)
                ax.plot(L[met], i + off, mk, ms=4.6, color=col, mec="white", mew=0.5,
                        label=lab if (i == 0 and pool == "test" and met == "ret") else None)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_title(title, fontsize=7.2, color=INK, loc="left")
        ax.tick_params(labelsize=6.5, length=2)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in (axs[0], axs[2]):
        ax.set_yticks(range(len(inc)))
        ax.set_yticklabels([l for _, l in inc], fontsize=6.5, color=INK)
    axs[0].set_ylim(len(inc) - 0.5, -0.5)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, -0.07),
               handletextpad=0.2, columnspacing=0.8)
    fig.savefig(out / "fig_e7.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_e7.png", bbox_inches="tight", dpi=200)
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
             lsize=5.3)
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


# ---- line / bar versions (23/11 revision): colour = rate model of the planner, used consistently in all figures
GRAY = "#8a8985"
ARMSTYLE = {"+mpc": ("calibrated MPC (ours)", BLUE, "o", "-", 1.8),
            "+mpcnc": ("nominal MPC", ORANGE, "s", "-", 1.3),
            "+mpcrh": ("RobustMPC-style MPC", GREEN, "^", "-", 1.3),
            "H": ("$H^*$ alone (no MPC)", GRAY, "D", "--", 1.1)}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(labelsize=6.5, length=2)


def lines(out):
    """Fig. 2: return per image vs an ordered setting (95% CI of the mean as band; differences are paired, see text)."""
    N = json.loads(NUMS.read_text())
    T = N["table1"]
    pan = (("(a) bandwidth", "total bandwidth (MHz)", [("BW:B0.5", 0.5), ("NET:N5x2", 1.0), ("BW:B2", 2.0)], 5),
           ("(b) UAVs ($M$=2)", "number of UAVs $K$", [("NET:N3x2", 3), ("NET:N5x2", 5), ("NET:N7x2", 7)], None),
           ("(c) channels ($K$=5)", "number of channels $M$", [("NET:N5x2", 2), ("NET:N5x3", 3), ("NET:N5x4", 4)], 5))
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw=dict(wspace=0.42))
    for ax, (title, xl, pts, kfix) in zip(axs, pan):
        for arm in ("H", "+mpcrh", "+mpcnc", "+mpc"):
            lab, col, mk, ls, lw = ARMSTYLE[arm]
            xs, ys, lo, hi = [], [], [], []
            for key, x in pts:
                K = kfix or x
                L = T[key]["level"][arm]
                xs.append(x); ys.append(L["ret"] / (52 * K))
                lo.append(L["ret_ci"][1] / (52 * K)); hi.append(L["ret_ci"][2] / (52 * K))
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xs, ys, ls, color=col, lw=lw, marker=mk, ms=4, mec="white", mew=0.6, label=lab)
        ax.set_xticks([x for _, x in pts])
        ax.set_xticklabels([f"{x:g}" for _, x in pts])
        ax.set_xlabel(xl, fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=7.2, color=INK, loc="left")
        _style(ax)
    axs[0].set_ylabel("return per image", fontsize=7, color=MUTED)
    # (d) feedback delay: calibrated MPC with delayed reports (no loss / 30% loss); references are flat lines
    ax = axs[3]
    n5 = T["NET:N5x2"]["level"]
    E5 = N["E5"]
    xs = [0, 1, 3, 10]
    ys = [n5["+mpc"]["ret"]] + [E5[f"fb{d}l0"]["level"][0] for d in (1, 3, 10)]
    cis = [n5["+mpc"]["ret_ci"]] + [E5[f"fb{d}l0"]["level"] for d in (1, 3, 10)]
    for arm in ("H", "+mpcnc"):
        lab, col, mk, ls, lw = ARMSTYLE[arm]
        ax.axhline(n5[arm]["ret"] / 260, color=col, ls=ls, lw=lw)
    lab, col, mk, ls, lw = ARMSTYLE["+mpc"]
    ax.fill_between(xs, [c[1] / 260 for c in cis], [c[2] / 260 for c in cis], color=col, alpha=0.12, lw=0)
    ax.plot(xs, [y / 260 for y in ys], ls, color=col, lw=lw, marker=mk, ms=4, mec="white", mew=0.6)
    ax.plot([1, 3], [E5[f"fb{d}l30"]["level"][0] / 260 for d in (1, 3)], "o", ms=4.2, mfc="white", mec=col, mew=1.1,
            label="calibrated, 30% reports lost")
    ax.set_xticks(xs)
    ax.set_xlabel("report delay (slots)", fontsize=7, color=MUTED)
    ax.set_title("(d) report delay ($K$=5, $M$=2)", fontsize=7.2, color=INK, loc="left")
    _style(ax)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h[::-1] + h2, l[::-1] + l2, loc="lower center", ncol=5, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.2), handlelength=2.2, columnspacing=1.2)
    fig.savefig(out / "fig_lines.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_lines.png", bbox_inches="tight", dpi=200)


def phy_bars(out):
    """Fig. 3: gain of the calibrated MPC over each comparator (colour = comparator) under PHY variants and on test-dev
    images, and its recall change vs H* alone; whiskers = paired 95% CIs; hatched = CI includes 0."""
    T = json.loads(NUMS.read_text())["table1"]
    rows = [("PHY:L", "LA (nominal)"), ("PHY:L_fade", "block fading"), ("PHY:L_bursty", "bursty interf."),
            ("PHY:L_badfb", "degraded CQI"), ("TD:L", "LA, test-dev"), ("PHY:S", "Shannon"), ("TD:S", "Shannon, test-dev")]
    comp = (("cal_minus_nom", "+mpcnc", "vs. nominal MPC"), ("cal_minus_rh", "+mpcrh", "vs. RobustMPC-style"),
            ("cal_minus_H", "H", "vs. $H^*$ alone"))
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(3.45, 2.45), sharey=True, gridspec_kw=dict(width_ratios=[2.3, 1],
                                                                                            wspace=0.08))
    bh = 0.26
    for j, (key, arm, lab) in enumerate(comp):
        col = ARMSTYLE[arm][1]
        for i, (r, _) in enumerate(rows):
            m, lo, hi = T[r][key]
            sig = lo > 0 or hi < 0
            y = i + (j - 1) * bh
            ax.barh(y, m, height=bh * 0.92, color=col if sig else "white", edgecolor=col, lw=0.8,
                    hatch=None if sig else "////", label=lab if i == 0 else None)
            ax.plot([lo, hi], [y, y], color=INK, lw=0.6)
    for i, (r, _) in enumerate(rows):
        m, lo, hi = T[r]["recall_cal_minus_H"]
        axr.barh(i, m, height=0.5, color=GRAY, edgecolor=GRAY, lw=0.8)
        axr.plot([lo, hi], [i, i], color=INK, lw=0.6)
    for a in (ax, axr):
        a.axvline(0, color=MUTED, lw=0.8)
        a.grid(axis="x", color=GRID, lw=0.5)
        a.tick_params(labelsize=6.5, length=2)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
    ax.axhline(4.5, color=GRID, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for _, l in rows], fontsize=6.5, color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlabel("return gain of calibrated MPC", fontsize=6.8, color=MUTED)
    axr.set_xlabel("recall change vs. $H^*$", fontsize=6.8, color=MUTED)
    ax.legend(fontsize=5.8, frameon=False, loc="lower right", handlelength=1.2)
    fig.savefig(out / "fig_phy.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_phy.png", bbox_inches="tight", dpi=200)


def e7_bars(out):
    """Fig. 4: B1 (vs incumbent alone, grey) and B2 (vs nominal MPC, orange) for each incumbent; solid = test,
    hatched = test-dev; (a) return, (b) recall; whiskers = paired 95% CIs over environment seeds."""
    E = json.loads(NUMS.read_text())["E7"]
    inc = [("H*", "$H^*$"), ("PPO-LA", "PPO-LA"), ("QR-DQN", "QR-DQN"), ("Discrete SAC", "Discr. SAC")]

    def get(pool, key):
        return E[pool]["H_ref"] if key == "H*" else E[pool]["B"][key]

    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.75), sharex=True, gridspec_kw=dict(hspace=0.25))
    bw = 0.19
    bars = (("B1", "test", GRAY, None, "vs. $X$ alone, test"), ("B1", "testdev", GRAY, "////", "vs. $X$ alone, test-dev"),
            ("B2", "test", ORANGE, None, "vs. $X$ + nominal MPC, test"),
            ("B2", "testdev", ORANGE, "////", "vs. $X$ + nominal MPC, test-dev"))
    for ax, suf, yl in ((axs[0], "", "return gain"), (axs[1], "_recall", "recall change")):
        for j, (b, pool, col, hatch, lab) in enumerate(bars):
            for i, (key, _) in enumerate(inc):
                m, lo, hi = get(pool, key)[b + suf]
                x = i + (j - 1.5) * bw
                ax.bar(x, m, width=bw * 0.92, color=col if hatch is None else "white", edgecolor=col, lw=0.8,
                       hatch=hatch, label=lab if i == 0 else None)
                ax.plot([x, x], [lo, hi], color=INK, lw=0.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(yl, fontsize=6.8, color=MUTED)
        _style(ax)
    axs[0].set_title("(a) calibrated MPC on incumbent $X$: return", fontsize=7, color=INK, loc="left")
    axs[1].set_title("(b) recall (points)", fontsize=7, color=INK, loc="left")
    axs[1].set_xticks(range(len(inc)))
    axs[1].set_xticklabels([l for _, l in inc], fontsize=6.8, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=5.9, bbox_to_anchor=(0.5, -0.1),
               handlelength=1.6, columnspacing=1.0)
    fig.savefig(out / "fig_e7bars.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_e7bars.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    o = Path(sys.argv[1])
    o.mkdir(parents=True, exist_ok=True)
    method(o)
    forest(o)
    e7(o)
    recall(o)
    lines(o)
    phy_bars(o)
    e7_bars(o)
    print("figures ->", o)
