"""Paper A independent-evaluation figures, table and supplementary exports. Reads ONLY the registered analysis
outputs runs/uav_semcom/paperA_ix/analysis/*.json and the stage-A training records (no experiment is re-run).

  Fig. 2  fig_ix_ablation.{pdf,png}  component ablation, ablation blocks of X1 / UAVDT-frozen / UAVDT-retrained:
          (a) N1 nominal-H*, A2 band-nominal, N2 guarded-H*, N3 calibrated-guarded; (b) A1 gate-nominal, A4 H4-H1,
          A4b H4-H0. Bars: unadjusted paired 95% t CI. Filled marker: Holm-rejected in the registered family
          (N1-N3: family N; A1, A2, A4, A4b: ablation family); hollow: not rejected.
  Fig. 3  fig_ix_tradeoff.{pdf,png}  return vs vehicle-recall differences. Inclusion rule (fixed before plotting):
          all seven Holm-family contrasts of the UAVDT ablation block (A1, A2, A4, A4b, N1, N2, N3), both arms; X1 has
          no recall and is excluded. x: paired 95% t CI of the return difference; y: sequence-cluster bootstrap 95%
          CI of the system-level vehicle recall difference (percentage points).
  Table 1 table_ix_results.tex  UAVDT-retrained: reference row = ablation block (N2, N3), six settings = grid block
          (G1, G2), learned incumbents = RL block (R1, R2).
  Data    runs/uav_semcom/paperA_ix/paper/: ix_paper_numbers.json (exact plotted / tabulated values),
          ix_contrasts_all.csv, ix_levels_all.csv (incl. controller cost), ix_mode_utilisation.csv,
          ix_train_action_hist.csv.

Usage: python paperA_ix_figures.py [FIGDIR]   (default paper/paperA/figs; the table goes to paper/paperA)"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
IX = REPO / "runs" / "uav_semcom" / "paperA_ix"
AN = IX / "analysis"
OUTD = IX / "paper"
PAPER = REPO / "paper" / "paperA"

for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
    "axes.linewidth": 0.7, "axes.edgecolor": "black", "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "legend.fontsize": 6.2,
    "pdf.fonttype": 42, "mathtext.default": "regular"})
INK, MUTED = "#1a1a1a", "#666666"
COND = {"x1": dict(block="x1_abl", color="#0072bd", marker="o", label="VisDrone X1"),
        "x2f": dict(block="x2f_abl", color="#d95319", marker="s", label="UAVDT frozen"),
        "x2r": dict(block="x2r_abl", color="#4d4d4d", marker="^", label="UAVDT retrained")}
PANEL_A = [("N1", "Nominal MPC $-$ $H^*$"), ("A2", "Band only $-$ nominal MPC"), ("N2", "Guarded MPC $-$ $H^*$"),
           ("N3", "Calibrated $-$ guarded MPC")]
PANEL_B = [("A1", "Gate only $-$ nominal MPC"), ("A4", "Look-ahead $H{=}4$ $-$ $H{=}1$"),
           ("A4b", "Look-ahead $H{=}4$ $-$ $H{=}0$")]
TRADE = ["N1", "A2", "N2", "N3", "A1", "A4", "A4b"]
SHORT = {"N1": "Nominal $-$ $H^*$", "A2": "Band only $-$ nominal", "N2": "Guarded $-$ $H^*$",
         "N3": "Calibrated $-$ guarded", "A1": "Gate only $-$ nominal", "A4": "$H{=}4$ $-$ $H{=}1$",
         "A4b": "$H{=}4$ $-$ $H{=}0$"}


def load(block):
    return json.loads((AN / f"{block}.json").read_text())


def con(block, key):
    c = load(block)["contrasts"][key]
    m, lo, hi = c["ci"]
    out = dict(block=block, contrast=key, a=c["a"], b=c["b"], cfg=c["cfg"], family=c["family"], d=m, lo=lo, hi=hi,
               p=c["p"], p_holm=c.get("holm", {}).get("p_holm"), holm_rejected=c.get("holm", {}).get("rejected"))
    r = c.get("recall")
    if r:
        for v, rv in r.items():
            out[f"rec_{v}"] = dict(seed=rv["seed"], cluster=rv["cluster"], levels=rv["levels"])
    return out


# ------------------------------------------------------------------ Fig. 2
def fig_ablation(figdir, nums):
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.75), sharex=True,
                             gridspec_kw=dict(height_ratios=[len(PANEL_A), len(PANEL_B)], hspace=0.08))
    off = {"x1": -0.25, "x2f": 0.0, "x2r": 0.25}
    nums["fig2"] = {}
    for ax, rows, tag in ((axes[0], PANEL_A, "(a)"), (axes[1], PANEL_B, "(b)")):
        for i, (k, lab) in enumerate(rows):
            for cnd, st in COND.items():
                c = con(st["block"], k)
                nums["fig2"][f"{cnd}|{k}"] = c
                rej = bool(c["holm_rejected"])
                ax.errorbar(c["d"], i + off[cnd], xerr=[[c["d"] - c["lo"]], [c["hi"] - c["d"]]], fmt=st["marker"],
                            color=st["color"], mfc=st["color"] if rej else "white", mew=0.8, ms=3.6, lw=0.8,
                            capsize=1.4)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for _, lab in rows], fontsize=6.3)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.015, 0.04 if tag == "(a)" else 0.06, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, fontweight="bold")
    axes[1].set_xlabel(r"$\Delta$ return per episode (paired 95% CI)")
    axes[1].set_xlim(-21, 29)
    hs = [Line2D([], [], marker=st["marker"], color=st["color"], lw=0.8, ms=3.6) for st in COND.values()]
    hs += [Line2D([], [], marker="o", color=MUTED, mfc="white", lw=0, ms=3.6)]
    axes[0].legend(hs, [st["label"] for st in COND.values()] + ["not Holm-rejected"], loc="lower center",
                   bbox_to_anchor=(0.42, 1.0), ncol=4, fontsize=5.7, handletextpad=0.25, columnspacing=0.8,
                   handlelength=1.4, borderpad=0.2)
    fig.savefig(figdir / "fig_ix_ablation.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(figdir / "fig_ix_ablation.png", bbox_inches="tight", dpi=250)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3
# label offsets (points) per arm and contrast, set by visual inspection only (no data selection)
LAB = {"x2f": {"N1": (5, -2, "left"), "A2": (6, -9, "left"), "N2": (6, 6, "left"), "N3": (-20, 13, "right"),
               "A1": (-20, -13, "right"), "A4": (-24, 1, "right"), "A4b": (8, 9, "left")},
       "x2r": {"N1": (6, 4, "left"), "A2": (6, -7, "left"), "N2": (6, 5, "left"), "N3": (-15, -12, "right"),
               "A1": (-16, 10, "right"), "A4": (-6, 6, "right"), "A4b": (-4, 6, "right")}}


def fig_tradeoff(figdir, nums):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True, gridspec_kw=dict(wspace=0.06))
    nums["fig3"] = {"rule": "all Holm-family contrasts of the UAVDT ablation block (A1, A2, A4, A4b, N1, N2, N3)",
                    "x": "paired 95% t CI of the return difference",
                    "y": "sequence-cluster bootstrap 95% CI of the vehicle recall difference (pp)"}
    for ax, cnd in zip(axes, ("x2f", "x2r")):
        st = COND[cnd]
        for k in TRADE:
            c = con(st["block"], k)
            r = c["rec_veh"]["cluster"]
            nums["fig3"][f"{cnd}|{k}"] = dict(ret=[c["d"], c["lo"], c["hi"]], recall_cluster=r,
                                              recall_seed=c["rec_veh"]["seed"], holm_rejected=c["holm_rejected"])
            rej = bool(c["holm_rejected"])
            ax.errorbar(c["d"], r[0], xerr=[[c["d"] - c["lo"]], [c["hi"] - c["d"]]], yerr=[[r[0] - r[1]], [r[2] - r[0]]],
                        fmt=st["marker"], color=st["color"], mfc=st["color"] if rej else "white", mew=0.8, ms=4.0,
                        lw=0.7, capsize=1.2, elinewidth=0.6)
            dx, dy, ha = LAB[cnd][k]
            lead = abs(dx) >= 15
            ax.annotate(SHORT[k], (c["d"], r[0]), xytext=(dx, dy), textcoords="offset points", fontsize=5.6,
                        ha=ha, va="center", color=INK,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1, shrinkB=2.5) if lead else None)
        ax.axvline(0, color="black", lw=0.6)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlim(-16, 29)
        ax.set_xlabel(r"$\Delta$ return per episode (paired 95% CI)")
        ax.set_title(st["label"], loc="left", fontsize=6.6, fontweight="bold", color=st["color"], pad=2)
    axes[0].set_ylim(-11.5, 22)
    axes[0].set_ylabel(r"$\Delta$ vehicle recall (pp)")
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(0.985, 0.97, tag, transform=ax.transAxes, ha="right", va="top", fontsize=7, fontweight="bold")
    fig.savefig(figdir / "fig_ix_tradeoff.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(figdir / "fig_ix_tradeoff.png", bbox_inches="tight", dpi=250)
    plt.close(fig)


# ------------------------------------------------------------------ Table 1
def sgn(v, nd=1):
    s = f"{v:+.{nd}f}"
    return s.replace("-", "$-$")


def rec_mark(c):
    rv = c["rec_veh"]
    s_ex = rv["seed"][1] > 0 or rv["seed"][2] < 0
    c_ex = rv["cluster"][1] > 0 or rv["cluster"][2] < 0
    return "\\rlap{$^*$}" if (s_ex and c_ex) else ("\\rlap{$^\\circ$}" if s_ex else "")


def cell(c):
    j = sgn(c["d"]) + ("\\rlap{$^\\dagger$}" if c["holm_rejected"] else "")
    ci = f"[{sgn(c['lo'])}, {sgn(c['hi'])}]"
    r = sgn(c["rec_veh"]["cluster"][0]) + rec_mark(c)
    return f"{j} & {ci} & {r}"


SETS = [("N3x2", "3", "2", "1"), ("N7x2", "7", "2", "1"), ("N5x3", "5", "3", "1"), ("N5x4", "5", "4", "1"),
        ("B0.5", "5", "2", "0.5"), ("B2", "5", "2", "2")]
RL = [("ppoix", "PPO"), ("qrix", "QR-DQN"), ("dsix", "Discrete SAC")]


def table(nums):
    T = {}
    rows = []
    a, b = con("x2r_abl", "N2"), con("x2r_abl", "N3")
    T["ref"] = dict(guarded_minus_inc=a, cal_minus_guarded=b)
    rows.append(f"5 & 2 & 1\\rlap{{$^a$}} & {cell(a)} & {cell(b)} \\\\")
    for cf, k, m, bw in SETS:
        a, b = con("x2r_grid", f"G1_{cf}"), con("x2r_grid", f"G2_{cf}")
        T[cf] = dict(guarded_minus_inc=a, cal_minus_guarded=b)
        rows.append(f"{k} & {m} & {bw} & {cell(a)} & {cell(b)} \\\\")
    rl = []
    for key, name in RL:
        a, b = con("x2r_rl", f"R1_{key}"), con("x2r_rl", f"R2_{key}")
        T[key] = dict(guarded_minus_inc=a, cal_minus_guarded=b)
        rl.append(f"\\multicolumn{{3}}{{@{{}}l}}{{{name}}} & {cell(a)} & {cell(b)} \\\\")
    nums["table1"] = T
    tex = [
        "% Generated by swarmbridge/experiments/uav_semcom/paperA_ix_figures.py from runs/uav_semcom/paperA_ix/analysis.",
        "% UAVDT-retrained (X2-retrained) only. Do not edit by hand.",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccc r c r r c r@{}}",
        "\\toprule",
        "\\multicolumn{3}{c}{Network} & \\multicolumn{3}{c}{Guarded $-$ incumbent} & "
        "\\multicolumn{3}{c}{Calibrated $-$ guarded} \\\\",
        "\\cmidrule(lr){1-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-9}",
        "$K$ & $M$ & $B$ (MHz) & $\\Delta J$ & 95\\% CI & $\\Delta R$ (pp) & $\\Delta J$ & 95\\% CI & $\\Delta R$ (pp) \\\\",
        "\\midrule",
        "\\multicolumn{9}{@{}l}{\\textit{Heuristic incumbent $H^*$ (re-tuned on UAVDT training data)}} \\\\",
        *rows,
        "\\midrule",
        "\\multicolumn{9}{@{}l}{\\textit{Learned incumbents trained on UAVDT ($K=5$, $M=2$, $B=1$ MHz)}} \\\\",
        *rl,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\par\\smallskip",
        "\\begin{minipage}{\\textwidth}\\footnotesize",
        "UAVDT test split, 100 paired seeds per block. $\\Delta R$: system-level vehicle recall, sequence-cluster mean. "
        "$^\\dagger$ Holm-rejected return test within its registered family; $^*$ recall: seed and cluster 95\\% CIs "
        "exclude zero; $^\\circ$ seed CI only. CIs are unadjusted. $^a$ Ablation block (seeds 32100--32199; family of "
        "three). Changed settings: grid block (seeds 32200--32299; family of 12). Learned incumbents: RL block "
        "(seeds 32000--32099; family of six; three checkpoints per algorithm averaged).",
        "\\end{minipage}", ""]
    (PAPER / "table_ix_results.tex").write_text("\n".join(tex), encoding="utf-8")


# ------------------------------------------------------------------ supplementary exports
BLOCKS = ["x1_ref", "x1_abl", "x2f_ref", "x2f_abl", "x2r_ref", "x2r_abl", "x2r_grid", "x2r_rl"]


def exports(nums):
    OUTD.mkdir(parents=True, exist_ok=True)
    with open(OUTD / "ix_contrasts_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["block", "contrast", "cfg", "a", "b", "family", "d_return", "ci_lo", "ci_hi", "p", "p_holm",
               "holm_rejected"]
        for v in ("veh", "cls_vancar", "cls_car"):
            hdr += [f"{v}_seed_d", f"{v}_seed_lo", f"{v}_seed_hi", f"{v}_cluster_d", f"{v}_cluster_lo",
                    f"{v}_cluster_hi", f"{v}_level_a", f"{v}_level_b"]
        w.writerow(hdr)
        for b in BLOCKS:
            for k in load(b)["contrasts"]:
                c = con(b, k)
                row = [b, k, c["cfg"], c["a"], c["b"], c["family"], c["d"], c["lo"], c["hi"], c["p"], c["p_holm"],
                       c["holm_rejected"]]
                for v in ("veh", "cls_vancar", "cls_car"):
                    rv = c.get(f"rec_{v}")
                    row += ([*rv["seed"], *rv["cluster"], *rv["levels"]] if rv else [""] * 8)
                w.writerow(row)
    keys = None
    with open(OUTD / "ix_levels_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        for b in BLOCKS:
            for arm, L in load(b)["levels"].items():
                flat = {}
                for k, v in L.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            flat[f"{k}_{kk}"] = vv
                    else:
                        flat[k] = v
                if keys is None:
                    keys = list(flat)
                    w.writerow(["block", "arm"] + keys)
                w.writerow([b, arm] + [flat.get(k, "") for k in keys])
    with open(OUTD / "ix_mode_utilisation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "key", "mode", "selected", "share", "compliant", "below_floor", "late_drop",
                    "no_mode", "finished_with_mode"])
        for b in BLOCKS:
            for key, u in load(b)["mode_utilisation"].items():
                for mode, s in u["modes"].items():
                    w.writerow([b, key, mode, s["selected"], s["share"], s["compliant"], s["below_floor"],
                                s["late_drop"], u["no_mode"], u["finished_with_mode"]])
    with open(OUTD / "ix_train_action_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        modes = None
        for p in sorted((IX / "train").glob("*.train.json")):
            r = json.loads(p.read_text())
            if modes is None:
                modes = r["action_log"]["modes"]
                w.writerow(["run", "phase", "total"] + modes)
            h = r["log"][-1]["train_action_hist"]
            parts = h.items() if isinstance(h, dict) else [("policy", h)]
            for ph, v in parts:
                w.writerow([p.name.replace(".train.json", ""), ph, sum(v)] + list(v))
    nums["sources"] = {b: str((AN / f"{b}.json").relative_to(REPO)) for b in BLOCKS}
    (OUTD / "ix_paper_numbers.json").write_text(json.dumps(nums, indent=1))


if __name__ == "__main__":
    figdir = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    N = {}
    fig_ablation(figdir, N)
    fig_tradeoff(figdir, N)
    table(N)
    exports(N)
    print("done", figdir)
