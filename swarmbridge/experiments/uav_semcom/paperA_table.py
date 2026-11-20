"""Paper A Table 1 rows (LaTeX) from runs/uav_semcom/paperA_r2/paper_numbers.json -> paper/paperA/table_main.tex.
Each cell: mean difference +- half-width of the 95% CI; bold if the CI excludes 0; '--' if not defined."""
import json

import a5_common as C

N = json.loads((C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "paper_numbers.json").read_text())
CONFS = [("S", "Shannon"), ("L", "LA"), ("L_fade", "LA, block fading"), ("L_bursty", "LA, bursty interf."),
         ("L_badfb", "LA, degraded CQI"), ("TD-S", "Shannon, test-dev"), ("TD-L", "LA, test-dev")]
COLS = ["A_return", "A_recall", "B_return", "B_recall", "C_return", "D_return", "E_return", "F_return"]


def cell(v):
    if v is None:
        return "--"
    m, lo, hi = v
    s = f"${m:+.1f}\\pm{(hi - lo) / 2:.1f}$"
    return f"$\\mathbf{{{m:+.1f}}}\\pm{(hi - lo) / 2:.1f}$" if (lo > 0 or hi < 0) else s


lines = []
for key, name in CONFS:
    lines.append(name + " & " + " & ".join(cell(N[key].get(c)) for c in COLS) + r" \\")
HEAD = r"""\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{A: PPO-S $-$ T:best} & \multicolumn{2}{c}{B: learned $-$ frozen $H^*$} & C: vs.\ re-tuned & D: cal.\ $-$ nominal & E: cal.\ $-$ RobustMPC-style & F: binned $-$ pooled\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Configuration & return & recall & return & recall & return & return & return & return\\
\midrule
"""
TAIL = "\\bottomrule\n\\end{tabular}\n"
(C.REPO / "paper" / "paperA" / "table_main.tex").write_text(HEAD + "\n".join(lines) + "\n" + TAIL, encoding="utf-8")
print("\n".join(lines))
