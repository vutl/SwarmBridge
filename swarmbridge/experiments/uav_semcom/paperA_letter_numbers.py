"""Paper A method letter (rewrite after E7): every number of the manuscript from stored result files (no new runs).
Output: runs/uav_semcom/paperA_r2/letter_numbers.json, paper/paperA/table_robust.tex, paper/paperA/table_e7.tex.

Table 1 (incumbent H*, paired 95% t-CI over 100 environment seeds):
  rows  E1 test pool (seeds 9900-9999; S, L, L_fade, L_bursty, L_badfb; H* frozen from S / nominal L),
        E3 test-dev (12000-12099; S, L), E4 network size (13000-13099; H* re-tuned per network),
        E6 bandwidth 0.5 / 2 MHz (13000-13099; H* re-tuned; 1 MHz = E4 N5x2)
  cols  cal - nominal MPC, cal - RobustMPC-style MPC, cal - H* (no MPC), Qmin recall cal - H*, slots > 50 ms (cal)
Table 2 (E7): from summary_e7.json (paperA_e7_summary.py).
Text: E2c ladder (summary_e2c.json), E5 feedback (e5.json vs e4.json), E1 decision-time statistics."""
import collections
import gzip
import json

import numpy as np

import a5_common as C
import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from ax2_summary import ci

PAPER = C.REPO / "paper" / "paperA"
ROWS = [("PHY", "Shannon", "e12", "S", None), ("PHY", "LA (nominal)", "e12", "L", None),
        ("PHY", "LA, block fading", "e12", "L_fade", None), ("PHY", "LA, bursty interf.", "e12", "L_bursty", None),
        ("PHY", "LA, degraded CQI", "e12", "L_badfb", None),
        ("TD", "Shannon", "e3", "S", None), ("TD", "LA (nominal)", "e3", "L", None),
        ("NET", "$K{=}3$, $M{=}2$", "e4", "N3x2", "H_net"), ("NET", "$K{=}5$, $M{=}2$", "e4", "N5x2", "H_net"),
        ("NET", "$K{=}7$, $M{=}2$", "e4", "N7x2", "H_net"), ("NET", "$K{=}5$, $M{=}3$", "e4", "N5x3", "H_net"),
        ("NET", "$K{=}5$, $M{=}4$", "e4", "N5x4", "H_net"),
        ("BW", "0.5 MHz", "e6", "B0.5", "H_bw"), ("BW", "2 MHz", "e6", "B2", "H_bw")]
GROUP = {"PHY": "Test pool, seeds 9900--9999 ($K{=}5$, $M{=}2$, 1 MHz)",
         "TD": "Test-dev pool, seeds 12000--12099",
         "NET": "Network size (1 MHz), LA, seeds 13000--13099",
         "BW": "Bandwidth ($K{=}5$, $M{=}2$), LA, seeds 13000--13099"}


def L(v):
    return [float(x) for x in v]


def main():
    AE.init(True)                                       # table with test and test-dev rows (as paperA_paper_numbers)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    cache = {}

    def load(name):
        if name not in cache:
            te = json.loads((R.OUT / f"{name}.json").read_text())
            dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
            by = collections.defaultdict(list)
            for r in te["results"]:
                by[(r["phy"], r["combo"])].append(r)
            cache[name] = (te, dl, by)
        return cache[name]

    def arm(name, phy, combo):
        te, dl, by = load(name)
        rs = by[(phy, combo)]
        assert len(rs) == 1 and len(rs[0]["rows"]) == 100, (name, phy, combo)
        r = rs[0]
        rec = []
        for ep in dl[f"{phy}|{combo}|{r['ckpt']}"]:
            tp = gt = 0.0
            for mi, mode, ok, ot in ep:
                rr = rows_of[mi]
                gt += T["ngt"][rr]
                if mode >= 0 and ok:
                    tp += T["gt"][rr, mode]
            rec.append(100 * tp / gt)
        return dict(seeds=r["seeds"], ret=np.array([x["return"] for x in r["rows"]]), rec=np.array(rec),
                    over50=float(np.mean([x["int_over50"] for x in r["rows"]])),
                    p99=np.array([x["ms_int_max_p99"] for x in r["rows"]]),
                    drop=float(np.mean([x["drop_rate"] for x in r["rows"]])))

    out = {"table1": {}}
    lines, last = [], None
    for grp, lab, name, phy, hkey in ROWS:
        te, _, _ = load(name)
        S = te["sched"]
        h = te[hkey][phy] if hkey else (te["H"]["S"] if phy == "S" else te["H"]["L"])
        a = {s: arm(name, phy, f"{S}|h={h}{s}") for s in ("", "+mpc", "+mpcnc", "+mpcrh")}
        assert all(a[s]["seeds"] == a[""]["seeds"] for s in a)
        row = dict(config=f"{name}:{phy}", H=h, seeds=[a[""]["seeds"][0], a[""]["seeds"][-1]],
                   cal_minus_nom=L(ci(a["+mpc"]["ret"] - a["+mpcnc"]["ret"])),
                   cal_minus_rh=L(ci(a["+mpc"]["ret"] - a["+mpcrh"]["ret"])),
                   cal_minus_H=L(ci(a["+mpc"]["ret"] - a[""]["ret"])),
                   recall_cal_minus_H=L(ci(a["+mpc"]["rec"] - a[""]["rec"])),
                   over50_cal=a["+mpc"]["over50"], p99_cal_mean=float(a["+mpc"]["p99"].mean()),
                   p99_cal_max=float(a["+mpc"]["p99"].max()),
                   level={s or "H": dict(ret=float(a[s]["ret"].mean()), recall=float(a[s]["rec"].mean()),
                                         drop=a[s]["drop"]) for s in a})
        out["table1"][f"{grp}:{phy}"] = row
        if grp != last:
            lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{GROUP[grp]}}}}}\\\\")
            last = grp
        lines.append(f"\\quad {lab} & " + " & ".join(cell(row[k]) for k in (
            "cal_minus_nom", "cal_minus_rh", "cal_minus_H", "recall_cal_minus_H")) + "\\\\")
        print(f"{grp:4s} {phy:9s} H={h:24s} nom {fmt(row['cal_minus_nom'])} rh {fmt(row['cal_minus_rh'])} "
              f"H {fmt(row['cal_minus_H'])} rec {fmt(row['recall_cal_minus_H'])} >50 {100 * row['over50_cal']:.1f}% "
              f"p99 {row['p99_cal_mean']:.1f} (max {row['p99_cal_max']:.1f})")
    head = (r"\begin{tabular}{@{}lrrrr@{}}" "\n" r"\toprule" "\n"
            r" & \multicolumn{3}{c}{Return: calibrated MPC minus} & Recall\\" "\n"
            r"\cmidrule(lr){2-4}" "\n"
            r"Configuration & nominal & RobustMPC-st. & $H^*$ alone & vs.\ $H^*$\\" "\n"
            r"\midrule" "\n")
    (PAPER / "table_robust.tex").write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # pass counts (LA configurations only for nominal / RobustMPC-style; all for H*)
    la = [k for k in out["table1"] if not k.endswith(":S")]
    out["counts"] = {c: dict(pos=sum(out["table1"][k][c][1] > 0 for k in la), neg=sum(out["table1"][k][c][2] < 0 for k in la),
                             of=len(la)) for c in ("cal_minus_nom", "cal_minus_rh", "cal_minus_H", "recall_cal_minus_H")}
    print("LA rows, CI>0 / CI<0:", out["counts"])

    # E1 decision time (all +mpc arms of e12 / e3 with H* or PPO incumbents)
    t = []
    for name in ("e12", "e3"):
        te, _, by = load(name)
        for (phy, combo), rs in by.items():
            if combo.endswith("+mpc"):
                for r in rs:
                    t += [(x["ms_int_max_p99"], x["int_over50"]) for x in r["rows"]]
    t = np.array(t)
    out["E1_latency"] = dict(p99_mean=float(t[:, 0].mean()), p99_min=float(t[:, 0].min()), p99_max=float(t[:, 0].max()),
                             over50_max=float(t[:, 1].max()), over50_mean=float(t[:, 1].mean()))
    print("E1/E3 +mpc latency:", out["E1_latency"])

    # E5 feedback delay / loss (N5x2, seeds 13000-13099)
    e4 = load("e4")[0]
    S, h = e4["sched"], e4["H_net"]["N5x2"]
    imm, nom, H0 = (arm("e4", "N5x2", f"{S}|h={h}{s}")["ret"] for s in ("+mpc", "+mpcnc", ""))
    out["E5"] = {}
    for fb in load("e5")[0]["fb"]:
        v = arm("e5", "N5x2", f"{S}|h={h}+mpc{fb}")["ret"]
        out["E5"][fb] = dict(minus_immediate=L(ci(v - imm)), minus_nominal=L(ci(v - nom)), minus_H=L(ci(v - H0)))
        print("E5", fb, {k: fmt(x) for k, x in out["E5"][fb].items()})

    # E2c ladder and E7 (copied from their summaries)
    out["E2c"] = json.loads((R.OUT / "summary_e2c.json").read_text())
    e7 = json.loads((R.OUT / "summary_e7.json").read_text())
    out["E7"] = e7
    json.dump(out, open(R.OUT / "letter_numbers.json", "w"), indent=1)

    # Table 2 (E7): absolute return / recall per arm (test), B1 / B2 (test, test-dev)
    lines = []
    for lab, key in (("$H^*$", "H*"), ("PPO-LA", "PPO-LA"), ("QR-DQN", "QR-DQN"), ("Discrete SAC", "Discrete SAC")):
        lv = e7["test"]["levels"]
        r = [f"{lv[f'{key}|{a}']['ret']:.1f}" for a in ("X", "X+cal", "X+nom")]
        rc = [f"{lv[f'{key}|{a}']['recall']:.1f}" for a in ("X", "X+cal", "X+nom")]
        if key == "H*":
            b = [e7["test"]["H_ref"]["B1"], e7["test"]["H_ref"]["B2"], e7["testdev"]["H_ref"]["B1"], e7["testdev"]["H_ref"]["B2"]]
        else:
            b = [e7["test"]["B"][key]["B1"], e7["test"]["B"][key]["B2"], e7["testdev"]["B"][key]["B1"], e7["testdev"]["B"][key]["B2"]]
        lines.append(f"{lab} & " + " / ".join(r) + " & " + " / ".join(rc) + " & " + " & ".join(cell(x) for x in b) + r"\\")
    head = (r"\begin{tabular}{@{}lccrrrr@{}}" "\n" r"\toprule" "\n"
            r" & Return (test) & Recall (test) & \multicolumn{2}{c}{Test} & \multicolumn{2}{c}{Test-dev}\\" "\n"
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
            r"Incumbent $X$ & $X$ / $+$cal / $+$nom & $X$ / $+$cal / $+$nom & B1 & B2 & B1 & B2\\" "\n"
            r"\midrule" "\n")
    (PAPER / "table_e7.tex").write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    print("\n".join(lines))


def fmt(v):
    m, lo, hi = v
    return f"{m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' '}"


def cell(v):
    m, lo, hi = v
    s = f"{m:+.1f}"
    return (f"$\\mathbf{{{s}}}\\pm{(hi - lo) / 2:.1f}$" if (lo > 0 or hi < 0) else f"${s}\\pm{(hi - lo) / 2:.1f}$")


if __name__ == "__main__":
    main()
