"""Paper A (ICT Express rewrite, 25/11): every number of the manuscript from STORED fd_v1 results only (no simulation).
Inputs: ablation / phase2 / phase3 / phase4 summary JSONs, supplement_extras.json, posthoc/{decomp,r5}.json; the two
recall contrasts not in any summary (ablation A8 ng - H*, A7 cal - ng) are computed with the registered recall
analysis (fd_posthoc.recall_pair: seed-level t-CI + video-cluster bootstrap B 2000, rng 0) from the stored deliveries.
Outputs: runs/uav_semcom/paperA_fd/paper/paper_numbers.json and paper/paperA/table_fd.tex.
Usage (WSL env): python paperA_fd_numbers.py"""
import json

import numpy as np

import fd_common as FD
import fd_posthoc as PH
from fd_next_summary import ci

R = FD.OUT
OUTD = R / "paper"
TEX = FD.C.REPO / "paper" / "paperA" / "table_fd.tex"
CFGS = ["N3x2", "N7x2", "N5x3", "N5x4", "B0.5", "B2"]
LABEL = {"N5x2": "$K{=}5$, $M{=}2$, 1 MHz (ref.)", "N3x2": "$K{=}3$, $M{=}2$", "N7x2": "$K{=}7$, $M{=}2$",
         "N5x3": "$K{=}5$, $M{=}3$", "N5x4": "$K{=}5$, $M{=}4$", "B0.5": "$B{=}0.5$ MHz", "B2": "$B{=}2$ MHz",
         "ppofd": "PPO", "qrfd": "QR-DQN", "dsfd": "Discrete SAC"}


def J(p):
    return json.loads((R / p).read_text())


def lvl(d):
    return dict(ret=d["ret"], seq_p99=d.get("ms_full_sum_p99"), par_p99=d.get("ms_int_max_p99"),
                seq_over=d.get("full_sum_over50", {}).get("mean"), par_over=d.get("full_max_over50", {}).get("mean"),
                ok=d.get("ok_rate"), drop=d.get("drop_rate"), qviol=d.get("q_viol_rate"), psnr=d.get("mean_psnr"))


def main():
    t, C = PH.ctx()
    N = {}
    # ---- ablation (reference N5x2, fresh seeds TD 24000-24099 / V 25000 / F 26000) ----
    ab = J("ablation/summary_ablation_test.json")
    dec = J("posthoc/decomp.json")
    ex = J("supplement_extras.json")
    H = J("phase2/tune.json")["H"]
    arms = {"H": "", "nc": "+mpcnc", "gate": "+mpcngg", "band": "+mpcngb", "ng": "+mpcng", "h1": "+mpcngh1",
            "h0": "+mpcngh0", "cal": "+mpc"}
    N["ablation"] = {}
    for pool in ("TD", "V", "F"):
        s = ab[pool]
        con = {k: dict(ci=v["ci"], holm=v.get("holm"), recall=v.get("recall")) for k, v in s["contrasts"].items()}
        con["nc-H"] = dict(ci=dec[f"ablation|{pool}|N5x2|nc-H*"]["diff"]["ret"], holm=None, recall=None)
        P = PH.load("ablation", f"test_{pool}", pool, C)
        con["cal-H"] = dict(ci=ci(P.ret("N5x2", f"h={H}+mpc") - P.ret("N5x2", f"h={H}")), holm=None, recall=None)
        for key, a, b in (("A8", "ng", "H"), ("A7", "cal", "ng"), ("nc-H", "nc", "H"), ("cal-H", "cal", "H")):
            con[key]["recall"] = PH.recall_pair(P, f"h={H}{arms[a]}", P, f"h={H}{arms[b]}")
        # post-hoc fixed-policy re-scoring with other mission-time weights (controllers NOT re-optimised):
        # J_w = 52K mean_util - 0.02 E - w K mission_s, from the stored per-episode aggregates
        def rew(a, w):
            return np.mean([[52 * 5 * r["mean_util"] - 0.02 * r["energy_kj"] - w * 5 * r["mission_s"] for r in v[0]]
                            for v in P.arms[("N5x2", f"h={H}{arms[a]}")].values()], 0)
        con["weights"] = {str(w): {f"{a}-{b}": ci(rew(a, w) - rew(b, w)) for a, b in
                                   (("nc", "H"), ("ng", "H"), ("band", "nc"), ("ng", "nc"), ("cal", "ng"))}
                          for w in (0.3, 0.15, 0.0)}
        levels = {k: lvl(v) for k, v in s["levels"].items()}
        for k, a in arms.items():
            levels[k]["recall"] = ex[f"ablation|test|{pool}|N5x2|h={H}{a}"]["recall"]
        N["ablation"][pool] = dict(contrasts=con, levels=levels)
    # ---- Phase 2 (reference, TD 20000-20099 etc.; confirmatory H1, H3b) ----
    p2 = J("phase2/summary_test.json")
    N["phase2"] = {pool: dict(ret=p2[pool]["return"], holm=p2[pool].get("holm"), recall=p2[pool].get("recall"))
                   for pool in ("TD", "F", "V", "TD_all")}
    for pool in ("TD", "F", "V", "TD_all"):
        P = PH.load("phase2", f"test_{pool}", pool, C)
        a = lambda s: f"h={H}{s}"
        N["phase2"][pool]["posthoc"] = dict(
            ng_H=dict(ci=ci(P.ret("N5x2", a("+mpcng")) - P.ret("N5x2", a(""))),
                      recall=PH.recall_pair(P, a("+mpcng"), P, a(""))),
            ng_rh=dict(ci=ci(P.ret("N5x2", a("+mpcng")) - P.ret("N5x2", a("+mpcrh")))),
            ng_rhnb=dict(ci=ci(P.ret("N5x2", a("+mpcng")) - P.ret("N5x2", a("+mpcrhnb")))),
            rh_H=dict(ci=ci(P.ret("N5x2", a("+mpcrh")) - P.ret("N5x2", a("")))),
            cal_H_recall=PH.recall_pair(P, a("+mpc"), P, a("")),
            cal_ng_recall=PH.recall_pair(P, a("+mpc"), P, a("+mpcng")))
    # ---- Phase 4 (TD 20000-20099, F 22000-22099; Holm over 12 on TD) ----
    g4 = J("phase4/summary_grid_test.json")
    N["phase4"] = {pool: {c: dict(G1=g4[pool]["contrasts"][f"G1_{c}"], G2=g4[pool]["contrasts"][f"G2_{c}"],
                                  G3=g4[pool]["contrasts"][f"G3_{c}"], G4=g4[pool]["contrasts"][f"G4_{c}"],
                                  levels={k: lvl(v) for k, v in g4[pool]["levels"][c].items()}) for c in CFGS}
                   for pool in ("TD", "F")}
    # ---- Phase 3 retrained (Holm over 6 on TD) + post-hoc R5 ----
    r3 = J("phase3/summary_rl_retrained.json")
    r5 = J("posthoc/r5.json")
    N["phase3"] = {}
    for pool in ("TD", "F", "V", "TD_all"):
        s = r3[pool]
        N["phase3"][pool] = {x: dict(R1=s["contrasts"][f"R1_{x}"], R2=s["contrasts"][f"R2_{x}"],
                                     R3=s["contrasts"][f"R3_{x}"],
                                     levels={a: lvl(s["levels"][x + a]) for a in ("", "+mpcnc", "+mpcng", "+mpc")},
                                     R5={tag: r5[f"retrained|{pool}|{x}|{tag}"] for tag in
                                         ("alone", "matched", "calibrated", "unguarded")})
                             for x in ("ppofd", "qrfd", "dsfd")}
    N["zeroshot_R5"] = {x: {tag: r5[f"zeroshot|TD|{x}|{tag}"] for tag in ("alone", "matched", "calibrated")}
                        for x in ("ppola", "qrla", "dsla")}
    N["decomp"] = {k: v for k, v in dec.items()}
    OUTD.mkdir(exist_ok=True)
    (OUTD / "paper_numbers.json").write_text(json.dumps(N, indent=1))
    table(N)
    print("written", OUTD / "paper_numbers.json", TEX)


def sg(v, dig=1):
    s = f"{v:+.{dig}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def fmt(ci, holm=None, dig=1):
    m, lo, hi = ci
    mark = "$^\\dagger$" if (holm and holm.get("rejected")) else ("$^*$" if (lo > 0 or hi < 0) else "")
    return f"{sg(m, dig)}{mark}"


def fci(ci, holm=None):
    m, lo, hi = ci
    return f"{fmt(ci, holm)} {{\\scriptsize[{sg(lo)}, {sg(hi)}]}}"


def frec(r):
    s, c = r["seed"], r["cluster"]
    sig_s, sig_c = (s[1] > 0 or s[2] < 0), (c[1] > 0 or c[2] < 0)
    return sg(s[0]) + ("$^*$" if sig_s and sig_c else "$^\\circ$" if sig_s else "")


def ftime(lv):
    return f"{lv['seq_p99']:.0f} / {100 * lv['seq_over']:.0f} / {100 * lv['par_over']:.1f}"


def frec2(seed, cluster):
    return frec(dict(seed=seed, cluster=cluster))


def table(N):
    """Table II: guarded - base, calibrated - guarded, calibrated - base (return) and timing of the guarded MPC."""
    rows = []
    p2 = N["phase2"]["TD"]
    ph = p2["posthoc"]
    rows.append(("Reference, pre-registered$^b$", fci(ph["ng_H"]["ci"]), frec(ph["ng_H"]["recall"]),
                 fci(p2["ret"]["H3b"], p2["holm"]["H3b"]), frec(ph["cal_ng_recall"]),
                 fci(p2["ret"]["H1"], p2["holm"]["H1"]), "---"))
    a = N["ablation"]["TD"]
    c = a["contrasts"]
    rows.append(("Reference, ablation seeds$^a$", fci(c["A8"]["ci"]), frec(c["A8"]["recall"]), fci(c["A7"]["ci"]),
                 frec(c["A7"]["recall"]), fci(c["cal-H"]["ci"]), ftime(a["levels"]["ng"])))
    for cf in CFGS:
        g = N["phase4"]["TD"][cf]
        rows.append((LABEL[cf], fci(g["G1"]["ci"], g["G1"]["holm"]), frec(g["G1"]["recall"]),
                     fci(g["G2"]["ci"], g["G2"]["holm"]), frec(g["G2"]["recall"]), fci(g["G3"]["ci"]),
                     ftime(g["levels"]["ng"])))
    rl = []
    for x in ("ppofd", "qrfd", "dsfd"):
        s = N["phase3"]["TD"][x]
        rl.append((LABEL[x], fci(s["R1"]["ci"], s["R1"]["holm"]), frec(s["R1"]["recall"]),
                   fci(s["R2"]["ci"], s["R2"]["holm"]), frec(s["R2"]["recall"]), fci(s["R3"]["ci"]),
                   ftime(s["levels"]["+mpcng"])))
    L = ["% generated by swarmbridge/experiments/uav_semcom/paperA_fd_numbers.py from stored results; do not edit",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Guarded MPC $-$ base} & \\multicolumn{2}{c}{Calibrated $-$ guarded MPC} "
         "& Calibrated $-$ base & Timing \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-7}",
         "Setting & $\\Delta$return [95\\% CI] & $\\Delta$recall & $\\Delta$return [95\\% CI] & $\\Delta$recall "
         "& $\\Delta$return [95\\% CI] & ms / \\% / \\% \\\\",
         "\\midrule", "\\multicolumn{7}{@{}l}{\\emph{Base: tuned heuristic $H^*$ (re-tuned per setting); "
         "reference $K{=}5$, $M{=}2$, $B{=}1$ MHz}} \\\\"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\midrule", "\\multicolumn{7}{@{}l}{\\emph{Base: learned incumbent $X$ alone (reference setting)}} \\\\"]
    L += [" & ".join(r) + " \\\\" for r in rl]
    L += ["\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
