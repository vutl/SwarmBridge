"""H1-CROP pilot 0B-TUNE evaluation (CPU only): conditions x budgets x alpha x tau_decl on precomputed records,
then the screening criteria C1-C5 (protocol sec 6, readiness v2). EXPLORATORY: alpha and tau_decl are chosen on the
same pilot sequences they are reported on (declared in every row), every condition gets the same tuning.

Resource model (review #9): B = joint UPLINK budget. DL bitmap bits of T2 / T2v are reported separately (main
rho = 0); rho in RHO_SENS converts DL bits into uplink-equivalent bits of a shared-time budget (sensitivity).
CPU (review #9): the primary per-condition figure encodes every candidate the condition may send (bit lengths
are only known after encoding); "sent only" is a supplementary optimistic figure.

Usage (records from h1_pilot0b_precompute.py; GT read through mdmt_adapter, never from the records):
    python h1_pilot0b_eval.py CONFIG_JSON PRECOMP_DIR OUT_JSON --zip MDMT_ZIP [--procs 5]
"""
import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import h1_pilot0b as P

ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)
TAUS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
RATES_MBPS = (2, 5, 10, 20)
RHO_SENS = (1.0, 4.0)
EPOCH_S = 1.0
C3_MAX_PRECISION_DROP = 0.05      # registered in readiness v2 before any MDMT result


def grid(levels, frame_level, qs):
    """cells (cond, q, B, rho)."""
    out = []
    for B in levels:
        out += [("T0", None, B, 0.0), ("T0+", None, B, 0.0), ("F", None, B, 0.0)]
        for q in qs:
            out += [(c, q, B, 0.0) for c in P.CROP_CONDS]
            out += [("T2", q, B, r) for r in RHO_SENS]
    out += [(c, None, frame_level, 0.0) for c in ("F", "T0", "T0+")]
    return out


def _new_acc():
    return {t: dict(v={}, va={}, u={}) for t in TAUS}


def eval_seq(args):
    """all grid cells for one sequence -> {(cond, q, B, rho, alpha): stats}."""
    pairs, cells = args
    res = {}
    ctx = [P.t2_context(*p) for p in pairs]
    ff = [{(v, kind): P.false_flags(r, g, kind) for v, (r, g) in enumerate(((p[0], p[2]), (p[1], p[3])))
           for kind in ("P1", "P2")} for p in pairs]
    for cond, q, B, rho in cells:
        qq = q if q is not None else P.FRAME_Q
        for a in ALPHAS:
            acc = _new_acc()
            st = dict(ul=[], dl=[], man=[], crops=[], false_crops=[], false_bits=[], drops=[], coord_ok=[],
                      cpu_all=[], cpu_sent=[], fp_view=[])
            frames, air = 0, {r: 0 for r in RATES_MBPS}
            tco = {r: [] for r in RATES_MBPS}
            for ei, (r1, r2, g1, g2) in enumerate(pairs):
                e = P.run_epoch(r1, r2, g1, g2, cond, qq, B, a, rho=rho, ctx=ctx[ei])
                st["ul"].append(e["ul"]); st["dl"].append(e["dl"]); st["man"].append(e["man_ul"])
                st["drops"].append(len(e["drop"])); st["coord_ok"].append(bool(e.get("coord_ok", False)))
                for r in RATES_MBPS:
                    air[r] += e["ul"] / (r * 1e6) > EPOCH_S
                    if cond in P.COORD_CONDS:                    # UAV1 manifest UL then DL bitmap (R_DL = R_UL)
                        tco[r].append(1e3 * (e["man_ul"] + e["dl"]) / (r * 1e6))
                nc = nf = bf = 0
                for v, (rec, s) in enumerate(((r1, e["sent"][0]), (r2, e["sent"][1]))):
                    frames += s["frame"]
                    st["cpu_all"].append(P.cpu_ms_all(rec, cond, qq)); st["cpu_sent"].append(P.cpu_ms(rec, qq, s))
                    for kind, k, bits in s["items"]:
                        if kind in ("P1", "P2"):
                            nc += 1
                            if ff[ei][(v, kind)][k]:
                                nf += 1; bf += bits
                st["crops"].append(nc); st["false_crops"].append(nf); st["false_bits"].append(bf)
                for t in TAUS:
                    v1 = P.view_counts(e["decl"][0], g1, t); v2 = P.view_counts(e["decl"][1], g2, t)
                    P.add(acc[t]["v"], v1); P.add(acc[t]["v"], v2)
                    P.add(acc[t]["u"], P.unique_counts(v1, v2, g1, g2))
                    P.add(acc[t]["va"], P.view_counts(e["decl"][0], g1, t, False))
                    P.add(acc[t]["va"], P.view_counts(e["decl"][1], g2, t, False))
            n = len(pairs)
            res[(cond, q, B, rho, a)] = dict(
                acc=acc, n_epochs=n, frame_frac=frames / (2 * n),
                **{k: float(np.mean(st[k])) for k in ("ul", "dl", "man", "crops", "false_crops", "false_bits",
                                                     "drops", "coord_ok")},
                cpu_all=st["cpu_all"], cpu_sent=st["cpu_sent"],
                airtime_over={r: air[r] / n for r in RATES_MBPS},
                t_coord_ms={r: (float(np.mean(tco[r])) if tco[r] else 0.0) for r in RATES_MBPS})
    return res


def screening(pairs_by_seq, qs):
    c12, c4, cpu = {}, dict(matched=0, redundant=0), {q: [] for q in qs}
    per_seq = {}
    for s, pairs in pairs_by_seq.items():
        acc = {}
        for r1, r2, g1, g2 in pairs:
            for rec, g in ((r1, g1), (r2, g2)):
                for k, v in P.c1_c2(rec, g).items():
                    acc[k] = acc.get(k, 0) + v
                for q in qs:
                    cpu[q].append(P.cpu_ms_all(rec, "T1", q))
            for k, v in P.c4_redundancy(r1, r2, g1, g2).items():
                c4[k] += v
        per_seq[s] = dict(acc, miss_rate=acc["missed"] / max(acc["n_gt"], 1),
                          cand_recall=acc["missed_covered"] / max(acc["missed"], 1),
                          false_cand_per_view_epoch=acc["false_cand"] / (2 * len(pairs)))
        for k, v in acc.items():
            c12[k] = c12.get(k, 0) + v
    n_views = 2 * sum(len(p) for p in pairs_by_seq.values())
    return dict(
        C1=dict(miss_rate=c12["missed"] / max(c12["n_gt"], 1), threshold=0.15, per_seq=per_seq),
        C2=dict(cand_recall=c12["missed_covered"] / max(c12["missed"], 1), threshold=0.5,
                false_cand_per_view_epoch=c12["false_cand"] / n_views,
                false_bits_per_view_epoch_q35=c12["false_bits"] / n_views,
                cand_per_view_epoch=c12["n_cand"] / n_views),
        C4_proposal_redundancy=dict(c4, frac=c4["redundant"] / max(c4["matched"], 1), threshold=0.2),
        C5=dict({f"q{q}": dict(median_ms=float(np.median(cpu[q])), p90_ms=float(np.percentile(cpu[q], 90)))
                 for q in qs}, epoch_ms=1000 * EPOCH_S,
                note="PRIMARY: YOLO11n + AVIF encode of EVERY P1 candidate (needed to know exact bits); "
                     "desktop CPU proxy, 1 thread"))


def _pooled(rows, t, key):
    acc = {}
    for r in rows:
        P.add(acc, r["acc"][t][key])
    return acc


def summarise(per_seq_res, cells):
    """choose (alpha, tau) per cell by MACRO (mean per-seq) per-view F1; report macro AND micro (pooled) F1."""
    seqs = sorted(per_seq_res)
    out = []
    for cond, q, B, rho in cells:
        best = None
        for a in ALPHAS:
            for t in TAUS:
                f = float(np.mean([P.f1(per_seq_res[s][(cond, q, B, rho, a)]["acc"][t]["v"]) for s in seqs]))
                if best is None or f > best[0] + 1e-12:
                    best = (f, a, t)
        f, a, t = best
        rows = [per_seq_res[s][(cond, q, B, rho, a)] for s in seqs]
        pr = [P.prf(r["acc"][t]["v"]) for r in rows]
        pv, pu = _pooled(rows, t, "v"), _pooled(rows, t, "u")
        n_ep = sum(r["n_epochs"] for r in rows)
        cpu_all = [x for r in rows for x in r["cpu_all"]]; cpu_sent = [x for r in rows for x in r["cpu_sent"]]
        out.append(dict(
            cond=cond, q=q, B=B, rho=rho, alpha=a, tau_decl=t,
            tuning="alpha, tau_decl chosen on the same EXPL sequences (exploratory)",
            f1_view_macro=f, p_view_macro=float(np.mean([x[0] for x in pr])),
            r_view_macro=float(np.mean([x[1] for x in pr])),
            f1_view_micro=P.f1(pv), p_view_micro=P.prf(pv)[0], r_view_micro=P.prf(pv)[1],
            fp_per_view_epoch=pv["fp"] / (2 * n_ep),
            f1_view_per_seq=[P.f1(r["acc"][t]["v"]) for r in rows],
            f1_view_class_agnostic_macro=float(np.mean([P.f1(r["acc"][t]["va"]) for r in rows])),
            f1_unique_macro=float(np.mean([P.f1(r["acc"][t]["u"]) for r in rows])), f1_unique_micro=P.f1(pu),
            f1_unique_per_seq=[P.f1(r["acc"][t]["u"]) for r in rows],
            **{k: float(np.mean([r[k] for r in rows])) for k in
               ("ul", "dl", "man", "crops", "false_crops", "false_bits", "frame_frac", "drops", "coord_ok")},
            cpu_all_median=float(np.median(cpu_all)), cpu_all_p90=float(np.percentile(cpu_all, 90)),
            cpu_sent_median_supplementary=float(np.median(cpu_sent)),
            airtime_over={str(rt): float(np.mean([r["airtime_over"][rt] for r in rows])) for rt in RATES_MBPS},
            t_coord_ms={str(rt): float(np.mean([r["t_coord_ms"][rt] for r in rows])) for rt in RATES_MBPS}))
    return out


def _cmp(a, b):
    """a vs b at the same budget: macro / micro differences, per-seq differences, precision and FP changes."""
    d = [x - y for x, y in zip(a["f1_view_per_seq"], b["f1_view_per_seq"])]
    return dict(diff_macro=a["f1_view_macro"] - b["f1_view_macro"], diff_micro=a["f1_view_micro"] - b["f1_view_micro"],
                dP_micro=a["p_view_micro"] - b["p_view_micro"], dR_micro=a["r_view_micro"] - b["r_view_micro"],
                dFP_per_view_epoch=a["fp_per_view_epoch"] - b["fp_per_view_epoch"],
                per_seq_diff=d, seqs_positive=int(sum(x > 0 for x in d)), n_seq=len(d),
                ul_a=a["ul"], ul_b=b["ul"])


def criteria(table, scr, levels, qs):
    def get(cond, q, B, rho=0.0):
        return next(r for r in table if r["cond"] == cond and r["q"] == q and r["B"] == B and r["rho"] == rho)
    c3, c4d = {}, {}
    for q in qs:
        rows = []
        for B in levels:
            t1, t0, t0p = get("T1", q, B), get("T0", None, B), get("T0+", None, B)
            base = t0 if t0["f1_view_macro"] >= t0p["f1_view_macro"] else t0p
            c = _cmp(t1, base)
            c.update(B=B, base=base["cond"],
                     pass_B=c["diff_macro"] > 0 and c["diff_micro"] > 0 and c["dP_micro"] >= -C3_MAX_PRECISION_DROP)
            rows.append(c)
            for cond in P.COORD_CONDS:
                for rho in ((0.0,) + RHO_SENS if cond == "T2" else (0.0,)):
                    x, y = get(cond, q, B, rho), get("T1", q, B)
                    c4d.setdefault(f"{cond}_q{q}_rho{rho:g}", []).append(dict(
                        B=B, diff_unique_macro=x["f1_unique_macro"] - y["f1_unique_macro"],
                        diff_unique_micro=x["f1_unique_micro"] - y["f1_unique_micro"],
                        per_seq_diff=[a - b for a, b in zip(x["f1_unique_per_seq"], y["f1_unique_per_seq"])],
                        dFP_per_view_epoch=x["fp_per_view_epoch"] - y["fp_per_view_epoch"], ul=x["ul"], dl=x["dl"]))
        c3[f"q{q}"] = dict(per_B=rows, pass_any_B=any(r["pass_B"] for r in rows))
    main = [v for k, v in c4d.items() if k.endswith("rho0")]
    return dict(
        C1_pass=scr["C1"]["miss_rate"] >= 0.15, C2_pass=scr["C2"]["cand_recall"] >= 0.5,
        C3={**c3, "pass": any(v["pass_any_B"] for v in c3.values()),
            "rule": f"T1 - max(T0, T0+) > 0 in macro AND micro F1, and precision drop <= {C3_MAX_PRECISION_DROP}"},
        C4={"proposal_redundancy_frac": scr["C4_proposal_redundancy"]["frac"], "unique_f1_diffs": c4d,
            "pass": scr["C4_proposal_redundancy"]["frac"] >= 0.2
                    and any(x["diff_unique_macro"] > 0 and x["diff_unique_micro"] > 0 for v in main for x in v),
            "note": "T2/T2v are GT-assisted suppression heuristics, not upper bounds: a non-positive difference "
                    "does not show that cooperative headroom is absent"},
        C5_pass=all(v["median_ms"] <= 1000 * EPOCH_S for k, v in scr["C5"].items() if k.startswith("q")),
        C5_note="primary = all candidates encoded; p90 reported in screening.C5")


def evaluate(pairs_by_seq, qs=(35, 50), procs=1):
    """pairs_by_seq: {seq: [(rec1, rec2, gt1, gt2), ...]}."""
    allpairs = [p for v in pairs_by_seq.values() for p in v]
    lv = P.budget_levels(allpairs)
    cells = grid(lv["levels"], lv["frame_level"], qs)
    jobs = [(pairs_by_seq[s], cells) for s in sorted(pairs_by_seq)]
    if procs > 1:
        with Pool(procs) as pool:
            res = pool.map(eval_seq, jobs)
    else:
        res = [eval_seq(j) for j in jobs]
    per = dict(zip(sorted(pairs_by_seq), res))
    table = summarise(per, cells)
    scr = screening(pairs_by_seq, qs)
    return dict(budget=lv, table=table, screening=scr, criteria=criteria(table, scr, lv["levels"], qs),
                n_seq=len(pairs_by_seq), n_epochs=len(allpairs), alphas=ALPHAS, taus=TAUS, rho_sens=RHO_SENS)


def load_pairs(precomp_dir, zip_path, allowed=("test",)):
    import mdmt_adapter as M
    ds = M.MDMT(zip_path, allowed=allowed)              # labels only: no rights note needed, no pixels served
    out, prov = {}, {}
    for f in sorted(Path(precomp_dir).glob("seq_*.pkl")):
        R = pickle.loads(f.read_bytes())
        s = R["seq"]
        g1, g2 = ds.labels(s, 1), ds.labels(s, 2)
        out[s] = [(R["recs"][1][i], R["recs"][2][i], g1.get(fr, {}), g2.get(fr, {}))
                  for i, fr in enumerate(R["frames"])]
        prov[s] = R["provenance"]
    return out, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("precomp"); ap.add_argument("out")
    ap.add_argument("--zip", required=True); ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    pairs, prov = load_pairs(a.precomp, a.zip, tuple(cfg["allowed_splits"]))
    res = evaluate(pairs, tuple(cfg["avif_q"]), a.procs)
    res.update(config=cfg, provenance=prov, status="EXPLORATORY (EXPL split; not a confirmation)")
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["criteria"], indent=1, default=str))


if __name__ == "__main__":
    main()
