"""H1-CROP pilot 0B-TUNE evaluation (CPU only): conditions x budgets x alpha x tau_decl on precomputed records,
then the screening criteria C1-C5 (protocol sec 6). EXPLORATORY: alpha and tau_decl are chosen on the same pilot
sequences they are reported on (declared in the output), every condition gets the same tuning.

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
EPOCH_S = 1.0


def grid(levels, frame_level, qs):
    out = []
    for B in levels:
        out += [("T0", None, B), ("T0+", None, B)]
        out += [(c, q, B) for c in P.CROP_CONDS for q in qs]
        out.append(("F", None, B))
    out.append(("F", None, frame_level))
    out += [(c, None, frame_level) for c in ("T0", "T0+")]
    return out


def eval_seq(args):
    """all grid cells for one sequence -> {(cond,q,B,alpha): {tau: counts...}, extras}."""
    pairs, cells = args
    res = {}
    # per-epoch caches (independent of condition / budget / alpha / q)
    t2c = [P.t2_drop(*p) for p in pairs]
    ff = [{(v, kind): P.false_flags(r, g, kind) for v, (r, g) in enumerate(((p[0], p[2]), (p[1], p[3])))
           for kind in ("P1", "P2")} for p in pairs]
    for cond, q, B in cells:
        qq = q if q is not None else P.FRAME_Q
        for a in ALPHAS:
            acc = {t: dict(v={}, va={}, u={}) for t in TAUS}
            ul, dl, ncrop, nfalse, bfalse, frames, air, cpu = [], [], [], [], [], 0, {r: 0 for r in RATES_MBPS}, []
            for ei, (r1, r2, g1, g2) in enumerate(pairs):
                e = P.run_epoch(r1, r2, g1, g2, cond, qq, B, a, t2=t2c[ei])
                ul.append(e["ul"]); dl.append(e["dl"])
                for r in RATES_MBPS:
                    air[r] += e["ul"] / (r * 1e6) > EPOCH_S
                nc = nf = bf = 0
                for v, (rec, s) in enumerate(((r1, e["sent"][0]), (r2, e["sent"][1]))):
                    frames += s["frame"]
                    cpu.append(P.cpu_ms(rec, qq, s))
                    for kind, k, bits in s["items"]:
                        if kind in ("P1", "P2"):
                            nc += 1
                            if ff[ei][(v, kind)][k]:
                                nf += 1; bf += bits
                ncrop.append(nc); nfalse.append(nf); bfalse.append(bf)
                for t in TAUS:
                    v1 = P.view_counts(e["decl"][0], g1, t); v2 = P.view_counts(e["decl"][1], g2, t)
                    P.add(acc[t]["v"], v1); P.add(acc[t]["v"], v2)
                    P.add(acc[t]["u"], P.unique_counts(v1, v2, g1, g2))
                    P.add(acc[t]["va"], P.view_counts(e["decl"][0], g1, t, False))
                    P.add(acc[t]["va"], P.view_counts(e["decl"][1], g2, t, False))
            res[(cond, q, B, a)] = dict(acc=acc, ul=float(np.mean(ul)), dl=float(np.mean(dl)),
                                        crops=float(np.mean(ncrop)), false_crops=float(np.mean(nfalse)),
                                        false_bits=float(np.mean(bfalse)), frame_frac=frames / (2 * len(pairs)),
                                        airtime_over={r: air[r] / len(pairs) for r in RATES_MBPS},
                                        cpu_sent_median=float(np.median(cpu)))
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
                    cpu[q].append(P.cpu_ms(rec, q))
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
        C4_redundancy=dict(c4, frac=c4["redundant"] / max(c4["matched"], 1), threshold=0.2),
        C5=dict({f"q{q}": dict(median_ms=float(np.median(cpu[q])), p90_ms=float(np.percentile(cpu[q], 90)))
                 for q in qs}, epoch_ms=1000 * EPOCH_S, note="all P1 candidates encoded; desktop CPU, 1 thread"))


def summarise(per_seq_res, cells):
    """choose (alpha, tau) per cell by mean per-seq per-view F1; report chosen numbers + per-seq values."""
    seqs = sorted(per_seq_res)
    out = []
    for cond, q, B in cells:
        best = None
        for a in ALPHAS:
            for t in TAUS:
                f = float(np.mean([P.f1(per_seq_res[s][(cond, q, B, a)]["acc"][t]["v"]) for s in seqs]))
                if best is None or f > best[0] + 1e-12:
                    best = (f, a, t)
        f, a, t = best
        rows = [per_seq_res[s][(cond, q, B, a)] for s in seqs]
        pr = [P.prf(r["acc"][t]["v"]) for r in rows]
        out.append(dict(
            cond=cond, q=q, B=B, alpha=a, tau_decl=t, chosen_on="same pilot sequences (exploratory)",
            f1_view=f, p_view=float(np.mean([x[0] for x in pr])), r_view=float(np.mean([x[1] for x in pr])),
            f1_view_per_seq=[P.f1(r["acc"][t]["v"]) for r in rows],
            f1_view_class_agnostic=float(np.mean([P.f1(r["acc"][t]["va"]) for r in rows])),
            f1_unique=float(np.mean([P.f1(r["acc"][t]["u"]) for r in rows])),
            f1_unique_per_seq=[P.f1(r["acc"][t]["u"]) for r in rows],
            **{k: float(np.mean([r[k] for r in rows])) for k in
               ("ul", "dl", "crops", "false_crops", "false_bits", "frame_frac", "cpu_sent_median")},
            airtime_over={str(rt): float(np.mean([r["airtime_over"][rt] for r in rows])) for rt in RATES_MBPS}))
    return out


def criteria(table, scr, levels, qs):
    def get(cond, q, B):
        return next(r for r in table if r["cond"] == cond and r["q"] == q and r["B"] == B)
    c3, c4d = {}, {}
    for q in qs:
        diffs = []
        for B in levels:
            t1, t0, t0p = get("T1", q, B), get("T0", None, B), get("T0+", None, B)
            base = t0 if t0["f1_view"] >= t0p["f1_view"] else t0p
            diffs.append(dict(B=B, diff=t1["f1_view"] - base["f1_view"], base=base["cond"],
                              dP=t1["p_view"] - base["p_view"], dR=t1["r_view"] - base["r_view"],
                              seqs_positive=int(sum(a > b for a, b in zip(t1["f1_view_per_seq"],
                                                                          base["f1_view_per_seq"])))))
            c4d.setdefault(f"q{q}", []).append(dict(
                B=B, diff=get("T2", q, B)["f1_unique"] - get("T1", q, B)["f1_unique"]))
        c3[f"q{q}"] = dict(per_B=diffs, max_diff=max(x["diff"] for x in diffs))
    return dict(
        C1_pass=scr["C1"]["miss_rate"] >= 0.15, C2_pass=scr["C2"]["cand_recall"] >= 0.5,
        C3={**c3, "pass": any(v["max_diff"] > 0 for v in c3.values()),
            "note": "screening; read with dP / dR (a gain bought by a large precision drop does not count)"},
        C4={"redundancy_frac": scr["C4_redundancy"]["frac"], "unique_f1_T2_minus_T1": c4d,
            "pass": scr["C4_redundancy"]["frac"] >= 0.2 and any(x["diff"] > 0 for v in c4d.values() for x in v)},
        C5_pass=all(v["median_ms"] <= 1000 * EPOCH_S for k, v in scr["C5"].items() if k.startswith("q")))


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
                n_seq=len(pairs_by_seq), n_epochs=len(allpairs), alphas=ALPHAS, taus=TAUS)


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
