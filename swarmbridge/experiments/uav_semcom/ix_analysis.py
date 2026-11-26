"""paperA_ix registered analysis (pre-registration §6–§7), written and tested BEFORE any test block exists.
Per block (each block is its own evidence line; Holm within each family of each block):
  ref  : H1 cal - H*, H3b cal - guarded (Holm over 2, confirmatory 'C'); H2 cal - rh, H2b cal - rhnb, H3 cal - nominal
         (supporting 'S')
  abl  : A1 gate - nominal, A2 band - nominal, A4 guarded H4 - H1, A4b guarded H4 - H0 (Holm over 4, 'C');
         A3, A5, A6, A7, A8, A9 (supporting); NEW family 'N': N1 nominal - H*, N2 guarded - H*, N3 calibrated -
         guarded (Holm over 3)
  grid : G1_c guarded - H*, G2_c calibrated - guarded per setting (Holm over 12, 'C'); G3, G4 supporting
  rl   : R1_X X+guarded - X, R2_X X+calibrated - X+guarded (Holm over 6, 'C'); R3, R4 supporting; per-checkpoint
         contrasts; R5 (vs the X2R-ref H* rows, same seeds)
Return: paired difference per environment seed (learned arms averaged over their 3 checkpoints first), 95% t-CI,
two-sided p (fd_next_summary.contrasts). Recall (UAVDT blocks only; X1 = 'unavailable'): for every C and N contrast and
every variant (veh primary, cls_vancar, cls_car), seed-level CI + sequence-cluster bootstrap (B 2000, rng 0) of the
Qmin-compliant recall (fd_next_summary.Pool.recall); original-image recall per variant reported first.
Principal finding (X1 principal; X2-retrained and X2-frozen under their own labels): verdict(N2, A2):
  'not supported'        if either is Holm-rejected with a NEGATIVE sign (takes precedence),
  'supported'            if both are Holm-rejected with a positive sign,
  'partially supported'  if exactly one is Holm-rejected positive (and none negative),
  'not supported'        otherwise.
Mode utilisation per arm: selection counts / shares of the 14 modes and outcome classes (compliant, below floor, late
or dropped, no mode decision).
Usage (WSL): python ix_analysis.py [--blocks x1_ref,...]   -> runs/uav_semcom/paperA_ix/analysis/"""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np

import a5_common as C
import ix_common as IX
from fd_next_summary import Pool, ci, contrasts, holm

KIND_FAMS = {"ref": ("C",), "abl": ("C", "N"), "grid": ("C",), "rl": ("C",)}


def specs(kind, H, cfgs=None):
    """(id, cfg, armA, armB, family) with the fd_v1 arm strings; H = H* name (ref / abl / rl) or {cfg: name} (grid)."""
    if kind in ("ref", "abl"):
        h = f"h={H}"
    if kind == "ref":
        a = {k: h + s for k, s in dict(H="", cal="+mpc", nom="+mpcnc", ng="+mpcng", rh="+mpcrh", rhnb="+mpcrhnb").items()}
        return [("H1", "N5x2", a["cal"], a["H"], "C"), ("H3b", "N5x2", a["cal"], a["ng"], "C"),
                ("H2", "N5x2", a["cal"], a["rh"], "S"), ("H2b", "N5x2", a["cal"], a["rhnb"], "S"),
                ("H3", "N5x2", a["cal"], a["nom"], "S")]
    if kind == "abl":
        a = {k: h + s for k, s in dict(H="", nc="+mpcnc", gate="+mpcngg", band="+mpcngb", ng="+mpcng", h1="+mpcngh1",
                                       h0="+mpcngh0", cal="+mpc").items()}
        return [("A1", "N5x2", a["gate"], a["nc"], "C"), ("A2", "N5x2", a["band"], a["nc"], "C"),
                ("A4", "N5x2", a["ng"], a["h1"], "C"), ("A4b", "N5x2", a["ng"], a["h0"], "C"),
                ("A3", "N5x2", a["ng"], a["nc"], "S"), ("A5", "N5x2", a["ng"], a["gate"], "S"),
                ("A6", "N5x2", a["ng"], a["band"], "S"), ("A7", "N5x2", a["cal"], a["ng"], "S"),
                ("A8", "N5x2", a["ng"], a["H"], "S"), ("A9", "N5x2", a["h1"], a["h0"], "S"),
                ("N1", "N5x2", a["nc"], a["H"], "N"), ("N2", "N5x2", a["ng"], a["H"], "N"),
                ("N3", "N5x2", a["cal"], a["ng"], "N")]
    if kind == "grid":
        out = []
        for cfg in cfgs:
            h = f"h={H[cfg]}"
            out += [(f"G1_{cfg}", cfg, h + "+mpcng", h, "C"), (f"G2_{cfg}", cfg, h + "+mpc", h + "+mpcng", "C"),
                    (f"G3_{cfg}", cfg, h + "+mpc", h, "S"), (f"G4_{cfg}", cfg, h + "+mpcng", h + "+mpcnc", "S")]
        return out
    out = []
    for x in H:                                   # rl: H = incumbent keys
        out += [(f"R1_{x}", "N5x2", x + "+mpcng", x, "C"), (f"R2_{x}", "N5x2", x + "+mpc", x + "+mpcng", "C"),
                (f"R3_{x}", "N5x2", x + "+mpc", x, "S"), (f"R4_{x}", "N5x2", x + "+mpcng", x + "+mpcnc", "S")]
    return out


def verdict(c):
    """Principal finding from the N2 and A2 contrasts (each with a 'holm' entry and a return CI)."""
    sig = {k: c[k]["holm"]["rejected"] for k in ("N2", "A2")}
    sign = {k: c[k]["ci"][0] > 0 for k in ("N2", "A2")}
    if any(sig[k] and not sign[k] for k in sig):
        return "not supported"
    pos = sum(sig[k] and sign[k] for k in sig)
    return {2: "supported", 1: "partially supported", 0: "not supported"}[pos]


def analyse(P, kind, H, rec=None, cfgs=None, P_ref=None, H_ref=None):
    """P: fd_next_summary.Pool of one block. rec: {variant: recall table} (UAVDT) or None (X1)."""
    integ = P.integrity()
    assert all(v == 0 for v in integ.values()), integ
    sp = specs(kind, H, cfgs)
    o = dict(integrity=integ, seeds=[P.seeds[0], P.seeds[-1]], contrasts=contrasts(P, sp, []))
    for fam in KIND_FAMS[kind]:
        hm = holm({k: v["p"] for k, v in o["contrasts"].items() if v["family"] == fam})
        for k, v in hm.items():
            o["contrasts"][k]["holm"] = v
    if rec is not None:
        o["recall_original"] = {}
        for v, r in rec.items():
            rows = np.array(sorted(P.pool_rows))
            o["recall_original"][v] = float(100 * np.nansum(r["tp_orig"][rows]) / np.nansum(r["ngt"][rows]))
        for hid, cfg, a, b, fam in sp:
            if fam in ("C", "N"):
                o["contrasts"][hid]["recall"] = {}
                for v, r in rec.items():
                    P.rec = r
                    o["contrasts"][hid]["recall"][v] = P.recall(cfg, a, b)
    else:
        o["recall"] = "unavailable (no public labels)"
    if kind == "abl":
        o["principal"] = verdict(o["contrasts"])
    if kind == "rl":
        o["per_checkpoint"] = {f"{h}_s{ck}": ci(P.ret(c, x, ck) - P.ret(c, y, ck))
                               for h, c, x, y, _ in sp if h[:2] in ("R1", "R2") for ck in P.arms[(c, x)]}
        if P_ref is not None and P_ref.seeds == P.seeds:
            o["R5"] = {x: dict(alone=ci(P.ret("N5x2", x) - P_ref.ret("N5x2", f"h={H_ref}")),
                               matched=ci(P.ret("N5x2", x + "+mpcng") - P_ref.ret("N5x2", f"h={H_ref}+mpcng")))
                       for x in H}
    o["levels"] = {f"{cfg}|{arm}": P.describe(cfg, arm) for (cfg, arm) in sorted(P.arms)}
    return o


def utilisation(deliv):
    """deliv: {key: [episodes of [man idx, mode, ok, ontime]]} -> {key: per-mode counts / shares / outcome classes}."""
    out = {}
    for key, eps in deliv.items():
        cnt, nomode = np.zeros((len(C.MODES), 3), int), 0
        for ep in eps:
            for _, m, ok, ot in ep:
                if m < 0:
                    nomode += 1
                else:
                    cnt[m, 0 if ok else (1 if ot else 2)] += 1
        tot = int(cnt.sum())
        out[key] = dict(no_mode=nomode, finished_with_mode=tot,
                        modes={md: dict(selected=int(cnt[j].sum()), share=float(cnt[j].sum() / tot) if tot else 0.0,
                                        compliant=int(cnt[j, 0]), below_floor=int(cnt[j, 1]), late_drop=int(cnt[j, 2]))
                               for j, md in enumerate(C.MODES)},
                        rare=[md for j, md in enumerate(C.MODES) if tot and cnt[j].sum() / tot <= 0.005])
    return out


def lines_of(block, o):
    L = [f"== {block}: seeds {o['seeds'][0]}..{o['seeds'][1]}, integrity {o['integrity']}"]
    if "recall_original" in o:
        L.append("   original-image recall (%): " + ", ".join(f"{k} {v:.2f}" for k, v in o["recall_original"].items()))
    for k, v in o["contrasts"].items():
        m, lo, hi = v["ci"]
        tag = (f"  Holm[{v['family']}] p {v['holm']['p_holm']:.4f} {'REJECT' if v['holm']['rejected'] else 'not rejected'}"
               if "holm" in v else "  supporting")
        L.append(f"   {k:10s} {v['a'][-20:]:>20s} - {v['b'][-20:]:20s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}] p {v['p']:.4f}{tag}")
        for var, r in v.get("recall", {}).items():
            L.append(f"   {'':10s} recall[{var}] seed {r['seed'][0]:+.2f} [{r['seed'][1]:+.2f},{r['seed'][2]:+.2f}] "
                     f"cluster {r['cluster'][0]:+.2f} [{r['cluster'][1]:+.2f},{r['cluster'][2]:+.2f}] ({r['n_sequences']} seq)")
    if "principal" in o:
        L.append(f"   PRINCIPAL FINDING (N2 & A2): {o['principal']}")
    return L


def main():
    import ix_eval as EV
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=",".join(EV.BLOCKS))
    a = ap.parse_args()
    IX.verify_lock()
    out_dir = IX.OUT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_cache, res, lines = {}, {}, []
    for block in a.blocks.split(","):
        stage, arm, kind, _ = EV.BLOCKS[block]
        pj, pd = EV.paths(block)
        if not EV.complete(block):
            lines.append(f"== {block}: incomplete / missing")
            continue
        if (stage, arm) not in ctx_cache:
            t, pools, rec, man_idx, st, man = IX.load_eval(stage, arm)
            ctx_cache[(stage, arm)] = (rec, {int(mi): r for r, mi in enumerate(man_idx)},
                                       {k: e["seq"] for k, e in enumerate(man)}, pools)
        rec, row_of_man, seq_of_man, pools = ctx_cache[(stage, arm)]
        P = Pool(pj, pd, EV.POOL[stage], None, row_of_man, seq_of_man, pools)
        cfgs = [c for c in EV.CFG if c != "N5x2"]
        H = (EV.h_star(arm, "N5x2") if kind in ("ref", "abl") else {c: EV.h_star(arm, c) for c in cfgs}
             if kind == "grid" else EV.LEARNED)
        P_ref = None
        if kind == "rl":
            rp, rd = EV.paths("x2r_ref")
            P_ref = Pool(rp, rd, EV.POOL[stage], None, row_of_man, seq_of_man, pools) if EV.complete("x2r_ref") else None
        o = analyse(P, kind, H, rec, cfgs, P_ref, EV.h_star(arm, "N5x2"))
        o["mode_utilisation"] = utilisation(json.load(gzip.open(pd, "rt")))
        res[block] = o
        lines += lines_of(block, o)
        json.dump(o, open(out_dir / f"{block}.json", "w"), indent=1)
    summary = {b: res[b]["principal"] for b in res if "principal" in res[b]}
    lines.append("PRINCIPAL FINDING by block (X1 principal; X2-retrained = method transfer; X2-frozen = artifact "
                 f"transfer): {summary}")
    (out_dir / "summary.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
