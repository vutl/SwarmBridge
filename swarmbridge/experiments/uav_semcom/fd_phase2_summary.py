"""Paper A full-data Phase 2 summary: implements section 7 of paperA-fd-phase2-prereg.md exactly (written and committed
BEFORE any test run).
Per pool (TD primary; F, V, TD_all reported alike):
  * return: paired difference per environment seed, 95% t-CI over seeds, for
      H1 cal - H*, H2 cal - rh, H2b cal - rhnb, H3 cal - nominal, H3b cal - nominal(matched design);
  * Qmin-compliant recall (F / TD / TD_all / V): (a) per-episode recall, paired t-CI over seeds; (b) video-cluster
    analysis: Delta = sum_e sum_s (TP^A_es - TP^B_es) / sum_e sum_s GT_es, percentile CI from B = 2000 bootstrap draws
    (numpy default_rng(0)) resampling the pool's sequences with replacement (terms of sequence s weighted by its
    multiplicity), for H1, H3, H3b;
  * descriptive levels per arm; mechanism log per arm (planned, gated, switched, aborts, robust checks / rejects,
    band and bias quantiles, decision time p99, slots > 50 ms);
  * wording verdicts of section 7 (H3b, H1, H2 / H2b) for TD.
Usage: python fd_phase2_summary.py [--name test] (reads phase2/<name>_<pool>.json + .deliv.json.gz)"""
import argparse
import gzip
import json

import numpy as np
from scipy import stats

import fd_common as FD
from ax2_summary import ci

ARMS = {"H": "", "cal": "+mpc", "nom": "+mpcnc", "ng": "+mpcng", "rh": "+mpcrh", "rhnb": "+mpcrhnb"}
HYP = [("H1", "cal", "H"), ("H2", "cal", "rh"), ("H2b", "cal", "rhnb"), ("H3", "cal", "nom"), ("H3b", "cal", "ng")]
PRIMARY = ["H1", "H3b"]                           # confirmatory family, Holm (review of 2cd2ec77)
RECALL_HYP = [("H1", "cal", "H"), ("H3", "cal", "nom"), ("H3b", "cal", "ng")]
POOLS = ["TD", "F", "V", "TD_all"]
B = 2000


def f(v):
    m, lo, hi = v
    return f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]{'*' if lo > 0 or hi < 0 else ' '}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test")
    ap.add_argument("--pools", default=",".join(POOLS))
    a = ap.parse_args()
    t, pools, rec, man_idx = FD.load()
    man = json.loads(FD.MANIFEST.read_text())["images"]
    row_of_man = {int(mi): r for r, mi in enumerate(man_idx)}
    seq_of_man = {k: e["seq"] for k, e in enumerate(man)}
    out, lines = {}, []
    for pool in a.pools.split(","):
        p = FD.OUT / "phase2" / f"{a.name}_{pool}.json"
        if not p.exists():
            lines.append(f"{pool}: no results")
            continue
        te = json.loads(p.read_text())
        dl = json.load(gzip.open(FD.OUT / "phase2" / f"{a.name}_{pool}.deliv.json.gz", "rt"))
        by = {r["combo"].split("|h=")[1][len(te["H"]):]: r for r in te["results"]}
        assert sorted(by) == sorted(ARMS.values()), sorted(by)
        seeds = by[""]["seeds"]
        assert all(by[s]["seeds"] == seeds for s in by) and (len(seeds) == 100 or a.name != "test")
        ret = {k: np.array([x["return"] for x in by[s]["rows"]]) for k, s in ARMS.items()}
        o = {"H": te["H"], "seeds": [seeds[0], seeds[-1]], "return": {}, "levels": {}, "mechanism": {}}
        lines.append(f"== {pool} (seeds {seeds[0]}..{seeds[-1]}, H* = {te['H']})")
        # integrity (review of 2cd2ec77): every episode accounts for K x 52 = 260 image tasks; arms with the same seed
        # have identical image occurrences; every delivered index maps to a table row of this pool
        n_task = 5 * 52
        eps = {k: dl[f"{pool}|{by[s]['combo']}"] for k, s in ARMS.items()}
        assert all(len(eps[k]) == len(seeds) for k in ARMS)
        bad_n = sum(len(ep) != n_task for k in ARMS for ep in eps[k])
        bad_set = sum(len({tuple(sorted(x[0] for x in eps[k][e])) for k in ARMS}) != 1 for e in range(len(seeds)))
        pool_rows = set(int(i) for i in pools[pool])
        bad_idx = sum(row_of_man.get(x[0]) not in pool_rows for k in ARMS for ep in eps[k] for x in ep)
        o["integrity"] = dict(episodes_not_260=bad_n, seeds_with_different_images=bad_set, deliveries_outside_pool=bad_idx)
        lines.append(f"   integrity: episodes != 260 tasks {bad_n}, seeds with differing image sets {bad_set}, "
                     f"deliveries outside the pool {bad_idx}")
        print(lines[-1], flush=True)
        assert bad_n == bad_set == bad_idx == 0
        for h, x, y in HYP:
            d = ret[x] - ret[y]
            o["return"][h] = [float(v) for v in ci(d)]
            o.setdefault("p", {})[h] = float(stats.ttest_1samp(d, 0.0).pvalue)
            lines.append(f"   {h:4s} {x}-{y:5s} return {f(o['return'][h])}  p {o['p'][h]:.4f}"
                         + ("  [confirmatory, Holm]" if h in PRIMARY else "  [supporting]"))
        # Holm over the two confirmatory contrasts (H1, H3b), two-sided, family-wise alpha 0.05
        ps = sorted(PRIMARY, key=lambda h: o["p"][h])
        holm, stop = {}, False
        for i, h in enumerate(ps):
            adj = min(1.0, max(o["p"][g] * (len(ps) - j) for j, g in enumerate(ps[:i + 1])))
            rej = (not stop) and o["p"][h] <= 0.05 / (len(ps) - i)
            stop = stop or not rej
            holm[h] = dict(p=o["p"][h], p_holm=adj, rejected=bool(rej), direction="+" if o["return"][h][0] > 0 else "-")
        o["holm"] = holm
        lines.append("   Holm (H1, H3b): " + ", ".join(f"{h} p_holm {holm[h]['p_holm']:.4f} "
                                                      f"{'REJECT' if holm[h]['rejected'] else 'not rejected'} "
                                                      f"({holm[h]['direction']})" for h in PRIMARY))
        has_recall = pool in ("TD", "F", "V", "TD_all")
        if has_recall:
            # per (arm, episode, sequence): TP and GT sums
            def tpgt(arm):
                eps = dl[f"{pool}|{by[ARMS[arm]]['combo']}"]
                res = []
                for ep in eps:
                    d = {}
                    for mi, mode, ok, ot in ep:
                        r = row_of_man[mi]
                        s = seq_of_man[mi]
                        tp, gt = d.get(s, (0.0, 0.0))
                        d[s] = (tp + (rec["tp"][r, mode] if (mode >= 0 and ok) else 0.0), gt + rec["ngt"][r])
                    res.append(d)
                return res
            TG = {k: tpgt(k) for k in ARMS}
            per_ep = {k: np.array([100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for d in TG[k]])
                      for k in TG}
            seqs = sorted({s for d in TG["H"] for s in d})
            idx = {s: i for i, s in enumerate(seqs)}
            rng = np.random.default_rng(0)
            draws = [np.bincount(rng.integers(len(seqs), size=len(seqs)), minlength=len(seqs)) for _ in range(B)]
            o["recall"] = {}
            for h, x, y in RECALL_HYP:
                o_seed = [float(v) for v in ci(per_ep[x] - per_ep[y])]
                dtp, gt = np.zeros(len(seqs)), np.zeros(len(seqs))
                for e in range(len(seeds)):
                    for s, (tpx, g) in TG[x][e].items():
                        assert abs(g - TG[y][e][s][1]) < 1e-9          # same images -> same GT per (episode, sequence)
                        dtp[idx[s]] += tpx - TG[y][e][s][0]
                        gt[idx[s]] += g
                est = 100 * dtp.sum() / gt.sum()
                bs = np.array([100 * (w * dtp).sum() / (w * gt).sum() for w in draws])
                o["recall"][h] = dict(seed_ci=o_seed, cluster=[float(est), float(np.percentile(bs, 2.5)),
                                                               float(np.percentile(bs, 97.5))], n_sequences=len(seqs))
                lines.append(f"   {h:4s} {x}-{y:5s} recall seed {f(o_seed)}  video-cluster {f(o['recall'][h]['cluster'])}"
                             f" ({len(seqs)} sequences)")
        for k, s in ARMS.items():
            R = by[s]["rows"]
            g = lambda key: [x[key] for x in R if x.get(key) is not None]
            L = dict(ret=float(np.mean(g("return"))), ret_per_image=float(np.mean(g("return")) / 260),
                     **{m: float(np.mean(g(m))) for m in ("ok_rate", "drop_rate", "late_rate", "q_viol_rate", "mean_psnr",
                                                          "energy_kj", "goodput_kbps", "ms_int_max_p99", "int_over50")
                        if g(m)})
            if has_recall:
                L["recall"] = float(per_ep[k].mean()) if k in per_ep else None
            o["levels"][k] = L
            if g("planned"):
                q = lambda key, i: float(np.mean([x[key][i] for x in R if x.get(key)])) if g(key) else None
                o["mechanism"][k] = dict(planned=float(np.mean(g("planned"))), gated=float(np.mean(g("gated"))),
                                         switched=float(np.mean(g("switched"))), aborts=float(np.mean(g("aborts"))),
                                         robust_checks=float(np.mean(g("robust_checks"))),
                                         robust_rejects=float(np.mean(g("robust_rejects"))),
                                         band_lo_q=[q("band_lo_q", i) for i in range(3)],
                                         band_hi_q=[q("band_hi_q", i) for i in range(3)],
                                         bias_q=[q("bias_q", i) for i in range(3)])
            lines.append(f"     {k:5s} ret {L['ret']:8.2f} ok {L.get('ok_rate', float('nan')):.3f} drop "
                         f"{L.get('drop_rate', float('nan')):.3f}" + (f" recall {L['recall']:.2f}" if L.get('recall') else "")
                         + (f" | switched {o['mechanism'][k]['switched']:.1f} rejects/checks "
                            f"{o['mechanism'][k]['robust_rejects']:.0f}/{o['mechanism'][k]['robust_checks']:.0f}"
                            if k in o["mechanism"] else "") + f" | p99 {L['ms_int_max_p99']:.1f} ms")
        out[pool] = o
    if "TD" in out:
        # confirmatory family {H1, H3b}: Holm-rejected + sign (review of 2cd2ec77); supporting H2 / H2b / H3: 95% CIs,
        # effect sizes only, never an additional confirmatory claim; a non-rejection is "no detected difference",
        # never equivalence
        r, hm = out["TD"]["return"], out["TD"]["holm"]
        cpos = lambda h: hm[h]["rejected"] and hm[h]["direction"] == "+"
        cneg = lambda h: hm[h]["rejected"] and hm[h]["direction"] == "-"
        pos = lambda h: r[h][1] > 0
        v = {"H3b": "CONFIRMATORY: the ACK-calibrated rate model improved return within a common control design" if cpos("H3b")
             else ("CONFIRMATORY: calibration reduced return relative to the nominal model in the same design" if cneg("H3b")
                   else "no detected difference between calibrated and nominal rate models within the common design "
                        "(not evidence of equivalence)"),
             "H1": "CONFIRMATORY: the calibrated selective-MPC stack improved the tuned incumbent" if cpos("H1")
             else ("CONFIRMATORY: the stack reduced return relative to the incumbent" if cneg("H1")
                   else "no detected difference from the tuned incumbent"),
             "H2/H2b (supporting)": "better than both implemented RobustMPC-style variants" if (pos("H2") and pos("H2b"))
             else ("better only than the band-limited RobustMPC-style variant (near-inactive baseline)" if pos("H2")
                   else "not shown better than the implemented RobustMPC-style variants"),
             "H3 (supporting)": "full stack vs nominal planner: " + ("CI>0" if pos("H3") else "CI incl./below 0")}
        v["replication_H3b"] = {p: ("CI>0" if out[p]["return"]["H3b"][1] > 0 else "CI<0" if out[p]["return"]["H3b"][2] < 0
                                    else "CI incl. 0") for p in out if p != "TD"}
        out["verdict_TD"] = v
        lines.append(f"VERDICT (TD): {v}")
    print("\n".join(lines))
    json.dump(out, open(FD.OUT / "phase2" / f"summary_{a.name}.json", "w"), indent=1)
    (FD.OUT / "phase2" / f"summary_{a.name}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
