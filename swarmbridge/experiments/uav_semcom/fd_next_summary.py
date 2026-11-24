"""Pre-registered analysis for the phases after Phase 2 (committed BEFORE their test runs):
  ablation : paperA-fd-ablation-prereg.md (+ amendment: H0 arm, Holm over A1, A2, A4, A4b)
  grid     : paperA-fd-phase4-prereg.md (Holm over G1 / G2 x 6 configurations on TD)
  rl       : paperA-fd-phase3-prereg.md (Holm over R1 / R2 x 3 families on TD, retrained; zero-shot supporting)
Common: integrity asserts (52 K tasks per episode, identical image occurrences across all arms / checkpoints per seed,
deliveries inside the pool); return = paired difference per environment seed, learned arms averaged over their 3
training-seed checkpoints first, 95% t-CI + two-sided p; Holm within the confirmatory family; supporting contrasts
unadjusted; Qmin-compliant recall at seed level + video-cluster bootstrap (B 2000, rng 0) for the confirmatory
contrasts; per-training-seed contrasts for learned incumbents; latency with the SEQUENTIAL full-sum p99 (scheduler + all
mode decisions + tracker / calibrator update) as the primary measured timing, max as hypothetical parallel, scheduler /
mode / update components and unrounded over-budget shares.
Usage: python fd_next_summary.py ablation|grid|rl [--name test|zeroshot|retrained] [--pools TD,F,...]"""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np
from scipy import stats

import fd_common as FD
from fd_next import CFG

B = 2000
LAT = ("ms_full_sum_p99", "ms_int_sum_p99", "ms_int_max_p99", "ms_slot_p99", "ms_mode_p99", "ms_upd_p99")
OVER = ("full_sum_over50", "int_sum_over50", "int_over50", "full_max_over50")
MECH = ("planned", "gated", "cold_unguarded", "switched", "aborts", "robust_checks", "robust_rejects")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def f(v):
    m, lo, hi = v
    return f"{m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]{'*' if lo > 0 or hi < 0 else ' '}"


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, stop = {}, False
    for i, h in enumerate(order):
        adj = min(1.0, max(ps[g] * (len(order) - j) for j, g in enumerate(order[:i + 1])))
        rej = (not stop) and ps[h] <= 0.05 / (len(order) - i)
        stop = stop or not rej
        out[h] = dict(p=ps[h], p_holm=adj, rejected=bool(rej))
    return out


class Pool:
    """results of one file: arm key (cfg, incumbent+suffix) -> list of checkpoints' rows / deliveries."""

    def __init__(self, path, deliv, pool, rec, row_of_man, seq_of_man, pools):
        te = json.loads(path.read_text())
        dl = json.load(gzip.open(deliv, "rt"))
        self.arms = defaultdict(dict)
        for r in te["results"]:
            key = (r.get("cfg", "N5x2"), r["combo"].split("|", 1)[1])
            dkey = (f"{r['cfg']}|{r['pool']}|{r['combo']}|{r['ckpt']}" if "cfg" in r else f"{r['pool']}|{r['combo']}")
            self.arms[key][r.get("ckpt", -1)] = (r["rows"], dl[dkey], r["seeds"])
        self.pool, self.rec, self.row_of_man, self.seq_of_man = pool, rec, row_of_man, seq_of_man
        self.pool_rows = set(int(i) for i in pools[pool])
        seeds = {tuple(v[2]) for a in self.arms.values() for v in a.values()}
        assert len(seeds) == 1, "arms evaluated on different seeds"
        self.seeds = list(seeds.pop())

    def integrity(self):
        bad_n = bad_idx = 0
        by_cfg = defaultdict(list)
        for (cfg, arm), cks in self.arms.items():
            for ck, (rows, eps, _) in cks.items():
                n_task = 52 * CFG[cfg]["K"]
                bad_n += sum(len(ep) != n_task for ep in eps)
                bad_idx += sum(self.row_of_man.get(x[0]) not in self.pool_rows for ep in eps for x in ep)
                by_cfg[cfg].append(eps)
        bad_set = 0
        for cfg, lst in by_cfg.items():
            for e in range(len(self.seeds)):
                bad_set += len({tuple(sorted(x[0] for x in eps[e])) for eps in lst}) != 1
        return dict(episodes_wrong_task_count=bad_n, seeds_with_differing_images=bad_set, deliveries_outside_pool=bad_idx)

    def ret(self, cfg, arm, ck=None):
        assert (cfg, arm) in self.arms, f"missing arm {cfg} {arm} in {self.pool}"
        cks = self.arms[(cfg, arm)]
        if ck is not None:
            return np.array([x["return"] for x in cks[ck][0]])
        return np.mean([[x["return"] for x in v[0]] for v in cks.values()], 0)

    def tpgt(self, cfg, arm):
        """per checkpoint-averaged: list over episodes of {sequence: (TP, GT)} (TP averaged over checkpoints)."""
        cks = list(self.arms[(cfg, arm)].values())
        out = []
        for e in range(len(self.seeds)):
            d = defaultdict(lambda: [0.0, 0.0])
            for rows, eps, _ in cks:
                for mi, mode, ok, ot in eps[e]:
                    r, s = self.row_of_man[mi], self.seq_of_man[mi]
                    d[s][0] += (self.rec["tp"][r, mode] if (mode >= 0 and ok) else 0.0) / len(cks)
                    d[s][1] += self.rec["ngt"][r] / len(cks)
            out.append(dict(d))
        return out

    def recall(self, cfg, a, b):
        A, Bb = self.tpgt(cfg, a), self.tpgt(cfg, b)
        per = lambda T: np.array([100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for d in T])
        seed_ci = ci(per(A) - per(Bb))
        seqs = sorted({s for d in A for s in d})
        idx = {s: i for i, s in enumerate(seqs)}
        dtp, gt = np.zeros(len(seqs)), np.zeros(len(seqs))
        for e in range(len(self.seeds)):
            for s, (tp, g) in A[e].items():
                assert abs(g - Bb[e][s][1]) < 1e-6
                dtp[idx[s]] += tp - Bb[e][s][0]
                gt[idx[s]] += g
        rng = np.random.default_rng(0)
        bs = [100 * (w * dtp).sum() / (w * gt).sum()
              for w in (np.bincount(rng.integers(len(seqs), size=len(seqs)), minlength=len(seqs)) for _ in range(B))]
        return dict(seed=seed_ci, cluster=[float(100 * dtp.sum() / gt.sum()), float(np.percentile(bs, 2.5)),
                                           float(np.percentile(bs, 97.5))], n_sequences=len(seqs),
                    levels=[float(per(A).mean()), float(per(Bb).mean())])

    def describe(self, cfg, arm):
        rows = [x for v in self.arms[(cfg, arm)].values() for x in v[0]]
        g = lambda k: [x[k] for x in rows if x.get(k) is not None]
        d = dict(ret=float(np.mean(g("return"))), ret_per_image=float(np.mean(g("return")) / (52 * CFG[cfg]["K"])))
        for k in ("ok_rate", "drop_rate", "late_rate", "q_viol_rate", "mean_psnr", "energy_kj", "goodput_kbps") + LAT + MECH:
            if g(k):
                d[k] = float(np.mean(g(k)))
        for k in OVER:
            if g(k):
                v = np.array(g(k))
                d[k] = dict(mean=float(v.mean()), max=float(v.max()), episodes_with_any=int((v > 0).sum()))
        return d


def load_pool(phase, name, pool, ctx):
    p = FD.OUT / phase / f"{name}_{pool}.json"
    if not p.exists():
        return None
    return Pool(p, FD.OUT / phase / f"{name}_{pool}.deliv.json.gz", pool, *ctx)


def phase2_pool(pool, ctx):
    """Phase 2 H* rows (same seeds) for the RL R5 contrasts."""
    p = FD.OUT / "phase2" / f"test_{pool}.json"
    return Pool(p, FD.OUT / "phase2" / f"test_{pool}.deliv.json.gz", pool, *ctx) if p.exists() else None


def contrasts(P, spec, lines, recall_for=()):
    """spec: list of (id, cfg, armA, armB, family). Returns {id: dict}."""
    out = {}
    for hid, cfg, a, b, fam in spec:
        d = P.ret(cfg, a) - P.ret(cfg, b)
        out[hid] = dict(cfg=cfg, a=a, b=b, family=fam, ci=ci(d), p=float(stats.ttest_1samp(d, 0.0).pvalue))
        if hid in recall_for:
            out[hid]["recall"] = P.recall(cfg, a, b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("ablation", "grid", "rl"))
    ap.add_argument("--name", default="test")
    ap.add_argument("--pools", default="")
    a = ap.parse_args()
    t, pools, rec, man_idx = FD.load()
    man = json.loads(FD.MANIFEST.read_text())["images"]
    ctx = (rec, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools)
    phase_dir = {"ablation": "ablation", "grid": "phase4", "rl": "phase3"}[a.phase]
    default = {"ablation": "TD,V,F", "grid": "TD,F", "rl": "TD,F,V,TD_all"}[a.phase]
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    out, lines = {}, []
    for pool in (a.pools or default).split(","):
        P = load_pool(phase_dir, a.name, pool, ctx)
        if P is None:
            lines.append(f"{pool}: no results")
            continue
        integ = P.integrity()
        lines.append(f"== {a.phase}/{a.name} {pool} (seeds {P.seeds[0]}..{P.seeds[-1]}) integrity {integ}")
        print(lines[-1], flush=True)
        assert all(v == 0 for v in integ.values())
        o = {"integrity": integ}
        if a.phase == "ablation":
            h = f"h={H}"
            arm = {k: h + s for k, s in dict(H="", nc="+mpcnc", gate="+mpcngg", band="+mpcngb", ng="+mpcng",
                                             h1="+mpcngh1", h0="+mpcngh0", cal="+mpc").items()}
            spec = [("A1", "N5x2", arm["gate"], arm["nc"], "C"), ("A2", "N5x2", arm["band"], arm["nc"], "C"),
                    ("A4", "N5x2", arm["ng"], arm["h1"], "C"), ("A4b", "N5x2", arm["ng"], arm["h0"], "C"),
                    ("A3", "N5x2", arm["ng"], arm["nc"], "S"), ("A5", "N5x2", arm["ng"], arm["gate"], "S"),
                    ("A6", "N5x2", arm["ng"], arm["band"], "S"), ("A7", "N5x2", arm["cal"], arm["ng"], "S"),
                    ("A8", "N5x2", arm["ng"], arm["H"], "S"), ("A9", "N5x2", arm["h1"], arm["h0"], "S")]
            o["contrasts"] = contrasts(P, spec, lines, recall_for=("A1", "A2", "A4", "A4b"))
            o["levels"] = {k: P.describe("N5x2", v) for k, v in arm.items()}
        elif a.phase == "grid":
            tune = json.loads((FD.OUT / "phase4" / "tune.json").read_text())["H"]
            spec, lv = [], {}
            for cfg in [c for c in CFG if c != "N5x2"]:
                h = f"h={tune[cfg]}"
                spec += [(f"G1_{cfg}", cfg, h + "+mpcng", h, "C"), (f"G2_{cfg}", cfg, h + "+mpc", h + "+mpcng", "C"),
                         (f"G3_{cfg}", cfg, h + "+mpc", h, "S"), (f"G4_{cfg}", cfg, h + "+mpcng", h + "+mpcnc", "S")]
                lv[cfg] = {k: P.describe(cfg, h + s) for k, s in dict(H="", nc="+mpcnc", ng="+mpcng", cal="+mpc").items()}
            o["contrasts"] = contrasts(P, spec, lines, recall_for=tuple(x[0] for x in spec if x[4] == "C"))
            o["levels"] = lv
        else:
            fams = ("ppola", "qrla", "dsla") if a.name == "zeroshot" else ("ppofd", "qrfd", "dsfd")
            fam = "S" if a.name == "zeroshot" else "C"
            spec = []
            for x in fams:
                spec += [(f"R1_{x}", "N5x2", x + "+mpcng", x, fam), (f"R2_{x}", "N5x2", x + "+mpc", x + "+mpcng", fam),
                         (f"R3_{x}", "N5x2", x + "+mpc", x, "S"), (f"R4_{x}", "N5x2", x + "+mpcng", x + "+mpcnc", "S")]
            o["contrasts"] = contrasts(P, spec, lines, recall_for=tuple(x[0] for x in spec if x[0][:2] in ("R1", "R2")))
            # per-training-seed contrasts (environment-seed CIs conditional on each trained policy)
            o["per_seed"] = {f"{h}_s{ck}": ci(P.ret(c, x, ck) - P.ret(c, y, ck))
                             for h, c, x, y, _ in spec if h[:2] in ("R1", "R2") for ck in P.arms[(c, x)]}
            # R5: against the heuristic (Phase 2 H* rows, same seeds)
            P2 = phase2_pool(pool, ctx)
            if P2 is not None and P2.seeds == P.seeds:
                o["R5"] = {}
                for x in fams:
                    o["R5"][x] = dict(alone=ci(P.ret("N5x2", x) - P2.ret("N5x2", f"h={H}")),
                                      matched=ci(P.ret("N5x2", x + "+mpcng") - P2.ret("N5x2", f"h={H}+mpcng")))
            o["levels"] = {x + s: P.describe("N5x2", x + s) for x in fams for s in ("", "+mpcnc", "+mpcng", "+mpc")}
        conf = {k: v["p"] for k, v in o["contrasts"].items() if v["family"] == "C"}
        if pool == "TD" and conf:
            hm = holm(conf)
            for k, v in hm.items():
                o["contrasts"][k]["holm"] = v
        for k, v in o["contrasts"].items():
            tag = ""
            if "holm" in v:
                tag = (f"  Holm p {v['holm']['p_holm']:.4f} {'REJECT' if v['holm']['rejected'] else 'not rejected'}"
                       f" ({'+' if v['ci'][0] > 0 else '-'})")
            lines.append(f"   {k:16s} {v['a'][-22:]:>22s} - {v['b'][-22:]:22s} return {f(v['ci'])}  p {v['p']:.4f}"
                         f" [{'confirmatory' if v['family'] == 'C' else 'supporting'}]{tag}")
            if "recall" in v:
                r = v["recall"]
                lines.append(f"   {'':16s} recall seed {f(r['seed'])}  video-cluster {f(r['cluster'])} ({r['n_sequences']} seq)")
        for k, v in o.get("R5", {}).items():
            lines.append(f"   R5 {k}: alone - H* {f(v['alone'])}   matched - H*matched {f(v['matched'])}")
        for k, v in o.get("per_seed", {}).items():
            lines.append(f"   per training seed {k}: {f(v)}")
        lv = o["levels"]
        flat = lv.items() if a.phase != "grid" else [(f"{c}:{k}", d) for c, x in lv.items() for k, d in x.items()]
        for k, d in flat:
            lines.append(f"     {k:28s} ret {d['ret']:8.2f} (per img {d['ret_per_image']:+.3f}) ok {d.get('ok_rate', 0):.3f}"
                         f" | full-sum p99 {d.get('ms_full_sum_p99', float('nan')):6.1f} sum {d.get('ms_int_sum_p99', 0):6.1f}"
                         f" max {d.get('ms_int_max_p99', 0):5.1f} sched {d.get('ms_slot_p99', 0):5.1f} mode {d.get('ms_mode_p99', 0):5.1f}"
                         f" upd {d.get('ms_upd_p99', float('nan')):4.2f} ms | >50 full-sum "
                         f"{d.get('full_sum_over50', {}).get('mean', float('nan')):.4f} (eps {d.get('full_sum_over50', {}).get('episodes_with_any', -1)})"
                         + (f" | sw {d['switched']:.1f} gated {d['gated']:.1f} rej/chk {d['robust_rejects']:.0f}/{d['robust_checks']:.0f}"
                            if "switched" in d else ""))
        out[pool] = o
    print("\n".join(lines))
    tag = f"{a.phase}_{a.name}"
    json.dump(out, open(FD.OUT / phase_dir / f"summary_{tag}.json", "w"), indent=1)
    (FD.OUT / phase_dir / f"summary_{tag}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
