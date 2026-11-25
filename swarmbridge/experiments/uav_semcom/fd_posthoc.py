"""Paper A full-data: POST-HOC analyses on stored results only (no simulation, no training). Requested after the
consolidated report (bc4d2736); not pre-registered, reported as post-hoc / descriptive.
  verify : FAIL-FAST coverage + integrity over all 27,600 fd_v1 evaluation episodes (Phase 2 2,400 + ablation 2,400 +
           Phase 4 4,800 + RL zero-shot 3,600 + retrained 14,400): manifest / profile sha256, file parses, exact seed
           lists, exact arm and checkpoint sets, one row + one delivery list per seed, 52 K tasks per episode, finite
           returns, the return identity, identical image multisets across arms per seed, deliveries inside the pool,
           identical seeds / images between Phase 3 and the Phase 2 H* rows used for R5. Exits 1 at the FIRST failure.
  decomp : return decomposition  return = SCE - violations - drops - uncaptured - energy cost - time cost
           (SCE = sum of 0.5 quality + 0.5 deadline terms over delivered images; each penalty is -1 per image;
           energy cost 0.02 / kJ; time cost = return residual, compared with 0.3 x K x mean mission time), plus mode
           choice (bits, PSNR) and a recall decomposition (images compliant in both arms / only one arm; detection
           yield per compliant image), for nc vs ng and related pairs, paired over environment seeds.
  r5     : R5 completed: X+cal - H*+cal (return, per training seed, recall), and video-cluster bootstrap (B 2000,
           rng 0, as the registered recall analysis) for every R5 recall contrast (alone, matched, calibrated).
Usage: python fd_posthoc.py verify|decomp|r5"""
import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

import fd_common as FD
from fd_next import AB_ARMS, AB_TEST, CFG, GRID_ARMS, GRID_TEST, RL_ARMS, RL_TEST
from fd_next_summary import Pool, ci

OUT = FD.OUT / "posthoc"
P2_ARMS = ["", "+mpc", "+mpcnc", "+mpcng", "+mpcrh", "+mpcrhnb"]
P2_TEST = {"TD": range(20000, 20100), "V": range(21000, 21100), "F": range(22000, 22100), "TD_all": range(20000, 20100)}
LEARNED = ("ppola", "qrla", "dsla", "ppofd", "qrfd", "dsfd")
W_ENERGY, W_TIME = 0.02, 0.3
B = 2000


def ctx():
    t, pools, rec, man_idx = FD.load()          # FD.load() asserts the manifest / profile sha256 first
    man = json.loads(FD.MANIFEST.read_text())["images"]
    return t, (rec, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools)


def files():
    """(phase, file stem, pool, seeds, [(cfg, arm key, checkpoints)]) for every evaluation file."""
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    T4 = json.loads((FD.OUT / "phase4" / "tune.json").read_text())["H"]
    h = lambda c: H if c == "N5x2" else T4[c]
    out = [("phase2", f"test_{p}", p, s, [("N5x2", f"h={H}{a}", [-1]) for a in P2_ARMS]) for p, s in P2_TEST.items()]
    out += [("ablation", f"test_{p}", p, s, [("N5x2", f"h={H}{a}", [-1]) for a in AB_ARMS]) for p, s in AB_TEST.items()]
    out += [("phase4", f"test_{p}", p, s, [(c, f"h={h(c)}{a}", [-1]) for c in CFG if c != "N5x2" for a in GRID_ARMS])
            for p, s in GRID_TEST.items()]
    out += [("phase3", "zeroshot_TD", "TD", RL_TEST["TD"], [("N5x2", x + a, [0, 1, 2]) for x in LEARNED[:3] for a in RL_ARMS])]
    out += [("phase3", f"retrained_{p}", p, s, [("N5x2", x + a, [0, 1, 2]) for x in LEARNED[3:] for a in RL_ARMS])
            for p, s in RL_TEST.items()]
    return out


def load(phase, stem, pool, C):
    return Pool(FD.OUT / phase / f"{stem}.json", FD.OUT / phase / f"{stem}.deliv.json.gz", pool, *C)


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def decompose(row, K):
    """exact return components of one episode row (sign: contribution to return)."""
    n = 52 * K
    U = n * row["mean_util"]
    n_drop = row["drop_rate"] * n
    n_unc = (1 - row["captured"]) * n
    n_ok = row["ok_rate"] * n
    n_viol = n - n_drop - n_unc - n_ok
    e_cost = W_ENERGY * row["energy_kj"]
    t_cost = U - e_cost - row["return"]
    return dict(ret=row["return"], sce=U + n_viol + n_drop + n_unc, viol=-n_viol, drop=-n_drop, uncapt=-n_unc,
                energy=-e_cost, time=-t_cost, time_model=-W_TIME * K * row["mission_s"], n_ok=n_ok,
                n_qviol=row["q_viol_rate"] * n, n_late=row["late_rate"] * n, mission_s=row["mission_s"],
                psnr=row["mean_psnr"], delay=row["mean_delay"])


def verify():
    t, C = ctx()
    print("manifest / profile sha256 OK")
    total = 0
    pools = {}
    for phase, stem, pool, seeds, arms in files():
        p = FD.OUT / phase / f"{stem}.json"
        if not p.exists():
            fail(f"missing {phase}/{stem}.json")
        try:
            P = load(phase, stem, pool, C)
        except Exception as e:                               # parse error or the Pool asserts (one seed list)
            fail(f"{phase}/{stem}: cannot load ({type(e).__name__}: {e})")
        if P.seeds != list(seeds):
            fail(f"{phase}/{stem}: seed list differs from the registered one")
        want = {(c, a) for c, a, _ in arms}
        if set(P.arms) != want:
            fail(f"{phase}/{stem}: arms differ (missing {sorted(want - set(P.arms))}, extra {sorted(set(P.arms) - want)})")
        n_ep = 0
        for cfg, arm, cks in arms:
            K = CFG[cfg]["K"]
            if sorted(P.arms[(cfg, arm)]) != sorted(cks):
                fail(f"{phase}/{stem} {cfg} {arm}: checkpoints {sorted(P.arms[(cfg, arm)])} != {cks}")
            for ck, (rows, eps, _) in P.arms[(cfg, arm)].items():
                if len(rows) != len(seeds) or len(eps) != len(seeds):
                    fail(f"{phase}/{stem} {cfg} {arm} ck {ck}: {len(rows)} rows / {len(eps)} delivery lists")
                for e, (r, ep) in enumerate(zip(rows, eps)):
                    if len(ep) != 52 * K:
                        fail(f"{phase}/{stem} {cfg} {arm} ck {ck} seed {seeds[e]}: {len(ep)} tasks != {52 * K}")
                    if not math.isfinite(r["return"]):
                        fail(f"{phase}/{stem} {cfg} {arm} ck {ck} seed {seeds[e]}: non-finite return")
                    d = decompose(r, K)
                    if abs(d["time"] - d["time_model"]) > 0.5 + 0.01 * abs(d["time_model"]):
                        fail(f"{phase}/{stem} {cfg} {arm} ck {ck} seed {seeds[e]}: return identity off "
                             f"(time residual {d['time']:.2f} vs 0.3 K mission {d['time_model']:.2f})")
                n_ep += len(rows)
        integ = P.integrity()
        if any(integ.values()):
            fail(f"{phase}/{stem}: integrity counters {integ}")
        total += n_ep
        pools[(phase, stem)] = P
        print(f"{phase}/{stem}: {n_ep} episodes OK; integrity {integ}", flush=True)
    # cross-file: the R5 pairing (Phase 3 vs Phase 2 H* rows) and the Phase 4 / Phase 2 seed reuse need identical images
    for pool in RL_TEST:
        stems = [("phase3", f"retrained_{pool}"), ("phase2", f"test_{pool}")] + ([("phase3", "zeroshot_TD")] if pool == "TD" else [])
        ref = None
        for key in stems:
            P = pools[key]
            arm = next(iter(P.arms.values()))
            ims = [Counter(x[0] for x in ep) for ep in next(iter(arm.values()))[1]]
            if ref is None:
                ref = (key, P.seeds, ims)
            elif P.seeds != ref[1] or ims != ref[2]:
                fail(f"{key} and {ref[0]}: different seeds or image multisets (R5 pairing invalid)")
        print(f"R5 pairing {pool}: Phase 3 and Phase 2 share seeds and images per seed", flush=True)
    for pool in GRID_TEST:
        a = pools[("phase4", f"test_{pool}")]
        b = pools[("phase2", f"test_{pool}")]
        if a.seeds != b.seeds:
            fail(f"Phase 4 {pool} and Phase 2 {pool} seeds differ (N5x2 reference reuse invalid)")
    print(f"Phase 4 reuses the Phase 2 N5x2 seeds: OK")
    if total != 27600:
        fail(f"total {total} episodes != 27,600")
    print(f"ALL OK: {total} evaluation episodes (Phase 2 2,400 + new 25,200)")


def comp(P, cfg, arm):
    """per seed (averaged over checkpoints) return components."""
    K = CFG[cfg]["K"]
    cks = list(P.arms[(cfg, arm)].values())
    out = []
    for e in range(len(P.seeds)):
        ds = [decompose(rows[e], K) for rows, _, _ in cks]
        out.append({k: float(np.mean([d[k] for d in ds])) for k in ds[0]})
    return out


def modes(P, t, cfg, arm):
    """per seed: mean payload (kbit), PSNR and detection yield (TP / GT) over Qmin-compliant on-time images."""
    rec, row_of = P.rec, P.row_of_man
    cks = list(P.arms[(cfg, arm)].values())
    out = []
    for e in range(len(P.seeds)):
        kb, ps, tp, gt, cnt = [], [], 0.0, 0.0, Counter()
        for _, eps, _ in cks:
            for mi, mode, ok, ot in eps[e]:
                r = row_of[mi]
                if mode >= 0 and ok:
                    kb.append(t.bits[r, mode] / 1e3)
                    ps.append(t.psnr[r, mode])
                    tp += rec["tp"][r, mode]
                    gt += rec["ngt"][r]
                    cnt[mode] += 1
        out.append(dict(kbit=float(np.mean(kb)), psnr_ok=float(np.mean(ps)), yield_ok=100 * tp / gt if gt else float("nan"),
                        mode_counts=cnt))
    return out


def recall_split(P, cfg, a, b):
    """paired recall decomposition b - a (pp): images compliant in both (mode effect), only in b, only in a.
    Occurrences of the same image within an episode are matched in delivery order."""
    rec, row_of = P.rec, P.row_of_man
    assert len(P.arms[(cfg, a)]) == 1 and len(P.arms[(cfg, b)]) == 1, "single-checkpoint arms only"
    _, ea, _ = next(iter(P.arms[(cfg, a)].values()))
    _, eb, _ = next(iter(P.arms[(cfg, b)].values()))
    res = defaultdict(list)
    for e in range(len(P.seeds)):
        ga, gb = defaultdict(list), defaultdict(list)
        for x in ea[e]:
            ga[x[0]].append(x)
        for x in eb[e]:
            gb[x[0]].append(x)
        gt = sum(rec["ngt"][row_of[x[0]]] for x in ea[e])
        s = Counter()
        for mi in ga:
            r = row_of[mi]
            for xa, xb in zip(ga[mi], gb[mi]):
                oka, okb = xa[1] >= 0 and xa[2], xb[1] >= 0 and xb[2]
                tpa = rec["tp"][r, xa[1]] if oka else 0.0
                tpb = rec["tp"][r, xb[1]] if okb else 0.0
                key = "both" if (oka and okb) else "only_b" if okb else "only_a" if oka else "neither"
                s[key] += tpb - tpa
                s["n_" + key] += 1
                if oka and okb and xa[1] != xb[1]:
                    s["n_both_diff_mode"] += 1
        for k in ("both", "only_b", "only_a"):
            res[k].append(100 * s[k] / gt)
        for k in ("n_both", "n_only_b", "n_only_a", "n_neither", "n_both_diff_mode"):
            res[k].append(s[k])
    return {k: (ci(v) if not k.startswith("n_") else float(np.mean(v))) for k, v in res.items()}


DECOMP = [  # (label, phase, stem, pool, cfg, arm a, arm b): contrast = b - a
    ("ablation", "ablation", "test_{p}", ("TD", "V", "F"), "N5x2", [("nc", "ng", "+mpcnc", "+mpcng"),
                                                                     ("nc", "band", "+mpcnc", "+mpcngb"),
                                                                     ("nc", "gate", "+mpcnc", "+mpcngg"),
                                                                     ("H*", "ng", "", "+mpcng"),
                                                                     ("H*", "nc", "", "+mpcnc"),
                                                                     ("ng", "cal", "+mpcng", "+mpc")]),
    ("phase2", "phase2", "test_{p}", ("TD", "F", "V", "TD_all"), "N5x2", [("nc", "ng", "+mpcnc", "+mpcng"),
                                                                          ("ng", "cal", "+mpcng", "+mpc")]),
]


def decomp():
    t, C = ctx()
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    T4 = json.loads((FD.OUT / "phase4" / "tune.json").read_text())["H"]
    jobs = []
    for label, phase, stem, pools, cfg, pairs in DECOMP:
        for p in pools:
            for na, nb, a, b in pairs:
                jobs.append((label, phase, stem.format(p=p), p, cfg, na, nb, f"h={H}{a}", f"h={H}{b}"))
    for p in GRID_TEST:
        for cfg in CFG:
            if cfg != "N5x2":
                jobs.append(("phase4", "phase4", f"test_{p}", p, cfg, "nc", "ng", f"h={T4[cfg]}+mpcnc", f"h={T4[cfg]}+mpcng"))
                jobs.append(("phase4", "phase4", f"test_{p}", p, cfg, "ng", "cal", f"h={T4[cfg]}+mpcng", f"h={T4[cfg]}+mpc"))
    for x in LEARNED[3:]:
        jobs.append(("phase3", "phase3", "retrained_TD", "TD", "N5x2", f"{x}+nc", f"{x}+ng", f"{x}+mpcnc", f"{x}+mpcng"))
    cache, out, lines = {}, {}, []
    KEYS = ("ret", "sce", "viol", "drop", "uncapt", "energy", "time")
    lines.append("Post-hoc return decomposition (stored rows; paired over environment seeds; contrast = B - A).")
    lines.append("return = SCE(quality+deadline terms of delivered images) + viol(-1 each) + drop(-1 each) + "
                 "uncapt(-1 each) + energy(-0.02/kJ) + time(-0.3 per UAV-second)")
    for label, phase, stem, pool, cfg, na, nb, a, b in jobs:
        if (phase, stem) not in cache:
            cache[(phase, stem)] = load(phase, stem, pool, C)
        P = cache[(phase, stem)]
        A, Bc = comp(P, cfg, a), comp(P, cfg, b)
        d = {k: ci([y[k] - x[k] for x, y in zip(A, Bc)]) for k in KEYS + ("n_ok", "n_qviol", "n_late", "mission_s", "psnr", "delay")}
        lv = {k: [float(np.mean([x[k] for x in A])), float(np.mean([x[k] for x in Bc]))] for k in KEYS + ("n_ok", "mission_s")}
        MA, MB = modes(P, t, cfg, a), modes(P, t, cfg, b)
        mo = {k: [float(np.nanmean([m[k] for m in MA])), float(np.nanmean([m[k] for m in MB]))] for k in ("kbit", "psnr_ok", "yield_ok")}
        mo["yield_diff"] = ci([y["yield_ok"] - x["yield_ok"] for x, y in zip(MA, MB)])
        ca, cb = sum((m["mode_counts"] for m in MA), Counter()), sum((m["mode_counts"] for m in MB), Counter())
        top = lambda c: {FD.C.MODES[k]: round(v / sum(c.values()), 3) for k, v in c.most_common(4)}
        mo["top_modes"] = [top(ca), top(cb)]
        rs = recall_split(P, cfg, a, b) if len(P.arms[(cfg, a)]) == 1 else None
        key = f"{label}|{pool}|{cfg}|{nb}-{na}"
        out[key] = dict(diff=d, levels=lv, modes=mo, recall_split=rs)
        f = lambda v: f"{v[0]:+7.2f}[{v[1]:+6.2f},{v[2]:+6.2f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
        lines.append(f"\n{key}")
        lines.append("  " + "  ".join(f"{k} {f(d[k])}" for k in KEYS))
        lines.append(f"  counts: ok {f(d['n_ok'])} qviol {f(d['n_qviol'])} late {f(d['n_late'])} mission_s {f(d['mission_s'])}")
        lines.append(f"  levels A/B: " + "; ".join(f"{k} {lv[k][0]:.1f}/{lv[k][1]:.1f}" for k in lv))
        lines.append(f"  compliant images: kbit {mo['kbit'][0]:.1f}/{mo['kbit'][1]:.1f}  PSNR {mo['psnr_ok'][0]:.2f}/"
                     f"{mo['psnr_ok'][1]:.2f}  yield TP/GT % {mo['yield_ok'][0]:.2f}/{mo['yield_ok'][1]:.2f} diff {f(mo['yield_diff'])}")
        lines.append(f"  top modes A {mo['top_modes'][0]}  B {mo['top_modes'][1]}")
        if rs:
            lines.append(f"  recall split (pp): both-compliant {f(rs['both'])} only-B {f(rs['only_b'])} only-A {f(rs['only_a'])}; "
                         f"n both {rs['n_both']:.0f} (diff mode {rs['n_both_diff_mode']:.0f}) only-B {rs['n_only_b']:.1f} "
                         f"only-A {rs['n_only_a']:.1f}")
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / "decomp.json", "w"), indent=1, default=lambda o: dict(o))
    (OUT / "decomp.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def recall_pair(PA, a, PB, b, cfg="N5x2"):
    """registered recall analysis (seed-level t-CI + video-cluster bootstrap, B 2000, rng 0) across two files."""
    A, Bb = PA.tpgt(cfg, a), PB.tpgt(cfg, b)
    per = lambda T: np.array([100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for d in T])
    seqs = sorted({s for d in A for s in d})
    idx = {s: i for i, s in enumerate(seqs)}
    dtp, gt = np.zeros(len(seqs)), np.zeros(len(seqs))
    for e in range(len(A)):
        assert set(A[e]) == set(Bb[e])
        for s, (tp, g) in A[e].items():
            assert abs(g - Bb[e][s][1]) < 1e-6, "different images in the paired episodes"
            dtp[idx[s]] += tp - Bb[e][s][0]
            gt[idx[s]] += g
    rng = np.random.default_rng(0)
    bs = [100 * (w * dtp).sum() / (w * gt).sum()
          for w in (np.bincount(rng.integers(len(seqs), size=len(seqs)), minlength=len(seqs)) for _ in range(B))]
    return dict(seed=ci(per(A) - per(Bb)), cluster=[float(100 * dtp.sum() / gt.sum()), float(np.percentile(bs, 2.5)),
                                                    float(np.percentile(bs, 97.5))],
                n_sequences=len(seqs), levels=[float(per(A).mean()), float(per(Bb).mean())])


def r5():
    t, C = ctx()
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    out, lines = {}, ["R5 completion (post-hoc; Phase 3 rows vs Phase 2 H* rows, same seeds and images; contrast = X - H*)."]
    f = lambda v: f"{v[0]:+7.2f} [{v[1]:+6.2f},{v[2]:+6.2f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
    sets = [("retrained", p) for p in RL_TEST] + [("zeroshot", "TD")]
    for name, pool in sets:
        P = load("phase3", f"{name}_{pool}", pool, C)
        P2 = load("phase2", f"test_{pool}", pool, C)
        assert P.seeds == P2.seeds
        fams = LEARNED[3:] if name == "retrained" else LEARNED[:3]
        lines.append(f"\n{name} {pool}")
        for x in fams:
            for tag, sa, sb in (("alone", "", ""), ("matched", "+mpcng", "+mpcng"), ("calibrated", "+mpc", "+mpc"),
                                ("unguarded", "+mpcnc", "+mpcnc")):
                a, b = x + sa, f"h={H}{sb}"
                ret = ci(P.ret("N5x2", a) - P2.ret("N5x2", b))
                per_seed = [ci(P.ret("N5x2", a, ck) - P2.ret("N5x2", b)) for ck in (0, 1, 2)]
                rc = recall_pair(P, a, P2, b)
                out[f"{name}|{pool}|{x}|{tag}"] = dict(ret=ret, ret_per_training_seed=per_seed, recall=rc,
                                                       levels_ret=[float(P.ret("N5x2", a).mean()), float(P2.ret("N5x2", b).mean())])
                lines.append(f"  {x:6s} {tag:10s} ret {f(ret)}  per seed " + " ".join(f"{v[0]:+.2f}{'*' if v[1] > 0 or v[2] < 0 else ''}" for v in per_seed)
                             + f"  | recall seed {f(rc['seed'])} cluster {f(rc['cluster'])} (levels {rc['levels'][0]:.2f}/{rc['levels'][1]:.2f}, {rc['n_sequences']} seq)")
    OUT.mkdir(exist_ok=True)
    json.dump(out, open(OUT / "r5.json", "w"), indent=1)
    (OUT / "r5.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "decomp", "r5"))
    dict(verify=verify, decomp=decomp, r5=r5)[ap.parse_args().cmd]()
