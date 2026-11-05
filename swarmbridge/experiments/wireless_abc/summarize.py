"""Summaries and decision gates for the wireless A/B/C pilot (pure python + numpy; runs anywhere).

A1 gate: the bridge (SB) must beat the strongest other feasible method by >= 5 % rate or >= 10 % relative outage at
the same pilots, with the same sign in all 3 seeds. Selection (method rule, NFE) is made on DEV per seed and config,
then read on TEST.  NFE is matched: for bridge at NFE k, competitors are allowed NFE <= k (classical ones have NFE 0).
B gate: coop (alpha chosen on dev) vs map, >= 5 % rate or >= 10 % relative outage, same sign in 3 seeds, and coop must
beat shuffle.
Usage: python summarize.py a1 A1_JSON | b B_JSON | c C_FEAS_JSON [C_TRACK_JSON TRACK_DIR]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def a1(path, test_split="test"):
    """test_split: test (truth-filtered links), test_all (every link of test episodes), test_deep (deep links only)."""
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    print("A1 data:", d["info"], "steps", d["steps"])
    key = lambda r: (r["proto"], r["n_p"], r["snr_db"], r["seed"])
    by = defaultdict(list)
    for r in rows:
        by[(r["split"],) + key(r)].append(r)
    configs = sorted({key(r)[:3] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = []
    print("\n| proto | Np | SNR | best non-bridge (test) | rate | outage | bridge best (test) | rate | outage | Δrate % | Δoutage % (rel) | seeds same sign |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for proto, n_p, snr in configs:
        per_seed = []
        for s in seeds:
            dev, te = by[("dev", proto, n_p, snr, s)], by[(test_split, proto, n_p, snr, s)]
            if not dev or not te:
                continue
            tidx = {(r["method"], r["rule"], r["nfe"]): r for r in te}

            def best(cands):
                c = max(cands, key=lambda r: r["rate"])
                return tidx[(c["method"], c["rule"], c["nfe"])]
            bridge = [r for r in dev if r["method"] == "bridge"]
            if not bridge:
                continue
            bb = best(bridge)
            others = [r for r in dev if r["method"] not in ("bridge",) and r["nfe"] <= bb["nfe"]]
            bo = best(others)
            per_seed.append((bo, bb))
        if not per_seed:
            continue
        dr = [100 * (b["rate"] - o["rate"]) / max(o["rate"], 1e-9) for o, b in per_seed]
        do = [100 * (o["outage"] - b["outage"]) / max(o["outage"], 1e-9) for o, b in per_seed]
        same = all(x > 0 for x in dr) or all(x < 0 for x in dr)
        o0, b0 = per_seed[0]
        names_o = ",".join(sorted({f"{o['method']}/{o['rule']}/{o['nfe']}" for o, _ in per_seed}))
        names_b = ",".join(sorted({f"{b['rule']}/{b['nfe']}" for _, b in per_seed}))
        mr = lambda k, i: np.mean([p[i][k] for p in per_seed])
        print(f"| {proto} | {n_p} | {snr:+d} | {names_o} | {mr('rate', 0):.3f} | {mr('outage', 0):.3f} | {names_b} | "
              f"{mr('rate', 1):.3f} | {mr('outage', 1):.3f} | {np.mean(dr):+.1f} | {np.mean(do):+.1f} | {same} |")
        genie = float(np.mean([o["rate"] + o["regret"] for o, _ in per_seed]))   # best DFT beam with perfect CSI
        gate = (np.mean(dr) >= 5 and all(x > 0 for x in dr)) or (np.mean(do) >= 10 and all(x > 0 for x in do))
        gate = gate and genie >= 0.01     # % differences between ~0 rates are meaningless (deep links: genie ~ 0)
        out.append(dict(proto=proto, n_p=n_p, snr_db=snr, d_rate_pct=dr, d_out_pct=do, gate=bool(gate), genie=genie))
    print(f"\nA1 gate ({test_split}) passed in", sum(o["gate"] for o in out), "of", len(out), "configs")
    print("genie (best DFT beam, perfect CSI) rate per config:", " ".join(f"{o['genie']:.3g}" for o in out))
    # method table (test, mean over seeds, best rule/NFE per method chosen on dev)
    print("\n| proto | Np | SNR | " + " | ".join(["ls", "sweep", "lmmse", "omp", "gmm", "map", "gauss", "ddpm", "flow", "bridge", "bridge0"]) + " |")
    print("|---|---|---|" + "---|" * 11)
    for proto, n_p, snr in configs:
        cells = []
        for meth in ["ls", "sweep", "lmmse", "omp", "gmm", "map", "gauss", "ddpm", "flow", "bridge", "bridge0"]:
            vals = []
            for s in seeds:
                dev = [r for r in by[("dev", proto, n_p, snr, s)] if r["method"] == meth]
                te = {(r["rule"], r["nfe"]): r for r in by[(test_split, proto, n_p, snr, s)] if r["method"] == meth}
                if dev:
                    c = max(dev, key=lambda r: r["rate"])
                    vals.append((te[(c["rule"], c["nfe"])]["rate"], te[(c["rule"], c["nfe"])]["outage"]))
            cells.append("-" if not vals else f"{np.mean([v[0] for v in vals]):.2f} / {np.mean([v[1] for v in vals]):.3f}")
        print(f"| {proto} | {n_p} | {snr:+d} | " + " | ".join(cells) + " |")
    # calibration and NMSE (test, seed-mean) for probabilistic methods at Np=4, all SNRs
    print("\nCalibration (test, Np=4, rule egain; coverage of 80 % / 95 % intervals of every beam's gain; NMSE dB; ms/sample):")
    for proto, n_p, snr in configs:
        if n_p != 4:
            continue
        line = []
        for meth, nfe in [("lmmse", 0), ("gmm", 0), ("gauss", 0), ("ddpm", 8), ("flow", 8), ("bridge", 8), ("bridge", 1), ("bridge0", 8)]:
            rs = [r for s in seeds for r in by[(test_split, proto, n_p, snr, s)]
                  if r["method"] == meth and r["rule"] == "egain" and r["nfe"] == nfe and "cov80" in r]
            if rs:
                line.append(f"{meth}{'' if nfe == 0 else '@' + str(nfe)} {np.mean([r['cov80'] for r in rs]):.2f}/"
                            f"{np.mean([r['cov95'] for r in rs]):.2f} nmse {np.mean([r['nmse_db'] for r in rs]):+.1f}")
        print(f"  {proto} SNR {snr:+d}: " + "; ".join(line))
    return out


def b(path):
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    print("B info:", d["info"])
    agg = defaultdict(lambda: np.zeros(4))
    for r in rows:
        k = (r["split"], r["seed"], r["proto"], r["n_p"], r["snr_db"], r["kind"], r["method"], r["rule"])
        agg[k] += [r["n"], r["rate_sum"], r["outage_sum"], r["msgs_sum"]]
    seeds = sorted({r["seed"] for r in rows})
    cfgs = sorted({(r["proto"], r["n_p"], r["snr_db"]) for r in rows})
    res = []
    for kind in ("convoy", "indep"):
        print(f"\n### {kind} (test; alpha and rule chosen on dev per seed)")
        print("| proto | Np | SNR | global | map | coop | shuffle | oracle | Δrate coop vs map % | Δoutage % | seeds same sign | msgs/decision |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for proto, n_p, snr in cfgs:
            cell = defaultdict(list)
            dr, do, msgs = [], [], []
            for s in seeds:
                def pick(prefix):
                    cands = [(m, ru) for (sp, ss, p, n, sn, kd, m, ru) in agg
                             if sp == "dev" and ss == s and p == proto and n == n_p and sn == snr and kd == kind
                             and (m == prefix or m.startswith(prefix + "_a"))]
                    if not cands:
                        return None
                    best = max(cands, key=lambda c: agg[("dev", s, proto, n_p, snr, kind) + c][1]
                               / max(agg[("dev", s, proto, n_p, snr, kind) + c][0], 1))
                    v = agg[("test", s, proto, n_p, snr, kind) + best]
                    return v[1] / max(v[0], 1), v[2] / max(v[0], 1), v[3] / max(v[0], 1)
                vals = {m: pick(m) for m in ("global", "map", "coop", "shuffle", "oracle")}
                if any(v is None for v in vals.values()):
                    continue
                for m, v in vals.items():
                    cell[m].append(v)
                dr.append(100 * (vals["coop"][0] - vals["map"][0]) / vals["map"][0])
                do.append(100 * (vals["map"][1] - vals["coop"][1]) / max(vals["map"][1], 1e-9))
                msgs.append(vals["coop"][2])
            if not dr:
                continue
            same = all(x > 0 for x in dr) or all(x < 0 for x in dr)
            f = lambda m: f"{np.mean([v[0] for v in cell[m]]):.2f} / {np.mean([v[1] for v in cell[m]]):.3f}"
            print(f"| {proto} | {n_p} | {snr:+d} | {f('global')} | {f('map')} | {f('coop')} | {f('shuffle')} | {f('oracle')} | "
                  f"{np.mean(dr):+.1f} | {np.mean(do):+.1f} | {same} | {np.mean(msgs):.1f} |")
            beats_shuffle = np.mean([v[0] for v in cell["coop"]]) > np.mean([v[0] for v in cell["shuffle"]])
            gate = ((np.mean(dr) >= 5 and all(x > 0 for x in dr)) or (np.mean(do) >= 10 and all(x > 0 for x in do))) and beats_shuffle
            res.append(dict(kind=kind, proto=proto, n_p=n_p, snr_db=snr, d_rate_pct=dr, d_out_pct=do, gate=bool(gate)))
    print("\nB gate passed in", sum(r["gate"] for r in res if r["kind"] == "convoy"), "of",
          sum(r["kind"] == "convoy" for r in res), "convoy configs")
    return res


def c(feas_path, track_path=None, track_dir=None):
    d = json.loads(Path(feas_path).read_text())
    print("C feasibility:", {k: v for k, v in d.items() if k != "policies"})
    print("| policy | delivered (all) | LoS frames | NLoS frames |")
    print("|---|---|---|---|")
    for p, s in d["policies"].items():
        print(f"| {p} | {s['deliv_frac']:.3f} | {s['los_frac']:.3f} | {s['nlos_frac']:.3f} |")
    if track_path:
        t = json.loads(Path(track_path).read_text())
        rows = t["rows"]
        print("\n| policy | delivered (UAV3D-mapped) | mAP | AMOTA (3 seeds) |")
        print("|---|---|---|---|")
        for pol in sorted({r["policy"] for r in rows}):
            rs = [r for r in rows if r["policy"] == pol]
            am = []
            for r in rs:
                f = Path(track_dir) / "tracking" / f"{pol}__m{r['seed']}" / "metrics_summary.json"
                if f.exists():
                    inp = json.loads((f.parent / "input.json").read_text())
                    assert inp["det_sha256"] == r["sha256"], f
                    am.append(json.loads(f.read_text())["amota"])
            amc = "-" if not am else f"{np.mean(am):.3f} ±{np.std(am):.3f} (n={len(am)})"
            print(f"| {pol} | {np.mean([r['deliv_frac'] for r in rs]):.3f} | {np.mean([r['mAP'] for r in rs]):.3f} | {amc} |")


if __name__ == "__main__":
    {"a1": a1, "b": b, "c": c}[sys.argv[1]](*sys.argv[2:])
