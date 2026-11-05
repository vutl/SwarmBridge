"""Summary of b_coop2.py: per setting and episode kind, test rate / outage of map, own (map + own history), coop
(+ decoded messages), shuffle, oracle; mixing weight chosen on DEV per (setting, seed, proto, snr, method).
Gate (vs the strongest non-cooperative baseline 'own'): >= 5 % rate or >= 10 % relative outage, same sign in 3 seeds,
and coop > shuffle. Also split by map coverage / message availability. Usage: python b2_summ.py B2_JSON
"""
import json
import sys
from collections import defaultdict

import numpy as np

d = json.load(open(sys.argv[1]))
rows = d["rows"]
print("bits per message:", d["info"]["bits_per_msg"])
agg = defaultdict(lambda: np.zeros(4))
for r in rows:
    k = (r["setting"], r["seed"], r["proto"], r["snr_db"], r["split"], r["kind"], r["has_map"], r["has_msgs"], r["method"])
    agg[k] += [r["n"], r["rate_sum"], r["outage_sum"], r["msgs_used_sum"]]


def pooled(setting, seed, proto, snr, split, kind, method, region=None):
    v = np.zeros(4)
    for (st, s, p, sn, sp, kd, hm, hc, m), a in agg.items():
        if (st, s, p, sn, sp, kd, m) == (setting, seed, proto, snr, split, kind, method) and \
                (region is None or (hm, hc) == region):
            v += a
    return v


def best(setting, seed, proto, snr, kind, prefix):
    cands = sorted({k[-1] for k in agg if k[-1] == prefix or k[-1].startswith(prefix + "_a")})
    return max(cands, key=lambda m: (lambda v: v[1] / max(v[0], 1))(pooled(setting, seed, proto, snr, "dev", kind, m)))


settings = sorted({r["setting"] for r in rows}, key=lambda s: (s != "main", s))
for setting in settings:
    for kind in ("convoy", "indep"):
        cfgs = sorted({(r["proto"], r["snr_db"]) for r in rows if r["setting"] == setting})
        seeds = sorted({r["seed"] for r in rows if r["setting"] == setting})
        print(f"\n### {setting} / {kind} (test; rate / outage; mixing weight chosen on dev)")
        print("| proto | SNR | map | own (map + own history) | coop | shuffle | oracle | coop vs own: Δrate % | Δoutage % | same sign | msgs used/decision |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for proto, snr in cfgs:
            vals = defaultdict(list)
            dr, do = [], []
            for s in seeds:
                ch = {m: best(setting, s, proto, snr, kind, m) for m in ("own", "coop", "shuffle", "oracle")}
                ch["map"] = "map"
                for m, mm in ch.items():
                    v = pooled(setting, s, proto, snr, "test", kind, mm)
                    vals[m].append((v[1] / v[0], v[2] / v[0], v[3] / v[0]))
                dr.append(100 * (vals["coop"][-1][0] - vals["own"][-1][0]) / vals["own"][-1][0])
                do.append(100 * (vals["own"][-1][1] - vals["coop"][-1][1]) / max(vals["own"][-1][1], 1e-9))
            f = lambda m: f"{np.mean([x[0] for x in vals[m]]):.2f} / {np.mean([x[1] for x in vals[m]]):.3f}"
            same = all(x > 0 for x in dr) or all(x < 0 for x in dr)
            print(f"| {proto} | {snr:+d} | {f('map')} | {f('own')} | {f('coop')} | {f('shuffle')} | {f('oracle')} | "
                  f"{np.mean(dr):+.1f} | {np.mean(do):+.1f} | {same} | {np.mean([x[2] for x in vals['coop']]):.1f} |")
        if kind == "convoy":
            print(f"\n  regions ({setting}, convoy, test, pooled over seeds/configs; dev-chosen weights):")
            for region in ((True, True), (False, True), (True, False), (False, False)):
                line, n = [], 0
                for m in ("map", "own", "coop", "shuffle", "oracle"):
                    v = np.zeros(4)
                    for proto, snr in cfgs:
                        for s in seeds:
                            mm = "map" if m == "map" else best(setting, s, proto, snr, kind, m)
                            v += pooled(setting, s, proto, snr, "test", kind, mm, region)
                    if v[0]:
                        line.append(f"{m} {v[1] / v[0]:.2f}/{v[2] / v[0]:.3f}")
                        n = int(v[0])
                if line:
                    print(f"    has_map={region[0]!s:5} has_msgs={region[1]!s:5} n={n:6d}: " + "; ".join(line))
