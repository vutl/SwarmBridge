"""Join b_coop and b_attn rows (Np=4): per kind / has_map region, test, pooled over seeds, SNRs and pilot types.
For b_coop methods, the (alpha, rule) variant is the one with the best DEV rate pooled the same way."""
import json, sys
from collections import defaultdict
bc = json.load(open(sys.argv[1]))["rows"]
ba = json.load(open(sys.argv[2]))["rows"]
rows = [r for r in bc if r["n_p"] == 4] + ba
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in rows:
    k = (r["split"], r["kind"], r["has_map"], r["has_coop"], r["method"], r["rule"], r["proto"], r["snr_db"])
    a = agg[k]; a[0] += r["n"]; a[1] += r["rate_sum"]; a[2] += r["outage_sum"]
def pooled(split, kind, hm, hc, meth, rule):
    n = rs = os_ = 0
    for k, a in agg.items():
        if k[:6] == (split, kind, hm, hc, meth, rule):
            n += a[0]; rs += a[1]; os_ += a[2]
    return (rs / n, os_ / n, n) if n else None
def best_variant(prefix, kind):
    cands = {(k[4], k[5]) for k in agg if k[4] == prefix or k[4].startswith(prefix + "_a")}
    def dev_rate(c):
        n = rs = 0
        for k, a in agg.items():
            if k[0] == "dev" and k[1] == kind and (k[4], k[5]) == c:
                n += a[0]; rs += a[1]
        return rs / max(n, 1)
    return max(cands, key=dev_rate)
for kind in ("convoy", "indep"):
    print("==", kind)
    vs = {m: best_variant(m, kind) for m in ("map", "coop", "oracle")}
    vs["attn"] = ("attn", "learned"); vs["attn_nomsg"] = ("attn_nomsg", "learned")
    for hm in (True, False):
        for hc in (True, False):
            line = []
            for m, (meth, rule) in vs.items():
                p = pooled("test", kind, hm, hc, meth, rule)
                if p:
                    line.append(f"{m} {p[0]:.2f}/{p[1]:.3f}")
            p = pooled("test", kind, hm, hc, "map", vs["map"][1])
            if p:
                print(f"  has_map={hm!s:5} has_msg={hc!s:5} n={p[2]:6d}: " + "; ".join(line))
    print("  variants:", vs)
