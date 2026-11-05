import json, sys
from collections import defaultdict
d = json.load(open(sys.argv[1]))
agg = defaultdict(lambda: [0, 0.0, 0.0])
for r in d["rows"]:
    if r["split"] != "test" or r["kind"] != "convoy" or r["has_map"] or not r["has_msgs"]:
        continue
    if r["setting"] not in ("main", "nomap", "map25") or r["proto"] != "random" or r["snr_db"] != 5:
        continue
    k = (r["setting"], r["method"]); a = agg[k]; a[0] += r["n"]; a[1] += r["rate_sum"]; a[2] += r["outage_sum"]
for st in ("main", "map25", "nomap"):
    print(st, "  ".join(f"{m}:{a[1]/a[0]:.2f}/{a[2]/a[0]:.3f}" for (s, m), a in sorted(agg.items()) if s == st and (m.startswith("own") or m.startswith("coop") or m.startswith("oracle_a0.0") or m in ("map", "global"))))
