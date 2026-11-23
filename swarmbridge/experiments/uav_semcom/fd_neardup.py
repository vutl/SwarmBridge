"""fd_v1 integrity, part 2 (review of 50ce9022: content duplicates between splits). Sequence prefixes do not always
identify a scene (DET-val 0000022_00000_d_0000004 and 0000023_00000_d_0000008 are byte-identical), so near-duplicate
frames are searched across ALL pools of manifest_fd_v1.json (+ excluded frames) with two perceptual hashes:
  dHash 16x16 (256 bits) on a grey 17x16 resize, and aHash 16x16 on a grey 16x16 resize.
A pair is reported when BOTH Hamming distances are <= T (T = 20 of 256 bits by default; the distribution of the
nearest cross-pool distances is printed so the threshold can be judged). Only pairs from different pools (TD and
TD_extra = one pool) matter for leakage; within-pool pairs are counted only.
Output: runs/uav_semcom/paperA_fd/neardup_fd_v1.json + .log. Usage (WSL): python fd_neardup.py [--t 20]"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fd_manifest import OUT, SRC


def hashes(p):
    with Image.open(p) as im:
        g = im.convert("L")
        d = np.asarray(g.resize((17, 16), Image.BILINEAR), np.int16)
        a = np.asarray(g.resize((16, 16), Image.BILINEAR), np.float32)
    dh = (d[:, 1:] > d[:, :-1]).ravel()
    ah = (a > a.mean()).ravel()
    return np.packbits(dh), np.packbits(ah)


def popcount(x):
    return np.unpackbits(x, axis=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=20)
    a = ap.parse_args()
    man = json.loads((OUT / "manifest_fd_v1.json").read_text())
    ents = man["images"] + man["excluded"]
    pool = [("TD_all" if e["split"] in ("TD", "TD_extra") else e["split"]) for e in ents]
    D, A = [], []
    for i, e in enumerate(ents):
        d, h = hashes(SRC[e["source"]] / "images" / e["name"])
        D.append(d); A.append(h)
        if i % 1000 == 0:
            print(f"hashed {i}/{len(ents)}", flush=True)
    D, A = np.stack(D), np.stack(A)
    pool = np.array(pool)
    pairs, nearest = [], []
    for i in range(len(ents)):
        dd = popcount(D[i] ^ D)
        da = popcount(A[i] ^ A)
        dd[i] = da[i] = 999
        other = pool != pool[i]
        m = np.maximum(dd, da)
        if other.any():
            nearest.append(int(m[other].min()))
        for j in np.where((dd <= a.t) & (da <= a.t))[0]:
            if j > i:
                pairs.append(dict(a=ents[i]["name"], b=ents[j]["name"], pool_a=str(pool[i]), pool_b=str(pool[j]),
                                  dhash=int(dd[j]), ahash=int(da[j]), cross=bool(pool[i] != pool[j])))
    cross = [p for p in pairs if p["cross"]]
    leak = [p for p in cross if not ({p["pool_a"], p["pool_b"]} <= {"x_cap", "x_valshared", "rl_train", "rl_val"}
                                     and "rl_val" not in (p["pool_a"], p["pool_b"]))]
    q = np.percentile(nearest, [0, 0.1, 1, 5, 50])
    lines = [f"images {len(ents)}, threshold {a.t}/256 on both hashes",
             f"nearest cross-pool max(dHash, aHash) distance percentiles (0, 0.1, 1, 5, 50 %): {q.tolist()}",
             f"near-duplicate pairs: {len(pairs)} (within pool {len(pairs) - len(cross)}, across pools {len(cross)})"]
    from collections import Counter
    lines.append("across-pool pairs by pool pair: " + str(dict(Counter(tuple(sorted((p['pool_a'], p['pool_b'])))
                                                                    for p in cross))))
    for p in cross[:40]:
        lines.append(f"   {p}")
    print("\n".join(lines))
    (OUT / "neardup_fd_v1.json").write_text(json.dumps(dict(threshold=a.t, pairs=pairs, nearest_pct=q.tolist()), indent=0))
    (OUT / "neardup_fd_v1.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
