"""Independent evaluation (namespace paperA_ix), data-independence audit. Read-only: no profiling, no detector, no
controller, no performance outcome is computed.

X1 = VisDrone2019-DET test-challenge (images only, no public labels). For every test-challenge frame:
  * sequence id (file-name prefix before the first '_') and resolution;
  * sha256 of the file bytes;
  * dHash + aHash (fd_neardup.hashes, 256 bits each) and the nearest distance to a REFERENCE set of every VisDrone
    image the project has on disk: DET-train, DET-val, DET-test-dev (all frames, not only the fd_v1 manifest) and
    VID-train / VID-val (used in R1, 10-11/11).
Matches: same sequence prefix as any DET reference frame; identical sha256; near-duplicate = BOTH Hamming distances
<= T (T = 20/256, the fd_v1 threshold, fd_neardup.py).
Output: runs/uav_semcom/paperA_ix/audit_x1.json + .log (names, ids, hashes and distances only; no image content).
Usage (WSL): python ix_audit.py x1 [--procs 6]"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from fd_manifest import REPO, SRC
from fd_neardup import hashes

OUT = REPO / "runs" / "uav_semcom" / "paperA_ix"
TC = Path("/home/uservutl/data/visdrone_tc/VisDrone2019-DET-test-challenge/images")
VID = [Path("/home/uservutl/data/visdrone_vid/VisDrone2019-VID-train/sequences"),
       Path("/home/uservutl/data/visdrone_vid/VisDrone2019-VID-val/sequences")]
T = 20


def seq_of(name):
    return name.split("_")[0]


def one(p):
    p = Path(p)
    b = p.read_bytes()
    with Image.open(p) as im:
        w, h = im.size
    d, a = hashes(p)
    return str(p), hashlib.sha256(b).hexdigest(), w, h, d, a


def run(paths, procs):
    with Pool(procs) as pool:
        return pool.map(one, [str(p) for p in paths], chunksize=64)


def x1(a):
    OUT.mkdir(parents=True, exist_ok=True)
    tc = sorted(TC.glob("*.jpg"))
    det = {k: sorted((v / "images").glob("*.jpg")) for k, v in SRC.items()}
    vid = sorted(p for d in VID for p in d.glob("*/*.jpg"))
    print(f"test-challenge {len(tc)}, DET refs {sum(map(len, det.values()))}, VID refs {len(vid)}", flush=True)
    q = run(tc, a.procs)
    ref_paths = [(k, p) for k, L in det.items() for p in L] + [("VID", p) for p in vid]
    r = run([p for _, p in ref_paths], a.procs)
    src = [k for k, _ in ref_paths]
    rd, ra = np.stack([x[4] for x in r]), np.stack([x[5] for x in r])
    rsha = defaultdict(list)
    for (k, p), x in zip(ref_paths, r):
        rsha[x[1]].append(f"{k}:{Path(p).name}")
    det_seq = {k: {seq_of(p.name) for p in L} for k, L in det.items()}
    rows = []
    for path, sha, w, h, d, ah in q:
        name = Path(path).name
        dd = np.unpackbits(rd ^ d, axis=1).sum(1)
        da = np.unpackbits(ra ^ ah, axis=1).sum(1)
        m = np.maximum(dd, da)
        j = int(np.argmin(m))
        near = np.flatnonzero((dd <= T) & (da <= T))
        rows.append(dict(name=name, seq=seq_of(name), w=w, h=h, sha256=sha,
                         dhash=d.tobytes().hex(), ahash=ah.tobytes().hex(),
                         prefix_in=[k for k, s in det_seq.items() if seq_of(name) in s],
                         sha_match=rsha.get(sha, []),
                         nearest=dict(ref=f"{src[j]}:{Path(ref_paths[j][1]).name}", dhash=int(dd[j]), ahash=int(da[j])),
                         near_dups=[dict(ref=f"{src[i]}:{Path(ref_paths[i][1]).name}", dhash=int(dd[i]), ahash=int(da[i]))
                                    for i in near[:20]]))
    # within test-challenge near-duplicates (counted only)
    qd, qa = np.stack([x[4] for x in q]), np.stack([x[5] for x in q])
    within = 0
    for i in range(len(q)):
        dd = np.unpackbits(qd[i + 1:] ^ qd[i], axis=1).sum(1)
        da = np.unpackbits(qa[i + 1:] ^ qa[i], axis=1).sum(1)
        within += int(((dd <= T) & (da <= T)).sum())
    seqs = Counter(x["seq"] for x in rows)
    bad = defaultdict(set)
    for x in rows:
        if x["prefix_in"]:
            bad["prefix"].add(x["seq"])
        if x["sha_match"]:
            bad["sha"].add(x["seq"])
        if x["near_dups"]:
            bad["near"].add(x["seq"])
    flagged = set().union(*bad.values()) if bad else set()
    nn = np.array([max(x["nearest"]["dhash"], x["nearest"]["ahash"]) for x in rows])
    summ = dict(frames=len(rows), sequences=len(seqs), frames_per_seq=dict(sorted(seqs.items())),
                resolutions=Counter(f"{x['w']}x{x['h']}" for x in rows).most_common(),
                refs=dict(det={k: len(v) for k, v in det.items()}, vid=len(vid)), threshold=T,
                flagged_sequences={k: sorted(v) for k, v in bad.items()},
                n_flagged_sequences=len(flagged),
                clean_sequences=len(set(seqs) - flagged),
                clean_frames=sum(n for s, n in seqs.items() if s not in flagged),
                clean_frames_cap30=sum(min(n, 30) for s, n in seqs.items() if s not in flagged),
                nearest_ref_pct={str(p): float(np.percentile(nn, p)) for p in (0, 1, 5, 50, 95)},
                within_tc_near_pairs=within)
    (OUT / "audit_x1.json").write_text(json.dumps(dict(summary=summ, frames=rows), indent=0))
    log = json.dumps(summ, indent=1)
    (OUT / "audit_x1.log").write_text(log + "\n")
    print(log)


# ---------------------------------------------------------------- full cross-dataset audit (after UAVDT acquisition)
UAVDT = Path("/home/uservutl/data/uavdt/UAV-benchmark-M")
UGT = Path("/home/uservutl/data/uavdt/UAV-benchmark-MOTD_v1.0/GT")
U_TEST = ("M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 "
          "M1302 M1303 M1401").split()
CACHE = Path("/home/uservutl/work/paperA_ix/hashcache.npz")


def pc64(x):
    """popcount of uint64 arrays (SWAR)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _near_chunk(args):
    """for query rows qi: indices j of reference rows with dHash <= T and aHash <= T, plus the nearest max-distance."""
    qd, qa, rd, ra, t = args
    out, nearest = [], []
    for i in range(len(qd)):
        dd = pc64(rd ^ qd[i]).sum(1)
        cand = np.flatnonzero(dd <= t)
        da = pc64(ra[cand] ^ qa[i]).sum(1) if len(cand) else np.zeros(0, np.uint64)
        out.append([(int(j), int(dd[j]), int(a)) for j, a in zip(cand, da) if a <= t])
        nearest.append(int(dd.min()) if len(dd) else 256)
    return out, nearest


def near(qd, qa, rd, ra, procs, t=T):
    qd, qa, rd, ra = (np.ascontiguousarray(x).view(np.uint64) for x in (qd, qa, rd, ra))
    ch = max(1, len(qd) // (procs * 8))
    jobs = [(qd[i:i + ch], qa[i:i + ch], rd, ra, t) for i in range(0, len(qd), ch)]
    with Pool(procs) as pool:
        res = pool.map(_near_chunk, jobs)
    return [m for r in res for m in r[0]], [m for r in res for m in r[1]]


def hashed(paths, procs):
    """(sha256, w, h, dhash, ahash) per path, cached outside git (hashes are of local files, not artifacts)."""
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = dict(zip(z["paths"].tolist(), zip(z["sha"].tolist(), z["w"].tolist(), z["h"].tolist(), z["d"], z["a"])))
    todo = [str(p) for p in paths if str(p) not in cache]
    if todo:
        print(f"hashing {len(todo)} files", flush=True)
        for p, sha, w, h, d, a in run(todo, procs):
            cache[p] = (sha, w, h, d, a)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        ks = sorted(cache)
        np.savez(CACHE, paths=np.array(ks, object), sha=np.array([cache[k][0] for k in ks], object),
                 w=np.array([cache[k][1] for k in ks]), h=np.array([cache[k][2] for k in ks]),
                 d=np.stack([cache[k][3] for k in ks]), a=np.stack([cache[k][4] for k in ks]))
    return [cache[str(p)] for p in paths]


def ugt_last_frame(seq):
    last = 0
    for f in (UGT / f"{seq}_gt_whole.txt", UGT / f"{seq}_gt_ignore.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                last = max(last, int(line.split(",")[0]))
    return last


def full(a):
    """Cross audit of VD_dev (history: DET train/val/test-dev + VID), X1 (test-challenge), U (UAVDT train / test).
    Deterministic exclusion direction (pre-registration §1.3): history is never excluded; the official UAVDT test split
    is preserved against any non-history pool; X1 (unofficial) yields to everything else.
      X1 seq ~ VD_dev        -> exclude X1 seq          U_test seq ~ VD_dev -> exclude U_test seq
      X1 seq ~ U (any)       -> exclude X1 seq          U_train seq ~ U_test seq -> exclude U_train seq
      U_train ~ VD_dev       -> no exclusion (U_train trains only the X2-retrained arm; VisDrone history is not tested)"""
    OUT.mkdir(parents=True, exist_ok=True)
    det = [(k, p) for k, v in SRC.items() for p in sorted((v / "images").glob("*.jpg"))]
    vid = [("VID", p) for d in VID for p in sorted(d.glob("*/*.jpg"))]
    tc = [("X1", p) for p in sorted(TC.glob("*.jpg"))]
    useq = sorted(d.name for d in UAVDT.iterdir() if d.is_dir())
    uu = [("U_test" if s in U_TEST else "U_train", p) for s in useq for p in sorted((UAVDT / s).glob("*.jpg"))]
    allp = det + vid + tc + uu
    H = hashed([p for _, p in allp], a.procs)
    grp = np.array([g if g in ("X1", "U_test", "U_train") else "VD" for g, _ in allp])
    seq = np.array([p.parent.name if g.startswith("U_") else (p.parent.name if g == "VID" else seq_of(p.name))
                    for g, p in allp])
    d = np.stack([h[3] for h in H])
    av = np.stack([h[4] for h in H])
    sha = np.array([h[0] for h in H])
    idx = {g: np.flatnonzero(grp == g) for g in ("VD", "X1", "U_test", "U_train")}
    links = []

    def check(qg, rg, same_seq_ok=False):
        qi, ri = idx[qg], idx[rg]
        m, nn = near(d[qi], av[qi], d[ri], av[ri], a.procs)
        sref = defaultdict(list)
        for k, j in enumerate(ri):
            sref[sha[j]].append(j)
        for k, i in enumerate(qi):
            hits = [(int(ri[j]), dd, aa, "near") for j, dd, aa in m[k]]
            hits += [(int(j), 0, 0, "sha") for j in sref.get(sha[i], [])]
            for j, dd, aa, kind in hits:
                if i == j or (same_seq_ok and seq[i] == seq[j]):
                    continue
                links.append(dict(q=qg, q_seq=str(seq[i]), q_file=str(allp[i][1].name), r=rg, r_src=allp[j][0],
                                  r_seq=str(seq[j]), r_file=str(allp[j][1].name), kind=kind, dhash=dd, ahash=aa))
        return nn

    print("checking", flush=True)
    nn = {}
    nn["X1~VD"] = check("X1", "VD")
    nn["U_test~VD"] = check("U_test", "VD")
    nn["U_train~VD"] = check("U_train", "VD")
    nn["X1~U_test"] = check("X1", "U_test")
    nn["X1~U_train"] = check("X1", "U_train")
    nn["U_train~U_test"] = check("U_train", "U_test")
    nn["X1~X1"] = check("X1", "X1", same_seq_ok=True)
    nn["U_test~U_test"] = check("U_test", "U_test", same_seq_ok=True)
    nn["U_train~U_train"] = check("U_train", "U_train", same_seq_ok=True)
    excl = defaultdict(set)
    for l in links:
        if l["q"] == "X1" and l["r"] in ("VD", "U_test", "U_train"):
            excl["X1"].add(l["q_seq"])
        elif l["q"] == "U_test" and l["r"] == "VD":
            excl["U_test"].add(l["q_seq"])
        elif l["q"] == "U_train" and l["r"] == "U_test":
            excl["U_train"].add(l["q_seq"])
    gt = {s: dict(images=len(list((UAVDT / s).glob("*.jpg"))), last_annotated=ugt_last_frame(s)) for s in useq}
    res_u = Counter(f"{H[i][1]}x{H[i][2]}" for i in np.r_[idx["U_test"], idx["U_train"]])
    pct = {k: {str(p): float(np.percentile(v, p)) for p in (0, 1, 5, 50)} for k, v in nn.items() if len(v)}
    cnt = Counter((l["q"], l["r"], l["kind"]) for l in links)
    summ = dict(threshold=T, sizes={g: int(len(v)) for g, v in idx.items()},
                uavdt_sequences=dict(test=[s for s in useq if s in U_TEST], train=[s for s in useq if s not in U_TEST]),
                uavdt_resolutions=res_u.most_common(), uavdt_gt_coverage=gt,
                link_counts={f"{q}~{r}:{k}": v for (q, r, k), v in sorted(cnt.items())},
                link_sequences={f"{q}~{r}": sorted({f"{l['q_seq']}~{l['r_seq']}" for l in links
                                                    if l["q"] == q and l["r"] == r})
                                for q, r in sorted({(l["q"], l["r"]) for l in links})},
                excluded={k: sorted(v) for k, v in excl.items()},
                nearest_dhash_pct=pct)
    (OUT / "audit_full.json").write_text(json.dumps(dict(summary=summ, links=links[:5000]), indent=0))
    log = json.dumps(summ, indent=1)
    (OUT / "audit_full.log").write_text(log + "\n")
    print(log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["x1", "full"])
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    x1(a) if a.what == "x1" else full(a)
