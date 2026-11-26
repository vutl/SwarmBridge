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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["x1"])
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    x1(a)
