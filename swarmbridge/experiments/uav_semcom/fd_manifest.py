"""Paper A full-data protocol fd_v1 (reports/phases/2026-11-23_paperA-fulldata/paperA-full-data-protocol.md, review of
50ce9022): deterministic manifest + integrity checks. Read-only on the data; writes
runs/uav_semcom/paperA_fd/manifest_fd_v1.json and integrity_fd_v1.log. Immutable once profiling starts (the profiler
refuses a manifest whose sha256 differs from the one recorded in its config).

Rules (all deterministic):
  * sequence = file-name prefix before the first '_';
  * DET-train sequences that also occur in DET-val (24)        -> 'x_valshared' (never used);
  * DET-train sequences of the YOLO checkpoint hold-out (18)     -> F (method-unseen supplementary pool);
  * remaining DET-train sequences (166): sorted, permuted with numpy default_rng(0), first round(0.15 n) -> rl_val,
    rest -> rl_train;
  * cap: at most 30 frames per sequence for rl_train / rl_val / F / TD, chosen as the frames at
    round(linspace(0, n-1, 30)) of the name-sorted sequence; uncapped DET-train frames -> 'x_cap' (not profiled);
    uncapped DET-test-dev frames -> TD_extra (TD_all = TD + TD_extra, secondary analysis);
  * DET-val: all 548 frames -> V (intentionally UNCAPPED; max 35 frames per sequence, so a 30-cap would bind);
  * gen_seed (generator seed of the generative modes) = position of the image in the profiling order
    (pools in the order rl_train, rl_val, F, TD, TD_extra, V; names sorted within a pool).
Checks: image/annotation stems match per split, annotation files parse, image sha256 duplicates within and across
splits, names unique, pools sequence-disjoint, YOLO train / hold-out lists equal the expected DET-train subsets.
Usage (WSL): python fd_manifest.py"""
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SRC = {"VisDrone2019-DET-train": Path("/home/uservutl/data/visdrone_dl/VisDrone2019-DET-train"),
       "VisDrone2019-DET-val": REPO / "data" / "visdrone" / "VisDrone2019-DET-val",
       "VisDrone2019-DET-test-dev": Path("/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev")}
YOLO = Path("/home/uservutl/data/visdrone_yolo/images")
OUT = REPO / "runs" / "uav_semcom" / "paperA_fd"
CAP, VAL_FRAC = 30, 0.15
ORDER = ["rl_train", "rl_val", "F", "TD", "TD_extra", "V"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_obj(ann):
    n = 0
    for line in Path(ann).read_text().splitlines():
        f = line.strip().strip(",").split(",")
        if len(f) < 6:
            continue
        x, y, w, h, score, cat = (int(v) for v in f[:6])
        if w > 0 and h > 0 and score != 0 and cat not in (0, 11):
            n += 1
    return n


def cap_idx(n):
    return sorted(set(np.round(np.linspace(0, n - 1, CAP)).astype(int))) if n > CAP else list(range(n))


def main():
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    ent = {}
    for src, d in SRC.items():
        imgs = sorted(p.name for p in (d / "images").iterdir() if p.suffix.lower() == ".jpg")
        anns = sorted(p.name for p in (d / "annotations").iterdir() if p.suffix == ".txt")
        si, sa = {Path(n).stem for n in imgs}, {Path(n).stem for n in anns}
        say(f"{src}: {len(imgs)} images, {len(anns)} annotations, stems equal: {si == sa} "
            f"(images without ann {len(si - sa)}, ann without image {len(sa - si)})")
        assert si == sa
        for n in imgs:
            assert n not in ent, f"duplicate name across splits: {n}"
            p = d / "images" / n
            with Image.open(p) as im:
                w, h = im.size
            ent[n] = dict(name=n, seq=n.split("_")[0], source=src, w=w, h=h, sha256=sha(p),
                          ann_sha256=sha(d / "annotations" / (n[:-4] + ".txt")),
                          n_obj=n_obj(d / "annotations" / (n[:-4] + ".txt")))
    # exact-content duplicates
    by_sha = defaultdict(list)
    for e in ent.values():
        by_sha[e["sha256"]].append(e["name"])
    dups = {k: v for k, v in by_sha.items() if len(v) > 1}
    cross = {k: v for k, v in dups.items() if len({ent[n]["source"] for n in v}) > 1}
    say(f"exact-duplicate image contents: {len(dups)} groups ({sum(len(v) for v in dups.values())} files), "
        f"across splits: {len(cross)}")
    for k, v in list(dups.items())[:20]:
        say(f"   dup {k[:12]}: {v}")
    # sequences
    seqs = defaultdict(lambda: defaultdict(list))
    for e in ent.values():
        seqs[e["source"]][e["seq"]].append(e["name"])
    tr, va, td = (seqs["VisDrone2019-DET-train"], seqs["VisDrone2019-DET-val"], seqs["VisDrone2019-DET-test-dev"])
    shared = sorted(set(tr) & set(va))
    hold_names = sorted(os.listdir(YOLO / "holdout"))
    ytrain_names = sorted(os.listdir(YOLO / "train"))
    hold = sorted({n.split("_")[0] for n in hold_names})
    say(f"DET-train seqs {len(tr)}, shared with DET-val {len(shared)} ({sum(len(tr[s]) for s in shared)} frames), "
        f"YOLO hold-out seqs {len(hold)} ({len(hold_names)} frames), test-dev seqs shared with train/val: "
        f"{len(set(td) & set(tr))}/{len(set(td) & set(va))}")
    exp_hold = sorted(n for s in hold for n in tr[s])
    exp_ytrain = sorted(n for s, ns in tr.items() if s not in shared and s not in hold for n in ns)
    say(f"YOLO hold-out list == DET-train frames of the hold-out seqs: {exp_hold == sorted(hold_names)}; "
        f"YOLO train list == DET-train minus val-shared minus hold-out: {exp_ytrain == ytrain_names}")
    assert exp_hold == sorted(hold_names) and exp_ytrain == ytrain_names
    assert not set(hold) & set(shared)
    rest = sorted(s for s in tr if s not in shared and s not in hold)
    perm = [rest[i] for i in np.random.default_rng(0).permutation(len(rest))]
    n_val = round(VAL_FRAC * len(rest))
    rl_val_seqs, rl_train_seqs = set(perm[:n_val]), set(perm[n_val:])
    split = {}
    for s, ns in tr.items():
        ns = sorted(ns)
        keep = {ns[i] for i in cap_idx(len(ns))}
        for n in ns:
            if s in shared:
                split[n] = "x_valshared"
            elif n not in keep:
                split[n] = "x_cap"
            else:
                split[n] = "F" if s in hold else ("rl_val" if s in rl_val_seqs else "rl_train")
    for s, ns in td.items():
        ns = sorted(ns)
        keep = {ns[i] for i in cap_idx(len(ns))}
        for n in ns:
            split[n] = "TD" if n in keep else "TD_extra"
    for s, ns in va.items():
        for n in ns:
            split[n] = "V"
    # profiling order + generator seeds
    images = []
    for pool in ORDER:
        for n in sorted(k for k, v in split.items() if v == pool):
            images.append(dict(ent[n], split=pool, gen_seed=len(images)))
    excluded = [dict(ent[n], split=v) for n, v in sorted(split.items()) if v.startswith("x_")]
    # pools sequence-disjoint (TD / TD_extra share sequences by design)
    pool_seqs = defaultdict(set)
    for e in images:
        pool_seqs["TD_all" if e["split"] in ("TD", "TD_extra") else e["split"]].add(e["seq"])
    for a in pool_seqs:
        for b in pool_seqs:
            if a < b:
                assert not pool_seqs[a] & pool_seqs[b], (a, b)
    say("pools are sequence-disjoint (TD and TD_extra form one pool, TD_all)")
    summ = {}
    for pool in ORDER + ["x_valshared", "x_cap"]:
        es = [e for e in images + excluded if e["split"] == pool]
        summ[pool] = dict(frames=len(es), sequences=len({e["seq"] for e in es}), objects=sum(e["n_obj"] for e in es),
                          max_frames_per_seq=max(Counter(e["seq"] for e in es).values()) if es else 0,
                          resolutions=dict(Counter(f"{e['w']}x{e['h']}" for e in es).most_common(4)))
        say(f"{pool:12s} {summ[pool]}")
    man = dict(version="fd_v1", rules=__doc__.split("Rules (all deterministic):")[1].split("Checks:")[0].strip(),
               cap=CAP, val_frac=VAL_FRAC, rl_val_seed=0, yolo_holdout_seqs=hold, val_shared_seqs=shared,
               summary=summ, images=images, excluded=excluded)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "manifest_fd_v1.json"
    p.write_text(json.dumps(man, indent=0))
    say(f"manifest written: {p.name} sha256 {sha(p)} ({len(images)} profiled images, {len(excluded)} excluded)")
    (OUT / "integrity_fd_v1.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
