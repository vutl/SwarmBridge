"""paperA_ix manifest (pre-registration §2). Deterministic; reads only file names, image headers / bytes and GT text.
  * X1: VisDrone2019-DET test-challenge minus the sequences excluded by audit_full (ix_audit.py full).
  * UAVDT (UAV-benchmark-M): official test = the 20 sequences with released baseline detections (ix_audit.U_TEST);
    the 30 train sequences, sorted, permuted with numpy default_rng(0); the first round(0.15 * 30) = 4 -> U_val, the
    rest -> U_train. Frames: only indices 1 .. last annotated frame of the sequence (gt_whole U gt_ignore; M0207 has
    885 images but annotations end at frame 571).
  * Cap: at most 30 frames per sequence at positions round(linspace(0, n - 1, 30)) of the name- (X1) or frame-index-
    (UAVDT) sorted list.
  * gen_seed = position in the order U_train, U_val, U_test, X1 (names sorted within a pool).
Entry fields: name, seq, source, split, w, h, sha256, (UAVDT) frame, n_obj (gt_whole boxes), n_ign (ignore regions),
ann_sha256 (sha256 of the frame's gt_whole + gt_ignore lines), gen_seed.
Output: runs/uav_semcom/paperA_ix/manifest_ix_v1.json. Usage (WSL): python ix_manifest.py"""
import hashlib
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from ix_audit import OUT, TC, U_TEST, UAVDT, UGT, seq_of

CAP = 30
ORDER = ["U_train", "U_val", "U_test", "X1"]


def cap(lst):
    n = len(lst)
    if n <= CAP:
        return lst
    return [lst[i] for i in sorted(set(np.round(np.linspace(0, n - 1, CAP)).astype(int).tolist()))]


def gt_lines(seq):
    by = defaultdict(lambda: {"whole": [], "ignore": []})
    for kind in ("whole", "ignore"):
        for line in (UGT / f"{seq}_gt_{kind}.txt").read_text().splitlines():
            if line.strip():
                by[int(line.split(",")[0])][kind].append(line.strip())
    return by


def entry(path, name, seq, source, split, extra=None):
    b = path.read_bytes()
    with Image.open(path) as im:
        w, h = im.size
    e = dict(name=name, seq=seq, source=source, split=split, w=w, h=h, sha256=hashlib.sha256(b).hexdigest())
    e.update(extra or {})
    return e


def main():
    audit = json.loads((OUT / "audit_full.json").read_text())["summary"]
    excl_x1 = set(audit["excluded"].get("X1", []))
    excl_ut = set(audit["excluded"].get("U_test", []))
    excl_utr = set(audit["excluded"].get("U_train", []))
    ents = []
    # X1
    by = defaultdict(list)
    for p in sorted(TC.glob("*.jpg")):
        by[seq_of(p.name)].append(p)
    for s in sorted(by):
        if s in excl_x1:
            continue
        for p in cap(by[s]):
            ents.append(entry(p, p.name, s, "VisDrone2019-DET-test-challenge", "X1"))
    # UAVDT
    seqs = sorted(d.name for d in UAVDT.iterdir() if d.is_dir())
    train = [s for s in seqs if s not in U_TEST and s not in excl_utr]
    perm = [train[i] for i in np.random.default_rng(0).permutation(len(train))]
    n_val = round(0.15 * len([s for s in seqs if s not in U_TEST]))
    val = set(perm[:n_val])
    for s in seqs:
        if s in U_TEST:
            if s in excl_ut:
                continue
            split = "U_test"
        elif s in excl_utr:
            continue
        else:
            split = "U_val" if s in val else "U_train"
        gl = gt_lines(s)
        last = max(gl)
        frames = sorted(p for p in (UAVDT / s).glob("img*.jpg") if int(p.stem[3:]) <= last)
        for p in cap(frames):
            f = int(p.stem[3:])
            g = gl.get(f, {"whole": [], "ignore": []})
            ann = hashlib.sha256("\n".join(g["whole"] + ["#"] + g["ignore"]).encode()).hexdigest()
            ents.append(entry(p, f"{s}/{p.name}", s, "UAVDT-M", split,
                              dict(frame=f, n_obj=len(g["whole"]), n_ign=len(g["ignore"]), ann_sha256=ann)))
    ents.sort(key=lambda e: (ORDER.index(e["split"]), e["name"]))
    for i, e in enumerate(ents):
        e["gen_seed"] = i
    summ = {}
    for sp in ORDER:
        L = [e for e in ents if e["split"] == sp]
        res = defaultdict(int)
        for e in L:
            res[f"{e['w']}x{e['h']}"] += 1
        summ[sp] = dict(frames=len(L), sequences=len({e["seq"] for e in L}), sequence_list=sorted({e["seq"] for e in L}),
                        objects=sum(e.get("n_obj", 0) for e in L) if sp != "X1" else None,
                        max_frames_per_seq=max(np.bincount(np.unique([e["seq"] for e in L],
                                                                     return_inverse=True)[1])),
                        resolutions=dict(res))
    man = dict(version="ix_v1", cap=CAP, val_rule="UAVDT train sequences sorted, default_rng(0) permutation, "
                                                   "first round(0.15*30)=4 -> U_val",
               excluded=dict(X1=sorted(excl_x1), U_test=sorted(excl_ut), U_train=sorted(excl_utr)),
               audit_sha256=hashlib.sha256((OUT / "audit_full.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               summary=summ, images=ents)
    p = OUT / "manifest_ix_v1.json"
    p.write_text(json.dumps(man, indent=0, default=int))
    print(json.dumps(summ, indent=1, default=int))
    print("manifest sha256", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
