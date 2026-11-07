"""A5.1: image manifest and fixed splits (written once, then read by every A5 script).

Images
  * all 548 VisDrone2019-DET-val frames (A4 used 100 of them: sorted by name, every 5th);
  * external held-out pool: a fixed subset of VisDrone2019-DET-test-dev (sorted by name, every 4th), if present.
    Test-dev comes from different video clips than val; its labels are public (used only by A5.2).
Split of the val frames: GROUPED BY VIDEO SEQUENCE (first field of the file name, e.g. 0000001_02999_d_0000005 ->
0000001), because frames of one sequence are near-duplicates; a frame-level random split would put look-alikes in
train and test. Sequences are shuffled with a fixed seed and assigned greedily to the split furthest below its
target share (train 0.55 / val 0.15 / test 0.30, by frames).
  train : fits the controller-side knowledge (mean table, probe predictor), heuristic tuning episodes, PPO training;
  val   : PPO checkpoint selection and the operating-regime sweep (regime choice);
  test  : final evaluation only. Frames of test sequences that were among A4's 100 images are EXCLUDED (split
          "test_a4_excluded", not used anywhere), so every A5 test frame is unseen by A4 (and by the A4 PPO models).
Every entry carries the SHA-256 of the image file, its resolution and its number of labelled objects (classes 1-10).
Usage: python a5_data.py OUT_JSON VAL_DIR [--testdev DIR]
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TARGET = dict(train=0.55, val=0.15, test=0.30)
SEED = 0


def a4_images(val_dir):
    files = sorted(p.name for p in Path(val_dir, "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    return set(files[::5][:100])


def entry(d, name, split):
    p = Path(d, "images", name)
    w, h = Image.open(p).size
    ann = Path(d, "annotations", Path(name).stem + ".txt")
    n_obj = 0
    if ann.exists():
        for l in ann.read_text().splitlines():
            f = l.strip().strip(",").split(",")
            if len(f) >= 6 and 1 <= int(f[5]) <= 10:
                n_obj += 1
    return dict(name=name, seq=name.split("_")[0], split=split, w=w, h=h, n_obj=n_obj,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("val_dir")
    ap.add_argument("--testdev", default=None)
    args = ap.parse_args()
    names = sorted(p.name for p in Path(args.val_dir, "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    a4 = a4_images(args.val_dir)
    by_seq = defaultdict(list)
    for n in names:
        by_seq[n.split("_")[0]].append(n)
    seqs = sorted(by_seq)
    order = np.random.default_rng(SEED).permutation(len(seqs))
    count = Counter()
    seq_split = {}
    for i in order:
        s = seqs[i]
        tot = max(sum(count.values()), 1)
        sp = min(TARGET, key=lambda k: (count[k] / tot - TARGET[k], k)) if sum(count.values()) else "train"
        seq_split[s] = sp
        count[sp] += len(by_seq[s])
    images = []
    for n in names:
        sp = seq_split[n.split("_")[0]]
        if sp == "test" and n in a4:
            sp = "test_a4_excluded"
        e = entry(args.val_dir, n, sp)
        e["in_a4"] = n in a4
        e["source"] = "VisDrone2019-DET-val"
        images.append(e)
    if args.testdev:
        td = sorted(p.name for p in Path(args.testdev, "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
        for n in td[::4]:
            e = entry(args.testdev, n, "testdev")
            e["in_a4"] = False
            e["source"] = "VisDrone2019-DET-test-dev"
            images.append(e)
    summ = {}
    for sp in ("train", "val", "test", "test_a4_excluded", "testdev"):
        rows = [e for e in images if e["split"] == sp]
        summ[sp] = dict(frames=len(rows), sequences=len({e["seq"] for e in rows}),
                        in_a4=sum(e["in_a4"] for e in rows), objects=sum(e["n_obj"] for e in rows),
                        resolutions=dict(Counter(f"{e['w']}x{e['h']}" for e in rows)))
    seq_sets = {sp: {e["seq"] for e in images if e["split"] == sp} for sp in ("train", "val", "test")}
    assert not (seq_sets["train"] & seq_sets["val"] or seq_sets["train"] & seq_sets["test"]
                or seq_sets["val"] & seq_sets["test"]), "a sequence is in two splits"
    assert not any(e["in_a4"] for e in images if e["split"] == "test")
    man = dict(version="A5", split_seed=SEED, target=TARGET, grouping="video sequence (file-name prefix)",
               a4_subset_rule="VisDrone2019-DET-val sorted by name, every 5th, first 100",
               testdev_rule="VisDrone2019-DET-test-dev sorted by name, every 4th", summary=summ, images=images)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(man, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
