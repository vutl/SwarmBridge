"""A5.2 detector: a VisDrone object detector trained by us, so that the A5 evaluation images (VisDrone2019-DET-val and
test-dev) are never seen in training or model selection.

  prep  : VisDrone2019-DET-train -> YOLO format. Categories 1..10 (pedestrian, people, bicycle, car, van, truck,
          tricycle, awning-tricycle, bus, motor) -> classes 0..9; 'ignored regions' (0) and 'others' (11) dropped
          (as the Ultralytics VisDrone conversion, which also drops score-0 rows). 10 % of the train SEQUENCES (seed 0)
          are held out ('holdout') for checkpoint selection; images are symlinked. Train frames whose sequence prefix
          also occurs in VisDrone2019-DET-val (same source video) are DROPPED (test-dev shares no prefix with train).
  train : Ultralytics YOLO11s, COCO-pretrained (yolo11s.pt), fixed recipe: imgsz 960 (VisDrone objects are small;
          the native frames are 960-1920 px wide), 40 epochs, batch 8, seed 0, deterministic, default augmentation;
          best.pt by holdout fitness is the detector used everywhere in A5.2.
Usage: python a5_det.py prep TRAIN_DIR OUT_DIR VAL_DIR ; python a5_det.py train OUT_DIR RUN_DIR
"""
import os
import sys
from pathlib import Path

import numpy as np

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def prep(train_dir, out, val_dir):
    from PIL import Image
    train_dir, out = Path(train_dir), Path(out)
    names = sorted(p.name for p in (train_dir / "images").iterdir() if p.suffix.lower() == ".jpg")
    # 24 of the 76 val sequence prefixes also occur in train (same source video, e.g. 0000072_..._d_0000001/2 in
    # train, _d_0000003 in val): drop those train frames so the detector never sees a scene of the evaluation images.
    val_seqs = {p.name.split("_")[0] for p in Path(val_dir, "images").iterdir()}
    dropped = [n for n in names if n.split("_")[0] in val_seqs]
    names = [n for n in names if n.split("_")[0] not in val_seqs]
    seqs = sorted({n.split("_")[0] for n in names})
    rng = np.random.default_rng(0)
    hold = set(rng.choice(seqs, size=max(1, round(0.1 * len(seqs))), replace=False))
    n_obj = {"train": 0, "holdout": 0}
    n_img = {"train": 0, "holdout": 0}
    for sp in n_obj:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for n in names:
        sp = "holdout" if n.split("_")[0] in hold else "train"
        src = train_dir / "images" / n
        dst = out / "images" / sp / n
        if not dst.exists():
            os.symlink(src, dst)
        w, h = Image.open(src).size
        lines = []
        for l in (train_dir / "annotations" / (Path(n).stem + ".txt")).read_text().splitlines():
            f = [int(x) for x in l.strip().strip(",").split(",")[:6]]
            x, y, bw, bh, score, cat = f
            if score == 0 or not 1 <= cat <= 10 or bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cat - 1} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / "labels" / sp / (Path(n).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_obj[sp] += len(lines)
        n_img[sp] += 1
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/holdout\nnames:\n" +
                                   "".join(f"  {i}: {c}\n" for i, c in enumerate(NAMES)))
    (out / "split.txt").write_text(f"holdout sequences (seed 0, 10%): {sorted(hold)}\nimages {n_img}\nobjects {n_obj}\n"
                                   f"dropped (sequence shared with val): {len(dropped)} frames, "
                                   f"{len({n.split('_')[0] for n in dropped})} sequences\n")
    print(n_img, n_obj, len(seqs), "sequences")


def train(out, run_dir):
    from ultralytics import YOLO
    m = YOLO("yolo11s.pt")
    m.train(data=str(Path(out) / "data.yaml"), imgsz=960, epochs=40, batch=8, workers=3, seed=0, deterministic=True,
            project=str(run_dir), name="yolo11s_visdrone_960", exist_ok=True, plots=False, cache=False, amp=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        train(sys.argv[2], sys.argv[3])
