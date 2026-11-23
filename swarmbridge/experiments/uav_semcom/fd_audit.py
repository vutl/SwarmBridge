"""Paper A full-data Phase 0 audit (read-only): sizes, sequences and overlaps of the VisDrone DET splits, the YOLO11s
training set (a5_det.py prep), and the current A5 pools. Output: runs/uav_semcom/paperA_fd/audit.log."""
import json
import os
from collections import Counter
from pathlib import Path

R = Path("/home/uservutl/data/visdrone_dl")
SPL = {"train": R / "VisDrone2019-DET-train", "val": Path("/mnt/c/Users/USER/Documents/SwarmBridge/data/visdrone/VisDrone2019-DET-val"),
       "testdev": R / "VisDrone2019-DET-test-dev"}
info = {}
for k, d in SPL.items():
    imgs = sorted(p for p in os.listdir(d / "images") if p.lower().endswith(".jpg"))
    anns = os.listdir(d / "annotations") if (d / "annotations").exists() else []
    seqs = Counter(n.split("_")[0] for n in imgs)
    # source video id: first field; clip = first two fields
    clips = Counter("_".join(n.split("_")[:2]) for n in imgs)
    info[k] = dict(n_img=len(imgs), n_ann=len(anns), n_seq=len(seqs), n_clip=len(clips),
                   frames_per_seq=dict(min=min(seqs.values()), median=sorted(seqs.values())[len(seqs) // 2],
                                       max=max(seqs.values())), seqs=set(seqs), seqcount=seqs)
for k, v in info.items():
    print(k, {a: b for a, b in v.items() if a not in ("seqs", "seqcount")})
for a in info:
    for b in info:
        if a < b:
            ov = info[a]["seqs"] & info[b]["seqs"]
            print(f"seq overlap {a}-{b}: {len(ov)} seqs; frames in {a}: {sum(info[a]['seqcount'][s] for s in ov)}, "
                  f"in {b}: {sum(info[b]['seqcount'][s] for s in ov)}")
# detector training data
yd = Path("/home/uservutl/data/visdrone_yolo")
for sp in ("train", "holdout"):
    p = yd / "images" / sp
    if p.exists():
        names = os.listdir(p)
        s = {n.split("_")[0] for n in names}
        print(f"YOLO {sp}: {len(names)} frames, {len(s)} seqs; overlap with val seqs {len(s & info['val']['seqs'])}, "
              f"with testdev seqs {len(s & info['testdev']['seqs'])}")
        info[f"yolo_{sp}"] = dict(seqs=s, n=len(names))
# current A5 manifest pools
m = json.load(open("/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/a5/manifest.json"))
pools = {}
for e in m["images"]:
    pools.setdefault(e["split"], set()).add(e["name"])
for k, v in pools.items():
    print("A5 pool", k, len(v), "seqs", len({n.split('_')[0] for n in v}))
td_used = {n.split("_")[0] for n in pools["testdev"]}
print("test-dev seqs touched by A5 test-dev pool:", len(td_used), "of", info["testdev"]["n_seq"])
print("test-dev frames in touched seqs:", sum(info["testdev"]["seqcount"][s] for s in td_used),
      "untouched seqs:", info["testdev"]["n_seq"] - len(td_used),
      "frames:", sum(c for s, c in info["testdev"]["seqcount"].items() if s not in td_used))
train_in_val = info["train"]["seqs"] & info["val"]["seqs"]
print("DET-train frames NOT in YOLO training (dropped because seq shared with val):",
      sum(info["train"]["seqcount"][s] for s in train_in_val))
# resolution / object counts (annotations) for DET-train sample stats
def nobj(d, n):
    a = d / "annotations" / (n[:-4] + ".txt")
    return sum(1 for line in open(a) if line.strip() and line.split(",")[5] not in ("0", "11"))
for k in ("train", "testdev"):
    d = SPL[k]
    imgs = sorted(p for p in os.listdir(d / "images") if p.lower().endswith(".jpg"))
    objs = [nobj(d, n) for n in imgs]
    print(k, "objects/frame mean", round(sum(objs) / len(objs), 1), "total", sum(objs))
