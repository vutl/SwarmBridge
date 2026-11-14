"""G-L2 0.4 (reports/archive/r1-visdrone/uav-semcom-g-l2.md): a second, independent detector measurement of every (image, codec mode).

Same codecs, settings, detector, operating point and scoring as a5_fidelity.py (A5.2 / L1), except that the detector
sees the decoded image FLIPPED horizontally (boxes flipped back before matching). The codecs are deterministic (the
payload is checked against the L1 rows), so both measurements see the same compressed content; their differences are
detector noise. Generative modes are not re-measured. Resumable (images already in the output are skipped).
Usage: python g2_flip.py OUT_JSONL --image-dir D --ann-dir A --weights best.pt [--splits train,val] [--n N]
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import a5_common as C
import imcodecs as IC
from a5_fidelity import CODECS, CONF, in_ignore, match, read_gt
from profile_a11 import load

MODES = ["original"] + CODECS + ["thumb64_up", "thumb128_up"]


def detect_flip(model, img):
    W = img.shape[1]
    f = np.ascontiguousarray(img[:, ::-1])
    r = model.predict(np.ascontiguousarray(f[..., ::-1]), imgsz=960, conf=0.01, iou=0.7, max_det=500, verbose=False)[0]
    b = r.boxes
    o = np.argsort(-b.conf.cpu().numpy())
    xy = b.xyxy.cpu().numpy()[o]
    xy = np.stack([W - xy[:, 2], xy[:, 1], W - xy[:, 0], xy[:, 3]], 1)
    return xy, b.conf.cpu().numpy()[o], b.cls.cpu().numpy().astype(int)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ref", default=str(C.A5.parent / "a6l1" / "fidelity" / "per_image_val.jsonl"))
    args = ap.parse_args()
    from ultralytics import YOLO
    ref = {}
    for line in open(args.ref):
        r = json.loads(line)
        ref[(r["image"], r["mode"])] = r
    splits = set(args.splits.split(","))
    sel = [e for e in C.manifest()["images"] if e["source"] == "VisDrone2019-DET-val" and e["split"] in splits]
    if args.n:
        sel = sel[:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["image"] for l in open(out)}
    codecs = {m.name: m for m in IC.default_modes("cuda") if m.name in CODECS}
    assert len(codecs) == len(CODECS)
    model = YOLO(args.weights)
    fo = open(out, "a")
    t0 = time.time()
    for ii, e in enumerate(sel):
        name = e["name"]
        if name in done:
            continue
        img = load(Path(args.image_dir) / name, 0)
        H, W = img.shape[:2]
        g, gc, ig = read_gt(Path(args.ann_dir) / (Path(name).stem + ".txt"))
        recs = {"original": (0, img)}
        for mname, cdc in codecs.items():
            data = cdc.encode(img)
            recs[mname] = (len(data), cdc.decode(data))
        for S in (64, 128):
            th = np.asarray(Image.fromarray(img).resize((max(1, int(round(S * W / H))), S), Image.LANCZOS))
            buf = io.BytesIO()
            Image.fromarray(th).save(buf, "AVIF", quality=50, speed=6)
            tb = buf.getvalue()
            recs[f"thumb{S}_up"] = (len(tb), np.asarray(Image.open(io.BytesIO(tb)).convert("RGB").resize((W, H), Image.BICUBIC)))
        small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 32 ** 2
        for mname in MODES:
            nb, rec = recs[mname]
            r0 = ref[(name, mname)]
            assert nb == r0["bytes"], (name, mname, nb, r0["bytes"])
            bx, sc, cl = detect_flip(model, rec)
            keep = ~in_ignore(bx, ig)
            bx, sc, cl = bx[keep], sc[keep], cl[keep]
            op = sc >= CONF
            mg, md = match(bx[op], cl[op], g, gc)
            fo.write(json.dumps(dict(image=name, mode=mname, bytes=nb, n_gt=int(len(g)), tp=int(mg.sum()),
                                     n_small=int(small.sum()), tp_small=int((mg & small).sum()), n_det=int(op.sum()),
                                     tp_ref=r0["tp"], n_det_ref=r0["n_det"])) + "\n")
        fo.flush()
        if ii % 20 == 0:
            print(f"IMG {ii}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    fo.close()
    print("FLIP_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
