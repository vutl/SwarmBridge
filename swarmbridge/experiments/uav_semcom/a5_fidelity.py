"""A5.2 task fidelity (SECONDARY evaluation; the reward is unchanged): does an image decoded from mode m still let the
ground station find the objects?

For every image of a pool (manifest) and every mode: real encode -> real decode (same codecs and settings as the
mode table; payload compared with the profile) -> the A5 detector (a5_det.py: YOLO11s trained on VisDrone-train only,
fixed weights, imgsz 960, conf >= 0.01, NMS IoU 0.7, max 500 detections) -> scored against the VisDrone labels of the ORIGINAL
image (classes 1-10; 'ignored regions' and 'others' are crowd/ignore boxes for every class).
  * pool level (pycocotools, maxDets 500 as the VisDrone toolkit): AP@[.5:.95], AP50, AP_small (< 32^2 px), AR500,
    AR_small, per mode and per subset (all / A5 test split / ...);
  * image level, operating point conf >= 0.25, IoU >= 0.5, class-aware greedy matching: recall, small-object
    recall, precision, and agreement with the detections on the ORIGINAL frame (F1 of decoded vs original detections)
    -> per_image.jsonl, joined later with the (image, mode) pairs each policy delivers.
'original' (no compression) is a mode here. Generative modes (BLIP caption + SD-Turbo, as profile_a12) only with
--gen N (first N images of the pool, generator seed = manifest index).
Usage: python a5_fidelity.py OUT_DIR --pool val|testdev --image-dir D --ann-dir A --weights best.pt [--gen 100]
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import a5_common as C
import imcodecs as IC
from profile_a11 import load

CODECS = ["avif_5", "avif_20", "avif_50", "avif_80", "mbt2018-mean_q1", "mbt2018-mean_q3", "mbt2018-mean_q5",
          "mbt2018-mean_q8", "jpeg2000_100", "jpeg2000_25"]
GEN = ["cap_t2i", "cap_thumb64_i2i_t0.6", "cap_thumb128_i2i_t0.3"]
CONF, IOU = 0.25, 0.5


def read_gt(path):
    """VisDrone annotation -> (boxes xyxy, classes 0..9), ignore boxes xyxy."""
    gt, cls, ign = [], [], []
    for l in Path(path).read_text().splitlines():
        f = [int(x) for x in l.strip().strip(",").split(",")[:6]]
        x, y, w, h, score, cat = f
        if w <= 0 or h <= 0:
            continue
        if score == 0 or cat in (0, 11):
            ign.append([x, y, x + w, y + h])
        else:
            gt.append([x, y, x + w, y + h])
            cls.append(cat - 1)
    return np.array(gt, float).reshape(-1, 4), np.array(cls, int), np.array(ign, float).reshape(-1, 4)


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def in_ignore(det, ign):
    """Detections with >= 50 % of their area inside an ignored region."""
    if len(det) == 0 or len(ign) == 0:
        return np.zeros(len(det), bool)
    x1 = np.maximum(det[:, None, 0], ign[None, :, 0]); y1 = np.maximum(det[:, None, 1], ign[None, :, 1])
    x2 = np.minimum(det[:, None, 2], ign[None, :, 2]); y2 = np.minimum(det[:, None, 3], ign[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) + 1e-9
    return (inter / area[:, None]).max(1) >= 0.5


def match(det, dcls, gt, gcls):
    """Greedy class-aware matching (detections in score order, given sorted) -> matched gt mask, matched det mask."""
    mg, md = np.zeros(len(gt), bool), np.zeros(len(det), bool)
    M = iou(det, gt)
    for i in range(len(det)):
        if len(gt) == 0:
            break
        cand = np.where((gcls == dcls[i]) & ~mg & (M[i] >= IOU))[0]
        if len(cand):
            j = cand[np.argmax(M[i, cand])]
            mg[j] = md[i] = True
    return mg, md


def detect(model, img):
    r = model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=960, conf=0.01, iou=0.7, max_det=500,
                      verbose=False)[0]
    b = r.boxes
    order = np.argsort(-b.conf.cpu().numpy())
    return (b.xyxy.cpu().numpy()[order], b.conf.cpu().numpy()[order], b.cls.cpu().numpy().astype(int)[order])


def coco_eval(images, gts, dets_by_mode, subset):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = [i for i, n in enumerate(images) if n in subset]
    cats = [dict(id=c + 1, name=str(c)) for c in range(10)]
    anns, aid = [], 1
    for i in ids:
        g, gc, ig = gts[images[i]]
        for bx, c in zip(g, gc):
            anns.append(dict(id=aid, image_id=i, category_id=int(c) + 1, bbox=[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                             area=float((bx[2] - bx[0]) * (bx[3] - bx[1])), iscrowd=0)); aid += 1
        for bx in ig:
            for c in range(10):
                anns.append(dict(id=aid, image_id=i, category_id=c + 1, bbox=[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                                 area=float((bx[2] - bx[0]) * (bx[3] - bx[1])), iscrowd=1)); aid += 1
    gt = COCO()
    gt.dataset = dict(images=[dict(id=i) for i in ids], annotations=anns, categories=cats)
    gt.createIndex()
    out, idset = {}, set(ids)
    for mode, dets in dets_by_mode.items():
        arr = np.concatenate(dets)
        res = [dict(image_id=int(r[0]), bbox=[float(x) for x in r[1:5]], score=float(r[5]), category_id=int(r[6]))
               for r in arr if int(r[0]) in idset]
        if not res:
            continue
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = ids
        E.params.maxDets = [1, 100, 500]
        E.evaluate(); E.accumulate(); E.summarize()
        s = E.stats
        out[mode] = dict(AP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AR500=s[8], AR_small=s[9],
                         n_images=len(ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pool", required=True, choices=["val", "testdev"])
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="first N images only (smoke test)")
    args = ap.parse_args()
    from ultralytics import YOLO
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = C.manifest()["images"]
    sel = [e for e in man if (e["source"] == "VisDrone2019-DET-val" if args.pool == "val" else e["split"] == "testdev")]
    if args.n:
        sel = sel[:args.n]
    man_idx = {e["name"]: i for i, e in enumerate(man)}
    prof = {}
    for p in (C.A5 / "profile" / f"a11_{args.pool}" / "profile.jsonl",):
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                prof[(r["image"], r["mode"])] = r["bytes"]
    dev = "cuda"
    codecs = {m.name: m for m in IC.default_modes(dev) if m.name in CODECS}
    assert len(codecs) == len(CODECS)
    model = YOLO(args.weights)
    gen = None
    if args.gen:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        from transformers import BlipForConditionalGeneration, BlipProcessor
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                            torch_dtype=torch.float16).to(dev).eval()
        t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                        variant="fp16").to(dev)
        t2i.set_progress_bar_config(disable=True)
        i2i = AutoPipelineForImage2Image.from_pipe(t2i)
        gen = (proc, blip, t2i, i2i)
    images, gts, dets = [], {}, {}
    fo = open(out / f"per_image_{args.pool}.jsonl", "w")
    t0 = time.time()
    for ii, e in enumerate(sel):
        name = e["name"]
        img = load(Path(args.image_dir) / name, 0)
        H, W = img.shape[:2]
        g, gc, ig = read_gt(Path(args.ann_dir) / (Path(name).stem + ".txt"))
        images.append(name)
        gts[name] = (g, gc, ig)
        recs = {"original": (0, img)}
        for mname, cdc in codecs.items():
            data = cdc.encode(img)
            recs[mname] = (len(data), cdc.decode(data))
        for S in (64, 128):
            th = np.asarray(Image.fromarray(img).resize((max(1, int(round(S * W / H))), S), Image.LANCZOS))
            buf = io.BytesIO(); Image.fromarray(th).save(buf, "AVIF", quality=50, speed=6); tb = buf.getvalue()
            recs[f"thumb{S}_up"] = (len(tb), np.asarray(Image.open(io.BytesIO(tb)).convert("RGB").resize((W, H), Image.BICUBIC)))
            if gen and ii < args.gen:
                recs[f"_thumb{S}_bytes"] = (len(tb), tb)
        if gen and ii < args.gen:
            proc, blip, t2i, i2i = gen
            gh, gw = 512, int(round(512 * W / H / 8)) * 8
            inp = proc(images=Image.fromarray(img), return_tensors="pt").to(dev, torch.float16)
            cap = proc.decode(blip.generate(**inp, max_new_tokens=40, num_beams=3)[0], skip_special_tokens=True).strip()
            seed = man_idx[name]
            r = t2i(prompt=cap, num_inference_steps=4, guidance_scale=0.0, height=gh, width=gw,
                    generator=torch.Generator(dev).manual_seed(seed)).images[0]
            recs["cap_t2i"] = (len(cap.encode()), np.asarray(r.resize((W, H), Image.BICUBIC)))
            for S, T in ((64, 0.6), (128, 0.3)):
                nb, tb = recs.pop(f"_thumb{S}_bytes")
                th = Image.open(io.BytesIO(tb)).convert("RGB").resize((gw, gh), Image.BICUBIC)
                r = i2i(prompt=cap, image=th, strength=T, num_inference_steps=max(4, int(np.ceil(1 / T))),
                        guidance_scale=0.0, generator=torch.Generator(dev).manual_seed(seed)).images[0]
                recs[f"cap_thumb{S}_i2i_t{T}"] = (nb + len(cap.encode()), np.asarray(r.resize((W, H), Image.BICUBIC)))
        recs = {k: v for k, v in recs.items() if not k.startswith("_")}
        base_det = None
        for mname, (nb, rec) in recs.items():
            bx, sc, cl = detect(model, rec)
            keep = ~in_ignore(bx, ig)
            bx, sc, cl = bx[keep], sc[keep], cl[keep]
            k01 = sc >= 0.01               # pool-level AP: detections down to conf 0.01, stored compactly
            dets.setdefault(mname, []).append(np.column_stack([np.full(k01.sum(), ii), bx[k01, :2],
                                                               bx[k01, 2:] - bx[k01, :2], sc[k01], cl[k01] + 1]))
            op = sc >= CONF
            mg, md = match(bx[op], cl[op], g, gc)
            small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 32 ** 2
            if mname == "original":
                base_det = (bx[op], cl[op])
            ag, _ = match(bx[op], cl[op], *base_det)
            n_b, n_d = len(base_det[0]), int(op.sum())
            mse = float(np.mean((rec.astype(np.float64) - img.astype(np.float64)) ** 2))
            fo.write(json.dumps(dict(
                image=name, mode=mname, bytes=nb, profile_bytes=prof.get((name, mname)),
                psnr=(10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")),
                n_gt=int(len(g)), tp=int(mg.sum()), n_small=int(small.sum()), tp_small=int((mg & small).sum()),
                n_det=n_d, recall=float(mg.mean()) if len(g) else None,
                recall_small=float(mg[small].mean()) if small.any() else None,
                precision=float(md.mean()) if n_d else None,
                agree_f1=float(2 * ag.sum() / (n_b + n_d)) if n_b + n_d else 1.0)) + "\n")
        fo.flush()
        if ii % 20 == 0:
            print(f"IMG {ii}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    fo.close()
    splits = {e["name"]: e["split"] for e in man}
    subsets = {"all": set(images)}
    if args.pool == "val":
        subsets["test"] = {n for n in images if splits[n] == "test"}
        subsets["in_a4"] = {n for n in images if any(e["name"] == n and e["in_a4"] for e in sel)}
    if args.gen:
        subsets["gen_subset"] = set(images[:args.gen])
    res = {}
    for sname, sub in subsets.items():
        dsub = {m: d for m, d in dets.items() if sname == "gen_subset" or m not in GEN}
        res[sname] = coco_eval(images, gts, dsub, sub)
    (out / f"coco_{args.pool}.json").write_text(json.dumps(dict(pool=args.pool, n=len(images), weights=args.weights,
                                                                conf_op=CONF, iou_match=IOU, results=res), indent=1))
    for sname, r in res.items():
        for m, s in r.items():
            print(f"{sname:10s} {m:24s} AP {s['AP']:.3f} AP50 {s['AP50']:.3f} APs {s['AP_small']:.3f} "
                  f"AR {s['AR500']:.3f} ARs {s['AR_small']:.3f}", flush=True)
    print("FIDELITY_DONE", time.time() - t0)


if __name__ == "__main__":
    main()
