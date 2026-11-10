"""R1b-0a precompute (protocol R1b-0a sec 2 / 6 / 11): per clip of one split set, at replay ticks f (native frame
first + 3 f), frames resized to long side LONG:
  * GT: classes 1..10 with score 1 -> (target_id, xyxy, cls); class 0 / 11 or score 0 -> ignored regions;
  * det_n: onboard YOLO11n (imgsz 640, conf >= 0.05, NMS 0.7, max 500), classes mapped 0..9 -> 1..10;
  * for each AVIF quality q: payload bits, encode ms (CPU, Pillow / libavif speed 6, 4:2:0), BS YOLO11s (imgsz 960,
    conf >= 0.05) on the DECODED frame;
  * --orig: BS YOLO11s on the uncompressed resized frame (TUNE only, for the FRAME-quality lock rule).
Output: OUT/<set>/<clip>.pkl (outside git) + a summary JSON. GPU: run under gpulock_run.py.
Usage: python r1b_precompute.py SPLIT VID_ROOT OUT --set tune --onboard N.pt --server S.pt --q 20,35 [--orig]
"""
import argparse
import io
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

LONG = 1344


def load(path):
    im = Image.open(path).convert("RGB")
    s = LONG / max(im.size)
    return np.asarray(im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)), s


def enc_dec(args):
    img, q = args
    t0 = time.perf_counter()
    b = io.BytesIO(); Image.fromarray(img).save(b, "AVIF", quality=q, speed=6, subsampling="4:2:0")
    ms = 1000 * (time.perf_counter() - t0)
    data = b.getvalue()
    return 8 * len(data), ms, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def detect(model, imgs, imgsz):
    out = []
    for i in range(0, len(imgs), 16):
        rs = model.predict([np.ascontiguousarray(x[..., ::-1]) for x in imgs[i:i + 16]], imgsz=imgsz, conf=0.05,
                           iou=0.7, max_det=500, verbose=False, device=0)
        for r in rs:
            b = r.boxes
            xy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int) + 1
            out.append([(xy[k].astype(np.float32), int(cl[k]), float(cf[k])) for k in range(len(cf))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("vid_root"); ap.add_argument("out")
    ap.add_argument("--set", required=True, choices=["tune", "gate", "aux", "quarantine"])
    ap.add_argument("--onboard", required=True); ap.add_argument("--server", required=True)
    ap.add_argument("--q", default="20"); ap.add_argument("--orig", action="store_true")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    from ultralytics import YOLO
    S = json.loads(Path(a.split).read_text())
    clips = {"tune": S["tune"], "gate": S["gate"], "aux": S["aux_val"], "quarantine": S["quarantine"]}[a.set]
    sp = "val" if a.set == "aux" else "train"
    qs = [int(x) for x in a.q.split(",")]
    od = Path(a.out) / a.set; od.mkdir(parents=True, exist_ok=True)
    mn, ms = YOLO(a.onboard), YOLO(a.server)
    summ = dict(set=a.set, long=LONG, q=qs, clips={})
    with Pool(a.procs) as pool:
        for c in clips:
            p = od / f"{c}.pkl"
            if p.exists():
                summ["clips"][c] = pickle.loads(p.read_bytes())["summary"]; continue
            root = Path(a.vid_root) / f"VisDrone2019-VID-{sp}"
            frames = sorted((root / "sequences" / c).glob("*.jpg"))
            idx = [int(f.stem) for f in frames]
            first = min(idx); by = {int(f.stem): f for f in frames}
            ticks = list(range(first, max(idx) + 1, 3))
            A = np.loadtxt(root / "annotations" / f"{c}.txt", delimiter=",", dtype=np.int64, ndmin=2)
            img0, s = load(by[ticks[0]])
            H, W = img0.shape[:2]
            gt, ign = [], []
            for k in ticks:
                R = A[A[:, 0] == k]
                g, ig = [], []
                for fi, tid, l, t, w, h, sc, cat, _, _ in R:
                    box = np.array([l, t, l + w, t + h], np.float32) * s
                    if 1 <= cat <= 10 and sc == 1:
                        g.append((int(tid), box, int(cat)))
                    elif cat in (0, 11) or sc == 0:
                        ig.append(box)
                gt.append(g); ign.append(ig)
            rec = dict(clip=c, W=W, H=H, scale=s, native_ticks=ticks, gt=gt, ign=ign, det_n=[],
                       bits={q: [] for q in qs}, enc_ms={q: [] for q in qs}, det_s={q: [] for q in qs})
            if a.orig:
                rec["det_s_orig"] = []
            for i0 in range(0, len(ticks), 64):           # bounded memory
                imgs = [load(by[k])[0] for k in ticks[i0:i0 + 64]]
                assert all(x.shape[:2] == (H, W) for x in imgs)
                rec["det_n"] += detect(mn, imgs, 640)
                for q in qs:
                    res = pool.map(enc_dec, [(x, q) for x in imgs], chunksize=4)
                    rec["bits"][q] += [r[0] for r in res]; rec["enc_ms"][q] += [r[1] for r in res]
                    rec["det_s"][q] += detect(ms, [r[2] for r in res], 960)
                if a.orig:
                    rec["det_s_orig"] += detect(ms, imgs, 960)
            rec["summary"] = dict(n_ticks=len(ticks), W=W, H=H, n_gt=int(sum(len(g) for g in gt)),
                                  bits_mean={q: float(np.mean(rec["bits"][q])) for q in qs},
                                  enc_ms_median={q: float(np.median(rec["enc_ms"][q])) for q in qs})
            p.write_bytes(pickle.dumps(rec, protocol=4))
            summ["clips"][c] = rec["summary"]
            print(c, rec["summary"], flush=True)
    Path(a.out, f"precomp_{a.set}.json").write_text(json.dumps(summ, indent=1, default=str))
    print("PRECOMP_DONE", a.set)


if __name__ == "__main__":
    main()
