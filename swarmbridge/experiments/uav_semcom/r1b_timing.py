"""R1b-0a onboard compute timing on VID TUNE frames (CPU only; UAV proxy = 6-thread desktop, NOT UAV hardware).

One frame (the middle one) from each of the first N TUNE clips of split_v1, resized to long side LONG px (protocol
processing resolution). Measures, median of N after 3 warm-up runs:
  * YOLO11n (onboard weights) at imgsz 640 / 960, torch threads = all and 1;
  * AVIF encode (Pillow / libavif, speed 6, 4:2:0) at qualities Q, and the payload bits;
  * AVIF decode.
Usage: CUDA_VISIBLE_DEVICES= python r1b_timing.py SPLIT_JSON VID_ROOT ONBOARD_PT OUT_JSON [--n 30] [--long 1344]
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("vid_root"); ap.add_argument("onboard"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--long", type=int, default=1344)
    ap.add_argument("--q", default="20,35,50")
    a = ap.parse_args()
    from ultralytics import YOLO
    S = json.loads(Path(a.split).read_text())
    imgs = []
    for c in S["tune"][:a.n]:
        fr = sorted((Path(a.vid_root) / "VisDrone2019-VID-train/sequences" / c).glob("*.jpg"))
        im = Image.open(fr[len(fr) // 2]).convert("RGB")
        s = a.long / max(im.size)
        imgs.append(np.asarray(im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)))
    model = YOLO(a.onboard)
    res = dict(n=len(imgs), long=a.long, cpu_count=os.cpu_count(), note="desktop CPU proxy, not UAV hardware")

    def med(fn, items):
        for x in items[:3]:
            fn(x)
        ts = []
        for x in items:
            t0 = time.perf_counter(); fn(x); ts.append(1000 * (time.perf_counter() - t0))
        return float(np.median(ts)), float(np.percentile(ts, 90))

    for th in (os.cpu_count(), 1):
        torch.set_num_threads(th)
        for sz in (640, 960):
            m, p = med(lambda x: model.predict(np.ascontiguousarray(x[..., ::-1]), imgsz=sz, conf=0.05, iou=0.7,
                                               max_det=500, verbose=False, device="cpu"), imgs)
            res[f"yolo11n_{sz}_cpu{th}_ms"] = dict(median=m, p90=p)
    torch.set_num_threads(os.cpu_count())
    for q in [int(x) for x in a.q.split(",")]:
        enc = []

        def e(x, q=q):
            b = io.BytesIO(); Image.fromarray(x).save(b, "AVIF", quality=q, speed=6, subsampling="4:2:0")
            enc.append(b.getvalue())
        m, p = med(e, imgs)
        bits = [8 * len(b) for b in enc[3:]]
        dm, dp = med(lambda b: np.asarray(Image.open(io.BytesIO(b)).convert("RGB")), enc[3:])
        res[f"avif_{q}"] = dict(encode_ms_median=m, encode_ms_p90=p, decode_ms_median=dm, bits_median=float(np.median(bits)),
                                bits_p90=float(np.percentile(bits, 90)), bits_mean=float(np.mean(bits)))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
