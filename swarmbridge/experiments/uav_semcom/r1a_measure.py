"""R1a measurements (reports/swarmbridge-r1a.md 0.3 / 0.4), GPU, every image of the A5 pools:

  * onboard YOLO11n detections at imgsz 640 and 960 (conf >= 0.05, NMS 0.7, max 500), original-image coordinates;
  * every 320 px tile (stride 288, last tile aligned to the border) encoded with the A5 avif_50 / avif_80 codecs
    (bytes, encode time), decoded, and detected by the BS YOLO11s at the SAME scale as the full frame
    (imgsz = 320 * 960 / long side, rounded to a multiple of 32), detections conf >= 0.01 shifted to image
    coordinates. Any tile subset (onboard-uncertainty rule or GT oracle) is then evaluated offline;
  * timing (run with --timing on an idle machine): onboard detector CPU (1 thread / all threads) and GPU, BS
    detector on a full frame and on a tile (GPU).
Output: OUT/measure.pkl (detections) and OUT/timing.json. Resumable per pool chunk is not needed (about 20 min).
Usage: python r1a_measure.py OUT --onboard best_n.pt --server best_s.pt --val-dir D --testdev-dir D2 [--n N]
       python r1a_measure.py OUT --timing --onboard .. --server .. --val-dir D --testdev-dir D2
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

import a5_common as C
import imcodecs as IC
from profile_a11 import load

TILE, STRIDE = 320, 288
POOLS = ("train", "val", "test", "testdev")


def tiles(W, H):
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE))
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE))
    if xs[-1] + TILE < W:
        xs.append(W - TILE)
    if ys[-1] + TILE < H:
        ys.append(H - TILE)
    return [(x, y, min(x + TILE, W), min(y + TILE, H)) for y in ys for x in xs]


def detect(model, img, imgsz, conf, device=0):
    r = model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=imgsz, conf=conf, iou=0.7, max_det=500,
                      verbose=False, device=device)[0]
    b = r.boxes
    return np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()]).astype(np.float32)


def tile_imgsz(W, H):
    return max(32, int(round(TILE * 960 / max(W, H) / 32)) * 32)


def images():
    man = C.manifest()["images"]
    return [e for e in man if e["split"] in POOLS]


def path_of(e, a):
    return Path(a.testdev_dir if e["split"] == "testdev" else a.val_dir) / e["name"]


def measure(a):
    from ultralytics import YOLO
    on, sv = YOLO(a.onboard), YOLO(a.server)
    codecs = {m.name: m for m in IC.default_modes("cuda") if m.name in ("avif_50", "avif_80")}
    sel = images()
    if a.n:     # smoke: first N images of every pool
        sel = [e for p in POOLS for e in [x for x in sel if x["split"] == p][:a.n]]
    out = {}
    t0 = time.time()
    for k, e in enumerate(sel):
        img = load(path_of(e, a), 0)
        H, W = img.shape[:2]
        rec = dict(W=W, H=H, onboard={u: detect(on, img, u, 0.05) for u in (640, 960)}, tiles=[])
        ts = tile_imgsz(W, H)
        for (x0, y0, x1, y1) in tiles(W, H):
            crop = np.ascontiguousarray(img[y0:y1, x0:x1])
            t = dict(box=(x0, y0, x1, y1))
            for q, cdc in codecs.items():
                s = time.perf_counter()
                data = cdc.encode(crop)
                enc = time.perf_counter() - s
                dec = cdc.decode(data)
                d = detect(sv, dec, ts, 0.01)
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
                t[q] = dict(bytes=len(data), enc_s=enc, dets=d)
            rec["tiles"].append(t)
        out[e["name"]] = rec
        if k % 50 == 0:
            print(f"IMG {k}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(tile=TILE, stride=STRIDE, images=out), open(Path(a.out) / "measure.pkl", "wb"))
    print("MEASURE_DONE", time.time() - t0, flush=True)


def timing(a):
    import torch
    from ultralytics import YOLO
    sel = images()[:a.n_time + 1]
    imgs = [load(path_of(e, a), 0) for e in sel]
    res = dict(cpu=os.cpu_count(), gpu=torch.cuda.get_device_name(0), n=a.n_time)
    on_g, sv_g = YOLO(a.onboard), YOLO(a.server)
    for u in (640, 960):
        ts = []
        for k, im in enumerate(imgs):
            s = time.perf_counter(); detect(on_g, im, u, 0.05, 0); ts.append(time.perf_counter() - s)
        res[f"onboard{u}_gpu_ms"] = float(np.median(ts[1:]) * 1e3)
    for th in (1, os.cpu_count()):
        torch.set_num_threads(th)
        on_c = YOLO(a.onboard)
        for u in (640, 960):
            ts = []
            for k, im in enumerate(imgs):
                s = time.perf_counter(); detect(on_c, im, u, 0.05, "cpu"); ts.append(time.perf_counter() - s)
            res[f"onboard{u}_cpu{th}_ms"] = float(np.median(ts[1:]) * 1e3)
    ts, tt = [], []
    for im in imgs:
        s = time.perf_counter(); detect(sv_g, im, 960, 0.01, 0); ts.append(time.perf_counter() - s)
        H, W = im.shape[:2]
        x0, y0, x1, y1 = tiles(W, H)[0]
        s = time.perf_counter(); detect(sv_g, np.ascontiguousarray(im[y0:y1, x0:x1]), tile_imgsz(W, H), 0.01, 0)
        tt.append(time.perf_counter() - s)
    res["server_frame_gpu_ms"] = float(np.median(ts[1:]) * 1e3)
    res["server_tile_gpu_ms"] = float(np.median(tt[1:]) * 1e3)
    res["onboard_params"] = int(sum(p.numel() for p in on_g.model.parameters()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "timing.json").write_text(json.dumps(res, indent=1))
    print("TIMING", json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--onboard", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--testdev-dir", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--n-time", type=int, default=30)
    a = ap.parse_args()
    (timing if a.timing else measure)(a)


if __name__ == "__main__":
    main()
