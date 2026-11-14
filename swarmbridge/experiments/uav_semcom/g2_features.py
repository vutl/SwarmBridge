"""G-L2 step 3 (reports/archive/r1-visdrone/uav-semcom-g-l2.md 0.5): on-board features of every manifest image (computed from the
ORIGINAL frame, before sending; no labels).

  B  cheap statistics on the grey frame resized to width 480: gradient magnitude mean / std, edge density, Laplacian
     variance, FFT energy share in 3 radial bands, mean / std of 8x8 block std, grey-level entropy, colourfulness;
  C  the A5 detector at imgsz 480 on the original frame (proxy of an on-board detector; SAME weights as the ground
     station -> optimistic): detection counts at conf >= 0.25 / 0.1, small boxes (< 32^2 px), confidence stats,
     box-size stats, class-group counts.
Timing: B on CPU (1 thread, numpy); C on GPU for every image, on CPU (torch threads 1 and all) for the first --cpu-n
images. Output: OUT.jsonl (one row per image) and OUT_timing.json.
Usage: python g2_features.py OUT --weights best.pt --val-dir D --testdev-dir D2 [--cpu-n 20]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import a5_common as C
from profile_a11 import load

GROUPS = {"person": (0, 1), "vehicle": (3, 4, 5, 8), "two_wheel": (2, 6, 7, 9)}


def cheap(img):
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    w, h = 480, max(8, int(round(H * 480 / W)))
    s = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), np.float64) / 255.0
    g = s @ np.array([0.299, 0.587, 0.114])
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    mag = np.hypot(gx, gy)
    lap = g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1]
    P = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None]
    fx = np.fft.rfftfreq(g.shape[1])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    tot = P[r > 0].sum() + 1e-12
    bands = [float(P[(r > a) & (r <= b)].sum() / tot) for a, b in ((0, 0.125), (0.125, 0.25), (0.25, 1.5))]
    hb, wb = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    blk = g[:hb, :wb].reshape(hb // 8, 8, wb // 8, 8).std(axis=(1, 3))
    hist = np.histogram(g, 64, (0, 1))[0] / g.size
    ent = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    R, Gc, Bc = s[..., 0], s[..., 1], s[..., 2]
    rg, yb = R - Gc, 0.5 * (R + Gc) - Bc
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    f = dict(grad_mean=float(mag.mean()), grad_std=float(mag.std()), edge_density=float((mag > 0.1).mean()),
             lap_var=float(lap.var()), fft_low=bands[0], fft_mid=bands[1], fft_high=bands[2],
             blk_std_mean=float(blk.mean()), blk_std_std=float(blk.std()), entropy=ent, colorful=colorful,
             log_pixels=float(np.log(H * W)), aspect=float(W / H))
    return f, time.perf_counter() - t0


def det_feats(model, img, device):
    t0 = time.perf_counter()
    r = model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=480, conf=0.05, iou=0.7, max_det=500,
                      verbose=False, device=device)[0]
    b = r.boxes
    sc, cl, xy = b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.xyxy.cpu().numpy()
    dt = time.perf_counter() - t0
    area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1]) if len(xy) else np.zeros(0)
    k25, k10 = sc >= 0.25, sc >= 0.10
    f = dict(n25=int(k25.sum()), n10=int(k10.sum()), n05=int(len(sc)), small25=int((k25 & (area < 32 ** 2)).sum()),
             small10=int((k10 & (area < 32 ** 2)).sum()), conf_sum10=float(sc[k10].sum()),
             conf_mean10=float(sc[k10].mean()) if k10.any() else 0.0,
             conf_q90=float(np.quantile(sc, 0.9)) if len(sc) else 0.0,
             logarea_med25=float(np.median(np.log(area[k25] + 1))) if k25.any() else 0.0,
             logarea_q10_25=float(np.quantile(np.log(area[k25] + 1), 0.1)) if k25.any() else 0.0,
             mid_conf10=int(((sc >= 0.10) & (sc < 0.40)).sum()))
    for gname, cs in GROUPS.items():
        f[f"n25_{gname}"] = int((k25 & np.isin(cl, cs)).sum())
    return f, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--testdev-dir", required=True)
    ap.add_argument("--cpu-n", type=int, default=20)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    man = C.manifest()["images"]
    if a.n:
        man = man[:a.n]
    gpu = YOLO(a.weights)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, tB, tC = [], [], []
    warm = None
    for k, e in enumerate(man):
        d = a.testdev_dir if e["split"] == "testdev" else a.val_dir
        img = load(Path(d) / e["name"], 0)
        if warm is None:
            det_feats(gpu, img, 0)
            warm = True
        fb, dtb = cheap(img)
        fc, dtc = det_feats(gpu, img, 0)
        tB.append(dtb)
        tC.append(dtc)
        rows.append(dict(image=e["name"], split=e["split"], seq=e["seq"], B=fb, C=fc))
        if k % 100 == 0:
            print(f"IMG {k}/{len(man)}", flush=True)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    timing = dict(B_cpu_ms_median=float(np.median(tB) * 1e3), C_gpu_ms_median=float(np.median(tC) * 1e3),
                  gpu=torch.cuda.get_device_name(0), cpu=os.cpu_count())
    cpu = YOLO(a.weights)
    for th in (1, os.cpu_count()):
        torch.set_num_threads(th)
        ts = []
        for k, e in enumerate(man[:a.cpu_n + 1]):
            d = a.testdev_dir if e["split"] == "testdev" else a.val_dir
            img = load(Path(d) / e["name"], 0)
            _, dt = det_feats(cpu, img, "cpu")
            if k:
                ts.append(dt)
        timing[f"C_cpu_threads{th}_ms_median"] = float(np.median(ts) * 1e3)
    n_par = sum(p.numel() for p in gpu.model.parameters())
    timing["C_params"] = int(n_par)
    Path(str(out) + "_timing.json").write_text(json.dumps(timing, indent=1))
    print("TIMING", json.dumps(timing), flush=True)
    print("FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main()
