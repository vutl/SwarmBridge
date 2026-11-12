"""H1-CROP pilot 0B-TUNE precompute: the real UAV and BS models, run once per (epoch, view). No label is read here.

Per frame (native resolution):
  1. onboard YOLO11n (R1a VisDrone weights) @640, conf >= 0.05, NMS 0.7, max 500, CPU with ONE torch thread, timed
     after warm-up; classes mapped VisDrone -> MDMT (h1_pilot0b.VISDRONE_TO_MDMT);
  2. P1 windows and P2 tiles from the uncertain detections (h1_pilot0b);
  3. every window is cropped and AVIF-encoded at each q (Pillow / libavif, speed 6, 4:2:0, max_threads 1): real bits
     and encode ms; the full frame too at q = FRAME_Q;
  4. BS YOLO11s (A5 VisDrone weights, GPU) on every DECODED crop (imgsz = side rounded up to 32, max 640) and on the
     decoded frame (imgsz 960); boxes mapped back to image coordinates.
Output: OUT/seq_<seq>.pkl with records in the format documented in h1_pilot0b.py, plus provenance.

Sources:
  --source mdmt    MDMT zip through mdmt_adapter (split guard; images only with --rights-note)
  --source images  smoke test on arbitrary image files (each image = one epoch; view 2 = horizontal flip). No GT.
GPU: run under swarmbridge/experiments/uav3d_comm/gpulock_run.py.
"""
import argparse
import io
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import h1_pilot0b as P


def enc_dec(args):
    img, q = args
    t0 = time.perf_counter()
    b = io.BytesIO()
    Image.fromarray(img).save(b, "AVIF", quality=q, speed=6, subsampling="4:2:0", max_threads=1)
    ms = 1000 * (time.perf_counter() - t0)
    data = b.getvalue()
    return 8 * len(data), ms, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _warm_worker(qs):
    """first libavif call in a process is slow (smoke test: 83 ms vs 21 ms); warm every worker before timing."""
    x = np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    for q in qs:
        for _ in range(2):
            enc_dec((x, q))


def imgsz_for(h, w, cap=640):
    return int(min(cap, max(32, -(-max(h, w) // 32) * 32)))


def yolo(model, imgs, imgsz, device):
    """list of RGB arrays (same imgsz) -> list of mapped dets in the input's pixel coords."""
    out = []
    for i in range(0, len(imgs), 32):
        rs = model.predict([np.ascontiguousarray(x[..., ::-1]) for x in imgs[i:i + 32]], imgsz=imgsz, conf=0.05,
                           iou=0.7, max_det=500, verbose=False, device=device)
        for r in rs:
            b = r.boxes
            out.append(P.map_visdrone(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int) + 1, b.conf.cpu().numpy()))
    return out


class Pipeline:
    def __init__(self, onboard, server, qs, procs, server_device):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.on, self.sv = YOLO(onboard), YOLO(server)
        self.qs, self.dev = qs, server_device
        self.pool = Pool(procs, initializer=_warm_worker, initargs=(sorted(set(qs) | {P.FRAME_Q}),))
        self._warm = False

    def onboard(self, img):
        self.torch.set_num_threads(1)
        x = np.ascontiguousarray(img[..., ::-1])
        if not self._warm:
            for _ in range(3):
                self.on.predict(x, imgsz=640, conf=0.05, iou=0.7, max_det=500, verbose=False, device="cpu")
            self._warm = True
        t0 = time.perf_counter()
        r = self.on.predict(x, imgsz=640, conf=0.05, iou=0.7, max_det=500, verbose=False, device="cpu")[0]
        ms = 1000 * (time.perf_counter() - t0)
        b = r.boxes
        return P.map_visdrone(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int) + 1, b.conf.cpu().numpy()), ms

    def record(self, img):
        H, W = img.shape[:2]
        det, det_ms = self.onboard(img)
        cands = dict(P1=P.p1_candidates(det, W, H), P2=P.p2_candidates(det, W, H))
        jobs, where = [], []
        for kind in ("P1", "P2"):
            for k, c in enumerate(cands[kind]):
                x0, y0, x1, y1 = c["win"]
                for q in self.qs:
                    jobs.append((np.ascontiguousarray(img[y0:y1, x0:x1]), q)); where.append((kind, k, q))
        jobs.append((img, P.FRAME_Q)); where.append(("frame", 0, P.FRAME_Q))
        res = self.pool.map(enc_dec, jobs, chunksize=2)
        by_sz = {}
        for (kind, k, q), (bits, ms, dec) in zip(where, res):
            if kind == "frame":
                frame = dict(bits=bits, enc_ms=ms)
                fdec = dec
                continue
            c = cands[kind][k]
            c.setdefault("bits", {})[q] = bits; c.setdefault("enc_ms", {})[q] = ms; c.setdefault("bs", {})
            by_sz.setdefault(imgsz_for(*dec.shape[:2]), []).append((kind, k, q, dec))
        for sz, lst in by_sz.items():
            ds = yolo(self.sv, [x[3] for x in lst], sz, self.dev)
            for (kind, k, q, _), d in zip(lst, ds):
                cands[kind][k]["bs"][q] = P.to_global(d, cands[kind][k]["win"])
        frame["bs"] = yolo(self.sv, [fdec], 960, self.dev)[0]
        return dict(W=W, H=H, det=det, det_ms=det_ms, cands=cands, frame=frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--source", choices=["mdmt", "images"], required=True)
    ap.add_argument("--zip"); ap.add_argument("--rights-note"); ap.add_argument("--images", nargs="*")
    ap.add_argument("--seqs", help="comma list; default: config stage order"); ap.add_argument("--stage")
    ap.add_argument("--onboard", required=True); ap.add_argument("--server", required=True)
    ap.add_argument("--procs", type=int, default=5); ap.add_argument("--device", default="0")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(a.onboard, a.server, cfg["avif_q"], a.procs, a.device)
    meta = dict(onboard=a.onboard, server=a.server, q=cfg["avif_q"], cpu_count=os.cpu_count(),
                note="desktop CPU proxy, not UAV hardware")
    if a.source == "images":
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in a.images]
        recs = {1: [], 2: []}
        for im in imgs:
            recs[1].append(pipe.record(im)); recs[2].append(pipe.record(np.ascontiguousarray(im[:, ::-1])))
        R = dict(seq="smoke", frames=list(range(len(imgs))), recs=recs, provenance=dict(meta, source=a.images))
        (od / "seq_smoke.pkl").write_bytes(pickle.dumps(R, protocol=4))
        print("SMOKE_DONE", len(imgs)); return
    import mdmt_adapter as M
    ds = M.MDMT(a.zip, allowed=tuple(cfg["allowed_splits"]), rights_note=a.rights_note)
    st = cfg["stages"][a.stage]
    seqs = [int(s) for s in a.seqs.split(",")] if a.seqs else ds.pilot_order(st["split"])[:st["n_seq"]]
    for s in seqs:
        p = od / f"seq_{s}.pkl"
        if p.exists():
            continue
        frames = ds.epochs(s, cfg["epoch_step"], st["max_epochs"])
        recs = {1: [], 2: []}
        for f in frames:
            for v in (1, 2):
                recs[v].append(pipe.record(ds.image(s, v, f)))
        R = dict(seq=s, frames=frames, recs=recs, provenance=dict(meta, **ds.provenance(), stage=a.stage))
        p.write_bytes(pickle.dumps(R, protocol=4))
        print(s, len(frames), flush=True)
    print("PRECOMP_DONE")


if __name__ == "__main__":
    main()
