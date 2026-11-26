"""paperA_ix profiling (pre-registration §2–§4): the fd_v1 pipeline (fd_profile.Pipe: same 14 modes, codecs, generator,
metrics, native resolution, one pass) on the paperA_ix manifest, plus the UAVDT vehicle-recall fidelity pass.

Stages (separate output files under WORK/profile, outside git; resumable with the fd_v1 cache rule):
  A      : U_train + U_val, profile only (stage A of the protocol; no recall);
  uavdt  : U_test, profile + frozen-YOLO fidelity (recall variants); REFUSES to start unless the L2 lock verifies;
  x1     : X1 (VisDrone test-challenge), profile only (no labels); REFUSES to start unless the L2 lock verifies.
Each image is checked against its manifest sha256 before use. Printed output is timing / counts only: no payload,
PSNR or recall value of a test image is printed by this script.

UAVDT fidelity (prereg §3): detector as fd_v1 (a5_fidelity.detect: imgsz 960, conf 0.01, NMS 0.7, max 500; operating
conf >= 0.25, IoU >= 0.5, greedy one-to-one by descending confidence, a5_fidelity.match). Detections with >= 50 % of
their area inside a UAVDT ignore region are removed; GT boxes with >= 50 % of their area inside an ignore region are
removed from the denominator; every other gt_whole box counts (any occlusion / out-of-view level). Variants:
  veh        : class-agnostic; predictions of VisDrone car / van / truck / bus; every eligible GT box  (primary)
  cls_vancar : class-aware; car <- {car, van}, truck <- truck, bus <- bus
  cls_car    : class-aware; car <- car, truck <- truck, bus <- bus (van predictions discarded)
Row fields (U_test): tp_<v>, n_gt_<v>, n_det_<v> per mode and for the 'original' row.

  python ix_profile.py run A|uavdt|x1 [--limit N]     (GPU; launch through fd_gpulock_win.py)"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import a5_common as C
import a5_fidelity as AF
import fd_profile as FP
import ix_common as IX
from ix_audit import TC, UAVDT, UGT
from profile_a11 import hardware, load, timed

HERE = Path(__file__).resolve().parent
# VisDrone detector classes (a5_fidelity.read_gt: category - 1): 3 car, 4 van, 5 truck, 8 bus
VD_CAR, VD_VAN, VD_TRUCK, VD_BUS = 3, 4, 5, 8
MAPS = {"veh": {VD_CAR: 0, VD_VAN: 0, VD_TRUCK: 0, VD_BUS: 0},
        "cls_vancar": {VD_CAR: 0, VD_VAN: 0, VD_TRUCK: 1, VD_BUS: 2},
        "cls_car": {VD_CAR: 0, VD_TRUCK: 1, VD_BUS: 2}}
GT_MAPS = {"veh": {0: 0, 1: 0, 2: 0}, "cls_vancar": {0: 0, 1: 1, 2: 2}, "cls_car": {0: 0, 1: 1, 2: 2}}


# ---------------------------------------------------------------- UAVDT ground truth and recall (pure; unit-tested)
def read_uavdt_gt(seq, frame, gt_dir=UGT):
    """-> GT boxes xyxy (float), classes 0 car / 1 truck / 2 bus, ignore boxes xyxy, for one frame."""
    g, c, ig = [], [], []
    for line in (Path(gt_dir) / f"{seq}_gt_whole.txt").read_text().splitlines():
        f = line.strip().split(",")
        if len(f) >= 9 and int(f[0]) == frame:
            x, y, w, h = (float(v) for v in f[2:6])
            if w > 0 and h > 0:
                g.append([x, y, x + w, y + h])
                c.append(int(f[8]) - 1)
    for line in (Path(gt_dir) / f"{seq}_gt_ignore.txt").read_text().splitlines():
        f = line.strip().split(",")
        if len(f) >= 6 and int(f[0]) == frame:
            x, y, w, h = (float(v) for v in f[2:6])
            if w > 0 and h > 0:
                ig.append([x, y, x + w, y + h])
    return np.array(g, float).reshape(-1, 4), np.array(c, int), np.array(ig, float).reshape(-1, 4)


def recall_variants(bx, sc, cl, g, gc, ig):
    """bx / sc / cl: detections sorted by descending score (a5_fidelity.detect). -> {variant: (tp, n_gt, n_det)}."""
    keep = ~AF.in_ignore(bx, ig)
    bx, sc, cl = bx[keep], sc[keep], cl[keep]
    op = sc >= AF.CONF
    bx, cl = bx[op], cl[op]
    g_ok = ~AF.in_ignore(g, ig)
    g, gc = g[g_ok], gc[g_ok]
    out = {}
    for v, mp in MAPS.items():
        dsel = np.array([c in mp for c in cl], bool)
        dcl = np.array([mp[c] for c in cl[dsel]], int)
        gcl = np.array([GT_MAPS[v][c] for c in gc], int)
        mg, md = AF.match(bx[dsel], dcl, g, gcl)
        out[v] = (int(mg.sum()), int(len(g)), int(dsel.sum()))
    return out


# ---------------------------------------------------------------- pipeline
class IxPipe(FP.Pipe):
    """fd_profile.Pipe with the reconstruction pass exposed (the code below is fd_profile.Pipe.image without the VisDrone
    fidelity block; equivalence with the stored fd_v1 rows is tested in test_ix_profile.py)."""

    def recon(self, path, seed):
        img = load(path, 0)
        H, W = img.shape[:2]
        ref_clip = self.met.clip_embed(img)
        out = {}
        for name, m in self.codecs.items():
            cuda = m.device == "cuda"
            data, te = timed(m.encode, img, cuda)
            rec, td = timed(m.decode, data, cuda)
            out[name] = dict(family=m.family, device=m.device, bytes=len(data), enc_s=te, dec_s=td, rec=rec)
        gh, gw = 512, int(round(512 * W / H / 8)) * 8
        gen = lambda: torch.Generator(self.dev).manual_seed(int(seed))
        self.sync(); t0 = time.perf_counter()
        cap = self.caption(img)
        self.sync(); t_cap = time.perf_counter() - t0
        cb = cap.encode("utf-8")
        self.sync(); t0 = time.perf_counter()
        r = self.t2i(prompt=cap, num_inference_steps=FP.STEPS, guidance_scale=0.0, height=gh, width=gw,
                     generator=gen()).images[0]
        rec = np.asarray(r.resize((W, H), Image.BICUBIC))
        self.sync(); td = time.perf_counter() - t0
        out["cap_t2i"] = dict(family="cap", device="cuda", bytes=len(cb), enc_s=t_cap, dec_s=td, rec=rec, caption=cap)
        for S, T in ((64, 0.6), (128, 0.3)):
            t0 = time.perf_counter()
            th = np.asarray(Image.fromarray(img).resize((max(1, int(round(S * W / H))), S), Image.LANCZOS))
            buf = io.BytesIO()
            Image.fromarray(th).save(buf, "AVIF", quality=FP.THUMB_Q, speed=6)
            tb = buf.getvalue()
            t_th = time.perf_counter() - t0
            if S == 64:
                t0 = time.perf_counter()
                up = np.asarray(Image.open(io.BytesIO(tb)).convert("RGB").resize((W, H), Image.BICUBIC))
                out["thumb64_up"] = dict(family="thumb64", device="cuda", bytes=len(tb), enc_s=t_th,
                                         dec_s=time.perf_counter() - t0, rec=up)
            self.sync(); t0 = time.perf_counter()
            thd = Image.open(io.BytesIO(tb)).convert("RGB").resize((gw, gh), Image.BICUBIC)
            r = self.i2i(prompt=cap, image=thd, strength=T, num_inference_steps=max(FP.STEPS, int(np.ceil(1 / T))),
                         guidance_scale=0.0, generator=gen()).images[0]
            rec = np.asarray(r.resize((W, H), Image.BICUBIC))
            self.sync(); td = time.perf_counter() - t0
            out[f"cap_thumb{S}_i2i_t{T}"] = dict(family="cap_thumb_i2i", device="cuda", bytes=len(cb) + len(tb),
                                                 enc_s=t_cap + t_th, dec_s=td, rec=rec, caption=cap)
        rows = []
        for name in C.MODES:
            o = out[name]
            q = self.met(img, o["rec"], ref_clip)
            rows.append(dict(mode=name, family=o["family"], device=o["device"], bytes=o["bytes"],
                             bpp=8 * o["bytes"] / H / W, enc_s=o["enc_s"], dec_s=o["dec_s"],
                             gen_seed=int(seed) if name in FP.SEEDED else None, caption=o.get("caption"), **q))
        return img, out, rows

    def image_ix(self, path, seed, gt=None):
        img, out, rows = self.recon(path, seed)
        if gt is not None:
            g, gc, ig = gt
            fid = {}
            for name, rec in [("original", img)] + [(r["mode"], out[r["mode"]]["rec"]) for r in rows]:
                bx, sc, cl = AF.detect(self.yolo, rec)
                rv = recall_variants(bx, sc, cl, g, gc, ig)
                fid[name] = {f"{k}_{v}": x for v, (tp, ng, nd) in rv.items()
                             for k, x in (("tp", tp), ("n_gt", ng), ("n_det", nd))}
            for r in rows:
                r.update(fid[r["mode"]])
            rows.append(dict(mode="original", bytes=0, psnr=float("inf"), gen_seed=None, **fid["original"]))
        return rows


def img_path(e):
    return (UAVDT / e["name"]) if e["source"] == "UAVDT-M" else (TC / e["name"])


def config(stage):
    cfg, _ = FP.config()
    cfg = dict(cfg, code={f: IX.sha_text(HERE / f) for f in ("ix_profile.py", "fd_profile.py", "imcodecs.py",
                                                             "metrics.py", "a5_fidelity.py", "profile_a11.py")},
               manifest_sha256=IX.sha_text(IX.MANIFEST), stage=stage,
               recall="UAVDT vehicle recall variants veh / cls_vancar / cls_car" if stage == "uavdt" else "none")
    return cfg, hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def run(a):
    stage = a.stage
    if stage in ("uavdt", "x1"):
        IX.verify_lock()                     # test profiling only after the L2 lock (prereg §4)
        print("L2 lock verified", flush=True)
    man = IX.read_manifest()
    sel = [e for e in man if e["split"] in IX.SPLITS[stage]]
    if a.limit:
        sel = sel[:a.limit]
    out = IX.PROFILE[stage]
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, ch = config(stage)
    (out.parent / f"config_{stage}_{ch}.json").write_text(json.dumps(dict(cfg, hardware=hardware()), indent=1))
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["sha256"], r["mode"], r.get("gen_seed"), r["config"]))
    recall = stage == "uavdt"

    def complete(e):
        need = list(C.MODES) + (["original"] if recall else [])
        return all((e["sha256"], m, e["gen_seed"] if m in FP.SEEDED else None, ch) in done for m in need)

    todo = [e for e in sel if not complete(e)]
    print(f"stage {stage} config {ch}: {len(sel)} selected, {len(sel) - len(todo)} cached, {len(todo)} to do", flush=True)
    if not todo:
        print("RUN_DONE nothing to do", flush=True)
        return
    pipe = IxPipe(recall=recall)
    t_start, fo = time.time(), open(out, "a")
    for ii, e in enumerate([todo[0]] + todo):          # first image twice, first pass discarded (warm-up)
        p = img_path(e)
        assert e["sha256"] == FP.sha_file(p), f"image changed: {e['name']}"
        gt = read_uavdt_gt(e["seq"], e["frame"]) if recall else None
        if recall:
            ann = hashlib.sha256("\n".join(
                [l.strip() for l in (UGT / f"{e['seq']}_gt_whole.txt").read_text().splitlines()
                 if l.strip() and int(l.split(",")[0]) == e["frame"]] + ["#"] +
                [l.strip() for l in (UGT / f"{e['seq']}_gt_ignore.txt").read_text().splitlines()
                 if l.strip() and int(l.split(",")[0]) == e["frame"]]).encode()).hexdigest()
            assert ann == e["ann_sha256"], f"annotation changed: {e['name']}"
        t0 = time.time()
        rows = pipe.image_ix(p, e["gen_seed"], gt)
        if ii == 0:
            continue
        for r in rows:
            fo.write(json.dumps(dict(image=e["name"], sha256=e["sha256"], split=e["split"], config=ch,
                                     wall_s=time.time() - t0, **r)) + "\n")
        fo.flush()
        if ii % 10 == 0 or ii == len(todo):
            el = time.time() - t_start
            print(f"IMG {ii}/{len(todo)} {el:.0f}s ({el / ii:.2f} s/img)", flush=True)
    fo.close()
    print(f"RUN_DONE {len(todo)} images {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("stage", choices=("A", "uavdt", "x1"))
    ap.add_argument("--limit", type=int, default=0)
    run(ap.parse_args())
