"""Paper A full-data protocol fd_v1, Phase 1: image-mode profiling + detector fidelity in ONE pass per image.

Same codecs, settings and metrics as the A5 tables (profile_a11.py, profile_a12.py, a5_fidelity.py; native
resolution), restricted to the 14 table modes (C.MODES). Differences from the A5 pipeline (pre-registered in the Phase 1
note of reports/phases/2026-11-23_paperA-fulldata/):
  * one pass: the decoded image whose PSNR / LPIPS is profiled is the SAME image the detector sees (in A5 the
    generative modes were generated twice, with different seeds for test-dev: profile seed = list index, fidelity
    seed = manifest index);
  * generator seed = manifest gen_seed (stored per row);
  * detector fidelity only for the recall pools F, TD, TD_extra, V (YOLO was trained on the rl_* scenes);
  * per-image cache / resume: a row is reused only if (image sha256, mode, gen_seed for generative modes, config
    hash) all match; config hash = code sha256s + model / weight identities + all codec / generator parameters +
    manifest sha256. A run refuses a manifest whose sha256 differs from the one fixed at the first run.
Rows: profile fields (bytes, bpp, enc_s, dec_s, psnr, ms_ssim, lpips, clip, caption) + fidelity fields (n_gt, tp,
n_small, tp_small, n_det, recall, recall_small, precision, agree_f1) for recall pools; one extra 'original' row per
recall-pool image (detector on the undecoded frame).

  python fd_profile.py run   OUT [--pools rl_train,...] [--limit N] [--names FILE]
  python fd_profile.py repro OUT --n 5          (smoke: first N DET-val images with the OLD A5 seeds, compared with
                                                 the A5 rows; not written to the main cache)
  python fd_profile.py cpu   OUT [--k 10]       (CPU timing of the learned codecs, first K rl_train images)"""
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
import imcodecs as IC
from fd_manifest import OUT as FD, SRC
from metrics import Metrics
from profile_a11 import hardware, load, timed

HERE = Path(__file__).resolve().parent
MANIFEST = FD / "manifest_fd_v1.json"
YOLO_W = Path("/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt")
BLIP, SD = "Salesforce/blip-image-captioning-large", "stabilityai/sd-turbo"
STEPS, THUMB_Q = 4, 50
CODEC_MODES = [m for m in C.MODES if m.startswith(("avif", "jpeg2000", "mbt2018"))]
GEN_MODES = ["cap_t2i", "thumb64_up", "cap_thumb64_i2i_t0.6", "cap_thumb128_i2i_t0.3"]
assert sorted(CODEC_MODES + GEN_MODES) == sorted(C.MODES)
SEEDED = {"cap_t2i", "cap_thumb64_i2i_t0.6", "cap_thumb128_i2i_t0.3"}
RECALL_POOLS = {"F", "TD", "TD_extra", "V"}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hf_rev(repo):
    p = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return p.read_text().strip() if p.exists() else "unknown"


def config():
    import compressai
    import diffusers
    import transformers
    import ultralytics
    cfg = dict(code={f: sha_file(HERE / f) for f in ("fd_profile.py", "imcodecs.py", "metrics.py", "a5_fidelity.py",
                                                     "profile_a11.py")},
               manifest_sha256=sha_file(MANIFEST), yolo_sha256=sha_file(YOLO_W), blip=[BLIP, hf_rev(BLIP)],
               sd=[SD, hf_rev(SD)], steps=STEPS, thumb_q=THUMB_Q, thumbs=[64, 128], strengths={"64": 0.6, "128": 0.3},
               guidance=0.0, gen_size="shorter side 512, width multiple of 8; output bicubic to native",
               caption="max_new_tokens 40, num_beams 3, fp16", preprocessing="native resolution",
               detector="imgsz 960, conf 0.01, NMS IoU 0.7, max_det 500; op conf 0.25 IoU 0.5",
               versions=dict(torch=torch.__version__, diffusers=diffusers.__version__,
                             transformers=transformers.__version__, ultralytics=ultralytics.__version__,
                             compressai=compressai.__version__))
    return cfg, hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


class Pipe:
    def __init__(self, recall):
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.dev = "cuda"
        self.codecs = {m.name: m for m in IC.default_modes(self.dev) if m.name in CODEC_MODES}
        assert sorted(self.codecs) == sorted(CODEC_MODES)
        self.proc = BlipProcessor.from_pretrained(BLIP)
        self.blip = BlipForConditionalGeneration.from_pretrained(BLIP, torch_dtype=torch.float16).to(self.dev).eval()
        self.t2i = AutoPipelineForText2Image.from_pretrained(SD, torch_dtype=torch.float16, variant="fp16").to(self.dev)
        self.t2i.set_progress_bar_config(disable=True)
        self.i2i = AutoPipelineForImage2Image.from_pipe(self.t2i)
        self.met = Metrics(self.dev)
        self.yolo = None
        if recall:
            from ultralytics import YOLO
            self.yolo = YOLO(str(YOLO_W))

    @staticmethod
    def sync():
        torch.cuda.synchronize()

    def caption(self, img):
        inp = self.proc(images=Image.fromarray(img), return_tensors="pt").to(self.dev, torch.float16)
        ids = self.blip.generate(**inp, max_new_tokens=40, num_beams=3)
        return self.proc.decode(ids[0], skip_special_tokens=True).strip()

    def image(self, path, seed, recall, ann=None):
        """-> list of row dicts (profile + fidelity fields) for the 14 modes (+ 'original' if recall)."""
        img = load(path, 0)
        H, W = img.shape[:2]
        ref_clip = self.met.clip_embed(img)
        out = {}
        for name, m in self.codecs.items():
            cuda = m.device == "cuda"
            data, te = timed(m.encode, img, cuda)
            rec, td = timed(m.decode, data, cuda)
            out[name] = dict(family=m.family, device=m.device, bytes=len(data), enc_s=te, dec_s=td, rec=rec)
        # generative / thumbnail modes (profile_a12 timing semantics: enc = caption (+ thumbnail) time,
        # dec = generation (+ resize) time)
        gh, gw = 512, int(round(512 * W / H / 8)) * 8
        gen = lambda: torch.Generator(self.dev).manual_seed(int(seed))
        self.sync(); t0 = time.perf_counter()
        cap = self.caption(img)
        self.sync(); t_cap = time.perf_counter() - t0
        cb = cap.encode("utf-8")
        self.sync(); t0 = time.perf_counter()
        r = self.t2i(prompt=cap, num_inference_steps=STEPS, guidance_scale=0.0, height=gh, width=gw,
                     generator=gen()).images[0]
        rec = np.asarray(r.resize((W, H), Image.BICUBIC))
        self.sync(); td = time.perf_counter() - t0
        out["cap_t2i"] = dict(family="cap", device="cuda", bytes=len(cb), enc_s=t_cap, dec_s=td, rec=rec, caption=cap)
        for S, T in ((64, 0.6), (128, 0.3)):
            t0 = time.perf_counter()
            th = np.asarray(Image.fromarray(img).resize((max(1, int(round(S * W / H))), S), Image.LANCZOS))
            buf = io.BytesIO()
            Image.fromarray(th).save(buf, "AVIF", quality=THUMB_Q, speed=6)
            tb = buf.getvalue()
            t_th = time.perf_counter() - t0
            if S == 64:
                t0 = time.perf_counter()
                up = np.asarray(Image.open(io.BytesIO(tb)).convert("RGB").resize((W, H), Image.BICUBIC))
                out["thumb64_up"] = dict(family="thumb64", device="cuda", bytes=len(tb), enc_s=t_th,
                                         dec_s=time.perf_counter() - t0, rec=up)
            self.sync(); t0 = time.perf_counter()
            thd = Image.open(io.BytesIO(tb)).convert("RGB").resize((gw, gh), Image.BICUBIC)
            r = self.i2i(prompt=cap, image=thd, strength=T, num_inference_steps=max(STEPS, int(np.ceil(1 / T))),
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
                             gen_seed=int(seed) if name in SEEDED else None, caption=o.get("caption"), **q))
        if recall:
            g, gc, ig = AF.read_gt(ann)
            recs = [("original", img)] + [(r["mode"], out[r["mode"]]["rec"]) for r in rows]
            base, fid = None, {}
            for name, rec in recs:
                bx, sc, cl = AF.detect(self.yolo, rec)
                keep = ~AF.in_ignore(bx, ig)
                bx, sc, cl = bx[keep], sc[keep], cl[keep]
                op = sc >= AF.CONF
                mg, md = AF.match(bx[op], cl[op], g, gc)
                small = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) < 32 ** 2
                if name == "original":
                    base = (bx[op], cl[op])
                ag, _ = AF.match(bx[op], cl[op], *base)
                nb, nd = len(base[0]), int(op.sum())
                fid[name] = dict(n_gt=int(len(g)), tp=int(mg.sum()), n_small=int(small.sum()),
                                 tp_small=int((mg & small).sum()), n_det=nd,
                                 recall=float(mg.mean()) if len(g) else None,
                                 recall_small=float(mg[small].mean()) if small.any() else None,
                                 precision=float(md.mean()) if nd else None,
                                 agree_f1=float(2 * ag.sum() / (nb + nd)) if nb + nd else 1.0)
            for r in rows:
                r.update(fid[r["mode"]])
            rows.append(dict(mode="original", bytes=0, psnr=float("inf"), gen_seed=None, **fid["original"]))
        return rows


def ann_path(e):
    return SRC[e["source"]] / "annotations" / (e["name"][:-4] + ".txt")


def img_path(e):
    return SRC[e["source"]] / "images" / e["name"]


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, ch = config()
    st = out / "state.json"
    if st.exists():
        s0 = json.loads(st.read_text())
        assert s0["manifest_sha256"] == cfg["manifest_sha256"], "manifest changed after profiling started"
    else:
        st.write_text(json.dumps(dict(manifest_sha256=cfg["manifest_sha256"], started=time.strftime("%F %T")), indent=1))
    (out / f"config_{ch}.json").write_text(json.dumps(dict(cfg, hardware=hardware()), indent=1))
    man = json.loads(MANIFEST.read_text())["images"]
    pools = set(a.pools.split(",")) if a.pools else None
    sel = [e for e in man if pools is None or e["split"] in pools]
    if a.names:
        keep = set(Path(a.names).read_text().split())
        sel = [e for e in sel if e["name"] in keep]
    if a.limit:
        sel = sel[:a.limit]
    prof = out / "profile.jsonl"
    done = set()
    if prof.exists():
        for line in open(prof):
            r = json.loads(line)
            done.add((r["sha256"], r["mode"], r.get("gen_seed"), r["config"]))

    def complete(e):
        need = list(C.MODES) + (["original"] if e["split"] in RECALL_POOLS else [])
        return all((e["sha256"], m, e["gen_seed"] if m in SEEDED else None, ch) in done for m in need)

    todo = [e for e in sel if not complete(e)]
    print(f"config {ch}: {len(sel)} selected, {len(sel) - len(todo)} cached, {len(todo)} to do", flush=True)
    if not todo:
        print("RUN_DONE nothing to do", flush=True)
        return
    pipe = Pipe(recall=any(e["split"] in RECALL_POOLS for e in todo))
    t_start, fo = time.time(), open(prof, "a")
    for ii, e in enumerate([todo[0]] + todo):          # first image twice, first pass discarded (warm-up)
        assert e["sha256"] == sha_file(img_path(e)), f"image changed: {e['name']}"
        t0 = time.time()
        rows = pipe.image(img_path(e), e["gen_seed"], e["split"] in RECALL_POOLS, ann_path(e))
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


def repro(a):
    """Smoke: the first N DET-val images, generated with the OLD A5 seeds, compared with the A5 rows."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val = [e for e in json.loads(MANIFEST.read_text())["images"] if e["split"] == "V"][:a.n]
    old_idx = {n: i for i, n in enumerate(json.loads((C.A5 / "profile" / "a12_val" / "meta.json").read_text())["images"])}
    man_idx = {e["name"]: i for i, e in enumerate(C.manifest()["images"])}
    old = {}
    for f in ("a11_val", "a12_val"):
        for line in open(C.A5 / "profile" / f / "profile.jsonl"):
            r = json.loads(line)
            old[(r["image"], r["mode"])] = r
    fid = {}
    for line in open(C.REPO / "runs" / "uav_semcom" / "a6l1" / "fidelity" / "per_image_val.jsonl"):
        r = json.loads(line)
        fid[(r["image"], r["mode"])] = r
    pipe = Pipe(recall=True)
    res, t_all = [], []
    for ii, e in enumerate([val[0]] + val):
        assert old_idx[e["name"]] == man_idx[e["name"]], "A5 profile and fidelity seeds differ for this DET-val image"
        t0 = time.time()
        rows = pipe.image(img_path(e), old_idx[e["name"]], True, ann_path(e))
        if ii == 0:
            continue
        t_all.append(time.time() - t0)
        for r in rows:
            o, f = old.get((e["name"], r["mode"])), fid.get((e["name"], r["mode"]))
            d = dict(image=e["name"], mode=r["mode"])
            if o:
                d.update(bytes_new=r["bytes"], bytes_old=o["bytes"], psnr_new=r["psnr"], psnr_old=o["psnr"],
                         lpips_new=r["lpips"], lpips_old=o["lpips"])
            if f:
                d.update(tp_new=r["tp"], tp_old=f["tp"], ndet_new=r["n_det"], ndet_old=f["n_det"])
            res.append(d)
    (out / "repro.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    lines = [f"repro smoke: {len(val)} DET-val images, old A5 seeds; wall {np.mean(t_all):.2f} s/img (with detector)"]
    for m in ["original"] + list(C.MODES):
        rs = [r for r in res if r["mode"] == m]
        if not rs:
            continue
        b = [r["bytes_new"] == r["bytes_old"] for r in rs if "bytes_old" in r]
        dp = [abs(r["psnr_new"] - r["psnr_old"]) for r in rs if "psnr_old" in r]
        dl = [abs(r["lpips_new"] - r["lpips_old"]) for r in rs if "lpips_old" in r]
        dt = [abs(r["tp_new"] - r["tp_old"]) for r in rs if "tp_old" in r]
        lines.append(f"  {m:24s} bytes equal {sum(b)}/{len(b)}  max|dPSNR| {max(dp) if dp else float('nan'):.4f}  "
                     f"max|dLPIPS| {max(dl) if dl else float('nan'):.4f}  max|dTP| {max(dt) if dt else 'n/a'}")
    print("\n".join(lines))
    (out / "repro.log").write_text("\n".join(lines) + "\n")


def cpu(a):
    out = Path(a.out)
    man = [e for e in json.loads(MANIFEST.read_text())["images"] if e["split"] == "rl_train"][:a.k]
    torch.set_num_threads(torch.get_num_threads())
    cms = [IC.CompressAICodec(m.arch, m.quality, "cpu") for m in IC.default_modes("cuda")
           if isinstance(m, IC.CompressAICodec) and m.name in CODEC_MODES]
    with open(out / "cpu_learned.jsonl", "w") as fo:
        for ii, e in enumerate([man[0]] + man):
            img = load(img_path(e), 0)
            for m in cms:
                data, te = timed(m.encode, img, False)
                rec, td = timed(m.decode, data, False)
                if ii:
                    fo.write(json.dumps(dict(image=e["name"], mode=m.name, bytes=len(data), enc_s=te, dec_s=td)) + "\n")
    print("CPU_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "repro", "cpu"))
    ap.add_argument("out")
    ap.add_argument("--pools", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", default="")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    dict(run=run, repro=repro, cpu=cpu)[a.cmd](a)


if __name__ == "__main__":
    main()
