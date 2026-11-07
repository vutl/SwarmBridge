"""A1.1: per-image rate / quality / latency profile of real image codecs on VisDrone UAV images.

Subset: VisDrone2019-DET-val images sorted by name, every 5th, first N (default 100) -> fixed, listed in the output.
Preprocessing: resize shorter side to SIZE (default 512, as in Huang et al.) with Lanczos, centre crop SIZExSIZE.
For every (image, mode): real encode -> bytes -> real decode -> PSNR / MS-SSIM / LPIPS / CLIP, encode and decode
wall time. The first image is run twice and its first pass discarded (warm-up). Learned codecs run on DEVICE
(GPU by default); --cpu-learned K additionally times the learned codecs on CPU for the first K images.

Usage: python profile_a11.py VISDRONE_IMAGES_DIR OUT_DIR [--n 100] [--size 512] [--cpu-learned 10]
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import imcodecs as C
from metrics import Metrics


def load(path, size):
    """size > 0: Lanczos resize of the shorter side to size, centre crop size x size (A1). size == 0: native frame."""
    im = Image.open(path).convert("RGB")
    if size == 0:
        return np.asarray(im)
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def timed(fn, arg, cuda):
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(arg)
    if cuda:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def hardware():
    info = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads())
    try:
        info["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        info["cpu"] = platform.processor()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import PIL.features as F
    info["pillow"] = {k: F.version(k) for k in ("jpg", "jpg_2000", "avif")}
    import compressai
    info["compressai"] = compressai.__version__
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cpu-learned", type=int, default=10)
    ap.add_argument("--save-samples", type=int, default=3)
    ap.add_argument("--modes", default="", help="comma list of mode names to profile (default: all)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))[::5][:args.n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modes = C.default_modes(dev)
    if args.modes:
        keep = set(args.modes.split(","))
        modes = [m for m in modes if m.name in keep]
        assert len(modes) == len(keep), "unknown mode name"
    met = Metrics(dev)
    meta = dict(hardware=hardware(), size=args.size, n=len(files), images=[f.name for f in files],
                modes=[m.name for m in modes],
                preprocessing="native resolution (no resize)" if args.size == 0 else
                f"Lanczos resize shorter side to {args.size}, centre crop")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    rows = open(out / "profile.jsonl", "w")
    t_start = time.time()
    for ii, f in enumerate([files[0]] + files):
        warm = ii == 0
        img = load(f, args.size)
        ref_clip = met.clip_embed(img)
        for m in modes:
            cuda = m.device == "cuda"
            data, te = timed(m.encode, img, cuda)
            rec, td = timed(m.decode, data, cuda)
            if warm:
                continue
            q = met(img, rec, ref_clip)
            rows.write(json.dumps(dict(image=f.name, mode=m.name, family=m.family, device=m.device,
                                       bytes=len(data), bpp=8 * len(data) / img.shape[0] / img.shape[1],
                                       enc_s=te, dec_s=td, **q)) + "\n")
            if ii <= args.save_samples:
                Image.fromarray(rec).save(out / "samples" / f"{f.stem}__{m.name}.png")
        if not warm and ii <= args.save_samples:
            Image.fromarray(img).save(out / "samples" / f"{f.stem}__original.png")
        rows.flush()
        if ii % 10 == 0:
            print(f"IMG {ii}/{len(files)} {time.time() - t_start:.0f}s", flush=True)
    rows.close()
    # CPU timing of the learned codecs (on-board UAV without GPU), first K images
    if args.cpu_learned and dev == "cuda":
        torch.set_num_threads(os.cpu_count())
        cpu_modes = [C.CompressAICodec(m.arch, m.quality, "cpu") for m in modes if isinstance(m, C.CompressAICodec)]
        with open(out / "cpu_learned.jsonl", "w") as fo:
            for ii, f in enumerate([files[0]] + files[:args.cpu_learned]):
                img = load(f, args.size)
                for m in cpu_modes:
                    data, te = timed(m.encode, img, False)
                    rec, td = timed(m.decode, data, False)
                    if ii:
                        fo.write(json.dumps(dict(image=f.name, mode=m.name, bytes=len(data), enc_s=te, dec_s=td)) + "\n")
    print(f"RUN_DONE a11 {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
