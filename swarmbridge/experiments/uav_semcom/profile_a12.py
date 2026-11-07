"""A1.2: generative (text / text + coarse visual) image modes with REAL payloads and REAL decoders, same images,
preprocessing and metrics as A1.1.

Modes (Huang-style, but every mode is an end-to-end pipeline whose receiver only uses the transmitted bytes, the
shared pretrained weights and a shared seed):
  cap_t2i            : BLIP caption (UTF-8 bytes) -> SD-Turbo text-to-image.                     (~ Huang mode 2)
  thumbS_up          : SxS thumbnail, AVIF q=Q -> bicubic upsample.            (non-generative control, no text)
  cap_thumbS_i2i_tT  : caption + SxS AVIF thumbnail -> SD-Turbo img2img from the upsampled thumbnail, strength T.
                       (text + coarse visual -> generative decoder; a compatible stand-in for Huang mode 3, whose
                        VQ-latent + conditional-diffusion models are not released)
Huang's exact mode-3 models are not public, so this is NOT a reproduction of Huang's modes; it is the closest
end-to-end pipeline built from public pretrained models whose decoder genuinely accepts both inputs.

--size 0 = native frames: thumbnails keep the aspect ratio (shorter side S); the generator runs at shorter side 512
(width rounded to a multiple of 8) and its output is resized to the native frame for scoring; decode time includes
that resize.
Usage: python profile_a12.py VISDRONE_IMAGES_DIR OUT_DIR [--n 100] [--size 512] [--thumbs 32,64,128]
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metrics import Metrics
from profile_a11 import hardware, load

THUMBS = (32, 64, 128)
THUMB_Q = 50
STRENGTHS = (0.3, 0.6)
STEPS = 4


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def avif_bytes(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "AVIF", quality=q, speed=6)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--save-samples", type=int, default=3)
    ap.add_argument("--thumbs", default=",".join(map(str, THUMBS)))
    ap.add_argument("--names", default="", help="file with image names (overrides the every-5th rule)")
    args = ap.parse_args()
    thumbs = tuple(int(x) for x in args.thumbs.split(","))
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))[::5][:args.n]
    if args.names:   # A5: explicit image list (one file name per line), in that order
        files = [Path(args.images) / n for n in Path(args.names).read_text().split()]
    blip_id, sd_id = "Salesforce/blip-image-captioning-large", "stabilityai/sd-turbo"
    proc = BlipProcessor.from_pretrained(blip_id)
    blip = BlipForConditionalGeneration.from_pretrained(blip_id, torch_dtype=torch.float16).to(dev).eval()
    t2i = AutoPipelineForText2Image.from_pretrained(sd_id, torch_dtype=torch.float16, variant="fp16").to(dev)
    t2i.set_progress_bar_config(disable=True)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    met = Metrics(dev)
    meta = dict(hardware=hardware(), size=args.size, n=len(files), images=[f.name for f in files],
                captioner=blip_id, generator=sd_id, steps=STEPS, thumbs=thumbs, thumb_avif_q=THUMB_Q,
                strengths=STRENGTHS, guidance_scale=0.0, seed="per-image index (shared, 0 bits)",
                note="caption payload = UTF-8 bytes of the caption; thumbnail payload = AVIF file bytes")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    fo = open(out / "profile.jsonl", "w")
    t_start = time.time()

    def caption(img):
        inp = proc(images=Image.fromarray(img), return_tensors="pt").to(dev, torch.float16)
        ids = blip.generate(**inp, max_new_tokens=40, num_beams=3)
        return proc.decode(ids[0], skip_special_tokens=True).strip()

    def gen(seed):
        return torch.Generator(dev).manual_seed(seed)

    for ii, f in enumerate([files[0]] + files):
        warm = ii == 0
        idx = max(ii - 1, 0)
        img = load(f, args.size)
        H, W = img.shape[:2]
        gh, gw = (args.size, args.size) if args.size else (512, int(round(512 * W / H / 8)) * 8)
        ref_clip = met.clip_embed(img)
        recs = {}
        # --- encoder side: caption ---
        sync(); t0 = time.perf_counter()
        cap = caption(img)
        sync(); t_cap = time.perf_counter() - t0
        cap_b = cap.encode("utf-8")
        # cap_t2i
        sync(); t0 = time.perf_counter()
        rec = t2i(prompt=cap_b.decode("utf-8"), num_inference_steps=STEPS, guidance_scale=0.0,
                  height=gh, width=gw, generator=gen(idx)).images[0]
        rec = np.asarray(rec if (gh, gw) == (H, W) else rec.resize((W, H), Image.BICUBIC))
        sync(); td = time.perf_counter() - t0
        recs["cap_t2i"] = (len(cap_b), t_cap, td, rec)
        for S in thumbs:
            tw, thh = (S, S) if args.size else (max(1, int(round(S * W / H))), S)
            sync(); t0 = time.perf_counter()
            th = np.asarray(Image.fromarray(img).resize((tw, thh), Image.LANCZOS))
            tb = avif_bytes(th, THUMB_Q)
            t_th = time.perf_counter() - t0
            sync(); t0 = time.perf_counter()
            th_dec = Image.open(io.BytesIO(tb)).convert("RGB")
            up = th_dec.resize((W, H), Image.BICUBIC)
            t_up = time.perf_counter() - t0
            recs[f"thumb{S}_up"] = (len(tb), t_th, t_up, np.asarray(up))
            for T in STRENGTHS:
                sync(); t0 = time.perf_counter()
                th_dec = Image.open(io.BytesIO(tb)).convert("RGB").resize((gw, gh), Image.BICUBIC)
                rec = i2i(prompt=cap_b.decode("utf-8"), image=th_dec, strength=T,
                          num_inference_steps=max(STEPS, int(np.ceil(1 / T))), guidance_scale=0.0,
                          generator=gen(idx)).images[0]
                rec = np.asarray(rec if (gh, gw) == (H, W) else rec.resize((W, H), Image.BICUBIC))
                sync(); td = time.perf_counter() - t0
                recs[f"cap_thumb{S}_i2i_t{T}"] = (len(cap_b) + len(tb), t_cap + t_th, td, rec)
        if warm:
            continue
        for name, (nb, te, td, rec) in recs.items():
            q = met(img, rec, ref_clip)
            fo.write(json.dumps(dict(image=f.name, mode=name, family=name.split("_")[0] if "i2i" not in name
                                     else "cap_thumb_i2i", device=dev, bytes=nb,
                                     bpp=8 * nb / img.shape[0] / img.shape[1], enc_s=te, dec_s=td,
                                     caption=cap, **q)) + "\n")
            if ii <= args.save_samples:
                Image.fromarray(rec).save(out / "samples" / f"{f.stem}__{name}.png")
        if ii <= args.save_samples:
            Image.fromarray(img).save(out / "samples" / f"{f.stem}__original.png")
        fo.flush()
        if ii % 10 == 0:
            print(f"IMG {ii}/{len(files)} {time.time() - t_start:.0f}s", flush=True)
    fo.close()
    print(f"RUN_DONE a12 {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
