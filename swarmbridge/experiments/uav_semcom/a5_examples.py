"""A5.2 qualitative examples: one A5-TEST frame, a crop around its densest labelled region, decoded from several modes,
with the A5 detector's output at the operating point (conf >= 0.25): labelled objects found (IoU >= 0.5, same class)
drawn in green, labelled objects missed in red. Frame choice (automatic, stated): among test-split frames with >= 40
labelled objects, the one with the largest small-object recall gap between avif_50 and avif_20 (per_image_val.jsonl),
i.e. where ~35 dB still works and ~28 dB does not.
Usage: python a5_examples.py WEIGHTS VAL_DIR
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import a5_common as C
import a5_fidelity as AF
import imcodecs as IC
from profile_a11 import load

MODES = ["original", "avif_80", "avif_50", "mbt2018-mean_q5", "avif_20", "thumb128_up", "cap_thumb128_i2i_t0.3"]


def main():
    weights, val = sys.argv[1], Path(sys.argv[2])
    from ultralytics import YOLO
    per = [json.loads(l) for l in open(C.A5 / "fidelity" / "per_image_val.jsonl")]
    split = {e["name"]: e["split"] for e in C.manifest()["images"]}
    by = {}
    for r in per:
        by.setdefault(r["image"], {})[r["mode"]] = r
    cand = [(by[n]["avif_50"]["recall_small"] - by[n]["avif_20"]["recall_small"], n) for n in by
            if split[n] == "test" and by[n]["original"]["n_gt"] >= 40 and by[n]["avif_20"]["recall_small"] is not None]
    gap, name = max(cand)
    img = load(val / "images" / name, 0)
    H, W = img.shape[:2]
    g, gc, ig = AF.read_gt(val / "annotations" / (Path(name).stem + ".txt"))
    cw, ch = min(640, W), min(400, H)
    cx = np.clip(np.median((g[:, 0] + g[:, 2]) / 2), cw / 2, W - cw / 2)
    cy = np.clip(np.median((g[:, 1] + g[:, 3]) / 2), ch / 2, H - ch / 2)
    x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
    model = YOLO(weights)
    codecs = {m.name: m for m in IC.default_modes("cuda")}
    recs = {"original": (0, img)}
    for m in ("avif_80", "avif_50", "mbt2018-mean_q5", "avif_20"):
        d = codecs[m].encode(img)
        recs[m] = (len(d), codecs[m].decode(d))
    th = np.asarray(Image.fromarray(img).resize((int(round(128 * W / H)), 128), Image.LANCZOS))
    buf = io.BytesIO(); Image.fromarray(th).save(buf, "AVIF", quality=50, speed=6); tb = buf.getvalue()
    recs["thumb128_up"] = (len(tb), np.asarray(Image.open(io.BytesIO(tb)).convert("RGB").resize((W, H), Image.BICUBIC)))
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                        torch_dtype=torch.float16).to("cuda").eval()
    t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    inp = proc(images=Image.fromarray(img), return_tensors="pt").to("cuda", torch.float16)
    cap = proc.decode(blip.generate(**inp, max_new_tokens=40, num_beams=3)[0], skip_special_tokens=True).strip()
    gh, gw = 512, int(round(512 * W / H / 8)) * 8
    r = i2i(prompt=cap, image=Image.open(io.BytesIO(tb)).convert("RGB").resize((gw, gh), Image.BICUBIC), strength=0.3,
            num_inference_steps=4, guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(0)).images[0]
    recs["cap_thumb128_i2i_t0.3"] = (len(tb) + len(cap.encode()), np.asarray(r.resize((W, H), Image.BICUBIC)))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), dpi=130)
    stats = {}
    for ax, m in zip(axes.flat, MODES):
        nb, rec = recs[m]
        bx, sc, cl = AF.detect(model, rec)
        keep = ~AF.in_ignore(bx, ig) & (sc >= AF.CONF)
        mg, _ = AF.match(bx[keep], cl[keep], g, gc)
        ax.imshow(rec[y0:y0 + ch, x0:x0 + cw])
        for b, ok in zip(g, mg):
            if b[2] < x0 or b[0] > x0 + cw or b[3] < y0 or b[1] > y0 + ch:
                continue
            ax.add_patch(mpatches.Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, lw=1.0,
                                            ec="#1baf7a" if ok else "#e34948"))
        mse = np.mean((rec.astype(float) - img.astype(float)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
        bpp = 8 * nb / (W * H)
        stats[m] = dict(bpp=bpp, psnr=psnr, recall=float(mg.mean()))
        ax.set_title(f"{m}\n{bpp:.3f} bpp, {psnr:.1f} dB, recall {mg.mean():.2f}", fontsize=8)
        ax.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.0, 0.6, f"{name}\n(tập test A5)\nxanh: vật được phát hiện\nđỏ: vật bị sót\n"
                       f"conf >= 0,25, IoU >= 0,5\ncaption: \"{cap[:60]}\"", fontsize=8, va="top", wrap=True)
    fig.tight_layout()
    out = C.REPO / "reports" / "figures" / "a5" / "examples.png"
    fig.savefig(out)
    (out.parent / "examples.json").write_text(json.dumps(dict(image=name, gap_small_recall=gap, crop=[x0, y0, cw, ch],
                                                              caption=cap, stats=stats), indent=1))
    print("saved", out, name, stats)


if __name__ == "__main__":
    main()
