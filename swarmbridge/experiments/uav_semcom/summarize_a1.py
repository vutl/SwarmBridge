"""Summarise A1 profiles: per-mode means, per-image spread, and how much the best mode varies across images.

Usage: python summarize_a1.py OUT_MD PROFILE_DIR [PROFILE_DIR ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    out_md = Path(sys.argv[1])
    rows, cpu, metas = [], [], []
    for d in sys.argv[2:]:
        d = Path(d)
        rows += [json.loads(l) for l in open(d / "profile.jsonl")]
        metas.append(json.loads((d / "meta.json").read_text()))
        if (d / "cpu_learned.jsonl").exists():
            cpu += [json.loads(l) for l in open(d / "cpu_learned.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["mode"]].append(r)
    imgs = sorted({r["image"] for r in rows})
    L = [f"# A1 — per-image mode profile on VisDrone ({len(imgs)} images, {metas[0]['size']}x{metas[0]['size']})", "",
         f"Hardware: {metas[0]['hardware'].get('gpu')} / {metas[0]['hardware'].get('cpu')}; "
         f"preprocessing: {metas[0]['preprocessing'] if 'preprocessing' in metas[0] else ''}", "",
         "Mean over images (p10–p90 in brackets for bytes and PSNR). Latency = wall time per image on the device "
         "column; Pillow codecs are single-process CPU.", "",
         "| mode | device | KB | bpp | PSNR dB | MS-SSIM | LPIPS ↓ | CLIP | enc ms | dec ms | decode fails |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(by, key=lambda m: (by[m][0]["family"], np.mean([r["bytes"] for r in by[m]])))
    for m in order:
        rs = by[m]
        b = np.array([r["bytes"] for r in rs]) / 1000
        p = np.array([r["psnr"] for r in rs])
        fails = int(np.sum(p < 12)) if "t2i" not in m and "i2i" not in m else "-"
        L.append(f"| {m} | {rs[0]['device']} | {b.mean():.1f} [{np.percentile(b, 10):.1f}–{np.percentile(b, 90):.1f}] | "
                 f"{np.mean([r['bpp'] for r in rs]):.3f} | {p.mean():.2f} [{np.percentile(p, 10):.1f}–"
                 f"{np.percentile(p, 90):.1f}] | {np.mean([r['ms_ssim'] for r in rs]):.3f} | "
                 f"{np.mean([r['lpips'] for r in rs]):.3f} | {np.mean([r['clip'] for r in rs]):.3f} | "
                 f"{1e3 * np.mean([r['enc_s'] for r in rs]):.0f} | {1e3 * np.mean([r['dec_s'] for r in rs]):.0f} | {fails} |")
    if cpu:
        L += ["", "Learned codecs timed on CPU (on-board UAV without GPU), first images:", "",
              "| mode | enc ms | dec ms |", "|---|---|---|"]
        cb = defaultdict(list)
        for r in cpu:
            cb[r["mode"]].append(r)
        for m, rs in cb.items():
            L.append(f"| {m} | {1e3 * np.mean([r['enc_s'] for r in rs]):.0f} | {1e3 * np.mean([r['dec_s'] for r in rs]):.0f} |")
    # per-image best mode under a byte budget (does the choice depend on the image?)
    L += ["", "## Does the best mode depend on the image?", "",
          "For each byte budget, per image: the mode with the highest metric among modes whose payload fits. "
          "'share' = fraction of images where the most common winner wins; 'winners' = distinct winning modes.", ""]
    per = defaultdict(dict)
    for r in rows:
        per[r["image"]][r["mode"]] = r
    for metric, sign in (("psnr", 1), ("lpips", -1)):
        L += [f"**{metric}**", "", "| budget KB | most common winner | share | winners | mean best | mean best of fixed winner |",
              "|---|---|---|---|---|---|"]
        for budget in (1, 2, 4, 8, 16, 32, 64):
            win, best = [], []
            for im in imgs:
                cand = [(sign * r[metric], m) for m, r in per[im].items() if r["bytes"] <= budget * 1000]
                if cand:
                    v, m = max(cand)
                    win.append(m)
                    best.append(sign * v)
            if not win:
                continue
            vals, cnt = np.unique(win, return_counts=True)
            top = vals[cnt.argmax()]
            fixed = [per[im][top][metric] for im in imgs if top in per[im] and per[im][top]["bytes"] <= budget * 1000]
            L.append(f"| {budget} | {top} | {cnt.max() / len(win):.2f} | {len(vals)} | {np.mean(best):.3f} | "
                     f"{np.mean(fixed):.3f} (fits {len(fixed)}/{len(imgs)}) |")
        L.append("")
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
