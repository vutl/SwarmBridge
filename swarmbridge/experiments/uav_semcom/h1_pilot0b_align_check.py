"""Visual alignment check for the 0B pilot (readiness sec 5 / 6): draw the XML labels of both views on the image the
adapter serves for the same frame, side by side, at the first / middle / last pilot epoch of each sequence.
Shared track ids get the same colour in both views, so a wrong frame offset shows up as boxes drifting off objects.
Output goes OUTSIDE git (e.g. ~/work/h1_pilot0b/align). Requires --rights-note (adapter rule).

Usage: python h1_pilot0b_align_check.py MDMT_ZIP OUT_DIR --seqs 52,68 --rights-note "..." [--step 10 --max-epochs 60]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mdmt_adapter as M


def colour(tid):
    r = np.random.default_rng(int(tid) * 7919 + 1)
    return tuple(int(x) for x in r.integers(64, 256, 3))


def panel(img, gt, scale):
    im = Image.fromarray(img).resize((max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale))))
    dr = ImageDraw.Draw(im)
    for tid, (x1, y1, x2, y2, _) in gt.items():
        c = colour(tid)
        dr.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=c, width=2)
        dr.text((x1 * scale, max(0, y1 * scale - 10)), str(tid), fill=c)
    return im


def render(ds, seq, frame, width=1920):
    g = {v: ds.labels(seq, v).get(frame, {}) for v in (1, 2)}
    imgs = {v: ds.image(seq, v, frame) for v in (1, 2)}
    scale = (width / 2) / max(imgs[1].shape[1], 1)
    p1, p2 = panel(imgs[1], g[1], scale), panel(imgs[2], g[2], scale)
    out = Image.new("RGB", (p1.width + p2.width, max(p1.height, p2.height)))
    out.paste(p1, (0, 0)); out.paste(p2, (p1.width, 0))
    ImageDraw.Draw(out).text((5, 5), f"seq {seq} frame {frame} | shared ids {len(set(g[1]) & set(g[2]))}",
                             fill=(255, 255, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip"); ap.add_argument("out")
    ap.add_argument("--seqs", default="52,68"); ap.add_argument("--rights-note", required=True)
    ap.add_argument("--step", type=int, default=10); ap.add_argument("--max-epochs", type=int, default=60)
    a = ap.parse_args()
    ds = M.MDMT(a.zip, rights_note=a.rights_note)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seqs.split(",")]:
        ep = ds.epochs(s, a.step, a.max_epochs)
        for f in sorted({ep[0], ep[len(ep) // 2], ep[-1]}):
            p = od / f"align_{s}_{f:04d}.jpg"
            render(ds, s, f).save(p, quality=90)
            print(p)


if __name__ == "__main__":
    main()
