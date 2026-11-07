"""A5 figures (PNG, reports/figures/a5/):
  rdt_<pool>.png : rate-distortion-task fidelity per codec family (one measure per panel, no dual axes):
                   (a) bpp vs PSNR, (b) bpp vs AP50, (c) bpp vs small-object recall @ conf 0.25; the uncompressed
                   original as a dashed reference line in (b)/(c).
Colours: categorical slots 1-3 of the reference palette (validated all-pairs for 3 series), muted grey for the
non-codec modes (thumbnails / generative), text in neutral ink; families also differ by marker shape.
Usage: python a5_figures.py [FIDELITY_DIR]
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import a5_common as C

OUT = C.REPO / "reports" / "figures" / "a5"
FID = Path(sys.argv[1]) if len(sys.argv) > 1 else C.A5 / "fidelity"
FAM = {"avif": ("#2a78d6", "o", "AVIF"), "mbt2018-mean": ("#eb6834", "s", "mbt2018-mean (learned)"),
       "jpeg2000": ("#1baf7a", "^", "JPEG 2000")}
OTHER = ("#8a8984", "D")
INK, INK2 = "#0b0b0b", "#52514e"


def fam(m):
    for f in FAM:
        if m.startswith(f + "_"):
            return f
    return None


def rdt(pool):
    coco = json.load(open(FID / f"coco_{pool}.json"))["results"]["all"]
    per = [json.loads(l) for l in open(FID / f"per_image_{pool}.jsonl")]
    man = {e["name"]: e for e in C.manifest()["images"]}
    agg = defaultdict(lambda: defaultdict(float))
    for r in per:
        a = agg[r["mode"]]
        if r["mode"] != "original":
            e = man[r["image"]]
            a["bpp"] += 8 * r["bytes"] / (e["w"] * e["h"])
            a["psnr"] += r["psnr"]
            a["n"] += 1
        a["tp_s"] += r["tp_small"]
        a["n_s"] += r["n_small"]
    pts = {}
    for m, a in agg.items():
        if m == "original" or m not in coco:
            continue
        pts[m] = (a["bpp"] / a["n"], a["psnr"] / a["n"], coco[m]["AP50"], a["tp_s"] / max(a["n_s"], 1))
    orig = (coco["original"]["AP50"], agg["original"]["tp_s"] / max(agg["original"]["n_s"], 1))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150)
    titles = ["(a) PSNR (dB)", "(b) AP50", "(c) recall vật nhỏ @ conf 0,25"]
    for k, ax in enumerate(axes):
        for f, (col, mk, lab) in FAM.items():
            ms = sorted((m for m in pts if fam(m) == f), key=lambda m: pts[m][0])
            if not ms:
                continue
            ax.plot([pts[m][0] for m in ms], [pts[m][k + 1] for m in ms], color=col, marker=mk, lw=2, ms=6,
                    label=lab)
        others = [m for m in pts if fam(m) is None]
        if others:
            ax.scatter([pts[m][0] for m in others], [pts[m][k + 1] for m in others], color=OTHER[0],
                       marker=OTHER[1], s=30, label="thumbnail / tạo sinh", zorder=3)
            if k:
                for i, m in enumerate(sorted(others, key=lambda m: pts[m][0])):
                    ax.annotate(m.replace("_i2i", "").replace("cap_", "c+"), (pts[m][0], pts[m][k + 1]),
                                fontsize=6, color=INK2, xytext=(3, 3 + 8 * i), textcoords="offset points")
        if k:
            ax.axhline(orig[k - 1], color=INK2, ls="--", lw=1)
            ax.annotate("ảnh gốc", (min(p[0] for p in pts.values()), orig[k - 1]), fontsize=7, color=INK2,
                        xytext=(0, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("bit / pixel")
        ax.set_title(titles[k], loc="left", color=INK, fontsize=9)
        ax.grid(True, color="#e4e3df", lw=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Rate–distortion–task fidelity, pool {pool} ({coco['original']['n_images']} ảnh, detector "
                 f"YOLO11s huấn luyện trên VisDrone-train)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"rdt_{pool}.png")
    print("saved", OUT / f"rdt_{pool}.png")


if __name__ == "__main__":
    for pool in ("val", "testdev"):
        if (FID / f"coco_{pool}.json").exists():
            rdt(pool)
