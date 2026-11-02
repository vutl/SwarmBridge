"""Render built BEV grids: rgb | obs (this UAV) | gt (swarm + boxes), one row per grid.
Colours: road grey, vehicle red, other dark green, unseen black.
White contour in gt = this UAV's own footprint (outside it = information from helpers).

    python scripts/show_bev.py data/coperception_uav/bev_debug/train out.png [n_rows]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PAL = {0: (40, 80, 40), 1: (150, 150, 150), 2: (230, 40, 40), 255: (0, 0, 0)}


def colour(lbl):
    out = np.zeros(lbl.shape + (3,), np.uint8)
    for k, c in PAL.items():
        out[lbl == k] = c
    return out


files = sorted(Path(sys.argv[1]).glob("*.npz"))
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
picks = files[:: max(1, len(files) // n)][:n]
rows = []
for f in picks:
    d = np.load(f)
    gt = colour(d["gt"])
    own = d["seen"][int(f.stem.rsplit("_", 1)[1])]
    gt[own & ~ndimage.binary_erosion(own, iterations=2)] = (255, 255, 255)
    rows.append(np.concatenate([d["rgb"], colour(d["obs"]), gt], 1))
img = Image.fromarray(np.concatenate(rows, 0))
img.resize((img.width // 2, img.height // 2)).save(sys.argv[2])
