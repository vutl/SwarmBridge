"""Label-free geometry and deterministic utilities shared by the matched pilot.

The cache uses [channel, x, y], in a world-aligned grid centred on UAV0.
General pose utilities support yaw; the mini metadata has identity rotations.
No convention is estimated from detection labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HALF, CELL, GRID = 102.4, 0.8, 256
DEV = ("scene_11", "scene_13", "scene_15")
SPLIT_SEED = 20261105


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def world_to_local(xy, pose):
    return (np.asarray(xy) - np.asarray(pose[:2])) @ rotation(pose[2])


def local_to_world(xy, pose):
    return np.asarray(xy) @ rotation(pose[2]).T + np.asarray(pose[:2])


def cells(xy, pose=(0, 0, 0)):
    q = world_to_local(xy, pose)
    ix = np.floor((q + HALF) / CELL).astype(int)
    valid = ((ix >= 0) & (ix < GRID)).all(axis=-1)
    return ix, valid


def warp_grid(feat, source_pose, target_pose):
    """A target cell samples its location in the source grid. Supports CHW/HW."""
    import torch
    import torch.nn.functional as F
    if np.allclose(source_pose, target_pose, atol=1e-9, rtol=0):
        return feat
    one = feat.ndim == 2
    x = feat[None] if one else feat
    v = (torch.arange(GRID, device=x.device, dtype=x.dtype) + 0.5) * CELL - HALF
    xx, yy = torch.meshgrid(v, v)
    local = torch.stack([xx, yy], -1)
    rd = torch.as_tensor(rotation(target_pose[2]), device=x.device, dtype=x.dtype)
    rs = torch.as_tensor(rotation(source_pose[2]), device=x.device, dtype=x.dtype)
    delta = torch.as_tensor(np.asarray(target_pose[:2]) - np.asarray(source_pose[:2]),
                            device=x.device, dtype=x.dtype)
    src = (local @ rd.T + delta) @ rs
    # grid_sample coordinates are (column=y, row=x), not (x,y).
    grid = torch.stack([src[..., 1] / HALF, src[..., 0] / HALF], -1)[None]
    z = F.grid_sample(x[None], grid, align_corners=False)[0]
    return z[0] if one else z


def nms_max(records, radius=2.0, cap=500):
    """Python stable sort deliberately matches the historical scratchpad."""
    keep = []
    for b in sorted(records, key=lambda x: -x["detection_score"]):
        x, y = b["translation"][:2]
        if all(np.hypot(x - k["translation"][0], y - k["translation"][1]) > radius for k in keep):
            keep.append(b)
    return keep[:cap]


def make_split(root):
    m = Path(root) / "v1.0-mini"
    scenes = json.loads((m / "scene.json").read_text())
    samples = {s["token"]: s for s in json.loads((m / "sample.json").read_text())}
    names = [s["name"] for s in scenes if s["name"] not in DEV]
    names.sort(key=lambda s: hashlib.sha256((str(SPLIT_SEED) + "|" + s).encode()).hexdigest())
    assert len(names) == 17
    groups = {"train": names[:13], "val": names[13:], "dev": list(DEV)}
    frames = {}
    for s in scenes:
        tok, seq = s["first_sample_token"], []
        while tok:
            seq.append(tok)
            tok = samples[tok]["next"]
        frames[s["name"]] = seq
    return dict(seed=SPLIT_SEED, scenes=groups, frames=frames,
                tokens={g: [t for s in ss for t in frames[s]] for g, ss in groups.items()},
                provenance="Existing ego/feature decoders were adapted on all 17 non-dev scenes. "
                           "The new split controls this adaptation only; all mini results are development evidence.")


def context(split, token, rng, k=None, random_tiles=None):
    scene = next(s for s, seq in split["frames"].items() if token in seq)
    seq = split["frames"][scene]
    t = seq.index(token)
    count = int(rng.integers(0, 5))
    helpers = sorted(int(j) for j in rng.choice(np.arange(1, 5), count, replace=False))
    sources = {}
    for j in helpers:
        age = int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        if t >= age:
            sources[str(j)] = dict(token=seq[t - age], age=age)
    return dict(token=token, helpers=sources, k=int(rng.choice([0, 16, 64])) if k is None else k,
                rank=("random" if rng.random() < 0.2 else "obj") if random_tiles is None else random_tiles)
