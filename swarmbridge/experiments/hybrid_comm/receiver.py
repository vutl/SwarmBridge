"""Shared 172-channel receiver input, constructed exclusively from decoded bytes.

Channels: ego 80 | helper 80 | box attributes 8 | tile mask 1 | box mask 1 |
tile age 1 | box age 1. GT is intentionally not accessible to this class.
"""
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from core import GRID, cells, rotation, warp_grid
import tilecodec as TC
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uav3d_comm/proto"))
import boxcodec as BC
import kalman as K

CHANNELS = 172
FEATURE_CHANNELS = slice(80, 160)
FEATURE_META = (168, 170)


class ReceiverData:
    def __init__(self, prepared, device="cuda"):
        self.root = Path(prepared)
        self.device = device
        self.geo = json.loads((self.root / "geometry.json").read_text())
        self.cache = OrderedDict()
        self.packet_cache = OrderedDict()

    def ego(self, token):
        if token not in self.cache:
            p = Path("/home/uservutl/work/bev_train") / (token + ".pt")
            if not p.exists(): p = Path("/home/uservutl/work/bev_heldout") / (token + ".pt")
            d = torch.load(str(p), map_location="cpu")
            # Clone the slice: otherwise it retains storage of all five agents.
            self.cache[token] = (d["bev"][0].clone(), d["metas"])
            while len(self.cache) > 6: self.cache.popitem(last=False)
        v = self.cache.pop(token); self.cache[token] = v
        return v

    def packet(self, token, sender, rank, kind):
        folder = "boxes" if kind == "box" else "tiles"
        name = token + "_%s" % sender + ("" if kind == "box" else "_" + rank) + ".bin"
        key = folder + "/" + name
        if key not in self.packet_cache:
            self.packet_cache[key] = (self.root / key).read_bytes()
            while len(self.packet_cache) > 64: self.packet_cache.popitem(last=False)
        return self.packet_cache[key]

    def build(self, ctx, with_tiles=True):
        token, count = ctx["token"], ctx["k"]
        ego, metas = self.ego(token)
        device = self.device
        x = torch.zeros(CHANNELS, GRID, GRID, device=device)
        x[:80] = ego.to(device=device, dtype=torch.float32)
        pose = self.geo["poses"][token]
        num = torch.zeros_like(x[:80]); den = torch.zeros_like(x[0]); age_num = torch.zeros_like(den)
        boxes = np.zeros((8, GRID, GRID), np.float32)
        scores = np.zeros((GRID, GRID), np.float32)
        box_age = np.zeros((GRID, GRID), np.float32)
        bits = 0
        for sender, source in sorted(ctx["helpers"].items()):
            src, age = source["token"], source["age"]
            assert 0 <= age <= 2
            packet = self.packet(src, sender, ctx["rank"], "box")
            bits += 8 * len(packet)
            _, b = BC.decode(packet)
            b = K.forecast(b, age)
            ix, valid = cells(b["xy"], pose)
            vel = b["vel"] @ rotation(pose[2])
            for i in np.where(valid)[0]:
                row, col = ix[i]
                s = b["score"][i]
                if s <= scores[row, col]: continue
                yaw = b["yaw"][i] - pose[2]
                boxes[:, row, col] = [*(b["size"][i] / 5), np.sin(yaw), np.cos(yaw), s,
                                       vel[i, 0] / 12.8, vel[i, 1] / 12.8]
                scores[row, col] = s
                box_age[row, col] = age / 2.0
            if not count or not with_tiles: continue
            packet = TC.prefix(self.packet(src, sender, ctx["rank"], "tile"), count)
            bits += len(packet) * 8
            _, idx, tl = TC.decode(packet)
            dense, mask = TC.scatter(idx, tl)
            f = torch.from_numpy(dense).to(device)
            m = torch.from_numpy(mask).to(device)
            spose = self.geo["poses"][src]
            f, m = warp_grid(f, spose, pose), warp_grid(m, spose, pose)
            num += f
            den += m
            age_num += m * (age / 2.0)
        if with_tiles:
            x[80:160] = num / den.clamp_min(1)[None]
            x[168] = den.clamp(max=1)
            x[170] = age_num / den.clamp_min(1)
        x[160:168] = torch.from_numpy(boxes).to(device)
        x[169] = torch.from_numpy((scores > 0).astype(np.float32)).to(device)
        x[171] = torch.from_numpy(box_age).to(device)
        return x, metas, bits


def expand_input(ds):
    old = ds[0]
    new = nn.Conv2d(CHANNELS, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :80] = old.weight
        if old.bias is not None: new.bias.copy_(old.bias)
    ds[0] = new


def forward(model, ds, x, metas):
    dec = model.decoder
    y = dec["neck"](dec["backbone"](ds(x[None])))
    return model.heads["object"](y, metas)


def train_modules(model, ds):
    model.eval()
    for module in [ds, model.decoder, model.heads["object"]]:
        module.train()
        for sub in module.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm): sub.eval()


def state(model, ds):
    return {"ds": ds.state_dict(), "decoder": model.decoder.state_dict(),
            "head": model.heads["object"].state_dict()}


def load_state(model, ds, data):
    ds.load_state_dict(data["ds"])
    model.decoder.load_state_dict(data["decoder"])
    model.heads["object"].load_state_dict(data["head"])
