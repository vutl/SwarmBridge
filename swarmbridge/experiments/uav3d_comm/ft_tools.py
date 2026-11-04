"""Helpers around the decoders saved by train_fusion.py (arm_s*/decoder.pt).

  dump  : decode EVERY UAV's cached BEV with a saved single-UAV decoder (ego arm) -> per-UAV detections for the
          given BEV dirs (train + held-out), so the box arm can be retrained on fine-tuned helper detections.
          python ft_tools.py dump CONFIG CKPT DECODER_PT INFOS OUT BEV_DIR [BEV_DIR ...]
  cap   : re-evaluate a saved arm on the held-out scenes (clean condition) with a different post_max_size cap.
          python ft_tools.py cap CONFIG CKPT ARM DECODER_PT INFOS AGENT_DETS BEV_HELD OUT CAP
Run in the legacy env from ~/work/uav3d_collab.
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

HALF, CELL, G, NBOX = 102.4, 0.8, 256, 6


def setup(cfg_path, ckpt, infos, arm, dec_pt):
    configs.load(cfg_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=cfg_path)
    cfg.data.test.dataset_root = "./data/uav3d/v1.0-mini/"
    cfg.data.test.ann_file = infos
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.encoders.camera.backbone.init_cfg = None
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt, map_location="cpu")
    model = model.cuda().eval()
    ds = model.encoders["camera"]["vtransform"].downsample
    if arm == "box":
        old = ds[0]
        ds[0] = nn.Conv2d(old.in_channels + NBOX, old.out_channels, 3, padding=1, bias=False).cuda()
    sd = torch.load(dec_pt, map_location="cuda")
    ds.load_state_dict(sd["ds"])
    model.decoder.load_state_dict(sd["decoder"])
    model.heads["object"].load_state_dict(sd["head"])
    return dataset, model, ds


def run_decode(model, ds, x, metas):
    head = model.heads["object"]
    y = model.decoder["neck"](model.decoder["backbone"](ds(x[None])))
    b, s, l = head.get_bboxes(head(y, metas), metas)[0]
    return {"boxes_3d": b.to("cpu"), "scores_3d": s.cpu(), "labels_3d": l.cpu()}


def dump(argv):
    cfg_path, ckpt, dec_pt, infos, out = argv[:5]
    dirs = argv[5:]
    dataset, model, ds = setup(cfg_path, ckpt, infos, "ego", dec_pt)
    if os.environ.get("CAP"):
        model.heads["object"].test_cfg["post_max_size"] = int(os.environ["CAP"])
        print("post_max_size", os.environ["CAP"], flush=True)
    where = {}
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(".pt"):
                where[f[:-3]] = d
    dataset.data_infos = [i for i in dataset.data_infos if i["token"] in where]
    res = {j: [] for j in range(5)}
    with torch.no_grad():
        for n, info in enumerate(dataset.data_infos):
            d = torch.load(os.path.join(where[info["token"]], info["token"] + ".pt"))
            bev = d["bev"].cuda().float()
            for j in range(5):
                res[j].append(run_decode(model, ds, bev[j], d["metas"]))
            if n % 50 == 0:
                print(f"STEP dump {n}/{len(dataset.data_infos)}", flush=True)
    for j, r in res.items():
        dataset.format_results(r, jsonfile_prefix=os.path.join(out, f"agent{j}"))
    print("RUN_DONE dump", flush=True)


def cap(argv):
    cfg_path, ckpt, arm, dec_pt, infos, agent_dets, bev_held, out, capn = argv[:9]
    dataset, model, ds = setup(cfg_path, ckpt, infos, arm, dec_pt)
    model.heads["object"].test_cfg["post_max_size"] = int(capn)
    info_by = {i["token"]: i for i in dataset.data_infos}
    held = {f[:-3] for f in os.listdir(bev_held) if f.endswith(".pt")}
    dataset.data_infos = [i for i in dataset.data_infos if i["token"] in held]
    dets = {}
    if arm == "box":
        for j in range(1, 5):
            R = json.load(open(os.path.join(agent_dets, f"agent{j}", "results_nusc.json")))["results"]
            for t, boxes in R.items():
                dets.setdefault(t, {})[j] = boxes

    def poture(tok):
        m = torch.zeros(NBOX, G, G); best = torch.zeros(G, G)
        off = info_by[tok]["ego2global_translation"][:2]
        for j in range(1, 5):
            for b in dets.get(tok, {}).get(j, []):
                s = b["detection_score"]
                if s < 0.2:
                    continue
                ix = int((b["translation"][0] - off[0] + HALF) // CELL)
                iy = int((b["translation"][1] - off[1] + HALF) // CELL)
                if not (0 <= ix < G and 0 <= iy < G) or s <= best[ix, iy]:
                    continue
                best[ix, iy] = s
                q = b["rotation"]; yaw = 2 * math.atan2(q[3], q[0]); w, l, h = b["size"]
                m[:, ix, iy] = torch.tensor([w / 5, l / 5, h / 5, math.sin(yaw), math.cos(yaw), s])
        return m

    # optional stress conditions via env: DELAY=d (helper message from t-d; features warped by UAV0 motion) and
    # DROPS=path:key (helper messages dropped with the shared Bernoulli draws)
    delay = int(os.environ.get("DELAY", "0"))
    drop_map = None
    if os.environ.get("DROPS"):
        p, key = os.environ["DROPS"].rsplit(":", 1)
        drop_map = json.load(open(p))[key]
    samples = json.load(open("./data/uav3d/v1.0-mini/v1.0-mini/sample.json"))
    prev = {s["token"]: s["prev"] for s in samples}

    def warp(f, t_from, t_to):
        a = info_by[t_from]["ego2global_translation"][:2]
        b = info_by[t_to]["ego2global_translation"][:2]
        dx, dy = a[0] - b[0], a[1] - b[1]
        H, W = f.shape[-2:]
        xs = torch.linspace(-1 + 1 / H, 1 - 1 / H, H) - dx / HALF
        ys = torch.linspace(-1 + 1 / W, 1 - 1 / W, W) - dy / HALF
        gx, gy = torch.meshgrid(xs, ys)
        grid = torch.stack([gy, gx], dim=-1)[None].to(f.device)
        return torch.nn.functional.grid_sample(f[None], grid, align_corners=False)[0]

    def poture_from(tok_msg, tok_now, helpers):
        m = torch.zeros(NBOX, G, G); best = torch.zeros(G, G)
        off = info_by[tok_now]["ego2global_translation"][:2]
        for j in helpers:
            for b in dets.get(tok_msg, {}).get(j, []):
                s = b["detection_score"]
                if s < 0.2:
                    continue
                ix = int((b["translation"][0] - off[0] + HALF) // CELL)
                iy = int((b["translation"][1] - off[1] + HALF) // CELL)
                if not (0 <= ix < G and 0 <= iy < G) or s <= best[ix, iy]:
                    continue
                best[ix, iy] = s
                q = b["rotation"]; yaw = 2 * math.atan2(q[3], q[0]); w, l, h = b["size"]
                m[:, ix, iy] = torch.tensor([w / 5, l / 5, h / 5, math.sin(yaw), math.cos(yaw), s])
        return m

    res = []
    with torch.no_grad():
        for info in dataset.data_infos:
            tok = info["token"]
            d = torch.load(os.path.join(bev_held, tok + ".pt"))
            bev = d["bev"].cuda().float()
            src = tok
            for _ in range(delay):
                src = prev.get(src) or src
            helpers = [j for j in range(1, 5) if not (drop_map and drop_map[tok][j - 1])]
            if arm == "ego":
                x = bev[0]
            elif arm == "feat":
                hb = torch.load(os.path.join(bev_held, src + ".pt"))["bev"].cuda().float() if src != tok else bev
                hs = [hb[j] if src == tok else warp(hb[j], src, tok) for j in helpers]
                x = torch.stack([bev[0]] + hs).mean(0)
            else:
                x = torch.cat([bev[0], poture_from(src, tok, helpers).cuda()])
            res.append(run_decode(model, ds, x, d["metas"]))
    dataset.format_results(res, jsonfile_prefix=out)
    print("RUN_DONE cap", arm, capn, "delay", delay, "drops", os.environ.get("DROPS", ""), flush=True)


if __name__ == "__main__":
    {"dump": dump, "cap": cap}[sys.argv[1]](sys.argv[2:])
