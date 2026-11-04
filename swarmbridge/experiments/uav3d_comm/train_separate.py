"""Controlled message-representation ablation on UAV3D (what ChatGPT/Codex asked for):
SAME frozen lowerbound encoder (cached per-UAV BEV features), SAME training scenes, SAME decoder init (lowerbound
decoder) and SAME training budget; only the message the receiver UAV0 gets from UAV1..4 differs:

  ego  : no message (control: shows the effect of fine-tuning alone)
  feat : dense BEV features of the helpers, fused by per-cell mean (upperbound rule)
  box  : helpers' detections (lowerbound decoded per UAV) rendered as a 6-channel BEV map at box centres
         [w, l, h, sin yaw, cos yaw, score], max-score merged over helpers, concatenated to the ego feature
         (CERF 'Poture'; extra input weights start at zero, so the box arm starts exactly at 'ego').

Train on the 17 non-held-out mini scenes (likely seen by the encoder in its own training; same for every arm),
evaluate on scene_11/13/15 under: clean, delay 1 / 2 frames (helper message from t-1 / t-2; features warped by
UAV0 ego motion, boxes kept in global coordinates), and packet loss p (same Bernoulli draws as feat_exp drops.json).

Run in the legacy env from ~/work/uav3d_collab:
  python train_fusion.py CONFIG CKPT BEV_TRAIN BEV_HELD INFOS AGENT_DET_DIR DROPS_JSON OUT --arm box --seed 0
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

HALF, CELL, G = 102.4, 0.8, 256
NBOX = 6


def parse():
    ap = argparse.ArgumentParser()
    for a in ("config", "ckpt", "bev_train", "bev_held", "infos", "agent_dets", "drops", "out"):
        ap.add_argument(a)
    ap.add_argument("--arm", choices=["ego", "feat", "box"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--axis", default="auto", choices=["auto", "xy", "yx"])
    ap.add_argument("--robust", action="store_true",
                    help="train with stale/lost messages: delay d~U{0,1,2} frames, each helper dropped with p=0.2")
    ap.add_argument("--cap", type=int, default=0, help="post_max_size used at evaluation (0 = config default 83)")
    ap.add_argument("--tag", default="", help="suffix of the output dir name")
    ap.add_argument("--age", action="store_true",
                    help="append one input channel = message age (frames / 2), so the decoder knows how stale it is")
    return ap.parse_args()


def main():
    args = parse()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data.test.dataset_root = "./data/uav3d/v1.0-mini/"
    cfg.data.test.ann_file = args.infos
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.encoders.camera.backbone.init_cfg = None
    dataset = build_dataset(cfg.data.test)
    idx_of = {i["token"]: k for k, i in enumerate(dataset.data_infos)}
    info_by = {i["token"]: i for i in dataset.data_infos}
    samples = json.load(open("./data/uav3d/v1.0-mini/v1.0-mini/sample.json"))
    prev_tok = {s["token"]: s["prev"] for s in samples if s["prev"]}

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = model.cuda()
    ds = model.encoders["camera"]["vtransform"].downsample
    dec, head = model.decoder, model.heads["object"]
    # Separate ego and aggregate helper channels; initialize as the pretrained ego decoder.
    extra = (ds[0].in_channels if args.arm == "feat" else NBOX if args.arm == "box" else 0) + (1 if args.age and args.arm != "ego" else 0)
    if extra:
        old = ds[0]
        new = nn.Conv2d(old.in_channels + extra, old.out_channels, 3, padding=1, bias=False).cuda()
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
        ds[0] = new
    params = list(ds.parameters()) + list(dec.parameters()) + list(head.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    # ---- helper detections (global frame) for the box message ----
    dets = {}
    for j in range(1, 5):
        R = json.load(open(os.path.join(args.agent_dets, f"agent{j}", "results_nusc.json")))["results"]
        for t, boxes in R.items():
            dets.setdefault(t, {})[j] = boxes

    def ego_xy(tok):
        return np.array(info_by[tok]["ego2global_translation"][:2])

    def poture(tok_msg, tok_now, helpers, axis):
        """Render helpers' boxes from frame tok_msg into the BEV grid of UAV0 at tok_now."""
        m = torch.zeros(NBOX, G, G)
        best = torch.zeros(G, G)
        off = ego_xy(tok_now)
        for j in helpers:
            for b in dets.get(tok_msg, {}).get(j, []):
                s = b["detection_score"]
                if s < 0.2:
                    continue
                x, y = b["translation"][0] - off[0], b["translation"][1] - off[1]
                ix, iy = int((x + HALF) // CELL), int((y + HALF) // CELL)
                if not (0 <= ix < G and 0 <= iy < G):
                    continue
                if axis == "yx":
                    ix, iy = iy, ix
                if s <= best[ix, iy]:
                    continue
                best[ix, iy] = s
                q = b["rotation"]
                yaw = 2 * math.atan2(q[3], q[0])
                w, l, h = b["size"]
                m[:, ix, iy] = torch.tensor([w / 5, l / 5, h / 5, math.sin(yaw), math.cos(yaw), s])
        return m

    def warp(f, tok_from, tok_to):
        d = torch.tensor(ego_xy(tok_from) - ego_xy(tok_to)).float()
        H, W = f.shape[-2:]
        xs = torch.linspace(-1 + 1 / H, 1 - 1 / H, H) - d[0] / HALF
        ys = torch.linspace(-1 + 1 / W, 1 - 1 / W, W) - d[1] / HALF
        gx, gy = torch.meshgrid(xs, ys)
        grid = torch.stack([gy, gx], dim=-1)[None].to(f.device)
        return torch.nn.functional.grid_sample(f[None], grid, align_corners=False)[0]

    def load(d, tok):
        return torch.load(os.path.join(d, tok + ".pt"))

    # ---- axis convention check: mean ego feature norm at GT-box cells under both conventions ----
    held = sorted(f[:-3] for f in os.listdir(args.bev_held) if f.endswith(".pt"))
    train = sorted(f[:-3] for f in os.listdir(args.bev_train) if f.endswith(".pt") and f[:-3] not in set(held))
    axis = args.axis
    if axis == "auto":
        score = {"xy": [], "yx": []}
        for tok in train[:10]:
            nrm = load(args.bev_train, tok)["bev"][0].float().norm(dim=0)
            gt = info_by[tok]["gt_boxes"]
            for ax in ("xy", "yx"):
                v = []
                for b in gt:
                    ix, iy = int((b[0] + HALF) // CELL), int((b[1] + HALF) // CELL)
                    if 0 <= ix < G and 0 <= iy < G:
                        v.append(float(nrm[ix, iy] if ax == "xy" else nrm[iy, ix]))
                score[ax].append(np.mean(v) / float(nrm.mean()))
        axis = max(score, key=lambda a: np.mean(score[a]))
        print("AXIS check (feature norm at GT cells / mean):", {k: round(float(np.mean(v)), 3) for k, v in score.items()},
              "->", axis, flush=True)

    def build_input(bev, tok_now, tok_msg, drop, age=0):
        """bev: [5,C,H,W] of the message frame (tok_msg) for helpers; ego always from tok_now."""
        ego = bev_now[0]
        helpers = [j for j in range(1, 5) if not drop[j - 1]]
        if args.arm == "ego":
            return ego
        if args.arm == "feat":
            hs = [bev[j] for j in helpers]
            if tok_msg != tok_now:
                hs = [warp(h, tok_msg, tok_now) for h in hs]
            # Helpers are pooled, but never mixed into the current ego observation.
            # No helper means a zero message rather than duplicating ego evidence.
            helper_mean = torch.stack(hs).mean(0) if hs else torch.zeros_like(ego)
            x = torch.cat([ego, helper_mean])
        else:
            x = torch.cat([ego, poture(tok_msg, tok_now, helpers, axis).cuda()])
        if args.age:
            x = torch.cat([x, torch.full_like(x[:1], age / 2.0)])
        return x

    def gt_of(tok):
        ann = dataset.get_ann_info(idx_of[tok])
        boxes = ann["gt_bboxes_3d"]
        keep = boxes.in_range_bev([-HALF, -HALF, HALF, HALF])
        return [boxes[keep].to("cuda")], [torch.as_tensor(ann["gt_labels_3d"])[keep.cpu()].cuda()]

    def forward(x, metas):
        y = dec["neck"](dec["backbone"](ds(x[None])))
        return head(y, metas)

    # BN layers keep their pretrained statistics (batch size 1)
    def set_train():
        model.eval()
        for mod in (ds, dec, head):
            mod.train()
            for m in mod.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    nodrop = [False] * 4
    t0 = time.time()
    for ep in range(args.epochs):
        random.shuffle(train)
        set_train()
        tot = 0.0
        for n, tok in enumerate(train):
            d = load(args.bev_train, tok)
            bev_now = d["bev"].cuda().float()
            if args.robust:
                src, age = tok, 0
                for _ in range(random.randint(0, 2)):
                    p = prev_tok.get(src)
                    if p and os.path.exists(os.path.join(args.bev_train, p + ".pt")):
                        src, age = p, age + 1
                bev_msg = load(args.bev_train, src)["bev"].cuda().float() if src != tok else bev_now
                x = build_input(bev_msg, tok, src, [random.random() < 0.2 for _ in range(4)], age)
            else:
                x = build_input(bev_now, tok, tok, nodrop)
            pred = forward(x, d["metas"])
            gb, gl = gt_of(tok)
            losses = head.loss(gb, gl, pred)
            # CenterHead.loss keys are e.g. 'heatmap/task0', 'bbox/task0' (the 'loss/' prefix is added by the model)
            loss = sum(v for k, v in losses.items() if torch.is_tensor(v) and v.requires_grad)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        print(f"STEP epoch {ep} loss {tot / len(train):.4f} {time.time() - t0:.0f}s", flush=True)

    # ---- evaluation on held-out under clean / delay / loss ----
    model.eval()
    if args.cap:
        head.test_cfg["post_max_size"] = args.cap
    drops = json.load(open(args.drops))
    conds = {"clean": (0, None), "delay1": (1, None), "delay2": (2, None)}
    for key in drops:
        conds[f"loss{key}"] = (0, key)
    results = {c: [] for c in conds}
    order = [i["token"] for i in dataset.data_infos if i["token"] in set(held)]
    with torch.no_grad():
        for tok in order:
            d = load(args.bev_held, tok)
            bev_now = d["bev"].cuda().float()
            for c, (dly, dk) in conds.items():
                src, age = tok, 0
                for _ in range(dly):
                    if prev_tok.get(src):
                        src, age = prev_tok[src], age + 1
                bev_msg = load(args.bev_held, src)["bev"].cuda().float() if src != tok else bev_now
                drop = drops[dk][tok] if dk else nodrop
                x = build_input(bev_msg, tok, src, drop, age)
                pred = forward(x, d["metas"])
                b, s, l = head.get_bboxes(pred, d["metas"])[0]
                results[c].append({"boxes_3d": b.to("cpu"), "scores_3d": s.cpu(), "labels_3d": l.cpu()})
    # ego arm: the fine-tuned single-UAV decoder applied to EVERY UAV's own feature -> per-UAV detections, so that
    # box late fusion can also be scored with the Town10-fine-tuned detector (fair vs the trained arms)
    per_agent = {j: [] for j in range(5)}
    if args.arm == "ego":
        with torch.no_grad():
            for tok in order:
                d = load(args.bev_held, tok)
                bev = d["bev"].cuda().float()
                for j in range(5):
                    b, s, l = head.get_bboxes(forward(bev[j], d["metas"]), d["metas"])[0]
                    per_agent[j].append({"boxes_3d": b.to("cpu"), "scores_3d": s.cpu(), "labels_3d": l.cpu()})
    dataset.data_infos = [info_by[t] for t in order]
    out = os.path.join(args.out, f"{args.arm}{args.tag}_s{args.seed}")
    for c, res in results.items():
        dataset.format_results(res, jsonfile_prefix=os.path.join(out, c))
    if args.arm == "ego":
        for j, res in per_agent.items():
            dataset.format_results(res, jsonfile_prefix=os.path.join(out, "agents", f"agent{j}"))
    torch.save({"ds": ds.state_dict(), "decoder": dec.state_dict(), "head": head.state_dict()},
               os.path.join(out, "decoder.pt"))
    print("RUN_DONE train_fusion", args.arm, args.seed, flush=True)


if __name__ == "__main__":
    sys.exit(main())
