"""Feature-message variants with the SAME frozen lowerbound detector, from cached per-UAV BEV features
(infer_agents.py --save-bev). Receiver = UAV0; helpers = UAV1..4. Fusion rule = per-cell mean over the agents
whose feature is available at that cell (equals the upperbound rule when everything is sent).

Variants and bit cost per frame (sum over 4 helpers; BEV 80 ch x 256 x 256 cells at 0.8 m):
  ego                 : 0
  dense_fp16          : 4 * 80*65536*16
  dense_q8 / dense_q4 : per-channel min-max uniform quantisation, 4 * (80*65536*b + 80*2*16)
  sel_norm@p          : each helper sends its top-p fraction of cells by feature L2 norm (Where2comm-like
                        'what I have'), cost 4 * p*65536*(80*8 + 16) at q8 + 16-bit cell index
  sel_need@p          : top-p cells by helper_norm - ego_norm (CoSDH-like supply-demand; needs the receiver's
                        coarse demand map: + 65536 bit, 1 bit per cell)
Writes OUT/<variant>/results_nusc.json via the dataset formatter and OUT/feat_bits.json.

Run in the legacy env from ~/work/uav3d_collab:
  python feat_exp.py CONFIG CKPT BEV_DIR INFOS_PKL OUT
"""
import json
import os
import sys

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

CELLS = 256 * 256
C = 80


def quant(x, b):
    """per-channel uniform min-max quantisation of x [C,H,W]"""
    lo = x.amin(dim=(1, 2), keepdim=True)
    hi = x.amax(dim=(1, 2), keepdim=True)
    s = (hi - lo).clamp_min(1e-6) / (2 ** b - 1)
    return torch.round((x - lo) / s) * s + lo


def main():
    cfg_path, ckpt, bev_dir, infos, out = sys.argv[1:6]
    configs.load(cfg_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=cfg_path)
    cfg.data.test.dataset_root = "./data/uav3d/v1.0-mini/"
    cfg.data.test.ann_file = infos
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.encoders.camera.backbone.init_cfg = None
    dataset = build_dataset(cfg.data.test)
    toks = sorted(f[:-3] for f in os.listdir(bev_dir) if f.endswith(".pt"))
    keep = set(toks)
    dataset.data_infos = [i for i in dataset.data_infos if i["token"] in keep]
    order = [i["token"] for i in dataset.data_infos]

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location="cpu")
    model = model.cuda().eval()
    head = model.heads["object"]

    def decode(f, metas):
        # mmcv>=1.3 + torch>=1.6: wrap_fp16_model keeps fp32 weights (autocast only inside auto_fp16 methods);
        # infer_agents.decode also ran this path in fp32, so the 'ego' variant here equals agent0 there.
        x = model.encoders["camera"]["vtransform"].downsample(f[None].float())
        x = model.decoder["neck"](model.decoder["backbone"](x))
        b, s, l = head.get_bboxes(head(x, metas), metas)[0]
        return {"boxes_3d": b.to("cpu"), "scores_3d": s.cpu(), "labels_3d": l.cpu()}

    def fuse(ego, sent):
        """sent: list of (feature [C,H,W], mask [H,W] bool). Per-cell mean over available agents."""
        num = ego.clone()
        cnt = torch.ones_like(ego[:1])
        for f, m in sent:
            m = m[None].to(ego.dtype)
            num = num + f * m
            cnt = cnt + m
        return num / cnt

    # --- delay / loss bookkeeping: ego (UAV0) translation per token, previous tokens in the same scene ---
    info_by = {i["token"]: i for i in dataset.data_infos}
    samples = json.load(open("./data/uav3d/v1.0-mini/v1.0-mini/sample.json"))
    prev_tok = {s["token"]: s["prev"] for s in samples if s["prev"] and s["token"] in info_by and s["prev"] in info_by}
    cache = {}

    def load_bev(tok):
        if tok not in cache:
            cache.clear() if len(cache) > 4 else None
            cache[tok] = torch.load(os.path.join(bev_dir, tok + ".pt"))["bev"].cuda().float()
        return cache[tok]

    def warp(f, tok_from, tok_to):
        """Shift a BEV map [C,H,W] (dims: x, y; 0.8 m cells, +-102.4 m) from UAV0(tok_from) to UAV0(tok_to) frame."""
        d = (torch.tensor(info_by[tok_from]["ego2global_translation"][:2]) -
             torch.tensor(info_by[tok_to]["ego2global_translation"][:2])).float()
        # a cell at x_to in the new frame corresponds to x_from = x_to - d in the old frame; grid coords in [-1, 1]
        H, W = f.shape[-2:]
        xs = torch.linspace(-1 + 1 / H, 1 - 1 / H, H) - d[0] / 102.4
        ys = torch.linspace(-1 + 1 / W, 1 - 1 / W, W) - d[1] / 102.4
        gx, gy = torch.meshgrid(xs, ys)
        grid = torch.stack([gy, gx], dim=-1)[None].to(f.device)       # grid_sample wants (w-index, h-index)
        return torch.nn.functional.grid_sample(f[None], grid, align_corners=False)[0]

    LOSS_P = (0.2, 0.5)
    SEEDS = (0, 1, 2)
    drops = {}   # (seed, p) -> {tok: [bool]*4}; written out so box experiments reuse the same draws
    for s in SEEDS:
        for p in LOSS_P:
            g = torch.Generator().manual_seed(1000 * s + int(100 * p))
            drops[(s, p)] = {t: (torch.rand(4, generator=g) < p).tolist() for t in order}

    P = (0.01, 0.05, 0.2)
    variants = {"ego": [], "dense_fp16": [], "dense_q8": [], "dense_q4": [],
                "dense_q8_delay1": [], "dense_q8_delay2": [], "dense_q8_delay1_nowarp": []}
    for s in SEEDS:
        for p in LOSS_P:
            variants[f"dense_q8_loss{p}_s{s}"] = []
    for p in P:
        variants[f"sel_norm@{p}"] = []
        variants[f"sel_need@{p}"] = []
    bits = {"ego": 0, "dense_fp16": 4 * C * CELLS * 16, "dense_q8": 4 * (C * CELLS * 8 + C * 32),
            "dense_q4": 4 * (C * CELLS * 4 + C * 32)}
    for p in P:
        k = int(p * CELLS)
        bits[f"sel_norm@{p}"] = 4 * k * (C * 8 + 16) + 4 * C * 32
        bits[f"sel_need@{p}"] = bits[f"sel_norm@{p}"] + CELLS

    with torch.no_grad():
        for n, tok in enumerate(order):
            d = torch.load(os.path.join(bev_dir, tok + ".pt"))
            bev = d["bev"].cuda().float()          # [5, C, H, W]
            metas = d["metas"]
            ego, hel = bev[0], bev[1:]
            full = torch.ones(bev.shape[-2:], dtype=torch.bool, device=bev.device)
            variants["ego"].append(decode(ego, metas))
            variants["dense_fp16"].append(decode(fuse(ego, [(h.half().float(), full) for h in hel]), metas))
            variants["dense_q8"].append(decode(fuse(ego, [(quant(h, 8), full) for h in hel]), metas))
            variants["dense_q4"].append(decode(fuse(ego, [(quant(h, 4), full) for h in hel]), metas))
            for dly in (1, 2):
                src = tok
                for _ in range(dly):
                    src = prev_tok.get(src, src)       # first frames of a scene: no older message -> use oldest
                hel_d = load_bev(src)[1:] if src != tok else hel
                hel_d = [warp(h, src, tok) if src != tok else h for h in hel_d]
                variants[f"dense_q8_delay{dly}"].append(decode(fuse(ego, [(quant(h, 8), full) for h in hel_d]), metas))
                if dly == 1:   # sanity check of the warp direction: unwarped delayed features must be worse
                    raw_d = load_bev(src)[1:] if src != tok else hel
                    variants["dense_q8_delay1_nowarp"].append(
                        decode(fuse(ego, [(quant(h, 8), full) for h in raw_d]), metas))
            for s in SEEDS:
                for p in LOSS_P:
                    dr = drops[(s, p)][tok]
                    variants[f"dense_q8_loss{p}_s{s}"].append(
                        decode(fuse(ego, [(quant(h, 8), full) for h, x in zip(hel, dr) if not x]), metas))
            ego_norm = ego.norm(dim=0)
            for p in P:
                k = int(p * CELLS)
                for kind in ("norm", "need"):
                    sent = []
                    for h in hel:
                        score = h.norm(dim=0) - (ego_norm if kind == "need" else 0)
                        idx = torch.topk(score.flatten(), k).indices
                        m = torch.zeros(CELLS, dtype=torch.bool, device=bev.device)
                        m[idx] = True
                        sent.append((quant(h, 8), m.view_as(score)))
                    variants[f"sel_{kind}@{p}"].append(decode(fuse(ego, sent), metas))
            if n % 20 == 0:
                print(f"STEP {n}/{len(order)}", flush=True)

    os.makedirs(out, exist_ok=True)
    for name, res in variants.items():
        dataset.format_results(res, jsonfile_prefix=os.path.join(out, name))
    with open(os.path.join(out, "feat_bits.json"), "w") as f:
        json.dump({k: v / 1000 for k, v in bits.items()}, f, indent=1)   # kbit per frame
    with open(os.path.join(out, "drops.json"), "w") as f:
        json.dump({f"{p}_s{s}": v for (s, p), v in drops.items()}, f)
    with open(os.path.join(out, "prev_tok.json"), "w") as f:
        json.dump(prev_tok, f)
    print("RUN_DONE feat_exp", flush=True)


if __name__ == "__main__":
    main()
