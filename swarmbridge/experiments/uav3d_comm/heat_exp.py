"""Evidence-map messages: each UAV runs the SAME frozen lowerbound decoder+CenterHead on its own BEV feature and
sends its object-centre probability map (CenterHead heatmap, sigmoid) plus the box regression at the cells it
sends. The receiver fuses the maps per cell BEFORE box decoding, so weak evidence seen by several UAVs can add up
(something box messages cannot do: the detector drops everything below score 0.1).

Variants (receiver UAV0, helpers UAV1..4), each with the official cap of 83 boxes/frame and with cap 500:
  ego        : UAV0 alone
  heat_max   : fused p = max_j p_j            (regression from the argmax UAV per cell)
  heat_nor   : fused p = 1 - prod_j (1 - p_j) (noisy-OR: independent weak evidence adds up)
  heat_nor@s : noisy-OR, helpers send only cells with p_j >= s (sparse message), s in {0.02, 0.05}
Bits (sparse): per sent cell 8 bit prob + 14 bit index + 8 regression channels x 8 bit.
Run in the legacy env from ~/work/uav3d_collab:  python heat_exp.py CONFIG CKPT BEV_DIR INFOS OUT
"""
import json
import os
import sys

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


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
    keep = {f[:-3] for f in os.listdir(bev_dir) if f.endswith(".pt")}
    dataset.data_infos = [i for i in dataset.data_infos if i["token"] in keep]
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt, map_location="cpu")
    model = model.cuda().eval()
    ds, dec, head = model.encoders["camera"]["vtransform"].downsample, model.decoder, model.heads["object"]
    cap0 = head.test_cfg["post_max_size"]

    def preds_of(f, metas):
        return head(dec["neck"](dec["backbone"](ds(f[None]))), metas)

    def fuse(P, rule, sparse=None):
        """P: list over agents of preds_dicts (tasks x [dict]); agent 0 = receiver."""
        out_p = []
        for task in range(len(P[0])):
            dicts = [p[task][0] for p in P]
            prob = torch.stack([d["heatmap"].sigmoid() for d in dicts])      # [A,B,C,H,W]
            if sparse is not None:
                mask = torch.ones_like(prob)
                mask[1:] = (prob[1:] >= sparse).float()
                prob = prob * mask
            if rule == "max":
                fp = prob.max(0).values
            else:
                fp = 1 - torch.prod(1 - prob, dim=0)
            arg = prob.max(0).indices                                         # [B,C,H,W] (C=1 class)
            fused = {}
            for k, v in dicts[0].items():
                if k == "heatmap":
                    fused[k] = torch.logit(fp.clamp(1e-6, 1 - 1e-6))
                else:
                    stack = torch.stack([d[k] for d in dicts])                # [A,B,K,H,W]
                    idx = arg[:, :1].expand(-1, stack.shape[2], -1, -1)[None]
                    fused[k] = torch.gather(stack, 0, idx)[0]
            out_p.append([fused])
        return out_p

    names = []
    for cap in (cap0, 500):
        for v in ("ego", "heat_max", "heat_nor", "heat_nor@0.02", "heat_nor@0.05"):
            names.append(f"{v}_cap{cap}")
    res = {n: [] for n in names}
    sent_cells = {0.02: [], 0.05: []}
    with torch.no_grad():
        for n, info in enumerate(dataset.data_infos):
            d = torch.load(os.path.join(bev_dir, info["token"] + ".pt"))
            bev, metas = d["bev"].cuda().float(), d["metas"]
            P = [preds_of(bev[j], metas) for j in range(5)]
            for s in (0.02, 0.05):
                sent_cells[s].append(sum(int((p[0][0]["heatmap"].sigmoid() >= s).sum()) for p in P[1:]))
            for cap in (cap0, 500):
                head.test_cfg["post_max_size"] = cap
                for v in ("ego", "heat_max", "heat_nor", "heat_nor@0.02", "heat_nor@0.05"):
                    if v == "ego":
                        pd = P[0]
                    elif v == "heat_max":
                        pd = fuse(P, "max")
                    elif v == "heat_nor":
                        pd = fuse(P, "nor")
                    else:
                        pd = fuse(P, "nor", sparse=float(v.split("@")[1]))
                    b, s_, l = head.get_bboxes(pd, metas)[0]
                    res[f"{v}_cap{cap}"].append({"boxes_3d": b.to("cpu"), "scores_3d": s_.cpu(), "labels_3d": l.cpu()})
            head.test_cfg["post_max_size"] = cap0
            if n % 20 == 0:
                print(f"STEP {n}/{len(dataset.data_infos)}", flush=True)
    os.makedirs(out, exist_ok=True)
    for name, r in res.items():
        dataset.format_results(r, jsonfile_prefix=os.path.join(out, name))
    H = int(P[0][0][0]["heatmap"].shape[-1]) * int(P[0][0][0]["heatmap"].shape[-2])
    reg_ch = sum(int(v.shape[1]) for k, v in P[0][0][0].items() if k != "heatmap")
    bits = {}
    for cap in (cap0, 500):
        bits[f"ego_cap{cap}"] = 0
        for v in ("heat_max", "heat_nor"):
            bits[f"{v}_cap{cap}"] = 4 * H * (8 + reg_ch * 8) / 1000
        for s in (0.02, 0.05):
            c = sum(sent_cells[s]) / len(sent_cells[s])
            bits[f"heat_nor@{s}_cap{cap}"] = c * (8 + 14 + reg_ch * 8) / 1000
    json.dump(bits, open(os.path.join(out, "heat_bits.json"), "w"), indent=1)
    print("heatmap cells", H, "reg channels", reg_ch, "mean sent cells", {s: sum(v) / len(v) for s, v in sent_cells.items()})
    print("RUN_DONE heat_exp", flush=True)


if __name__ == "__main__":
    main()
