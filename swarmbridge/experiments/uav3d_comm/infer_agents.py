"""Run a UAV3D collaborative BEVFusion checkpoint on v1.0-mini and save, per sample:
  - 'official': the model's own output (exactly what tools/test.py produces for this com type),
  - 'agent{j}' : UAV j's own BEV feature (mean of its 5 cameras, already in the UAV0 frame) decoded alone,
  - 'mean5'    : mean of the 5 per-UAV BEV features decoded (= upperbound fusion rule).
Each variant is written as a nuScenes-format results JSON (global frame) via the dataset's own formatter.

Run inside the legacy env from the UAV3D collaborative code dir (python 3.8, torch 1.10):
  cd ~/work/uav3d_collab && ~/envs/uav3d/bin/python <this file> CONFIG CKPT OUT_DIR [--limit N]
"""
import argparse
import os
import pickle
import sys
import time

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

ROOT = "./data/uav3d/v1.0-mini/"


def merged_infos(path):
    infos = []
    for split in ("train", "val"):
        with open(ROOT + f"uav3d_infos_{split}.pkl", "rb") as f:
            infos += pickle.load(f)["infos"]
    infos.sort(key=lambda i: i["timestamp"])
    with open(path, "wb") as f:
        pickle.dump({"infos": infos, "metadata": {"version": "v1.0-mini"}}, f)
    return len(infos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokens", default="", help="optional file with sample tokens to keep (one per line)")
    ap.add_argument("--save-bev", default="", help="dir: save per-UAV BEV features [5,C,H,W] fp16 + metas per token")
    ap.add_argument("--scenes", default="", help="comma list of scene names to keep (needs v1.0-mini scene.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    ann = os.path.abspath(os.path.join(args.out, "infos_all.pkl"))
    print("infos:", merged_infos(ann), flush=True)
    cfg.data.test.dataset_root = ROOT
    cfg.data.test.ann_file = ann
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    # the Swin init_cfg points to a URL; weights come from the checkpoint anyway
    cfg.model.encoders.camera.backbone.init_cfg = None

    dataset = build_dataset(cfg.data.test)
    if args.tokens:
        keep = set(open(args.tokens).read().split())
        dataset.data_infos = [i for i in dataset.data_infos if i["token"] in keep]
    if args.scenes:
        import json
        meta = ROOT + "v1.0-mini/"
        names = {s["token"]: s["name"] for s in json.load(open(meta + "scene.json"))}
        scene_of = {s["token"]: names[s["scene_token"]] for s in json.load(open(meta + "sample.json"))}
        want = set(args.scenes.split(","))
        dataset.data_infos = [i for i in dataset.data_infos if scene_of[i["token"]] in want]
    if args.limit:
        dataset.data_infos = dataset.data_infos[: args.limit]
    if args.save_bev:
        os.makedirs(args.save_bev, exist_ok=True)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model.CLASSES = dataset.CLASSES
    core = model
    model = MMDataParallel(model, device_ids=[0]).eval()

    cap = {}
    core.encoders["camera"]["vtransform"].register_forward_hook(lambda m, i, o: cap.__setitem__("bev", o))
    core.heads["object"].register_forward_pre_hook(lambda m, i: cap.__setitem__("metas", i[1]))

    def decode(bev):
        x = core.encoders["camera"]["vtransform"].downsample(bev)
        x = core.decoder["backbone"](x)
        x = core.decoder["neck"](x)
        head = core.heads["object"]
        boxes, scores, labels = head.get_bboxes(head(x, cap["metas"]), cap["metas"])[0]
        return {"boxes_3d": boxes.to("cpu"), "scores_3d": scores.cpu(), "labels_3d": labels.cpu()}

    names = ["official"] + [f"agent{j}" for j in range(5)] + ["mean5"]
    results = {n: [] for n in names}
    t0 = time.time()
    for k, data in enumerate(loader):
        with torch.no_grad():
            results["official"] += model(return_loss=False, rescale=True, **data)
            bev = cap["bev"]                                  # [B, 25, C, H, W]
            per_agent = torch.stack([s.mean(dim=1) for s in torch.split(bev, 5, dim=1)], dim=1)  # [B, 5, C, H, W]
            for j in range(5):
                results[f"agent{j}"].append(decode(per_agent[:, j]))
            results["mean5"].append(decode(per_agent.mean(dim=1)))
            if args.save_bev:
                tok = dataset.data_infos[k]["token"]
                torch.save({"bev": per_agent[0].half().cpu(), "metas": cap["metas"]},
                           os.path.join(args.save_bev, tok + ".pt"))
        if k % 20 == 0:
            print(f"STEP {k}/{len(dataset)} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)

    tokens = [i["token"] for i in dataset.data_infos]
    with open(os.path.join(args.out, "tokens.txt"), "w") as f:
        f.write("\n".join(tokens))
    for n in names:
        files, _ = dataset.format_results(results[n], jsonfile_prefix=os.path.join(args.out, n))
        path = files["object"] if isinstance(files, dict) else files
        print("STEP wrote", n, path, flush=True)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
