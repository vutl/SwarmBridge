"""Milestone 0/1a of the hybrid object-feature pilot: tile codec + feature receiver sanity on the dev scenes.

No wireless yet (every message arrives intact at age 0). Answers: does the trained feature decoder (train_fusion
'feat' arm, cap 500) still work when helpers send (a) everything, (b) everything at q8, (c) only K tiles chosen by a
causal sender score, decoded from real bytes? And what do codec / decoder cost in bytes and milliseconds?

Sender tile score (causal, sender-side only):
  obj   max objectness (sigmoid heatmap) of the SENDER's own single-UAV detector (ego-fine-tuned decoder) in the tile
  norm  mean L2 norm of the sender's feature in the tile (control)
Receiver input per cell = mean over ego + helpers whose decoded tile covers the cell (equals the training rule when
everything is sent). Variants (OUT/<variant>/results_nusc.json):
  feat_ego          ego feature only through the feature decoder (what the receiver has before any tile arrives)
  feat_dense        all helpers, float (reproduces the 'feat' arm)
  feat_dense_q8     all helpers, per-channel q8 over the whole map (diagnostic 'full feature q8')
  tiles_<rank>_<K>  K in {16 (M2), 64 (M3), 256 (diagnostic)}; K = 16 / 64 decoded from prefixes of ONE 64-tile message
Writes OUT/m0_profile.json (bytes per message, encode/decode/decoder ms p50/p95, tile overlap obj vs norm).

Run in the legacy env from ~/work/uav3d_collab:
  python m0_receiver.py CONFIG CKPT INFOS BEV_HELD FEAT_DEC EGO_DEC OUT [--cap 500]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "uav3d_comm"))
import tilecodec as tc  # noqa: E402
from ft_tools import run_decode, setup  # noqa: E402


def heatmap_of(pred):
    """Collect sigmoid heatmaps from a CenterHead output (nested lists/dicts) -> [H, W] max over tasks/classes."""
    maps = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "heatmap" and torch.is_tensor(v):
                    maps.append(v.sigmoid().amax(dim=1)[0])
                else:
                    walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(pred)
    return torch.stack(maps).amax(0)


def main():
    ap = argparse.ArgumentParser()
    for a in ("config", "ckpt", "infos", "bev_held", "feat_dec", "ego_dec", "out"):
        ap.add_argument(a)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--frames", type=int, default=0, help="limit for a smoke run")
    args = ap.parse_args()
    dataset, fmodel, fds = setup(args.config, args.ckpt, args.infos, "feat", args.feat_dec)
    _, emodel, eds = setup(args.config, args.ckpt, args.infos, "ego", args.ego_dec)
    for m in (fmodel, emodel):
        m.heads["object"].test_cfg["post_max_size"] = args.cap
    held = {f[:-3] for f in os.listdir(args.bev_held) if f.endswith(".pt")}
    dataset.data_infos = [i for i in dataset.data_infos if i["token"] in held]
    if args.frames:
        dataset.data_infos = dataset.data_infos[:args.frames]
    variants = ["feat_ego", "feat_dense", "feat_dense_q8"] + \
               [f"tiles_{r}_{k}" for r in ("obj", "norm") for k in (16, 64, 256)]
    res = {v: [] for v in variants}
    prof = {k: [] for k in ("score_ms", "encode_ms", "decode_ms", "decoder_ms", "overlap64")}
    nbytes = {}
    # Fixed by the pooling coordinate order and metadata, never by dev labels.
    half, cell = 102.4, 0.8
    axis = "xy"
    for info in dataset.data_infos:
        assert np.allclose(info["ego2global_rotation"], [1, 0, 0, 0])
        assert np.allclose(info["lidar2ego_rotation"], [1, 0, 0, 0])
    print("AXIS xy (pooling coordinates; identity pose rotations)", flush=True)
    sent = {}     # token -> rank -> helper -> top-256 tile ids (rank order); prefixes give M2 / M3

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    with torch.no_grad():
        for n, info in enumerate(dataset.data_infos):
            tok = info["token"]
            d = torch.load(os.path.join(args.bev_held, tok + ".pt"))
            bev = d["bev"].cuda().float()
            metas = d["metas"]
            ego = bev[0]

            def decode_with(x):
                t0 = sync()
                r = run_decode(fmodel, fds, x, metas)
                prof["decoder_ms"].append(1e3 * (sync() - t0))
                return r
            res["feat_ego"].append(decode_with(ego))
            res["feat_dense"].append(decode_with(bev.mean(0)))
            # dense q8 (per-channel min-max over the whole map, fp16 parameters)
            qs = []
            for j in range(1, 5):
                h = bev[j]
                lo = h.amin(dim=(1, 2), keepdim=True).half().float()
                st = ((h.amax(dim=(1, 2), keepdim=True) - lo) / 255).half().float().clamp_min(1e-12)
                qs.append(torch.round((h - lo) / st).clamp(0, 255) * st + lo)
            res["feat_dense_q8"].append(decode_with(torch.stack([ego] + qs).mean(0)))
            # sender tile scores
            orders = {"obj": {}, "norm": {}}
            for j in range(1, 5):
                t0 = sync()
                y = emodel.decoder["neck"](emodel.decoder["backbone"](eds(bev[j][None])))
                hm = heatmap_of(emodel.heads["object"](y, metas)).cpu().numpy()
                orders["obj"][j] = np.argsort(-tc.tile_scores(hm), kind="stable")
                prof["score_ms"].append(1e3 * (sync() - t0))
                nrm = bev[j].norm(dim=0).cpu().numpy()
                tsn = nrm.reshape(tc.NT, tc.T, tc.NT, tc.T).mean(axis=(1, 3)).reshape(-1)
                orders["norm"][j] = np.argsort(-tsn, kind="stable")
                prof["overlap64"].append(len(set(orders["obj"][j][:64]) & set(orders["norm"][j][:64])) / 64)
            sent[tok] = {r: {str(j): [int(i) for i in orders[r][j][:256]] for j in range(1, 5)} for r in orders}
            sent[tok]["ego_xy"] = [float(v) for v in info["ego2global_translation"][:2]]
            for rank in ("obj", "norm"):
                msgs = {}
                for j in range(1, 5):
                    for planned in (64, 256):
                        idx = orders[rank][j][:planned]
                        t0 = time.perf_counter()
                        tl = tc.tiles_of(bev[j].cpu().numpy())[idx]          # sender-side tile extraction
                        msgs[(j, planned)] = tc.encode(tl, idx, sender=j, t_sense_ms=int(n * 200), msg_id=n)
                        prof["encode_ms"].append(1e3 * (time.perf_counter() - t0))
                for k, planned in ((16, 64), (64, 64), (256, 256)):
                    num, den = ego.clone(), torch.ones_like(ego[0])
                    for j in range(1, 5):
                        buf = tc.prefix(msgs[(j, planned)], k)
                        nbytes.setdefault(k, []).append(len(buf))
                        t0 = time.perf_counter()
                        _, di, dt = tc.decode(buf)
                        f, m = tc.scatter(di, dt)
                        prof["decode_ms"].append(1e3 * (time.perf_counter() - t0))
                        mt = torch.from_numpy(m).cuda()
                        num += torch.from_numpy(f).cuda() * mt
                        den += mt
                    res[f"tiles_{rank}_{k}"].append(decode_with(num / den))
            if n % 10 == 0:
                print(f"STEP {n}/{len(dataset.data_infos)}", flush=True)
    for v, r in res.items():
        dataset.format_results(r, jsonfile_prefix=os.path.join(args.out, v))
    pct = lambda a: {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2)}
    summary = {k: pct(v) for k, v in prof.items() if k != "overlap64"}
    summary["overlap_obj_norm_top64"] = round(float(np.mean(prof["overlap64"])), 3)
    summary["bytes_per_msg"] = {str(k): int(np.mean(v)) for k, v in nbytes.items()}
    summary["kbit_per_msg"] = {str(k): round(8 * np.mean(v) / 1e3, 1) for k, v in nbytes.items()}
    summary["frames"] = len(dataset.data_infos)
    summary["cap"] = args.cap
    summary["axis"] = axis
    json.dump(summary, open(os.path.join(args.out, "m0_profile.json"), "w"), indent=1)
    json.dump(dict(axis=axis, half=half, cell=cell, tile=tc.T, sent=sent), open(os.path.join(args.out, "sent_tiles.json"), "w"))
    print(json.dumps(summary), flush=True)
    print("RUN_DONE m0_receiver", flush=True)


if __name__ == "__main__":
    main()
