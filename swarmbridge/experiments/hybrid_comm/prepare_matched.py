"""Freeze scene split, sender predictions and real byte messages before training.

Run only via run_matched.sh prepare (legacy environment and shared GPU lock).
Helpers are decoded independently using the frozen ego decoder. No label enters
objectness ranks, quantization, sender Kalman filters or packets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "uav3d_comm"))
sys.path.insert(0, str(HERE.parent / "uav3d_comm/proto"))
import core as C
import tilecodec as TC
import boxcodec as BC
import box_proto as BP
import kalman as K
from fusion import load_variant, select
from ft_tools import setup, run_decode
from m0_receiver import heatmap_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out) / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    split = C.make_split(repo / "data/v1.0-mini")
    init = repo / "runs/uav3d_comm/fusion/ego_s2/decoder.pt"
    base = repo / "external/uav3d_ckpt/lowerbound.pth"
    infos = repo / "runs/uav3d_comm/lowerbound/infos_all.pkl"
    cfg = "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/lowerbound/default.yaml"
    fingerprint = dict(init=C.sha(init), base=C.sha(base), infos=C.sha(infos), split=split,
                       code={p.name: C.sha(p) for p in [Path(__file__), HERE / "core.py", HERE / "tilecodec.py"]})
    fingerprint_sha = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    done = out / "prepared.json"
    if done.exists():
        assert json.loads(done.read_text())["fingerprint"] == fingerprint_sha, "Prepared inputs changed"
        print("SKIP prepared", flush=True); return
    if (out / "split.json").exists():
        assert json.loads((out / "split.json").read_text()) == split
    else:
        C.save_json(out / "split.json", split)
    dataset, model, ds = setup(cfg, str(base), str(infos), "ego", str(init))
    model.heads["object"].test_cfg["post_max_size"] = 500
    by = {i["token"]: i for i in dataset.data_infos}
    for i in by.values():
        assert np.allclose(i["ego2global_rotation"], [1, 0, 0, 0])
        assert np.allclose(i["lidar2ego_rotation"], [1, 0, 0, 0])
    poses = {t: [float(x) for x in by[t]["ego2global_translation"][:2]] + [0.0] for t in by}
    C.save_json(out / "geometry.json", dict(axis="xy", shape=[80, 256, 256], half=C.HALF, cell=C.CELL,
                poses=poses, calibration="Metadata rotations identity for all 400 samples. bev_pool uses x as "
                "the first spatial index; no GT-based axis selection.",
                assumption="Cached cameras of each helper were projected into UAV0's world-aligned grid. "
                "A common known reference grid is assumed; not an independent distributed encoder reproduction."))
    tiles = out / "tiles"; tiles.mkdir(exist_ok=True)
    predictions = {j: [] for j in range(5)}
    # Match dataset formatter order, not filesystem or token lexical order.
    timings, message_hashes = [], {}
    with torch.no_grad():
        for n, info in enumerate(dataset.data_infos):
            tok = info["token"]
            cache = Path("/home/uservutl/work/bev_train") / (tok + ".pt")
            if not cache.exists(): cache = Path("/home/uservutl/work/bev_heldout") / (tok + ".pt")
            d = torch.load(str(cache), map_location="cpu")
            bev = d["bev"].cuda().float()
            assert tuple(bev.shape) == (5, 80, 256, 256)
            for j in range(5):
                y = model.decoder["neck"](model.decoder["backbone"](ds(bev[j][None])))
                pred = model.heads["object"](y, d["metas"])
                b, s, l = model.heads["object"].get_bboxes(pred, d["metas"])[0]
                predictions[j].append(dict(boxes_3d=b.to("cpu"), scores_3d=s.cpu(), labels_3d=l.cpu()))
                if not j: continue
                heat = heatmap_of(pred).cpu().numpy()
                ranks = np.argsort(-TC.tile_scores(heat), kind="stable")[:64]
                seed = int(hashlib.sha256((tok + "|" + str(j)).encode()).hexdigest()[:8], 16)
                ranks_random = np.random.default_rng(seed).permutation(TC.NT ** 2)[:64]
                raw = bev[j].cpu().numpy()
                for name, idx in [("obj", ranks), ("random", ranks_random)]:
                    t0 = time.perf_counter()
                    msg = TC.encode(TC.tiles_of(raw)[idx], idx, sender=j, msg_id=n, t_sense_ms=n * 200)
                    p = tiles / (tok + "_%d_%s.bin" % (j, name))
                    p.write_bytes(msg)
                    message_hashes[p.name] = hashlib.sha256(msg).hexdigest()
                    timings.append(time.perf_counter() - t0)
            if n % 40 == 0:
                print("PREP", n, len(by), "seconds", round(time.monotonic() - start, 1), flush=True)
    for j in range(5):
        dataset.format_results(predictions[j], jsonfile_prefix=str(out / "agents" / ("agent%d" % j)))
    A = {j: load_variant(out / "agents" / ("agent%d" % j) / "results_nusc.json") for j in range(5)}
    positions = BP.uav_positions(repo / "data/v1.0-mini")
    boxes = out / "boxes"; boxes.mkdir(exist_ok=True)
    for scene, seq in split["frames"].items():
        for j in range(1, 5):
            tr = K.run_sequence([A[j][t] for t in seq])
            for n, tok in enumerate(seq):
                m = select(BP.base(tr[n]), tr[n]["score"] >= BP.TAU_SEND)
                msg, _ = BC.encode(j, n, positions[(tok, j)], m)
                BC.decode(msg)  # Fail before training if the serialized packet is invalid.
                p = boxes / (tok + "_%d.bin" % j); p.write_bytes(msg)
                message_hashes["boxes/" + p.name] = hashlib.sha256(msg).hexdigest()
    C.save_json(out / "message_hashes.json", message_hashes)
    C.save_json(done, dict(fingerprint=fingerprint_sha, inputs=fingerprint,
                         seconds=time.monotonic() - start, frames=len(by), packets=len(message_hashes),
                         peak_vram_bytes=torch.cuda.max_memory_allocated(),
                         versions=dict(torch=torch.__version__, numpy=np.__version__),
                         encode_s_median=float(np.median(timings))))
    print("PREPARED", len(by), "seconds", round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__": main()
