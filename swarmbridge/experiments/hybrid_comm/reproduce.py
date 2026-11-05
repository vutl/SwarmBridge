"""Recreate legacy NMS predictions without overwriting the original artifacts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from core import nms_max, save_json, sha
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uav3d_comm"))
from evaldet import MiniDB, evaluate, load_nusc_results
from hybrid_merge import tile_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    old = repo / "runs/hybrid_comm/m0"
    st = json.loads((old / "sent_tiles.json").read_text())
    box = json.loads((repo / "runs/uav3d_proto/box/dets/kf_nor/sync_s0/results_nusc.json").read_text())["results"]
    dense = json.loads((old / "feat_dense/results_nusc.json").read_text())["results"]
    expected = json.loads((old / "track_summary.json").read_text())
    db = MiniDB(repo / "data/v1.0-mini")
    toks = db.tokens(["scene_11", "scene_13", "scene_15"])
    rows = {}
    def write(name, res, expect=None):
        p = out / name / "results_nusc.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Retain historical byte order and separators for exact SHA256 matching.
        p.write_bytes(json.dumps({"meta": {"use_camera": True}, "results": res}).encode())
        digest = sha(p)
        row = dict(pred=p.as_posix(), sha256=digest, **evaluate(db, load_nusc_results(p), toks))
        row["per_scene"] = {s: evaluate(db, load_nusc_results(p), db.tokens([s]))["mAP"] for s in ("scene_11", "scene_13", "scene_15")}
        if expect:
            row["expected_sha256"] = expected[expect]["det_sha256"]
            row["hash_match"] = digest == row["expected_sha256"]
            if not row["hash_match"]:
                raise RuntimeError("Legacy hash mismatch: " + name)
        rows[name] = row
    for k in (16, 64, 256):
        feat = json.loads((old / ("tiles_obj_%d/results_nusc.json" % k)).read_text())["results"]
        for suffix, src in (("nms", feat), ("densein_nms", dense)):
            results = {}
            for tok, s in st["sent"].items():
                union = set().union(*(s["obj"][str(j)][:k] for j in range(1, 5)))
                fb = [b for b in src[tok] if tile_of(b, s["ego_xy"], st["half"], st["cell"], st["tile"], st["axis"]) in union]
                results[tok] = nms_max(fb + box[tok])
            name = ("m%d_%s" % ({16: 2, 64: 3, 256: 256}[k], suffix))
            write(name, results, name if name in expected else None)
    write("m1_same_nms", {t: nms_max(box[t]) for t in st["sent"]})
    for name, path in [("m1_kf_nor", repo / "runs/uav3d_proto/box/dets/kf_nor/sync_s0/results_nusc.json"),
                       ("feat_dense", old / "feat_dense/results_nusc.json"),
                       ("feat_dense_q8", old / "feat_dense_q8/results_nusc.json")]:
        rows[name] = dict(pred=path.as_posix(), sha256=sha(path), **evaluate(db, load_nusc_results(path), toks))
        if name in expected:
            assert rows[name]["sha256"] == expected[name]["det_sha256"]
            rows[name]["hash_match"] = True
    save_json(out / "reproduction.json", rows)
    print(json.dumps({k: {j: v[j] for j in ("mAP", "hash_match") if j in v} for k, v in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
