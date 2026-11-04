"""Track detections with the official UAV3D CenterPoint greedy tracker (PubTracker, max_age 3) and score AMOTA etc.
with the official UAV3D TrackingEval, on the held-out mini scenes (scene_11/13/15).

Run in the legacy env from ~/work/uav3d_track (a copy of tracking/CenterPoint/tools/nusc_tracking):
  python track_run.py OUT_DIR NAME=DET_JSON [NAME=DET_JSON ...]
Writes OUT_DIR/<NAME>/tracking_result.json + metrics_summary.json and prints one line per input.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
HELD = ["scene_11", "scene_13", "scene_15"]
ROOT = "/mnt/c/Users/USER/Documents/SwarmBridge/data/v1.0-mini/"

import projects.splits as S  # noqa: E402
_orig = S.create_splits_scenes
S.create_splits_scenes = lambda verbose=False: dict(_orig(verbose), mini_val=HELD)
for mod in ("projects.common.loaders", "projects.tracking.loaders", "projects.evaluate", "projects.utils"):
    try:
        m = __import__(mod, fromlist=["x"])
        if hasattr(m, "create_splits_scenes"):
            m.create_splits_scenes = S.create_splits_scenes
    except ImportError:
        pass

from nuscenes import NuScenes  # noqa: E402
from nuscenes.eval.common.config import config_factory  # noqa: E402
from pub_tracker import PubTracker  # noqa: E402
from projects.evaluate import TrackingEval  # noqa: E402

nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
frames = []
for scene in nusc.scene:
    if scene["name"] not in HELD:
        continue
    tok = scene["first_sample_token"]
    first = True
    while tok:
        s = nusc.get("sample", tok)
        frames.append({"token": tok, "timestamp": s["timestamp"] * 1e-9, "first": first})
        first = False
        tok = s["next"]


def run(name, det_path, out_dir):
    preds = json.load(open(det_path))["results"]
    tracker = PubTracker(max_age=3, hungarian=False)
    res = {"results": {}, "meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                                   "use_map": False, "use_external": False}}
    last = None
    for f in frames:
        if f["first"]:
            tracker.reset()
            last = f["timestamp"]
        lag = f["timestamp"] - last
        last = f["timestamp"]
        dets = [dict(d) for d in preds.get(f["token"], [])]
        out = tracker.step_centertrack(dets, lag)
        res["results"][f["token"]] = [{
            "sample_token": f["token"], "translation": o["translation"], "size": o["size"],
            "rotation": o["rotation"], "velocity": o["velocity"], "tracking_id": str(o["tracking_id"]),
            "tracking_name": o["detection_name"], "tracking_score": o["detection_score"]}
            for o in out if o["active"] != 0]
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    rp = os.path.join(d, "tracking_result.json")
    json.dump(res, open(rp, "w"))
    ev = TrackingEval(config=config_factory("tracking_nips_2019"), result_path=rp, eval_set="mini_val",
                      output_dir=d, nusc_version="v1.0-mini", nusc_dataroot=ROOT, verbose=False)
    m = ev.main() if hasattr(ev, "main") else None
    summ = json.load(open(os.path.join(d, "metrics_summary.json")))
    keys = ["amota", "amotp", "mota", "motp", "recall", "ids", "frag", "tid", "lgd"]
    print("TRACK", name, " ".join(f"{k}={summ.get(k, float('nan')):.3f}" if isinstance(summ.get(k), float)
                                  else f"{k}={summ.get(k)}" for k in keys), flush=True)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        run(name, path, out_dir)
