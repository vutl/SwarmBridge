"""Cross-check: run the OFFICIAL UAV3D DetectionEval (projects/evaluate.py) on the held-out mini scenes.
Run in the legacy env from ~/work/uav3d_collab:  python eval_official.py RESULTS_JSON [RESULTS_JSON ...]
The working copy's projects/splits.py mini_val is overridden in memory to the held-out scenes.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
from projects import splits  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]
splits.mini_val = HELD
_orig = splits.create_splits_scenes
splits.create_splits_scenes = lambda verbose=False: dict(_orig(verbose), mini_val=HELD)
import projects.loaders as L  # noqa: E402
L.create_splits_scenes = splits.create_splits_scenes

from nuscenes import NuScenes  # noqa: E402
from nuscenes.eval.common.config import config_factory  # noqa: E402
from projects.evaluate import DetectionEval  # noqa: E402

nusc = NuScenes(version="v1.0-mini", dataroot="./data/uav3d/v1.0-mini/", verbose=False)
held_tokens = {s["token"] for s in nusc.sample if nusc.get("scene", s["scene_token"])["name"] in HELD}
for path in sys.argv[1:]:
    res = json.load(open(path))
    res["results"] = {t: v for t, v in res["results"].items() if t in held_tokens}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json")
        json.dump(res, open(p, "w"))
        ev = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=p,
                           eval_set="mini_val", output_dir=d, verbose=False)
        metrics, _ = ev.evaluate()
        s = metrics.serialize()
        ap = s["label_aps"]["car"]
        print(f"OFFICIAL {path}: mAP {s['mean_ap']:.4f} AP@0.5/1/2/4 = "
              + "/".join(f"{ap[str(k)]:.4f}" if str(k) in ap else f"{ap[k]:.4f}" for k in (0.5, 1.0, 2.0, 4.0)),
              flush=True)
