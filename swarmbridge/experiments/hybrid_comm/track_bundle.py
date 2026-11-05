"""CPU official tracking, resumable by prediction SHA, aggregate and per scene.

Run from ~/work/uav3d_track in the legacy environment. No GPU is used.
Jobs: python track_bundle.py REPO MANIFEST_JSON OUT [--part N --parts M].
The manifest is {name: prediction_path}; disjoint workers partition jobs before
importing the official evaluator. Skips only complete results with matching hash.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "uav3d_comm"))
from core import DEV, save_json, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("manifest"); ap.add_argument("out")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    entries = json.loads(Path(args.manifest).read_text())
    os.environ["TRACK_DT"] = "0.2"
    os.environ["TRACK_ROOT"] = str(repo / "data/v1.0-mini")
    os.environ["TRACK_SCENES"] = ",".join(DEV)
    os.environ["OMP_NUM_THREADS"] = "2"
    import track_run as TR
    original_frames = TR.frames
    # Resolve token -> scene without inference/GT labels.
    scene_of = {s["token"]: TR.nusc.get("scene", s["scene_token"])["name"] for s in TR.nusc.sample}
    groups = [("all", list(DEV))] if args.aggregate_only else [("all", list(DEV))] + [(s,[s]) for s in DEV]
    jobs = [(name, p, group, scenes) for name,p in entries.items() for group,scenes in groups]
    summaries = {}
    for ji, (name, p, group, scenes) in enumerate(jobs):
        if ji % args.parts != args.part: continue
        p = Path(str(p).replace("\\", "/"))
        if not p.is_absolute(): p = repo / p
        name_group = name + "__" + group
        dest = out / name_group
        digest = sha(p)
        if (dest / "input.json").exists() and (dest / "metrics_summary.json").exists():
            md = json.loads((dest / "input.json").read_text())
            if md["det_sha256"] == digest and md["dt"] == "0.2" and md["scenes"] == scenes:
                print("TRACK_SKIP", name_group, flush=True)
                continue
        TR.HELD = scenes
        TR.frames = [f for f in original_frames if scene_of[f["token"]] in scenes]
        t0 = time.monotonic()
        TR.run(name_group, str(p), str(out))
        m = json.loads((dest / "metrics_summary.json").read_text())
        summaries[name_group] = dict(amota=m["amota"], ids=m["ids"], seconds=time.monotonic()-t0, sha256=digest)
        save_json(out / ("worker_%d.json" % args.part), summaries)
        print("TRACK_DONE", name_group, summaries[name_group], flush=True)


if __name__ == "__main__": main()
