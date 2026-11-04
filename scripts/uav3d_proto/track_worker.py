"""Tracking worker with a shared claim queue, so WSL and Windows workers can share one job list (CPU only).

Usage: python -u track_worker.py REPO_ROOT WORKER_ID DIR [DIR ...]   (DIR repo-relative, e.g. runs/uav3d_proto/box)
Run from the UAV3D tracking code directory (imports track_run, which loads the devkit DB once per worker).
Env: TRACK_DT=0.2, TRACK_ROOT=<REPO_ROOT>/data/v1.0-mini/.
A job = one prediction file DIR/dets/<a>/<b>/results_nusc.json -> DIR/tracking/<a>__<b>/. It is skipped when the
existing tracking input.json records the same sha256, and claimed by atomically creating
REPO_ROOT/runs/uav3d_proto/.claims/<dir>__<name> (O_CREAT | O_EXCL). Clear .claims/ before a fresh launch.
Jobs are taken in DIR order; env TRACK_PRIORITY=<file> with lines "<dir> <name>" puts those jobs first.
"""
import glob
import hashlib
import json
import os
import sys
import time

repo, wid, dirs = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
sys.path.insert(0, os.path.join(repo, "swarmbridge", "experiments", "uav3d_comm"))
sys.path.insert(0, os.getcwd())
import track_run as TR  # noqa: E402

claims = os.path.join(repo, "runs", "uav3d_proto", ".claims")
os.makedirs(claims, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jobs():
    pri = []
    if os.environ.get("TRACK_PRIORITY"):
        for line in open(os.environ["TRACK_PRIORITY"]):
            if line.strip():
                d, name = line.split()
                pri.append((d, name))
    seen = set(pri)
    for d, name in pri:
        a, b = name.split("__")
        yield d, os.path.join(repo, d, "dets", a, b, "results_nusc.json"), name
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(repo, d, "dets", "*", "*", "results_nusc.json"))):
            rel = os.path.relpath(os.path.dirname(f), os.path.join(repo, d, "dets")).replace("\\", "/")
            name = rel.replace("/", "__")
            if (d, name) not in seen:
                yield d, f, name


done = 0
for d, f, name in jobs():
        base = os.path.join(repo, d)
        if not os.path.exists(f):
            continue
        tdir = os.path.join(base, "tracking", name)
        if os.path.exists(os.path.join(tdir, "metrics_summary.json")) and os.path.exists(os.path.join(tdir, "input.json")):
            try:
                if json.load(open(os.path.join(tdir, "input.json")))["det_sha256"] == sha(f):
                    continue
            except Exception:
                pass
        claim = os.path.join(claims, d.replace("/", "_") + "__" + name)
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{wid} {time.time():.0f}".encode())
        os.close(fd)
        t0 = time.time()
        try:
            TR.run(name, f, os.path.join(base, "tracking"))
            done += 1
            print(f"DONE {wid} {d} {name} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # leave the claim so the job is not retried in a loop; report it
            print(f"FAIL {wid} {d} {name} {type(e).__name__}: {e}", flush=True)
print(f"WORKER_END {wid} done={done}", flush=True)
