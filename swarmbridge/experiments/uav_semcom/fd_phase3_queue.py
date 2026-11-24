"""Paper A full-data Phase 3 training queue (pre-registration paperA-fd-phase3-prereg.md). Requires phase3/AUTHORIZED.
  1. PPO-LA on fd_v1 (a5_train.py --fd; the A-X9 command otherwise), seeds 0, 1, 2, one at a time (6 env workers);
  2. QR-DQN (lr 1e-4) and Discrete SAC (entropy ratio 0.3) on fd_v1 (offpolicy_train.py --fd), seeds 0, 1, 2, two at a
     time (2 env workers each); no hyper-parameter search (E7 choices);
  3. phase3/ckpts.json {ppofd, qrfd, dsfd: [3 paths]} = best validation checkpoint of each run (train.json; rl_val
     seeds 3000..3019; ties: earlier) + selection.json + copies of every train.json.
Resumable: runs whose train.json reaches 2M decisions are skipped; interrupted runs are moved aside and restarted.
Weights stay in /home/uservutl/work/fd_phase3 (not in git)."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import fd_common as FD

PY = "/home/uservutl/envs/semcom-a/bin/python"
WORK = Path("/home/uservutl/work/fd_phase3")
REC = FD.OUT / "phase3"


def frozen():
    import a5_common as C
    return json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]


def jobs():
    f = frozen()
    extra = json.dumps(dict(la_mode="sched", la_interf="persistent", obs_la=True))
    out = []
    for s in range(3):
        out.append(("ppofd", s, 1, [PY, "a5_train.py", str(WORK / f"ppofd_s{s}"), "--regime", "K5_la", "--rl-part", "mode",
                                    "--heur", "fixed_avif_20", "--seed", str(s), "--steps", "2e6", "--cfg-extra", extra,
                                    "--slot-move", f["move"], "--slot-ja", f["member"], "--fd"]))
    for s in range(3):
        out.append(("qrfd", s, 2, [PY, "offpolicy_train.py", str(WORK / f"qrfd_s{s}"), "--algo", "qrdqn", "--seed", str(s),
                                   "--lr", "1e-4", "--fd"]))
        out.append(("dsfd", s, 2, [PY, "offpolicy_train.py", str(WORK / f"dsfd_s{s}"), "--algo", "dsac", "--seed", str(s),
                                   "--ent-ratio", "0.3", "--fd"]))
    return out


def done(d):
    tj = d / "train.json"
    return tj.exists() and json.loads(tj.read_text())["last_steps"] >= 2_000_000


def run_group(group, width):
    (REC / "logs").mkdir(parents=True, exist_ok=True)
    pending, running = [], []
    for key, s, _, argv in group:
        d = Path(argv[2])
        if done(d):
            print(f"[{time.strftime('%H:%M')}] skip {key}_s{s} (complete)", flush=True)
            continue
        if d.exists():
            dst = d.with_name(d.name + time.strftime("_interrupted_%m%d%H%M"))
            d.rename(dst)
            print(f"[{time.strftime('%H:%M')}] {key}_s{s}: partial run moved to {dst.name}, restarting", flush=True)
        pending.append((f"{key}_s{s}", argv))
    while pending or running:
        while pending and len(running) < width:
            name, argv = pending.pop(0)
            f = open(REC / "logs" / f"{name}.log", "w")
            running.append((name, subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT), f))
            print(f"[{time.strftime('%H:%M')}] start {name}", flush=True)
        time.sleep(20)
        for item in list(running):
            name, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(f"[{time.strftime('%H:%M')}] done {name} exit {p.returncode}", flush=True)
                assert p.returncode == 0, f"{name} failed (see phase3/logs/{name}.log)"


def main():
    assert (REC / "AUTHORIZED").exists(), "Phase 3 needs the approved pre-registration (phase3/AUTHORIZED)"
    WORK.mkdir(parents=True, exist_ok=True)
    js = jobs()
    run_group([j for j in js if j[2] == 1], 1)
    run_group([j for j in js if j[2] == 2], 2)
    ck, sel = {}, {}
    (REC / "train").mkdir(parents=True, exist_ok=True)
    for key, s, _, argv in js:
        d = Path(argv[2])
        tj = json.loads((d / "train.json").read_text())
        assert tj["last_steps"] >= 2_000_000
        ext = ".zip" if key == "ppofd" else ".pt"
        ck.setdefault(key, []).append(str(d / f"ckpt_{tj['best_steps']}{ext}"))
        sel.setdefault(key, {})[s] = dict(best_steps=tj["best_steps"], best_val=max(r["val_return"] for r in tj["log"]))
        shutil.copy(d / "train.json", REC / "train" / f"{key}_s{s}.train.json")
    (REC / "ckpts.json").write_text(json.dumps(ck, indent=1))
    (REC / "selection.json").write_text(json.dumps(sel, indent=1))
    print("checkpoints:", ck, flush=True)


if __name__ == "__main__":
    main()
