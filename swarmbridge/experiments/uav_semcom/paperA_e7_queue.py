"""Paper A E7 queue (pre-registration: reports/phases/2026-11-18_paperA-writeup/paperA-e7-rl-families-prereg.md).
1. wait until E4 / E5 / E6 have finished (no CPU contention with their latency measurements);
2. T1 (seed 0): QR-DQN lr in {1e-4, 5e-4}, Discrete SAC entropy ratio in {0.3, 0.6}; choose per algorithm by the
   best-checkpoint validation return (train.json "log", selection statistic fixed in advance);
3. T2: seeds 1, 2 with the chosen setting;
4. ckpts.json (selected checkpoint of each training seed) + selection.json; copies of train.json into the repo;
5. paperA_r2.py e7 (evaluation).
Two training runs at a time, 2 environment workers each. Weights stay in /home/uservutl/work/e7 (not in git)."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import a5_common as C

PY = "/home/uservutl/envs/semcom-a/bin/python"
WORK = Path("/home/uservutl/work/e7")
REC = C.REPO / "runs" / "uav_semcom" / "paperA_rl"
T1 = {"qrdqn": [("lr1e-4", ["--lr", "1e-4"]), ("lr5e-4", ["--lr", "5e-4"])],
      "dsac": [("ent0.3", ["--ent-ratio", "0.3"]), ("ent0.6", ["--ent-ratio", "0.6"])]}
KEY = {"qrdqn": "qrla", "dsac": "dsla"}


def busy():
    r = subprocess.run(["pgrep", "-f", r"paperA_r2.py e[456]|run_e[56].sh"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def run_parallel(jobs, width=2):
    """jobs: list of (name, argv); runs `width` at a time; each job logs to REC/logs/<name>.log."""
    (REC / "logs").mkdir(parents=True, exist_ok=True)
    pending, running = list(jobs), []
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
                assert p.returncode == 0, f"{name} failed (see logs/{name}.log)"


def job(algo, tag, args, seed):
    d = WORK / f"{algo}_{tag}_s{seed}"
    return (f"{algo}_{tag}_s{seed}", [PY, "offpolicy_train.py", str(d), "--algo", algo, "--seed", str(seed)] + args), d


def best_val(d):
    tj = json.loads((d / "train.json").read_text())
    assert tj["last_steps"] >= 2_000_000, d
    return max(r["val_return"] for r in tj["log"]), tj["best_steps"]


def main():
    while busy():
        time.sleep(60)
    print("E4-E6 finished; starting E7 training", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    t1 = {a: [job(a, tag, args, 0) for tag, args in T1[a]] for a in T1}
    run_parallel([j for a in t1 for j, _ in t1[a]])
    sel = {}
    for a in T1:
        scores = {tag: best_val(d)[0] for (tag, _), (_, d) in zip(T1[a], t1[a])}
        tag = max(scores, key=scores.get)           # ties: first listed setting
        sel[a] = dict(chosen=tag, t1_best_val=scores, args=dict(T1[a])[tag])
        print(f"T1 {a}: {scores} -> {tag}", flush=True)
    t2 = {a: [job(a, sel[a]["chosen"], sel[a]["args"], s) for s in (1, 2)] for a in T1}
    run_parallel([j for a in t2 for j, _ in t2[a]])
    ck = {}
    for a in T1:
        dirs = [WORK / f"{a}_{sel[a]['chosen']}_s{s}" for s in (0, 1, 2)]
        ck[KEY[a]] = []
        sel[a]["seeds"] = {}
        for s, d in enumerate(dirs):
            bv, bs = best_val(d)
            ck[KEY[a]].append(str(d / f"ckpt_{bs}.pt"))
            sel[a]["seeds"][s] = dict(best_val=bv, best_steps=bs)
            (REC / "train").mkdir(parents=True, exist_ok=True)
            shutil.copy(d / "train.json", REC / "train" / f"{d.name}.train.json")
        for (tag, _), (_, d) in zip(T1[a], t1[a]):
            if tag != sel[a]["chosen"]:
                shutil.copy(d / "train.json", REC / "train" / f"{d.name}.train.json")
    (REC / "ckpts.json").write_text(json.dumps(ck, indent=1))
    (REC / "selection.json").write_text(json.dumps(sel, indent=1))
    print("checkpoints:", ck, flush=True)
    with open(C.REPO / "runs" / "uav_semcom" / "paperA_r2" / "e7.log", "w") as f:
        r = subprocess.run([PY, "paperA_r2.py", "e7", "--procs", "6"], stdout=f, stderr=subprocess.STDOUT)
    print(f"e7 evaluation exit {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
