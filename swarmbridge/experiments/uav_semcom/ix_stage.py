"""paperA_ix stage A (X2-retrained fitting on UAVDT U_train / U_val) and the L2 lock (pre-registration §3.2, §4).
Commands, in order (each refuses to run if its inputs are missing; none touches X1 or U_test):
  fit      : retrained controller state from the stage-A profile + the registered CPU estimate -> state_retrained.json
  tune     : H* among the same 22 rules (ax9_run.H_FAMILY) on U_train, training seeds C.TRAIN_SEEDS, for the reference
             setting N5x2 and the 6 grid settings (fd_v1 rule: best mean train return, ties -> first listed) -> tune.json
  train    : 9 RL runs (PPO-LA a5_train.py, QR-DQN / Discrete SAC offpolicy_train.py; fd_v1 commands with --ix
             --log-actions instead of --fd), PPO one at a time with 6 env workers, off-policy two at a time; resumable
  select   : best validation checkpoint per run (U_val, seeds C.PPO_VAL_SEEDS, ties -> earlier; recorded by the trainers)
             -> ckpts.json {ppoix, qrix, dsix: [3 paths]}, selection.json (sha256 of every checkpoint), train/*.train.json
  lock     : L2_lock.json (ix_common.lock_items: manifest, stage-A profile, CPU estimate, retrained state, frozen state,
             H*, checkpoints, locked code). Written once; commit it before stage B.
  verify   : verify the L2 lock (also done by every stage-B command)
  status   : completeness of the stage-A inputs
Weights stay in WORK/rl (outside git)."""
import argparse
import json
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import ax2_eval as X2
import ix_common as IX
from fd_next import CFG

PY = "/home/uservutl/envs/semcom-a/bin/python"
LA = dict(la_mode="sched", la_interf="persistent")
RL_WORK = IX.WORK / "rl"
KEYS = ("ppoix", "qrix", "dsix")


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}", f


def fit():
    h = IX.fit_retrained()
    print("retrained state fitted on U_train:", h, flush=True)


def _init():
    IX.init_stage_a()
    X2.DELIV_EXT = True


def _job(j):
    cfg, combo, seeds = j
    return cfg, combo, [X2.episode("K5", "K5_la", combo, -1, s, dict(LA, obs_la=False, **CFG[cfg]), None, "train")[0]
                        ["return"] for s in seeds]


def tune(procs):
    import ax9_run as A9
    assert not IX.TUNE_FILE.exists(), "tune.json exists (H* is chosen once)"
    S, _ = sched()
    t0 = time.time()
    jobs = [(cfg, f"{S}|h={h}", C.TRAIN_SEEDS[x:x + 10]) for cfg in CFG for h in A9.H_FAMILY
            for x in range(0, len(C.TRAIN_SEEDS), 10)]
    with Pool(procs, initializer=_init) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    rets = {cfg: {} for cfg in CFG}
    for cfg, combo, r in res:
        rets[cfg].setdefault(combo.split("|h=")[1], []).extend(r)
    score = {cfg: {h: float(np.mean(v)) for h, v in d.items()} for cfg, d in rets.items()}
    assert all(len(v) == len(C.TRAIN_SEEDS) for d in rets.values() for v in d.values())
    best = {cfg: max(A9.H_FAMILY, key=lambda h: score[cfg][h]) for cfg in CFG}
    IX.TUNE_FILE.write_text(json.dumps(dict(sched=S, seeds=C.TRAIN_SEEDS, pool="U_train", family=list(A9.H_FAMILY),
                                            train_return=score, H=best,
                                            state_hash=IX.state_hash(IX.load_state(IX.STATE_FILE)),
                                            wall_s=time.time() - t0), indent=1))
    print("H* per setting (U_train):", best, f"({time.time() - t0:.0f}s)", flush=True)


def jobs():
    _, f = sched()
    extra = json.dumps(dict(la_mode="sched", la_interf="persistent", obs_la=True))
    out = []
    for s in range(3):
        out.append(("ppoix", s, 1, [PY, "a5_train.py", str(RL_WORK / f"ppoix_s{s}"), "--regime", "K5_la", "--rl-part",
                                    "mode", "--heur", "fixed_avif_20", "--seed", str(s), "--steps", "2e6",
                                    "--cfg-extra", extra, "--slot-move", f["move"], "--slot-ja", f["member"],
                                    "--ix", "--log-actions"]))
    for s in range(3):
        out.append(("qrix", s, 2, [PY, "offpolicy_train.py", str(RL_WORK / f"qrix_s{s}"), "--algo", "qrdqn", "--seed",
                                   str(s), "--lr", "1e-4", "--ix", "--log-actions"]))
        out.append(("dsix", s, 2, [PY, "offpolicy_train.py", str(RL_WORK / f"dsix_s{s}"), "--algo", "dsac", "--seed",
                                   str(s), "--ent-ratio", "0.3", "--ix", "--log-actions"]))
    return out


def done(d):
    tj = d / "train.json"
    return tj.exists() and json.loads(tj.read_text())["last_steps"] >= 2_000_000


def run_group(group, width):
    (IX.OUT / "train_logs").mkdir(parents=True, exist_ok=True)
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
            f = open(IX.OUT / "train_logs" / f"{name}.log", "w")
            running.append((name, subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT), f))
            print(f"[{time.strftime('%H:%M')}] start {name}", flush=True)
        time.sleep(20)
        for item in list(running):
            name, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(f"[{time.strftime('%H:%M')}] done {name} exit {p.returncode}", flush=True)
                assert p.returncode == 0, f"{name} failed (see paperA_ix/train_logs/{name}.log)"


def train():
    assert IX.STATE_FILE.exists() and IX.TUNE_FILE.exists(), "run fit and tune first"
    RL_WORK.mkdir(parents=True, exist_ok=True)
    js = jobs()
    run_group([j for j in js if j[2] == 1], 1)
    run_group([j for j in js if j[2] == 2], 2)
    print("TRAIN_DONE", flush=True)


def select():
    assert not IX.SELECTION.exists(), "selection.json exists"
    ck, sel = {}, {}
    (IX.OUT / "train").mkdir(parents=True, exist_ok=True)
    for key, s, _, argv in jobs():
        d = Path(argv[2])
        tj = json.loads((d / "train.json").read_text())
        assert tj["last_steps"] >= 2_000_000 and tj["table"].startswith("paperA_ix"), (key, s)
        ckp = d / f"ckpt_{tj['best_steps']}{'.zip' if key == 'ppoix' else '.pt'}"
        assert ckp.exists(), ckp
        ck.setdefault(key, []).append(str(ckp))
        last = tj["log"][-1]
        sel.setdefault(key, {})[s] = dict(best_steps=tj["best_steps"], best_val=max(r["val_return"] for r in tj["log"]),
                                          ckpt=str(ckp), ckpt_sha256=IX.sha_file(ckp), command=argv, seed=s,
                                          last_steps=tj["last_steps"], train_wall_s=last.get("sec"), table=tj["table"])
        shutil.copy(d / "train.json", IX.OUT / "train" / f"{key}_s{s}.train.json")
    IX.CKPTS.write_text(json.dumps(ck, indent=1))
    IX.SELECTION.write_text(json.dumps(dict(selection=sel, rule="best mean validation return on U_val seeds "
                                            "3000..3019, ties -> earlier checkpoint"), indent=1))
    print("checkpoints:", ck, flush=True)


def lock():
    assert not IX.LOCK.exists(), "L2_lock.json exists (written once)"
    items = IX.lock_items()
    IX.LOCK.write_text(json.dumps(dict(created=time.strftime("%F %T"), items=items,
                                       note="L2: every fitted component of stage A; stage B verifies all items"),
                                  indent=1))
    IX.verify_lock()
    print("L2 written and verified:", IX.LOCK, flush=True)


def verify():
    lk = IX.verify_lock()
    print("L2 verified (created", lk["created"] + ")", flush=True)


def status():
    man = IX.read_manifest()
    need = {e["name"] for e in man if e["split"] in IX.SPLITS["A"]}
    have = {}
    if IX.PROFILE["A"].exists():
        for line in open(IX.PROFILE["A"]):
            r = json.loads(line)
            have.setdefault(r["image"], set()).add(r["mode"])
    ok = sum(1 for n in need if have.get(n, set()) >= set(C.MODES))
    print(f"stage-A profile: {ok}/{len(need)} images complete")
    for f in (IX.CPU_EST, IX.STATE_FILE, IX.TUNE_FILE, IX.CKPTS, IX.SELECTION, IX.LOCK):
        print(f"  {f.name:42s} {'present' if f.exists() else 'missing'}")
    for key, s, _, argv in jobs():
        d = Path(argv[2])
        tj = d / "train.json"
        print(f"  {key}_s{s}: {json.loads(tj.read_text())['last_steps'] if tj.exists() else 0} steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "tune", "train", "select", "lock", "verify", "status"))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"fit": fit, "tune": lambda: tune(a.procs), "train": train, "select": select, "lock": lock, "verify": verify,
     "status": status}[a.cmd]()
