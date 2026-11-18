"""Review #44 (A-X9): the final bounded test of the DRL hypothesis. Protocol (pre-registered):
reports/phases/2026-11-17_paperA-scheduling/swarmbridge-ax9-protocol.md.

Environment: K5_la, scheduler-aware LA, persistent interference. The observation includes the causal link-adaptation
features (delayed CQI report, report age, measured INR, OLLA offset, NACK rates; obs_la = true) for the LA-trained PPO;
the Shannon-trained PPO keeps its own layout (obs_la = false). The flag only changes the observation, not the dynamics.

  sched : TRAIN pool, seeds 0..29. Grid: movement {mission, pacedB1.5, pacedB2, pacedB3} x channel scheduler {12 JA
          members (ja_sched.family("stat")), least-slack}, each under 3 representative mode rules (fixed_mbt2018-mean_q3,
          fixed_avif_20, greedy+budget1). Frozen choice = (movement, JA member) with the best train return averaged
          over the 3 mode rules. -> runs/uav_semcom/ax9/sched.json
  heur  : TRAIN pool, frozen scheduler / movement: 22 mode rules (H) + 8 calibrated (HC); then the 5 best by train
          return as incumbents of the calibrated selective MPC -> best MPC incumbent I*. -> runs/uav_semcom/ax9/heur.json
  train : prints the 3 a5_train commands (PPO mode-only, 2M steps, best-val checkpoint) and, with --collect, writes
          runs/uav_semcom/ax9/ppo_la_ckpts.json from their train.json files.
  test  : TEST pool, held-out seeds 8300..8399 (never used before), frozen scheduler / movement.
Usage: python ax9_run.py sched|heur|test --procs 6 ; python ax9_run.py train [--collect]
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import ax2_eval as X2
import ja_sched as JA

OUT = C.REPO / "runs" / "uav_semcom" / "ax9"
BASE_EXTRA = dict(la_mode="sched", la_interf="persistent")
MOVES = ("mission", "pacedB1.5", "pacedB2", "pacedB3")
MODES3 = ("fixed_mbt2018-mean_q3", "fixed_avif_20", "greedy+budget1")
TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
H_FAMILY = [f"fixed_{m}" for m in C.MODES] + ["greedy"] + [f"greedy+budget{t:g}" for t in TAUS]
HC_FAMILY = ["greedy"] + [f"greedy+budget{t:g}" for t in TAUS]
TEST_SEEDS = "8300:8400"
TRAIN_DIR = "/home/uservutl/work/ax9"


def extra_for(combo):
    """the LA-trained PPO sees the causal LA features; everything else (incl. the Shannon PPO) the K5 layout."""
    return dict(BASE_EXTRA, obs_la=("ppola" in combo.split("|")[1]))


def _job(j):
    combo, i, seeds, pool = j
    import torch
    torch.set_num_threads(1)
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, i, s, extra_for(combo), None, pool)
        rows.append(m)
        dl.append(d)
    return dict(combo=combo, ckpt=i, seeds=list(seeds), rows=rows, deliv=dl)


def run(combos, seeds, pool, procs, chunk):
    chunks = [seeds[x:x + chunk] for x in range(0, len(seeds), chunk)]
    jobs = [(c, i, ch, pool) for c in combos for i in (range(3) if X2.needs_ckpt(c) else [-1]) for ch in chunks]
    with Pool(procs, initializer=AE.init, initargs=(False,)) as p:
        res = p.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["combo"], r["ckpt"]), dict(combo=r["combo"], ckpt=r["ckpt"], seeds=[], rows=[], deliv=[]))
        m["seeds"] += r["seeds"]
        m["rows"] += r["rows"]
        m["deliv"] += r["deliv"]
    return sorted(merged.values(), key=lambda x: (x["combo"], x["ckpt"]))


def mret(res, combo):
    rs = [r for r in res if r["combo"] == combo]
    return float(np.mean([np.mean([x["return"] for x in r["rows"]]) for r in rs]))


def strip(res):
    return [{k: v for k, v in r.items() if k != "deliv"} for r in res]


def sched_stage(a):
    OUT.mkdir(parents=True, exist_ok=True)
    scheds = [f"ja~{m}" for m in JA.family("stat")] + ["ls"]
    combos = [f"{s}@{mv}|h={md}" for mv in MOVES for s in scheds for md in MODES3]
    t0 = time.time()
    res = run(combos, list(C.TRAIN_SEEDS), "train", a.procs, 10)
    score = {(s, mv): float(np.mean([mret(res, f"{s}@{mv}|h={md}") for md in MODES3])) for mv in MOVES for s in scheds}
    best = max(((s, mv) for (s, mv) in score if s != "ls"), key=lambda x: score[x])
    ref = ("ja~JA:stat:NR:a1:m0", "pacedB2")
    out = dict(wall_s=time.time() - t0, seeds=list(C.TRAIN_SEEDS), modes3=MODES3,
               score={f"{s}@{mv}": v for (s, mv), v in score.items()},
               frozen=dict(sched=best[0], move=best[1], member=best[0].split("~")[1], score=score[best]),
               reference_A_X8=dict(sched=ref[0], move=ref[1], score=score[ref]),
               least_slack_same_move=score[("ls", best[1])], results=strip(res))
    json.dump(out, open(OUT / "sched.json", "w"), indent=1)
    print(f"frozen: {best[0]} @ {best[1]}  score {score[best]:.2f} | A-X8 JA@pacedB2 {score[ref]:.2f} | "
          f"least-slack@{best[1]} {score[('ls', best[1])]:.2f}")
    for mv in MOVES:
        b = max((s for s in scheds if s != "ls"), key=lambda s: score[(s, mv)])
        print(f"  {mv:10s} best JA {b:28s} {score[(b, mv)]:8.2f}  least-slack {score[('ls', mv)]:8.2f}")


def frozen():
    f = json.loads((OUT / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def heur_stage(a):
    S = frozen()
    seeds = list(C.TRAIN_SEEDS)
    t0 = time.time()
    combos = [f"{S}|h={n}" for n in H_FAMILY] + [f"{S}|hc={n}" for n in HC_FAMILY]
    res = run(combos, seeds, "train", a.procs, 10)
    rets = {c.split("|")[1]: mret(res, c) for c in combos}
    ranked = sorted(rets, key=lambda k: -rets[k])
    top5 = ranked[:5]
    res2 = run([f"{S}|{m}+mpc" for m in top5], seeds, "train", a.procs, 10)
    mpc = {m: mret(res2, f"{S}|{m}+mpc") for m in top5}
    inc = max(mpc, key=mpc.get)
    best_h = max((k for k in rets if k.startswith("h=")), key=lambda k: rets[k])
    best_any = ranked[0]
    out = dict(wall_s=time.time() - t0, sched=S, seeds=seeds, heur_returns=rets, mpc_incumbent_returns=mpc,
               H_best=best_h, heuristic_best=best_any, mpc_incumbent=inc, results=strip(res + res2))
    json.dump(out, open(OUT / "heur.json", "w"), indent=1)
    print(f"{S}: best heuristic {best_any} ({rets[best_any]:.2f}); best uncalibrated {best_h} ({rets[best_h]:.2f})")
    for m in top5:
        print(f"  incumbent {m:30s} alone {rets[m]:8.2f}  +MPC {mpc[m]:8.2f}" + ("  <- I*" if m == inc else ""))


def train_stage(a):
    f = json.loads((OUT / "sched.json").read_text())["frozen"]
    if a.collect:
        paths = []
        for s in range(3):
            tj = json.loads(Path(f"{TRAIN_DIR}/ppo_la_s{s}/train.json").read_text())
            paths.append(f"{TRAIN_DIR}/ppo_la_s{s}/ckpt_{tj['best_steps']}.zip")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ppo_la_ckpts.json").write_text(json.dumps(paths, indent=1))
        print(paths)
        return
    extra = json.dumps(dict(BASE_EXTRA, obs_la=True))
    for s in range(3):
        print(f"python a5_train.py {TRAIN_DIR}/ppo_la_s{s} --regime K5_la --rl-part mode --heur fixed_avif_20 "
              f"--seed {s} --steps 2e6 --cfg-extra '{extra}' --slot-move {f['move']} --slot-ja '{f['member']}'")


def test_stage(a):
    S = frozen()
    h = json.loads((OUT / "heur.json").read_text())
    H, I = h["heuristic_best"], h["mpc_incumbent"]
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    combos = list(dict.fromkeys([f"{S}|{H}", f"{S}|{I}", f"{S}|{I}+mpc", f"{S}|{I.replace('hc=', 'h=')}+mpcnc",
                                 f"{S}|ppola", f"{S}|ppola+mpc", f"{S}|ppo", f"{S}|ppo+mpc"]))
    t0 = time.time()
    res = run(combos, seeds, "test", a.procs, 25)
    json.dump(dict(wall_s=time.time() - t0, sched=S, seeds=seeds, heuristic_best=H, mpc_incumbent=I,
                   results=strip(res)), open(OUT / "test.json", "w"))
    json.dump({f"{r['combo']}|{r['ckpt']}": r["deliv"] for r in res}, gzip.open(OUT / "test.deliv.json.gz", "wt"))
    for r in res:
        print(f"{r['combo']:50s} ckpt {r['ckpt']:2d} ret {np.mean([x['return'] for x in r['rows']]):8.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sched", "heur", "train", "test"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", default=TEST_SEEDS)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    dict(sched=sched_stage, heur=heur_stage, train=train_stage, test=test_stage)[a.cmd](a)


if __name__ == "__main__":
    main()
