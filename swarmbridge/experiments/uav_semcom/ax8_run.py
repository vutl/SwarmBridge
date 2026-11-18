"""Review #43 (A-X8): fair-baseline study of the scheduling + mode controller under realistic link adaptation.
No training; existing K5 PPO mode-only checkpoints (3) only.

Environment: K5_la with scheduler-aware LA, persistent interference, K5 observation layout (obs_la false).
Channel scheduler: JA (JA:stat:NR:a1:m0, causal statistical joint assignment) unless stated; least-slack (ls / tbest_env)
only for the interaction check. Movement options (both stacks get both): paced (= ja / ls: movement + power of
T:best@K5, pacedB2, hovers while the backlog is long) and mission (= jae / tbest_env: T:best@K5_la, mission flight).

  tune : TRAIN pool, seeds 0..29, for each movement
           H   heuristic mode family: 14 fixed modes, greedy, greedy+budget{0.5,0.75,1,1.25,1.5,2,3}     (22)
           HC  the same rule families with the MPC's own online calibrated rate (fixed modes are rate-free)  (8)
           PPO zero-shot, PPO + calibrated selective MPC (3 checkpoints)
         then HC* + calibrated selective MPC (heuristic incumbent). Selection: best mean train return per movement;
         each stack also picks its best movement on train seeds. -> runs/uav_semcom/ax8/tune.json
  test : TEST pool, held-out fresh seeds 8900..8999 (never used before), both movements:
           JA x {h=H*, hc=HC*, ppo, ppo+mpc, hc=HC*+mpc, h=H*+mpc}, least-slack x {h=H*, hc=HC*, ppo, ppo+mpc}
         -> runs/uav_semcom/ax8/test.json (+ deliveries for detector recall)
Usage: python ax8_run.py tune --procs 6 ; python ax8_run.py test --procs 6
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import ax2_eval as X2

OUT = C.REPO / "runs" / "uav_semcom" / "ax8"
EXTRA = dict(la_mode="sched", la_interf="persistent", obs_la=False)
JA_MEMBER = "JA:stat:NR:a1:m0"
MOVES = {"paced": dict(ja="ja", ls="ls"), "mission": dict(ja="jae", ls="tbest_env")}
TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
H_FAMILY = [f"fixed_{m}" for m in C.MODES] + ["greedy"] + [f"greedy+budget{t:g}" for t in TAUS]
HC_FAMILY = ["greedy"] + [f"greedy+budget{t:g}" for t in TAUS]


def _job(j):
    combo, i, seeds, pool = j
    import torch
    torch.set_num_threads(1)
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, i, s, EXTRA, JA_MEMBER, pool)
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


def mean_ret(results, combo):
    rs = [r for r in results if r["combo"] == combo]
    return float(np.mean([np.mean([x["return"] for x in r["rows"]]) for r in rs]))


def tune(a):
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(C.TRAIN_SEEDS)
    t0 = time.time()
    combos = []
    for mv, sc in MOVES.items():
        combos += [f"{sc['ja']}|h={n}" for n in H_FAMILY] + [f"{sc['ja']}|hc={n}" for n in HC_FAMILY]
        combos += [f"{sc['ja']}|ppo", f"{sc['ja']}|ppo+mpc"]
    res = run(combos, seeds, "train", a.procs, 10)
    sel = {}
    for mv, sc in MOVES.items():
        h = {n: mean_ret(res, f"{sc['ja']}|h={n}") for n in H_FAMILY}
        hc = {n: mean_ret(res, f"{sc['ja']}|hc={n}") for n in HC_FAMILY}
        best_h = max(h, key=h.get)
        cal_pool = dict({f"hc={n}": v for n, v in hc.items()}, **{f"h={n}": h[n] for n in H_FAMILY if n.startswith("fixed_")})
        best_hc = max(cal_pool, key=cal_pool.get)
        sel[mv] = dict(H=f"h={best_h}", H_ret=h[best_h], HC=best_hc.replace("h=", "hc=", 1) if best_hc.startswith("h=")
                       else best_hc, HC_ret=cal_pool[best_hc], H_all=h, HC_all=hc,
                       PPO_ret=mean_ret(res, f"{sc['ja']}|ppo"), PPOMPC_ret=mean_ret(res, f"{sc['ja']}|ppo+mpc"))
    res2 = run([f"{MOVES[mv]['ja']}|{sel[mv]['HC']}+mpc" for mv in MOVES], seeds, "train", a.procs, 10)
    for mv in MOVES:
        sel[mv]["HCMPC_ret"] = mean_ret(res2, f"{MOVES[mv]['ja']}|{sel[mv]['HC']}+mpc")
    stack_mv = {k: max(MOVES, key=lambda mv: sel[mv][k + "_ret"]) for k in ("H", "HC", "PPO", "PPOMPC", "HCMPC")}
    out = dict(wall_s=time.time() - t0, seeds=seeds, pool="train", ja=JA_MEMBER, env_extra=EXTRA, selected=sel,
               stack_movement=stack_mv,
               results=[{k: v for k, v in r.items() if k != "deliv"} for r in res + res2])
    json.dump(out, open(OUT / "tune.json", "w"), indent=1)
    for mv in MOVES:
        s = sel[mv]
        print(f"{mv:8s} H* {s['H']:28s} {s['H_ret']:8.2f} | HC* {s['HC']:28s} {s['HC_ret']:8.2f} | PPO {s['PPO_ret']:8.2f}"
              f" | PPO+MPC {s['PPOMPC_ret']:8.2f} | HC*+MPC {s['HCMPC_ret']:8.2f}")
    print("stack movement:", stack_mv)


def test(a):
    sel = json.loads((OUT / "tune.json").read_text())["selected"]
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    combos = []
    for mv, sc in MOVES.items():
        H, HC = sel[mv]["H"], sel[mv]["HC"]
        combos += [f"{sc['ja']}|{H}", f"{sc['ja']}|{HC}", f"{sc['ja']}|ppo", f"{sc['ja']}|ppo+mpc",
                   f"{sc['ja']}|{HC}+mpc", f"{sc['ja']}|{H}+mpc"]
        combos += [f"{sc['ls']}|{H}", f"{sc['ls']}|{HC}", f"{sc['ls']}|ppo", f"{sc['ls']}|ppo+mpc"]
    combos = list(dict.fromkeys(combos))
    t0 = time.time()
    res = run(combos, seeds, "test", a.procs, 25)
    json.dump(dict(wall_s=time.time() - t0, seeds=seeds, pool="test", ja=JA_MEMBER, env_extra=EXTRA, selected=sel,
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]), open(OUT / "test.json", "w"))
    json.dump({f"{r['combo']}|{r['ckpt']}": r["deliv"] for r in res}, gzip.open(OUT / "test.deliv.json.gz", "wt"))
    for r in res:
        print(f"{r['combo']:40s} ckpt {r['ckpt']:2d} ret {np.mean([x['return'] for x in r['rows']]):8.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "test"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seeds", default="8900:9000")
    a = ap.parse_args()
    (tune if a.cmd == "tune" else test)(a)


if __name__ == "__main__":
    main()
