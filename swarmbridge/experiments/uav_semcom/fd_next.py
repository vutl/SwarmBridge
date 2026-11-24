"""Paper A full-data, phases after Phase 2 (reports/phases/2026-11-23_paperA-fulldata/): one runner for
  * ablation (Phase 2b, mechanism ablation around the matched nominal design, paperA-fd-ablation-prereg.md),
  * grid     (Phase 4, network size / bandwidth, paperA-fd-phase4-prereg.md),
  * rl       (Phase 3, learned incumbents: zero-shot old checkpoints and fd_v1-retrained ones, paperA-fd-phase3-prereg.md).
Same simulator as Phase 2 (fd_common table, K5_la nominal LA, frozen JA on pacedB3, ax2_eval.episode); every test
command refuses to run without phaseX/AUTHORIZED. Outputs: runs/uav_semcom/paperA_fd/<phase>/<name>.json (+ .deliv)."""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import ax2_eval as X2
import fd_common as FD

LA = dict(la_mode="sched", la_interf="persistent")
CFG = {"N5x2": dict(K=5, M=2), "N3x2": dict(K=3, M=2), "N7x2": dict(K=7, M=2), "N5x3": dict(K=5, M=3),
       "N5x4": dict(K=5, M=4), "B0.5": dict(K=5, M=2, B_total=0.5e6), "B2": dict(K=5, M=2, B_total=2e6)}
LEARNED = ("ppola", "qrla", "dsla", "ppofd", "qrfd", "dsfd")
AB_ARMS = ["", "+mpcnc", "+mpcngg", "+mpcngb", "+mpcng", "+mpcngh1", "+mpc"]
AB_TEST = {"TD": range(24000, 24100), "V": range(25000, 25100), "F": range(26000, 26100)}
GRID_ARMS = ["", "+mpcnc", "+mpcng", "+mpc"]
GRID_TEST = {"TD": range(20000, 20100), "F": range(22000, 22100)}
RL_ARMS = ["", "+mpcnc", "+mpcng", "+mpc"]
RL_TEST = {"TD": range(20000, 20100), "F": range(22000, 22100), "V": range(21000, 21100), "TD_all": range(20000, 20100)}


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def _init():
    FD.init()
    X2.DELIV_EXT = True


def _job(j):
    cfg, combo, ckpt, seeds, pool = j
    learned = combo.split("|")[1].split("+")[0] in LEARNED
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, ckpt, s, dict(LA, obs_la=learned, **CFG[cfg]), None, pool)
        rows.append(m)
        dl.append(d)
    return dict(cfg=cfg, combo=combo, ckpt=ckpt, pool=pool, seeds=list(seeds), rows=rows, deliv=dl)


def run(items, seeds, pool, procs, chunk):
    """items: (cfg, combo); learned incumbents run every training-seed checkpoint 0..2."""
    seeds = list(seeds)
    jobs = [(cfg, c, i, seeds[x:x + chunk], pool) for cfg, c in items
            for i in (range(3) if X2.needs_ckpt(c) else [-1]) for x in range(0, len(seeds), chunk)]
    with Pool(procs, initializer=_init) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["cfg"], r["combo"], r["ckpt"]), dict(cfg=r["cfg"], combo=r["combo"], ckpt=r["ckpt"],
                                                                        pool=pool, seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    return list(merged.values())


def save(out, name, res, extra):
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(extra, results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]),
              open(out / f"{name}.json", "w"))
    json.dump({f"{r['cfg']}|{r['pool']}|{r['combo']}|{r['ckpt']}": r["deliv"] for r in res},
              gzip.open(out / f"{name}.deliv.json.gz", "wt"))


def report(res):
    for r in sorted(res, key=lambda r: (r["cfg"], r["combo"], r["ckpt"])):
        R = r["rows"]
        g = lambda k: [x[k] for x in R if x.get(k) is not None]
        line = (f"  {r['cfg']:5s} {r['pool']:6s} {r['combo'].split('|')[1]:34s} ck {r['ckpt']:2d} ret {np.mean(g('return')):8.2f}")
        if g("planned"):
            line += (f" planned {np.mean(g('planned')):5.1f} gated {np.mean(g('gated')):5.1f} cold {np.mean(g('cold_unguarded')):5.1f}"
                     f" sw {np.mean(g('switched')):5.1f} rej/chk {np.mean(g('robust_rejects')):.0f}/{np.mean(g('robust_checks')):.0f}")
        line += (f" p99 max {np.mean(g('ms_int_max_p99')):5.1f} sum {np.mean(g('ms_int_sum_p99')):6.1f} full-sum "
                 f"{np.mean(g('ms_full_sum_p99')):6.1f} upd {np.mean(g('ms_upd_p99')):4.2f} ms")
        print(line, flush=True)


def phase_dir(p):
    return FD.OUT / p


def authorized(p):
    assert (phase_dir(p) / "AUTHORIZED").exists(), f"{p}: test runs need the approved pre-registration (AUTHORIZED)"


def h_star(cfg):
    if cfg == "N5x2":
        return json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    return json.loads((phase_dir("phase4") / "tune.json").read_text())["H"][cfg]


# ---------------------------------------------------------------- Phase 2b: mechanism ablation
def ablation(cmd, procs):
    S, h = sched(), h_star("N5x2")
    if cmd == "smoke":
        res = run([("N5x2", f"{S}|h={h}{a}") for a in AB_ARMS], range(3000, 3005), "val", procs, 1)
        save(phase_dir("ablation"), "smoke_val", res, dict(sched=S, H=h, pool="rl_val"))
        report(res)
        return
    authorized("ablation")
    for pool, seeds in AB_TEST.items():
        t0 = time.time()
        res = run([("N5x2", f"{S}|h={h}{a}") for a in AB_ARMS], seeds, pool, procs, 25)
        save(phase_dir("ablation"), f"test_{pool}", res, dict(sched=S, H=h, pool=pool, seeds=list(seeds),
                                                               wall_s=time.time() - t0))
        report(res)


# ---------------------------------------------------------------- Phase 4: network size / bandwidth grid
def grid(cmd, procs):
    import ax9_run as A9
    S = sched()
    cfgs = [c for c in CFG if c != "N5x2"]
    if cmd == "tune":
        t0 = time.time()
        res = run([(c, f"{S}|h={x}") for c in cfgs for x in A9.H_FAMILY], C.TRAIN_SEEDS, "train", procs, 10)
        score = {c: {} for c in cfgs}
        for r in res:
            score[r["cfg"]][r["combo"].split("|h=")[1]] = float(np.mean([x["return"] for x in r["rows"]]))
        best = {c: max(A9.H_FAMILY, key=lambda x: score[c][x]) for c in cfgs}
        save(phase_dir("phase4"), "tune", res, dict(sched=S, seeds=C.TRAIN_SEEDS, pool="rl_train", train_return=score,
                                                    H=best, wall_s=time.time() - t0))
        print("H* per configuration:", best, flush=True)
        return
    if cmd == "smoke":
        res = run([(c, f"{S}|h={h_star(c)}{a}") for c in cfgs for a in GRID_ARMS], range(3000, 3003), "val", procs, 1)
        save(phase_dir("phase4"), "smoke_val", res, dict(sched=S, pool="rl_val"))
        report(res)
        return
    authorized("phase4")
    for pool, seeds in GRID_TEST.items():
        t0 = time.time()
        res = run([(c, f"{S}|h={h_star(c)}{a}") for c in cfgs for a in GRID_ARMS], seeds, pool, procs, 25)
        save(phase_dir("phase4"), f"test_{pool}", res, dict(sched=S, pool=pool, seeds=list(seeds),
                                                            H={c: h_star(c) for c in cfgs}, wall_s=time.time() - t0))
        report(res)


# ---------------------------------------------------------------- Phase 3: learned incumbents
def rl(cmd, procs):
    S = sched()
    if cmd == "smoke":                       # zero-shot old checkpoints on rl_val (validation only)
        res = run([("N5x2", f"{S}|{x}{a}") for x in ("ppola", "qrla", "dsla") for a in RL_ARMS], range(3000, 3002),
                  "val", procs, 1)
        save(phase_dir("phase3"), "smoke_val_zeroshot", res, dict(sched=S, pool="rl_val"))
        report(res)
        return
    authorized("phase3")
    which = ("ppola", "qrla", "dsla") if cmd == "zeroshot" else ("ppofd", "qrfd", "dsfd")
    pools = {"TD": RL_TEST["TD"]} if cmd == "zeroshot" else RL_TEST
    for pool, seeds in pools.items():
        t0 = time.time()
        res = run([("N5x2", f"{S}|{x}{a}") for x in which for a in RL_ARMS], seeds, pool, procs, 25)
        save(phase_dir("phase3"), f"{cmd}_{pool}", res, dict(sched=S, pool=pool, seeds=list(seeds), incumbents=which,
                                                             wall_s=time.time() - t0))
        report(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("ablation", "grid", "rl"))
    ap.add_argument("cmd", choices=("tune", "smoke", "test", "zeroshot", "retrained"))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    dict(ablation=ablation, grid=grid, rl=rl)[a.phase](a.cmd, a.procs)
