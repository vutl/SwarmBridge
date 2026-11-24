"""Paper A full-data Phase 2 (pre-registration: reports/phases/2026-11-23_paperA-fulldata/paperA-fd-phase2-prereg.md).
K = 5, M = 2, 1 MHz, nominal LA (K5_la, scheduler-aware LA, persistent interference), frozen JA scheduler on pacedB3,
fd_v1 table (fd_common). Arms (incumbent H*): H*, +mpc (calibrated, the method), +mpcnc (nominal), +mpcng (nominal in
the same gate / band design), +mpcrh (RobustMPC-style).
  tune  : H* = best of the 22-rule family (ax9_run.H_FAMILY) on rl_train, seeds 0..29 (mean return; ties: first listed)
  smoke : all arms on rl_val, seeds 3000..3004 (no test pool)
  test  : all arms on TD 20000..20099, V 21000..21099, F 22000..22099, TD_all 23000..23099 (ONLY after approval)
Outputs: runs/uav_semcom/paperA_fd/phase2/<name>.json (+ .deliv.json.gz: per delivery [manifest idx, mode, ok, ontime])."""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import ax2_eval as X2
import fd_common as FD

OUT = FD.OUT / "phase2"
LA = dict(la_mode="sched", la_interf="persistent")
SUFFIX = ["", "+mpc", "+mpcnc", "+mpcng", "+mpcrh", "+mpcrhnb"]
TEST = {"TD": range(20000, 20100), "V": range(21000, 21100), "F": range(22000, 22100), "TD_all": range(23000, 23100)}


def sched():
    f = json.loads((C.REPO / "runs" / "uav_semcom" / "ax9" / "sched.json").read_text())["frozen"]
    return f"{f['sched']}@{f['move']}"


def _init():
    FD.init()
    X2.DELIV_EXT = True


def _job(j):
    combo, seeds, pool = j
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, -1, s, dict(LA, obs_la=False), None, pool)
        rows.append(m)
        dl.append(d)
    return dict(combo=combo, pool=pool, seeds=list(seeds), rows=rows, deliv=dl)


def run(combos, seeds, pool, procs, chunk):
    seeds = list(seeds)
    jobs = [(c, seeds[x:x + chunk], pool) for c in combos for x in range(0, len(seeds), chunk)]
    with Pool(procs, initializer=_init) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault(r["combo"], dict(combo=r["combo"], pool=pool, seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    return list(merged.values())


def save(name, res, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(extra, results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]),
              open(OUT / f"{name}.json", "w"))
    json.dump({f"{r['pool']}|{r['combo']}": r["deliv"] for r in res}, gzip.open(OUT / f"{name}.deliv.json.gz", "wt"))


def tune(procs):
    import ax9_run as A9
    S, t0 = sched(), time.time()
    res = run([f"{S}|h={h}" for h in A9.H_FAMILY], C.TRAIN_SEEDS, "train", procs, 10)
    score = {r["combo"].split("|h=")[1]: float(np.mean([x["return"] for x in r["rows"]])) for r in res}
    best = max(A9.H_FAMILY, key=lambda h: score[h])           # ties: first listed in H_FAMILY
    save("tune", res, dict(sched=S, seeds=C.TRAIN_SEEDS, pool="rl_train", train_return=score, H=best,
                           wall_s=time.time() - t0))
    for h in sorted(score, key=score.get, reverse=True)[:6]:
        print(f"  {h:28s} {score[h]:8.2f}")
    print(f"H* = {best}  ({time.time() - t0:.0f}s)", flush=True)
    return best


def load_h():
    return json.loads((OUT / "tune.json").read_text())["H"]


def report(res):
    for r in res:
        R = r["rows"]
        g = lambda k: [x[k] for x in R if x.get(k) is not None]
        line = f"  {r['pool']:7s} {r['combo'].split('|')[1]:36s} ret {np.mean(g('return')):8.2f}"
        if g("planned"):
            lo = [x["band_lo_q"][1] for x in R if x.get("band_lo_q")]
            hi = [x["band_hi_q"][1] for x in R if x.get("band_hi_q")]
            line += (f" planned {np.mean(g('planned')):6.1f} gated {np.mean(g('gated')):6.1f} switched "
                     f"{np.mean(g('switched')):5.1f} aborts {np.mean(g('aborts')):4.1f} band med lo/hi "
                     f"{np.mean(lo) if lo else float('nan'):.2f}/{np.mean(hi) if hi else float('nan'):.2f}")
        line += f" int p99 {np.mean(g('ms_int_max_p99')):5.1f} ms >50 {np.mean(g('int_over50')):.3f}"
        print(line, flush=True)


def smoke(procs):
    S, h, t0 = sched(), load_h(), time.time()
    res = run([f"{S}|h={h}{s}" for s in SUFFIX], range(3000, 3005), "val", procs, 1)
    save("smoke_val", res, dict(sched=S, H=h, seeds=list(range(3000, 3005)), pool="rl_val", wall_s=time.time() - t0))
    report(res)
    print(f"smoke done {time.time() - t0:.0f}s", flush=True)


def test(procs):
    assert (OUT / "AUTHORIZED").exists(), "Phase 2 test runs need the approved pre-registration (create AUTHORIZED)"
    S, h = sched(), load_h()
    for pool, seeds in TEST.items():
        t0 = time.time()
        res = run([f"{S}|h={h}{s}" for s in SUFFIX], seeds, pool, procs, 25)
        save(f"test_{pool}", res, dict(sched=S, H=h, seeds=list(seeds), pool=pool, wall_s=time.time() - t0))
        report(res)
        print(f"test {pool} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "smoke", "test"))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    dict(tune=tune, smoke=smoke, test=test)[a.cmd](a.procs)
