"""paperA_ix stage B: the 8 registered test blocks (pre-registration §5; 10,200 episodes). Every command verifies the
L2 lock first (ix_common.verify_lock; workers load the test table through ix_common.load_eval, which verifies again
and never fits). Same simulator and arms as fd_v1 (ax2_eval.episode, K5_la + scheduler-aware LA, frozen JA on
pacedB3).

  block     table   arm        arms                                              seeds
  x1_ref    X1      frozen     H*, +mpc, +mpcnc, +mpcng, +mpcrh, +mpcrhnb         30000-30099
  x1_abl    X1      frozen     8 ablation arms (fd_next.AB_ARMS)                  30100-30199
  x2f_ref   U_test  frozen     6 Phase 2 arms                                     31000-31099
  x2f_abl   U_test  frozen     8 ablation arms                                    31100-31199
  x2r_ref   U_test  retrained  6 Phase 2 arms, H* from ix tune (N5x2)             32000-32099
  x2r_abl   U_test  retrained  8 ablation arms                                    32100-32199
  x2r_grid  U_test  retrained  6 settings x {H*, +mpcnc, +mpcng, +mpc}            32200-32299
  x2r_rl    U_test  retrained  {ppoix, qrix, dsix} x 3 ckpts x {X, +mpcnc, +mpcng, +mpc}   32000-32099
Frozen blocks use the fd_v1 H* (paperA_fd/phase2/tune.json); retrained blocks the U_train H* (paperA_ix/tune.json).
X1 blocks are runnable independently of the UAVDT blocks (separate commands and files).
Outputs: runs/uav_semcom/paperA_ix/eval/<block>.json (per-episode metrics, no image content) and
WORK/eval/<block>.deliv.json.gz (per-delivery records, outside git). Resumable: complete blocks are skipped.
Usage (WSL): python ix_eval.py x1|uavdt|<block> [--procs 6]"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import ax2_eval as X2
import ix_common as IX
from fd_next import AB_ARMS, CFG, GRID_ARMS, RL_ARMS
from fd_phase2 import SUFFIX

LA = dict(la_mode="sched", la_interf="persistent")
LEARNED = ("ppoix", "qrix", "dsix")
BLOCKS = {
    "x1_ref": ("x1", "frozen", "ref", range(30000, 30100)),
    "x1_abl": ("x1", "frozen", "abl", range(30100, 30200)),
    "x2f_ref": ("uavdt", "frozen", "ref", range(31000, 31100)),
    "x2f_abl": ("uavdt", "frozen", "abl", range(31100, 31200)),
    "x2r_ref": ("uavdt", "retrained", "ref", range(32000, 32100)),
    "x2r_abl": ("uavdt", "retrained", "abl", range(32100, 32200)),
    "x2r_grid": ("uavdt", "retrained", "grid", range(32200, 32300)),
    "x2r_rl": ("uavdt", "retrained", "rl", range(32000, 32100)),
}
GROUPS = {"x1": ["x1_ref", "x1_abl"],
          "uavdt": ["x2f_ref", "x2f_abl", "x2r_ref", "x2r_abl", "x2r_grid", "x2r_rl"]}
POOL = {"x1": "X1", "uavdt": "U_test"}
N_EPISODES = {"ref": 6, "abl": 8, "grid": 24, "rl": 36}


def sched():
    from fd_next import sched as s
    return s()


def h_star(arm, cfg):
    if arm == "frozen":
        assert cfg == "N5x2"
        return json.loads((IX.REPO / "runs" / "uav_semcom" / "paperA_fd" / "phase2" / "tune.json").read_text())["H"]
    return json.loads(IX.TUNE_FILE.read_text())["H"][cfg]


def items(block):
    stage, arm, kind, _ = BLOCKS[block]
    S = sched()
    if kind == "ref":
        return [("N5x2", f"{S}|h={h_star(arm, 'N5x2')}{s}", -1) for s in SUFFIX]
    if kind == "abl":
        return [("N5x2", f"{S}|h={h_star(arm, 'N5x2')}{s}", -1) for s in AB_ARMS]
    if kind == "grid":
        return [(c, f"{S}|h={h_star(arm, c)}{s}", -1) for c in CFG if c != "N5x2" for s in GRID_ARMS]
    return [("N5x2", f"{S}|{x}{s}", i) for x in LEARNED for s in RL_ARMS for i in range(3)]


_CTX = {}


def _init(stage, arm):
    IX.init_eval(stage, arm)
    X2.DELIV_EXT = True
    _CTX["stage"] = stage


def _job(j):
    cfg, combo, ckpt, seeds, pool = j
    learned = combo.split("|")[1].split("+")[0] in LEARNED
    rows, dl = [], []
    for s in seeds:
        m, d = X2.episode("K5", "K5_la", combo, ckpt, s, dict(LA, obs_la=learned, **CFG[cfg]), None, pool)
        rows.append(m)
        dl.append(d)
    return dict(cfg=cfg, combo=combo, ckpt=ckpt, pool=pool, seeds=list(seeds), rows=rows, deliv=dl)


def paths(block):
    return IX.OUT / "eval" / f"{block}.json", IX.WORK / "eval" / f"{block}.deliv.json.gz"


def complete(block):
    pj, pd = paths(block)
    seeds = list(BLOCKS[block][3])
    try:
        te = json.loads(pj.read_text())
        dl = json.load(gzip.open(pd, "rt"))
    except (OSError, ValueError, EOFError):
        return False
    n = len(items(block))
    return (te.get("seeds") == seeds and len(te["results"]) == n == len(dl)
            and all(len(r["rows"]) == len(seeds) for r in te["results"]) and all(len(v) == len(seeds) for v in dl.values()))


def run_block(block, procs, chunk=25):
    stage, arm, kind, seeds = BLOCKS[block]
    if complete(block):
        print(f"skip {block}: complete", flush=True)
        return
    lk = IX.verify_lock()
    seeds, pool = list(seeds), POOL[stage]
    its = items(block)
    jobs = [(cfg, c, ck, seeds[x:x + chunk], pool) for cfg, c, ck in its for x in range(0, len(seeds), chunk)]
    t0 = time.time()
    with Pool(procs, initializer=_init, initargs=(stage, arm)) as pl:
        res = pl.map(_job, jobs, chunksize=1)
    merged = {}
    for r in res:
        m = merged.setdefault((r["cfg"], r["combo"], r["ckpt"]), dict(cfg=r["cfg"], combo=r["combo"], ckpt=r["ckpt"],
                                                                        pool=pool, seeds=[], rows=[], deliv=[]))
        for k in ("seeds", "rows", "deliv"):
            m[k] += r[k]
    res = list(merged.values())
    assert all(r["seeds"] == seeds for r in res) and len(res) == len(its)
    pj, pd = paths(block)
    pj.parent.mkdir(parents=True, exist_ok=True)
    pd.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(block=block, stage=stage, arm=arm, kind=kind, pool=pool, seeds=seeds, sched=sched(),
                   L2_created=lk["created"], wall_s=time.time() - t0,
                   results=[{k: v for k, v in r.items() if k != "deliv"} for r in res]), open(pj, "w"))
    json.dump({f"{r['cfg']}|{r['pool']}|{r['combo']}|{r['ckpt']}": r["deliv"] for r in res}, gzip.open(pd, "wt"))
    print(f"{block}: {len(res)} arm rows x {len(seeds)} seeds done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=list(GROUPS) + list(BLOCKS))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    for b in GROUPS.get(a.what, [a.what]):
        run_block(b, a.procs)
