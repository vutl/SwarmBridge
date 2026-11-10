"""R1b-0a driver (protocol reports/swarmbridge-r1b0a-protocol.md).

  lock  : FRAME quality rule (sec 6) on TUNE: q20 unless BS YOLO11s class-aware recall (conf >= 0.25, IoU 0.5) on the
          decoded frames is > 5 points below the uncompressed frames, then q35; frame_bits0 = TUNE mean bits; c_enc =
          idle CPU median (timing_tune.json); R_F = (median bits + packet headers) / (D_max - c_det - c_trk - c_enc -
          d_ul); R_ul = {0.5 (report only), 1, 2, 4} x R_F.  -> runs/uav_semcom/r1b/lock_v1.json
  tune  : every condition x R_ul level, full grid (sec 7) on TUNE clips x tuning seeds, scenario N, DL_cap 64.
          Objective: pooled (micro) class-aware F1; ties -> lower uplink.  -> tune_v1.json (+ all grid rows)
  gate  : chosen parameters on G7 (5 locked channel seeds) and AUX; per-source / pooled results, H1 verdict.
Pooled = micro F1 from summed TP / FP / FN over all clips x seeds; per-source = mean F1 over seeds.
"""
import argparse
import itertools
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from r1b_metrics import Scorer
from r1b_sim import Cfg, Clip, run_clip
from r1b_track import match

# amendment v1.1 (before any tuning outcome; compute budget): 64-point grid, one tuning seed. Measured ~50 s per grid
# point per condition per level per seed over the 48 TUNE clips on one core -> the 256 x 2-seed grid was ~33 h.
GRID = dict(th_T=[0.5, 1, 2, 4], th_F=[2, 4, 8, float("inf")], N_max=[30, 10 ** 9], w_lost=[0, 1], w_drift=[1],
            w_unc=[0])
PRIMARY = ["A1", "B", "Bp", "C1", "D"]
SECONDARY = ["A0", "C0"]
TUNE_SEEDS = [1000]
GATE_SEEDS = [0, 1, 2, 3, 4]


def load_clip(path, q):
    r = pickle.loads(Path(path).read_bytes())
    return Clip(r["clip"], r["W"], r["H"], r["gt"], r["ign"], r["det_n"], r["det_s"][q], r["bits"][q])


def grid():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*[GRID[k] for k in keys])]


# ---------------------------------------------------------------- lock
def recall_ca(gt, dets, thr=0.25):
    tp = n = 0
    for g, d in zip(gt, dets):
        d = [x for x in d if x[2] >= thr]
        n += len(g)
        for c in {x[2] for x in g}:
            gi = [x[1] for x in g if x[2] == c]; di = [x[0] for x in d if x[1] == c]
            p, _, _ = match(di, gi, 0.5)
            tp += len(p)
    return tp, n


def cmd_lock(a):
    pk = sorted(Path(a.precomp, "tune").glob("*.pkl"))
    tim = json.loads(Path(a.timing).read_text())
    rec = {"orig": [0, 0], 20: [0, 0], 35: [0, 0]}; bits = {20: [], 35: []}
    for p in pk:
        r = pickle.loads(p.read_bytes())
        for k, dets in [("orig", r["det_s_orig"]), (20, r["det_s"][20]), (35, r["det_s"][35])]:
            tp, n = recall_ca(r["gt"], dets); rec[k][0] += tp; rec[k][1] += n
        for q in (20, 35):
            bits[q] += r["bits"][q]
    R = {k: 100 * v[0] / max(v[1], 1) for k, v in rec.items()}
    q = 20 if R["orig"] - R[20] <= 5 else 35
    c_det, c_trk, d_ul, D_max = tim["yolo11n_640_cpu6_ms"]["median"], 2.0, 10.0, 300.0
    c_enc = tim[f"avif_{q}"]["encode_ms_median"]
    bmed = float(np.median(bits[q]))
    pay = bmed + 96 * int(np.ceil(bmed / 9600))
    R_F = pay / ((D_max - c_det - c_trk - c_enc - d_ul) / 1000.0)
    L = dict(version="r1b-lock-v1", n_tune_clips=len(pk), n_tune_ticks=len(bits[q]), recall_ca_conf025=R,
             rule="q20 unless recall(orig) - recall(q20) > 5 points", frame_q=q, frame_long=1344,
             frame_bits0=float(np.mean(bits[q])), frame_bits_median=bmed, frame_bits_p90=float(np.percentile(bits[q], 90)),
             c_det=c_det, c_trk=c_trk, c_enc=c_enc, d_ul=d_ul, D_max=D_max, R_F=R_F,
             R_ul_levels={"0.5xR_F (report only)": 0.5 * R_F, "1xR_F": R_F, "2xR_F": 2 * R_F, "4xR_F": 4 * R_F},
             DL_cap=64e3, tune_seeds=TUNE_SEEDS, gate_seeds=GATE_SEEDS, grid=GRID, grid_size=len(grid()))
    Path(a.out).write_text(json.dumps(L, indent=1, default=str))
    print(json.dumps(L, indent=1, default=str))


# ---------------------------------------------------------------- tune / gate
_CLIPS = {}


def _init(paths, q):
    global _CLIPS
    _CLIPS = {Path(p).stem: load_clip(p, q) for p in paths}


def base_cfg(L, cond, R_ul, seed, params):
    return Cfg(cond=cond, R_ul=R_ul, DL_cap=L["DL_cap"], c_det=L["c_det"], c_trk=L["c_trk"], c_enc=L["c_enc"],
               d_ul=L["d_ul"], D_max=L["D_max"], frame_bits0=L["frame_bits0"], seed=seed, **params)


def _job(args):
    L, cond, R_ul, params, seeds, names = args
    tp = fp = fn = 0; ul = dl = 0.0; t0 = time.time()
    rows = []
    for s in seeds:
        for nm in names:
            r = run_clip(_CLIPS[nm], base_cfg(L, cond, R_ul, s, params))
            tp += r["tp"]; fp += r["fp"]; fn += r["fn"]; ul += r["ul_bits"]; dl += r["dl_bits"]
            rows.append({k: r[k] for k in r if not isinstance(r[k], (list, dict))})
    P = tp / max(tp + fp, 1); Rr = tp / max(tp + fn, 1)
    return dict(cond=cond, R_ul=R_ul, params=params, F1=100 * 2 * P * Rr / max(P + Rr, 1e-12), tp=tp, fp=fp, fn=fn,
                ul_bits=ul, dl_bits=dl, secs=time.time() - t0, rows=rows)


def cmd_tune(a):
    L = json.loads(Path(a.lock).read_text())
    S = json.loads(Path(a.split).read_text())
    names = S["tune"] if not a.subset else S["tune"][::a.subset]
    paths = [str(Path(a.precomp, "tune", f"{c}.pkl")) for c in names]
    conds = a.conds.split(",")
    levels = list(L["R_ul_levels"].items())
    if a.levels:
        levels = [lv for lv in levels if lv[0] in a.levels.split(",")]
    G = grid() if not a.limit else grid()[:a.limit]
    jobs = [(L, c, R, p, TUNE_SEEDS, names) for c in conds for _, R in levels for p in G]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.procs, initializer=_init, initargs=(paths, L["frame_q"])) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            r.pop("rows")
            rows.append(r)
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} {r['cond']} {r['R_ul']:.0f} F1 {r['F1']:.2f} {r['secs']:.1f}s", flush=True)
    best = {}
    for c in conds:
        for lname, R in levels:
            cand = [r for r in rows if r["cond"] == c and r["R_ul"] == R]
            b = max(cand, key=lambda r: (round(r["F1"], 6), -r["ul_bits"]))
            best[f"{c}|{lname}"] = dict(params=b["params"], F1_tune=b["F1"], ul_bits=b["ul_bits"], dl_bits=b["dl_bits"])
    json.dump(dict(lock=a.lock, clips=names, seeds=TUNE_SEEDS, best=best), open(out, "w"), indent=1, default=str)
    json.dump(rows, open(str(out).replace(".json", "_grid.json"), "w"), default=str)
    print("TUNE_DONE", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock"); p.add_argument("precomp"); p.add_argument("timing"); p.add_argument("out")
    p = sub.add_parser("tune"); p.add_argument("split"); p.add_argument("precomp"); p.add_argument("lock")
    p.add_argument("out"); p.add_argument("--conds", default=",".join(PRIMARY + SECONDARY)); p.add_argument("--levels")
    p.add_argument("--procs", type=int, default=6); p.add_argument("--subset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"lock": cmd_lock, "tune": cmd_tune}[a.cmd](a)


if __name__ == "__main__":
    main()
