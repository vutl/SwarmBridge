"""A6.0-L1 (reports/archive/uav-semcom-a/uav-semcom-a6-l1.md section 0): tune heuristics and evaluate policies under the TASK reward.

  tune --regimes ..   every A5 family (base, budget, reuse) + the task-valued family a55_heur.family('tv'), train pool,
                      seeds 0..29, task reward -> runs/uav_semcom/a6l1/tune/<regime>.json (rows; tuned member per
                      family; best_task = best over all)
  eval OUT --regimes .. --pool test|testdev|val --seeds 9000:9100 --policies SPEC,..
       -> OUT (per-episode metrics incl. both returns and the task metrics), OUT.deliv.json.gz (every planned frame:
          [table image index, delivered mode or -1]) and OUT.dec.json.gz (state at every mode decision).
Specs: H:NAME (A5 or a55 names), T6:FAMILY (L1 tuned: base / budget / reuse / tv / fixed / best), T5:FAMILY (A5
PSNR-tuned), P:LABEL=PATH:PART:HEUR:COMBO with HEUR 'T6~best' / 'T5~best' / a name.
"""
import argparse
import gzip
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import a5_eval as AE
import a55_heur as H
import a6l1_task as TK
import env_ev as V
import gym_env_ev as GE
import mode_heur as MH
import policies as P
from tune_heur import family as base_family

OUT = C.A5.parent / "a6l1"
FINAL_SEEDS = list(range(9000, 9100))
BOOT_SEED0 = 30000          # video cluster bootstrap: replicate b, env seeds 30000 + n b + i
KEEP = AE.KEEP + ("psnr_minus_task", "recall", "recall_small", "fp", "fp_per_deliv", "precision", "task_score",
                  "task_score_b0", "task_score_b1", "recall_rel_orig", "bits_sent", "bits_deliv", "n_planned", "n_ok")
G = {}


def families():
    return dict(base=base_family(C.MODES), budget=MH.family(), reuse=MH.reuse_family(C.MODES), tv=H.family("tv"))


def tune_path(regime):
    return OUT / "tune" / f"{regime}.json"


def tuned(regime, fam):
    d = json.loads(tune_path(regime).read_text())
    return d["best_task"] if fam == "best" else d["tuned"][fam]


def resolve(regime, name):
    if name.startswith("T6:") or name.startswith("T6~"):
        return tuned(regime, name[3:])
    if name.startswith("T5:") or name.startswith("T5~"):
        return AE.tuned(regime, name[3:])
    return name[2:] if name.startswith("H:") else name


def init(testdev, boot_pool=None, B=0):
    import torch
    torch.set_num_threads(1)
    G["table"], G["pools"] = C.load_table(testdev)
    G["models"] = {}
    if boot_pool:
        seq = {e["name"]: e["seq"] for e in C.manifest()["images"]}
        by = {}
        for i in G["pools"][boot_pool]:
            by.setdefault(seq[G["table"].images[i]], []).append(int(i))
        seqs = sorted(by)
        for b in range(B):
            rng = np.random.default_rng([BOOT_SEED0, b, 0 if boot_pool == "test" else 1])
            draw = rng.choice(len(seqs), size=len(seqs), replace=True)
            G["pools"][f"{boot_pool}#{b}"] = np.array([i for j in draw for i in by[seqs[j]]])


def model(path):
    from sb3_contrib import MaskablePPO
    import timeppo  # noqa: F401
    if path not in G["models"]:
        G["models"][path] = MaskablePPO.load(path, device="cpu")
    return G["models"][path]


def dec_state(core, k, task):
    """Causal state at a mode decision (for the mode-choice analysis)."""
    tab = core.tab
    cx = math.log10(tab.bits[task.img, tab.probe_j])
    vh = core.task_value_hat(task.img)
    share = P.channel_share(core)
    r = max(P.expected_rate(core, k) * share, 1.0)
    u = core.uavs[k]
    snr = 10 * math.log10(core.c["p_levels"][-1] * core.expected_gain(u.pos) / core.noise_w())
    return [round(cx, 4), round(float(vh.max()), 4), round(P.backlog_s(core, k, r), 3), round(snr, 2),
            round(task.t_cap + task.dmax - core.t, 3), round(task.qmin, 2)]


def finish(core, m, dec):
    ok = {(tk.uav, tk.wp): tk.mode for tk in core.finished if tk.ok}
    K, W = core.task_img.shape[:2]
    deliv = [[int(core.task_img[k, w, 0]), int(ok.get((k, w), -1))] for k in range(K) for w in range(W)]
    m = dict(m)
    m["return_psnr"] = m["return"] + m["psnr_minus_task"]
    return {k: float(m[k]) for k in KEEP + ("return_psnr",)}, deliv, dec


def heur_episode(regime, pool, name, seed):
    sp, mp = H.from_name(name, C.MODES)
    core = TK.TaskSurveyEnv(G["table"], G["pools"][pool], C.regime_cfg(regime))
    dec = []

    def mp_log(env, k, task):
        mode = mp(env, k, task)
        dec.append(dec_state(env, k, task) + [int(mode)])
        return mode
    m = V.run_episode_ev(core, sp, mp_log, seed)
    return finish(core, m, dec)


def rl_episode(regime, pool, path, part, heur, combo, seed):
    cfg = C.regime_cfg(regime)
    sp, mp = H.from_name(heur, C.MODES)
    kw = dict(movement=dict(mode_pol=mp), mode=dict(slot_pol=sp), joint={})[part]
    e = GE.GymUavSemcomEv(G["table"], G["pools"][pool], cfg, seed=seed, seeds=[seed], core_cls=TK.TaskSurveyEnv, **kw)
    mdl = model(path)
    assert mdl.observation_space.shape == e.observation_space.shape
    assert list(mdl.action_space.nvec) == list(e.action_space.nvec)
    obs, _ = e.reset()
    done, tot, dec = False, 0.0, []
    while not done:
        a, _ = mdl.predict(obs, deterministic=True, action_masks=e.action_masks())
        a = np.array(a)
        if e.phase == "slot" and "H" in combo[:3]:
            ha = sp(e.core)
            for k in range(e.K):
                for j in range(3):
                    if combo[j] == "H":
                        a[3 * k + j] = ha[k][j]
        elif e.phase == "mode":
            if combo[3] == "H":
                a[-1] = mp(e.core, *e.pending)
            dec.append(dec_state(e.core, *e.pending) + [int(a[-1])])
        obs, r, done, _, info = e.step(a)
        tot += r
    m = e.core.metrics()
    m["return"] = tot
    return finish(e.core, m, dec)


def job(j):
    regime, pool, spec, seed = j
    if spec.startswith("P:"):
        label, rest = spec[2:].split("=", 1)
        path, part, heur, combo = rest.split(":")
        out = rl_episode(regime, pool, path, part, resolve(regime, heur), combo, seed)
    else:
        out = heur_episode(regime, pool, resolve(regime, spec), seed)
    return (regime, spec, seed), out


def run_pool(jobs, procs, testdev, chunk, boot_pool=None, B=0):
    res, t0 = [], time.time()
    with Pool(procs, initializer=init, initargs=(testdev, boot_pool, B)) as p:
        for i, x in enumerate(p.imap(job, jobs, chunksize=chunk)):
            res.append(x)
            if (i + 1) % max(len(jobs) // 20, 1) == 0:
                el = time.time() - t0
                print(f"progress {i + 1}/{len(jobs)} {el:.0f}s eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s", flush=True)
    return res


def run_tune(args):
    fams = families()
    names = list(dict.fromkeys(sum(fams.values(), [])))
    regimes = args.regimes.split(",")
    jobs = [(r, "train", "H:" + n, s) for r in regimes for n in names for s in C.TRAIN_SEEDS]
    res = run_pool(jobs, args.procs, False, 8)
    for r in regimes:
        rows = {}
        for (rg, spec, s), (m, _, _) in res:
            if rg == r:
                rows.setdefault(spec[2:], []).append(m)
        summ = {n: {k: float(np.mean([e[k] for e in eps])) for k in ("return", "return_psnr", "ok_rate", "mean_psnr",
                                                                    "recall", "task_score", "mission_s", "energy_kj")}
                for n, eps in rows.items()}
        best = {f: max(fams[f], key=lambda n: summ[n]["return"]) for f in fams}
        best["fixed"] = max((n for n in fams["base"] if n.startswith("fixed_")), key=lambda n: summ[n]["return"])
        bt = max(names, key=lambda n: summ[n]["return"])
        tune_path(r).parent.mkdir(parents=True, exist_ok=True)
        tune_path(r).write_text(json.dumps(dict(regime=r, cfg=C.regime_cfg(r), seeds=C.TRAIN_SEEDS, pool="train",
                                                beta=TK.BETA, tuned=best, best_task=bt, rows=summ), indent=1))
        print(r, "best_task", bt, f"{summ[bt]['return']:+.2f}", " ".join(f"{f}={best[f]}({summ[best[f]]['return']:+.2f})"
                                                                          for f in best), flush=True)


def run_eval(args):
    regimes = args.regimes.split(",")
    a, b = map(int, args.seeds.split(":"))
    seeds = list(range(a, b))
    if args.pool in ("test", "testdev"):
        assert set(seeds) <= set(FINAL_SEEDS), "final seeds only"
    specs = args.policies.split(",")
    jobs = [(r, args.pool, sp, s) for r in regimes for sp in specs for s in seeds]
    res = run_pool(jobs, args.procs, args.pool == "testdev", args.chunk)
    out, dl, dc = {}, {}, {}
    for (r, spec, s), (m, deliv, dec) in res:
        out.setdefault(r, {}).setdefault(spec, []).append(m)
        dl.setdefault(r, {}).setdefault(spec, []).append(deliv)
        dc.setdefault(r, {}).setdefault(spec, []).append(dec)
    names = {r: {sp: (resolve(r, sp) if not sp.startswith("P:") else sp) for sp in specs} for r in regimes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tab = G.get("table") or C.load_table(args.pool == "testdev")[0]
    json.dump(dict(pool=args.pool, seeds=seeds, resolved=names, results=out), open(args.out, "w"))
    base = args.out[:-len(".json")]
    with gzip.open(base + ".deliv.json.gz", "wt") as f:
        json.dump(dict(images=tab.images, note="[table image index, mode index or -1] per planned frame",
                       results=dl), f)
    with gzip.open(base + ".dec.json.gz", "wt") as f:
        json.dump(dict(fields=["log10_probe_bits", "vhat_max", "backlog_s", "snr_db", "slack_s", "qmin", "mode"],
                       results=dc), f)
    for r in regimes:
        for sp in specs:
            e = out[r][sp]
            f = lambda k: np.mean([x[k] for x in e])
            print(f"{r:14s} {sp[:44]:44s} ({names[r][sp][:28]:28s}) R_task {f('return'):+7.2f} R_psnr "
                  f"{f('return_psnr'):+7.2f} ok {f('ok_rate'):.3f} recall {f('recall'):.3f} small {f('recall_small'):.3f}"
                  f" fp/deliv {f('fp_per_deliv'):.1f} Mbit {f('bits_sent') / 1e6:.1f} psnr {np.nanmean([x['mean_psnr'] for x in e]):.1f}",
                  flush=True)


def run_boot(args):
    regimes = args.regimes.split(",")
    specs = args.policies.split(",")
    jobs = [(r, f"{args.pool}#{b}", sp, BOOT_SEED0 + args.n * b + i)
            for r in regimes for b in range(args.B) for sp in specs for i in range(args.n)]
    res = run_pool(jobs, args.procs, args.pool == "testdev", args.chunk, boot_pool=args.pool, B=args.B)
    keys = ("return", "return_psnr", "recall", "recall_small", "task_score", "bits_sent", "ok_rate", "fp_per_deliv")
    out = {}
    for (r, spec, s), (m, _, _) in res:
        b = (s - BOOT_SEED0) // args.n
        out.setdefault(r, {}).setdefault(str(b), {}).setdefault(spec, []).append({k: m[k] for k in keys})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pool=args.pool, B=args.B, n=args.n, seed0=BOOT_SEED0,
                   resolved={r: {sp: (resolve(r, sp) if not sp.startswith("P:") else sp) for sp in specs}
                             for r in regimes}, results=out), open(args.out, "w"))
    print("BOOT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--regimes", required=True)
    t.add_argument("--procs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("out")
    e.add_argument("--regimes", required=True)
    e.add_argument("--pool", required=True, choices=["test", "testdev", "val", "train"])
    e.add_argument("--seeds", default="9000:9100")
    e.add_argument("--policies", required=True)
    e.add_argument("--procs", type=int, default=6)
    e.add_argument("--chunk", type=int, default=2)
    b = sub.add_parser("boot")
    b.add_argument("out")
    b.add_argument("--regimes", required=True)
    b.add_argument("--pool", required=True, choices=["test", "testdev"])
    b.add_argument("--policies", required=True)
    b.add_argument("--procs", type=int, default=6)
    b.add_argument("--chunk", type=int, default=2)
    b.add_argument("--B", type=int, default=50)
    b.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    dict(tune=run_tune, eval=run_eval, boot=run_boot)[args.cmd](args)


if __name__ == "__main__":
    main()
