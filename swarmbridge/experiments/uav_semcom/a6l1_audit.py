"""A6.0-L1 method audit (pre-registered: reports/archive/uav-semcom-a/uav-semcom-a6-l1-audit.md section 0). No PPO training.

  tune --regimes ..             every L1 heuristic family under the task reward of each audit config (xi_q in
                                {0.5, 1, 2, 4} x Qmin on/off), train pool, seeds 0..29 -> audit/tune/<regime>_<cfg>.json
  cross --pool test|testdev     0.2: PPO task / PPO PSNR (L1 / A5 best-val models) with BOTH slot parts, and the
                                heuristics with crossed slot / mode parts, seeds 9000..9099 -> audit/cross_<pool>.json
  val                           0.4: best_task of every config + comparators on the val pool, seeds 3000..3049
  final --pool test|testdev     0.4: the config chosen on val (rule 0.4) + comparators, seeds 9000..9099
Specs: as a6l1_eval (H:, T6:, T5:, P:LABEL=PATH:mode:HEUR:PPPP) plus TC:CFG:best (best_task of an audit config) and
X:SLOTSPEC|MODESPEC (slot part of one heuristic, mode part of another). A job is (regime, cfg, pool, spec, seed);
cfg '' = the L1 configuration (xi_q 0.5, Qmin on).
Extra metric recall_ontime: TP of frames delivered by their deadline regardless of Qmin / total objects.
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import a5_common as C
import a55_heur as H
import a6l1_eval as LE
import a6l1_task as TK
import env_ev as V
import gym_env_ev as GE

OUT = LE.OUT / "audit"
XIQ = (0.5, 1.0, 2.0, 4.0)
CONFIGS = {}
for _q in (True, False):
    for _x in XIQ:
        CONFIGS[f"xq{_x:g}" + ("" if _q else "_noqmin")] = dict(xi_q=_x, **({} if _q else dict(qmin=(0.0, 0.0))))
SEL = ["ref", "K5", "phy_mcs_blind"]
KEEP = LE.KEEP + ("return_psnr", "recall_ontime", "recall_small_ontime", "n_ontime", "bits_ontime")


def cfg_of(regime, cfgname):
    return C.regime_cfg(regime, **CONFIGS[cfgname]) if cfgname else C.regime_cfg(regime)


def tune_file(regime, cfgname):
    return OUT / "tune" / f"{regime}_{cfgname}.json"


def resolve(regime, s):
    if s.startswith("TC:"):
        cfgname, fam = s[3:].rsplit(":", 1)
        d = json.loads(tune_file(regime, cfgname).read_text())
        return d["best_task"] if fam == "best" else d["tuned"][fam]
    return LE.resolve(regime, s)


def heur_parts(regime, spec):
    if spec.startswith("X:"):
        s1, s2 = spec[2:].split("|")
        sp, _ = H.from_name(resolve(regime, s1), C.MODES)
        _, mp = H.from_name(resolve(regime, s2), C.MODES)
        return sp, mp
    return H.from_name(resolve(regime, spec), C.MODES)


def extra(core, m):
    tt = core.tt
    planned = core.task_img.ravel()
    ngt, nsm = float(np.sum(tt.ngt[planned])), float(np.sum(tt.nsmall[planned]))
    ont = [tk for tk in core.finished if not tk.dropped and tk.delay <= tk.dmax]
    if ont:
        oi = np.array([tk.img for tk in ont], int)
        om = np.array([tk.mode for tk in ont], int)
        tp, tps = float(tt.tp[oi, om].sum()), float(tt.tps[oi, om].sum())
    else:
        tp = tps = 0.0
    m = dict(m)
    m["return_psnr"] = m["return"] + m["psnr_minus_task"]
    m.update(recall_ontime=tp / max(ngt, 1.0), recall_small_ontime=tps / max(nsm, 1.0), n_ontime=len(ont),
             bits_ontime=float(sum(tk.bits for tk in ont)))
    ok = {(tk.uav, tk.wp): tk.mode for tk in core.finished if tk.ok}
    on = {(tk.uav, tk.wp): tk.mode for tk in ont}
    K, W = core.task_img.shape[:2]
    deliv = [[int(core.task_img[k, w, 0]), int(ok.get((k, w), -1)), int(on.get((k, w), -1))]
             for k in range(K) for w in range(W)]
    modes = np.bincount([tk.mode for tk in core.finished if not tk.dropped], minlength=len(C.MODES)).tolist()
    return {k: float(m[k]) for k in KEEP}, deliv, modes


def episode(regime, cfgname, pool, spec, seed):
    cfg = cfg_of(regime, cfgname)
    table, pools = LE.G["table"], LE.G["pools"]
    if spec.startswith("P:"):
        _, rest = spec[2:].split("=", 1)
        path, part, heur, combo = rest.split(":")
        assert part == "mode" and combo == "PPPP"
        sp, _ = H.from_name(resolve(regime, heur.replace("~", ":")), C.MODES)
        e = GE.GymUavSemcomEv(table, pools[pool], cfg, seed=seed, seeds=[seed], core_cls=TK.TaskSurveyEnv,
                              slot_pol=sp)
        mdl = LE.model(path)
        assert mdl.observation_space.shape == e.observation_space.shape
        obs, _ = e.reset()
        done, tot = False, 0.0
        while not done:
            a, _ = mdl.predict(obs, deterministic=True, action_masks=e.action_masks())
            obs, r, done, _, _ = e.step(np.array(a))
            tot += r
        core = e.core
        m = core.metrics()
        m["return"] = tot
    else:
        sp, mp = heur_parts(regime, spec)
        core = TK.TaskSurveyEnv(table, pools[pool], cfg)
        m = V.run_episode_ev(core, sp, mp, seed)
    return extra(core, m)


def job(j):
    return j, episode(*j)


def run(jobs, procs, testdev, chunk=2):
    res, t0 = [], time.time()
    with Pool(procs, initializer=LE.init, initargs=(testdev,)) as p:
        for i, x in enumerate(p.imap(job, jobs, chunksize=chunk)):
            res.append(x)
            if (i + 1) % max(len(jobs) // 20, 1) == 0:
                el = time.time() - t0
                print(f"progress {i + 1}/{len(jobs)} {el:.0f}s eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s", flush=True)
    return res


def save(res, out, meta):
    R, D, M = {}, {}, {}
    for (regime, cfgname, pool, spec, seed), (m, deliv, modes) in res:
        key = f"{cfgname}|{spec}" if cfgname else spec
        R.setdefault(regime, {}).setdefault(key, []).append(m)
        D.setdefault(regime, {}).setdefault(key, []).append(deliv)
        M.setdefault(regime, {}).setdefault(key, []).append(modes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta, results=R, modes=M), open(out, "w"))
    with gzip.open(str(out)[:-5] + ".deliv.json.gz", "wt") as f:
        json.dump(dict(note="[table image index, mode delivered ok or -1, mode delivered on time or -1]", results=D), f)
    for regime in R:
        for key, eps in R[regime].items():
            f = lambda k: np.mean([x[k] for x in eps])
            print(f"{regime:14s} {key[:70]:70s} R_task {f('return'):+7.2f} ok {f('ok_rate'):.3f} recall {f('recall'):.4f}"
                  f" ontime {f('recall_ontime'):.4f} Mbit {f('bits_sent') / 1e6:6.1f}", flush=True)


def ppo_specs(regime, kind):
    """Best-val PPO specs of L1 final ('task' or 'psnr'), as evaluated there."""
    d = json.loads((LE.OUT / "final" / f"test_{regime}.json").read_text())
    return sorted(s for s in d["resolved"][regime] if s.startswith(f"P:{kind}_best_s"))


def swap_slot(spec, new):
    lab, rest = spec[2:].split("=", 1)
    path, part, heur, combo = rest.split(":")
    return f"P:{lab}@{new.replace('~', '')}={path}:{part}:{new}:{combo}"


def cross_specs(regime):
    task, psnr = ppo_specs(regime, "task"), ppo_specs(regime, "psnr")
    return (task + [swap_slot(s, "T5~best") for s in task] + psnr + [swap_slot(s, "T6~best") for s in psnr]
            + ["T6:best", "T5:best", "X:T5:best|T6:best", "X:T6:best|T5:best"])


def cmd_tune(a):
    fams = LE.families()
    names = list(dict.fromkeys(sum(fams.values(), [])))
    regimes = a.regimes.split(",")
    cfgs = a.configs.split(",") if a.configs else list(CONFIGS)
    jobs = [(r, c, "train", "H:" + n, s) for r in regimes for c in cfgs for n in names for s in C.TRAIN_SEEDS]
    res = run(jobs, a.procs, False, 8)
    for r in regimes:
        for c in cfgs:
            rows = {}
            for (rg, cg, _, spec, _), (m, _, _) in res:
                if rg == r and cg == c:
                    rows.setdefault(spec[2:], []).append(m)
            summ = {n: {k: float(np.mean([e[k] for e in eps])) for k in ("return", "ok_rate", "recall", "recall_ontime",
                                                                        "bits_sent")} for n, eps in rows.items()}
            best = {f: max(fams[f], key=lambda n: summ[n]["return"]) for f in fams}
            best["fixed"] = max((n for n in fams["base"] if n.startswith("fixed_")), key=lambda n: summ[n]["return"])
            bt = max(names, key=lambda n: summ[n]["return"])
            tune_file(r, c).parent.mkdir(parents=True, exist_ok=True)
            tune_file(r, c).write_text(json.dumps(dict(regime=r, config=c, cfg=cfg_of(r, c), seeds=C.TRAIN_SEEDS,
                                                       tuned=best, best_task=bt, rows=summ), indent=1))
            print(r, c, "best_task", bt, f"{summ[bt]['return']:+.2f} recall_ontime {summ[bt]['recall_ontime']:.4f}"
                  f" Mbit {summ[bt]['bits_sent'] / 1e6:.1f}", flush=True)


def cmd_cross(a):
    regimes = a.regimes.split(",")
    seeds = LE.FINAL_SEEDS[:a.n]
    jobs = [(r, "", a.pool, sp, s) for r in regimes for sp in cross_specs(r) for s in seeds]
    res = run(jobs, a.procs, a.pool == "testdev")
    save(res, a.out or OUT / f"cross_{a.pool}.json", dict(pool=a.pool, seeds=seeds,
                                                          resolved={r: {sp: (sp if sp.startswith("P:") else
                                                                             "|".join(resolve(r, x) for x in sp[2:].split("|"))
                                                                             if sp.startswith("X:") else resolve(r, sp))
                                                                        for sp in cross_specs(r)} for r in regimes}))


def val_specs(regime):
    return ([(c, "TC:" + c + ":best") for c in CONFIGS] +
            [("", s) for s in ["T5:best", "T6:best"] + ppo_specs(regime, "psnr") + ppo_specs(regime, "task")])


def cmd_val(a):
    regimes = a.regimes.split(",")
    jobs = [(r, c, "val", sp, s) for r in regimes for c, sp in val_specs(r) for s in C.VAL_SEEDS]
    res = run(jobs, a.procs, False)
    save(res, OUT / "val.json", dict(pool="val", seeds=C.VAL_SEEDS,
                                     resolved={r: {c + "|" + sp if c else sp: resolve(r, sp) if not sp.startswith("P:")
                                                   else sp for c, sp in val_specs(r)} for r in regimes}))


def choose(regime, val):
    """Rule 0.4: highest val recall_ontime among configs with val bits <= PPO PSNR val bits (ties -> fewer bits)."""
    R = val["results"][regime]
    mean = lambda key, k: float(np.mean([e[k] for e in R[key]]))
    psnr = ppo_specs(regime, "psnr")
    cap = float(np.mean([mean(s, "bits_sent") for s in psnr]))
    rows = [(c, mean(f"{c}|TC:{c}:best", "recall_ontime"), mean(f"{c}|TC:{c}:best", "bits_sent")) for c in CONFIGS]
    ok = [r for r in rows if r[2] <= cap]
    pick = max(ok or rows, key=lambda r: (round(r[1], 12), -r[2]))
    return pick[0], bool(ok), cap, rows


def cmd_final(a):
    val = json.load(open(OUT / "val.json"))
    regimes = a.regimes.split(",")
    picks = {r: choose(r, val) for r in regimes}
    jobs = []
    for r in regimes:
        c = picks[r][0]
        specs = [(c, f"TC:{c}:best"), ("", "T6:best")] + [("", s) for s in ppo_specs(r, "psnr")]
        jobs += [(r, cg, a.pool, sp, s) for cg, sp in specs for s in LE.FINAL_SEEDS]
    res = run(jobs, a.procs, a.pool == "testdev")
    save(res, OUT / f"final_{a.pool}.json", dict(pool=a.pool, seeds=LE.FINAL_SEEDS,
                                                 picks={r: dict(config=p[0], feasible=p[1], psnr_val_bits=p[2],
                                                                rows=p[3]) for r, p in picks.items()},
                                                 resolved={r: resolve(r, f"TC:{picks[r][0]}:best") for r in regimes}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tune", "cross", "val", "final"):
        p = sub.add_parser(name)
        p.add_argument("--regimes", default=",".join(SEL))
        p.add_argument("--procs", type=int, default=6)
        if name in ("cross", "final"):
            p.add_argument("--pool", required=True, choices=["test", "testdev"])
        if name == "cross":
            p.add_argument("--n", type=int, default=100)
            p.add_argument("--out", default="")
        if name == "tune":
            p.add_argument("--configs", default="")
    a = ap.parse_args()
    dict(tune=cmd_tune, cross=cmd_cross, val=cmd_val, final=cmd_final)[a.cmd](a)


if __name__ == "__main__":
    main()
