"""A5 evaluator (heuristic tuning, operating-regime sweep, final test), on the A5 pools (a5_common).

  tune  : every member of the heuristic families -- base (v2 slot family + event-greedy modes, incl. fixed_*),
          budget (slot x backlog-budgeted mode), reuse (base + budget with channel reuse, the A4 post-hoc rule) --
          on the TRAIN pool and seeds 0..29 of each regime; the member with the best mean train return is the
          family's tuned heuristic (chosen before any validation / test / PPO result of that regime is seen).
  eval  : policies on a pool (val | test | testdev) and seeds, paired (same seeds for every policy). Policy specs:
            H:NAME[@INFO]            heuristic by name (mode_heur.from_name), optionally with another mode_info
            T:FAMILY[@INFO]          the tuned member of FAMILY (base | budget | reuse | fixed) for that regime
            P:LABEL=PATH:PART:HEUR:COMBO
                                     a PPO model (PART = joint | mode | movement as trained); HEUR = heuristic
                                     name whose slot part drives mode-only RL's slots (and movement-only RL's modes
                                     come from its mode part) and whose parts replace PPO's where COMBO (4 letters
                                     over move/channel/power/mode, P = PPO, H = heuristic) says H; HEUR may be
                                     T:FAMILY. Zero-shot use of a model trained elsewhere is allowed only when its
                                     observation / action shapes match (asserted).
Per episode: the metrics, and (in a separate .deliv.json.gz) every finished frame as (manifest index, mode, ok)
for the A5.2 task-fidelity join.
Usage:
  python a5_eval.py tune --regimes ref,K5 [--procs 6]
  python a5_eval.py eval OUT --regimes ref,K5 --pool val --seeds 3000:3050 --policies T:base,T:reuse,H:greedy
"""
import argparse
import gzip
import json
from multiprocessing import Pool

import numpy as np

import a5_common as C
import env_ev as V
import env_survey as SV
import gym_env_ev as GE
import mode_heur as MH
import timeppo  # noqa: F401  (models saved by TimeMaskablePPO)
from sb3_contrib import MaskablePPO
from tune_heur import family as base_family

G = {}
KEEP = ("return", "mean_util", "ok_rate", "drop_rate", "q_viol_rate", "late_rate", "mean_psnr", "mean_delay",
        "mission_s", "makespan_s", "energy_kj", "tx_energy_j", "captured", "airtime_s")


def families():
    b = base_family(C.MODES)
    return dict(base=b, budget=MH.family(), reuse=MH.reuse_family(C.MODES))


def tune_path(regime):
    return C.A5 / "tune" / f"{regime}.json"


def tuned(regime, fam):
    d = json.loads(tune_path(regime).read_text())
    return d["tuned"][fam]


def init(testdev):
    G["table"], G["pools"] = C.load_table(testdev)
    man = {e["name"]: i for i, e in enumerate(C.manifest()["images"])}
    G["man_idx"] = np.array([man.get(n, -1) for n in G["table"].images])
    G["models"] = {}


def model(path):
    if path not in G["models"]:
        G["models"][path] = MaskablePPO.load(path, device="cpu")
    return G["models"][path]


def resolve(regime, name):
    """'T:FAMILY' -> tuned member name; anything else unchanged."""
    return tuned(regime, name[2:]) if name.startswith("T:") else name


def deliveries(core):
    """Every finished frame (delivered or dropped): (manifest index, mode or -1, ok)."""
    return [[int(G["man_idx"][tk.img]), int(tk.mode), int(tk.ok)] for tk in core.finished]


def heur_episode(regime, pool, name, seed):
    base, _, info = name.partition("@")
    cfg = C.regime_cfg(regime + ("@" + info if info else ""))
    sp, mp = MH.from_name(base, C.MODES)
    core = SV.UavSurveyEnv(G["table"], G["pools"][pool], cfg)
    m = V.run_episode_ev(core, sp, mp, seed)
    return m, deliveries(core)


def rl_episode(regime, pool, path, part, heur, combo, seed):
    cfg = C.regime_cfg(regime)
    sp, mp = MH.from_name(heur, C.MODES)
    kw = dict(movement=dict(mode_pol=mp), mode=dict(slot_pol=sp), joint={})[part]
    e = GE.GymUavSemcomEv(G["table"], G["pools"][pool], cfg, seed=seed, seeds=[seed], core_cls=SV.UavSurveyEnv, **kw)
    mdl = model(path)
    assert mdl.observation_space.shape == e.observation_space.shape, "obs shape mismatch (incompatible regime)"
    assert list(mdl.action_space.nvec) == list(e.action_space.nvec), "action space mismatch (incompatible regime)"
    obs, _ = e.reset()
    done, tot = False, 0.0
    while not done:
        a, _ = mdl.predict(obs, deterministic=True, action_masks=e.action_masks())
        a = np.array(a)
        if e.phase == "slot" and "H" in combo[:3]:
            ha = sp(e.core)
            for k in range(e.K):
                for j in range(3):
                    if combo[j] == "H":
                        a[3 * k + j] = ha[k][j]
        elif e.phase == "mode" and combo[3] == "H":
            a[-1] = mp(e.core, *e.pending)
        obs, r, done, _, info = e.step(a)
        tot += r
    info["return"] = tot
    return info, deliveries(e.core)


def job(j):
    regime, pool, spec, seed = j
    if spec.startswith("P:"):
        label, rest = spec[2:].split("=", 1)
        path, part, heur, combo = rest.split(":")
        heur = resolve(regime, "T:" + heur[2:]) if heur.startswith("T~") else heur
        m, dl = rl_episode(regime, pool, path, part, heur, combo, seed)
    else:
        name = spec[2:]
        if spec.startswith("T:"):
            fam, _, info = name.partition("@")
            name = tuned(regime, fam) + ("@" + info if info else "")
        m, dl = heur_episode(regime, pool, name, seed)
    return (regime, spec, seed), {k: float(m[k]) for k in KEEP}, dl


def run_tune(args):
    fams = families()
    names = list(dict.fromkeys(fams["base"] + fams["budget"] + fams["reuse"]))
    regimes = args.regimes.split(",")
    jobs = [(r, "train", "H:" + n, s) for r in regimes for n in names for s in C.TRAIN_SEEDS]
    with Pool(args.procs, initializer=init, initargs=(False,)) as p:
        res = p.map(job, jobs, chunksize=8)
    for r in regimes:
        rows = {}
        for (rg, spec, s), m, _ in res:
            if rg == r:
                rows.setdefault(spec[2:], []).append(m)
        summ = {n: {k: float(np.mean([e[k] for e in eps])) for k in ("return", "ok_rate", "mean_psnr", "mission_s",
                                                                    "energy_kj")} for n, eps in rows.items()}
        best = {f: max(fams[f], key=lambda n: summ[n]["return"]) for f in fams}
        best["fixed"] = max((n for n in fams["base"] if n.startswith("fixed_")), key=lambda n: summ[n]["return"])
        tune_path(r).parent.mkdir(parents=True, exist_ok=True)
        tune_path(r).write_text(json.dumps(dict(regime=r, cfg=C.regime_cfg(r), seeds=C.TRAIN_SEEDS, pool="train",
                                                tuned=best, rows=summ), indent=1))
        print(r, " ".join(f"{f}={best[f]}({summ[best[f]]['return']:+.2f})" for f in best), flush=True)


def run_eval(args):
    regimes = args.regimes.split(",")
    a, b = map(int, args.seeds.split(":"))
    seeds = list(range(a, b))
    if args.pool != "train":
        assert not set(seeds) & set(C.TRAIN_SEEDS)
    if args.pool == "test" or args.pool == "testdev":
        assert set(seeds) <= set(C.TEST_SEEDS)
    specs = args.policies.split(",")
    jobs = [(r, args.pool, sp, s) for r in regimes for sp in specs for s in seeds]
    with Pool(args.procs, initializer=init, initargs=(args.pool == "testdev",)) as p:
        res = p.map(job, jobs, chunksize=args.chunk)
    out, dl = {}, {}
    for (r, spec, s), m, d in res:
        out.setdefault(r, {}).setdefault(spec, []).append(m)
        dl.setdefault(r, {}).setdefault(spec, []).append(d)
    names = {r: {sp: (tuned(r, sp[2:].partition("@")[0]) if sp.startswith("T:") else sp) for sp in specs}
             for r in regimes}
    json.dump(dict(pool=args.pool, seeds=seeds, regimes={r: C.regime_cfg(r) for r in regimes}, resolved=names,
                   results=out), open(args.out, "w"))
    with gzip.open(args.out.replace(".json", "") + ".deliv.json.gz", "wt") as f:
        json.dump(dict(manifest_index_note="index into manifest['images']", results=dl), f)
    for r in regimes:
        for sp in specs:
            eps = out[r][sp]
            print(f"{r:14s} {sp[:60]:60s} ({names[r][sp][:28]:28s}) return {np.mean([e['return'] for e in eps]):+8.2f}"
                  f"  ok {np.mean([e['ok_rate'] for e in eps]):.3f}  psnr {np.nanmean([e['mean_psnr'] for e in eps]):.1f}"
                  f"  mission {np.mean([e['mission_s'] for e in eps]):.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--regimes", required=True)
    t.add_argument("--procs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("out")
    e.add_argument("--regimes", required=True)
    e.add_argument("--pool", required=True, choices=["train", "val", "test", "testdev"])
    e.add_argument("--seeds", required=True)
    e.add_argument("--policies", required=True)
    e.add_argument("--procs", type=int, default=6)
    e.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()
    (run_tune if args.cmd == "tune" else run_eval)(args)


if __name__ == "__main__":
    main()
