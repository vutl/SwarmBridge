"""A4 confirmation on FRESH test seeds (default 5000..5099), never used while designing or tuning any policy or
heuristic (train/tune seeds 0..29, validation 2000..2009, earlier test 1000..1029).

Evaluates on the same seeds (paired):
  * every PPO model given (--model LABEL=PATH[:PART]; PART = joint | movement | mode as trained);
  * the heuristic families (base v2 family + event-greedy modes, and slot x budget_mode), so the train-tuned members
    chosen BEFORE this run (tune_v20 / tune_budget_v20 JSON) and every other member are reported;
  * for joint models, action-replacement ablation: each of {move, channel, power, mode} is taken from PPO ('P') or
    from the heuristic ('H': slot parts from the train-tuned slot heuristic, i.e. assign_channels / max power /
    its pacing move; mode from event-greedy), 2^4 combinations on the trained policy (no retraining; a replaced part
    puts the policy off its own state distribution, which this cannot correct for).
Usage: python final_a4.py OUT_JSON --model joint_s0=/path/ppo_s0.zip:joint ... --slot-heur pacedB1.5 --profiles ...
"""
import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np

import env as E
import env_ev as V
import env_survey as SV
import gym_env_ev as GE
import mode_heur as MH
import policies as P
import timeppo  # noqa: F401  (so models saved by TimeMaskablePPO unpickle)
from sb3_contrib import MaskablePPO
from tune_heur import family

G = {}
PARTS = ("move", "chan", "pow", "mode")


def init(profiles, modes, cpu_file, cfg, slot_heur):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["te"] = G["table"].split()[1]
    G["modes"], G["cfg"], G["slot_heur"], G["models"] = modes, cfg, slot_heur, {}


def model(path):
    if path not in G["models"]:
        G["models"][path] = MaskablePPO.load(path, device="cpu")
    return G["models"][path]


def rl_episode(path, part, combo, s):
    heur = P.from_name(G["slot_heur"], G["modes"])
    kw = dict(movement=dict(mode_pol=V.greedy_mode), mode=dict(slot_pol=heur), joint={})[part]
    e = GE.GymUavSemcomEv(G["table"], G["te"], G["cfg"], seed=s, seeds=[s], core_cls=SV.UavSurveyEnv, **kw)
    obs, _ = e.reset()
    m, done, tot = model(path), False, 0.0
    while not done:
        a, _ = m.predict(obs, deterministic=True, action_masks=e.action_masks())
        a = np.array(a)
        if e.phase == "slot" and "H" in combo[:3]:
            ha = heur(e.core)
            for k in range(e.K):
                for j in range(3):
                    if combo[j] == "H":
                        a[3 * k + j] = ha[k][j]
        elif e.phase == "mode" and combo[3] == "H":
            a[-1] = V.greedy_mode(e.core, *e.pending)
        obs, r, done, _, info = e.step(a)
        tot += r
    info["return"] = tot
    return info


def job(j):
    kind, label, arg, part, combo, s = j
    if kind == "rl":
        m = rl_episode(arg, part, combo, s)
    else:
        sp, mp = MH.from_name(arg, G["modes"])
        m = V.run_episode_ev(SV.UavSurveyEnv(G["table"], G["te"], G["cfg"]), sp, mp, s)
    keep = ("return", "mean_util", "ok_rate", "drop_rate", "q_viol_rate", "late_rate", "mean_psnr", "mean_delay",
            "mission_s", "energy_kj", "tx_energy_j", "captured")
    return (label, combo), {k: float(m[k]) for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", action="append", default=[], help="LABEL=PATH[:PART]")
    ap.add_argument("--slot-heur", required=True)
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--no-heur", action="store_true")
    ap.add_argument("--no-mix", action="store_true")
    args = ap.parse_args()
    cfg, modes = json.loads(args.cfg), args.modes.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.n))
    assert not set(seeds) & (set(range(30)) | set(range(1000, 1030)) | set(range(2000, 2010)))
    jobs = []
    for spec in args.model:
        label, rest = spec.split("=", 1)
        path, part = rest.rsplit(":", 1) if rest.endswith((":joint", ":movement", ":mode")) else (rest, "joint")
        combos = ["".join(c) for c in itertools.product("PH", repeat=4)] if part == "joint" and not args.no_mix \
            else ["PPPP"]
        jobs += [("rl", label, path, part, c, s) for c in combos for s in seeds]
    if not args.no_heur:
        names = list(dict.fromkeys(family(modes) + MH.family()))
        jobs += [("heur", "H:" + n, n, None, "-", s) for n in names for s in seeds]
    with Pool(args.procs, initializer=init, initargs=(args.profiles, modes, args.cpu_file, cfg, args.slot_heur)) as p:
        res = p.map(job, jobs, chunksize=4)
    out = {}
    for (label, combo), m in res:
        out.setdefault(label, {}).setdefault(combo, []).append(m)
    json.dump(dict(cfg=cfg, seeds=seeds, slot_heur=args.slot_heur, parts=PARTS, results=out), open(args.out, "w"))
    for label, d in out.items():
        for combo, eps in d.items():
            if combo in ("PPPP", "-") or label.startswith("joint") or label.startswith("td_joint"):
                print(f"{label:28s} {combo}  return {np.mean([e['return'] for e in eps]):+8.2f}  "
                      f"ok {np.mean([e['ok_rate'] for e in eps]):.3f}  psnr {np.nanmean([e['mean_psnr'] for e in eps]):.1f}"
                      f"  mission {np.mean([e['mission_s'] for e in eps]):.0f}s", flush=True)


if __name__ == "__main__":
    main()
