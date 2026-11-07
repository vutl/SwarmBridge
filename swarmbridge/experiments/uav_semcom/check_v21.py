"""A2 v2.1 impact check: run the same policies (heuristics and the A3 v2 MaskablePPO models, unchanged) on the frozen
env v2 and on env v2.1, same 30 test seeds, and report return / utility / delay plus how much airtime is spent
co-channel with interference. No training.
Usage: python check_v21.py OUT_JSON --profiles ... --cpu-file ... --modes ... --models-dir DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

import env as E
import gym_env as G
import legacy_env_v2 as V2
import policies as P

# the v2 env has no causal-probe stage: everything without a mode awaits a decision
V2.UavSemcomEnv.awaiting = lambda self, u: [tk for tk in u.queue if tk.mode < 0]


def ppo_policy(model, per):
    def pol(env):
        a, _ = model.predict(env.obs(), deterministic=True, action_masks=G.action_masks(env, per))
        return G.decode_action(env, a)
    pol.__name__ = "ppo"
    return pol


def episode(env_cls, table, te, cfg, pol, seed):
    env = env_cls(table, te, cfg)
    m = P.run_episode(env, pol, seed)
    if env_cls is E.UavSemcomEnv:
        air = sum(s[3] - s[2] for s in env.segments)
        m["cochannel_airtime_frac"] = sum(s[3] - s[2] for s in env.segments if s[4] > 0) / air if air else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--models-dir", required=True)
    args = ap.parse_args()
    modes = args.modes.split(",")
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    tr, te = table.split()
    seeds = range(1000, 1030)
    res = {}
    for mi in ("predicted", "oracle"):
        for wt in (0.0, 0.3):
            cfg = dict(mode_info=mi, w_time=wt)
            key = f"{mi}_wt{wt}"
            g = G.GymUavSemcom(table, te, cfg, seed=0)
            model = MaskablePPO.load(str(Path(args.models_dir) / f"a3v2_{mi}_wt{wt}_s0" / "ppo_s0.zip"), device="cpu")
            pols = [ppo_policy(model, g.per), P.greedy(), P.fixed(modes.index("mbt2018-mean_q8")), P.paced(4),
                    P.staggered(3, "near"), P.staggered(None, "near")]
            res[key] = {}
            for pol in pols:
                row = {}
                for tag, cls in (("v2", V2.UavSemcomEnv), ("v2.1", E.UavSemcomEnv)):
                    eps = [episode(cls, table, te, cfg, pol, s) for s in seeds]
                    row[tag] = {k: float(np.mean([e[k] for e in eps])) for k in
                                ("return", "mean_util", "ok_rate", "mean_delay", "mission_s")}
                    row[tag]["return_all"] = [e["return"] for e in eps]
                    if tag == "v2.1":
                        row[tag]["cochannel_airtime_frac"] = float(np.mean([e["cochannel_airtime_frac"] for e in eps]))
                res[key][pol.__name__] = row
                print(f"{key:16s} {pol.__name__:12s} v2 {row['v2']['return']:+7.2f} -> v2.1 {row['v2.1']['return']:+7.2f}"
                      f" | delay {row['v2']['mean_delay']:.2f} -> {row['v2.1']['mean_delay']:.2f} s"
                      f" | co-channel airtime {row['v2.1']['cochannel_airtime_frac']:.3f}", flush=True)
            # PPO minus best heuristic, paired over seeds, on each env version
            for tag in ("v2", "v2.1"):
                best = max((n for n in res[key] if not n.startswith("ppo")), key=lambda n: res[key][n][tag]["return"])
                d = np.array(res[key]["ppo"][tag]["return_all"]) - np.array(res[key][best][tag]["return_all"])
                h = 2.045 * d.std(ddof=1) / np.sqrt(len(d))
                res[key][f"ppo_minus_best_{tag}"] = dict(best=best, mean=float(d.mean()), lo=float(d.mean() - h),
                                                          hi=float(d.mean() + h))
                print(f"{key:16s} PPO - best heuristic ({best}) on {tag}: {d.mean():+.2f} [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}]")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
