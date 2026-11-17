"""A-E1 (reports/two-track/swarmbridge-two-track-plan.md, review #27): online decision cost of PPO mode-only vs H* vs SC.

Same code paths as A5.5 (a55_eval.job): heuristics via a55_heur.from_name + env_ev loop; PPO mode-only via
gym_env_ev.GymUavSemcomEv(slot_pol = slot part of the A5 T:best heuristic) and the A5.5 best-val checkpoints.
Per decision we time, with time.perf_counter_ns and one torch thread (as in A5/A5.5 evaluation):
  heuristic  slot decision  = slot_pol(core)                        (channel / power / movement for all UAVs)
             mode decision  = mode_pol(core, k, task)
  PPO        slot decision  = slot_pol(core) inside the wrapper     (same heuristic slot part, timed the same way)
             mode decision  = observation build + action masks + model.predict   ("controller cost")
             forward only   = policy forward on a prepared tensor (distribution + deterministic action)
The first WARMUP decisions of every (process, policy) are discarded. Returns are recomputed and compared with the
A5.5 confirmatory files for the same seeds (sanity: the timed loop must not change behaviour).
Usage: python a9_latency.py OUT_JSON [--regimes ref,K5,phy_mcs_blind] [--seeds 8000:8020] [--procs 3]"""
import argparse
import json
import platform
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import a55_eval as A55
import a55_heur as H
import env_ev as V
import env_survey as SV
import gym_env_ev as GE

WARMUP = 50
A55_FINAL = C.A5.parent / "a55" / "final"


def ppo_specs(regime):
    d = json.loads((A55_FINAL / f"test_ppo_{regime}.json").read_text())
    return [s for s in d["results"][regime] if s.startswith("P:mode_best_")]


class Timer:
    def __init__(self):
        self.t = {}

    def wrap(self, key, f):
        def g(*a):
            t0 = time.perf_counter_ns()
            out = f(*a)
            self.t.setdefault(key, []).append(time.perf_counter_ns() - t0)
            return out
        return g

    def add(self, key, ns):
        self.t.setdefault(key, []).append(ns)


def heur_episodes(regime, spec, seeds, tm):
    name = A55.resolve(regime, spec)
    sp, mp = H.from_name(name, C.MODES)
    sp, mp = tm.wrap("slot", sp), tm.wrap("mode", mp)
    rets = []
    for s in seeds:
        core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))
        t0 = time.perf_counter_ns()
        m = V.run_episode_ev(core, sp, mp, s)
        tm.add("episode", time.perf_counter_ns() - t0)
        rets.append(float(m["return"]))
    return name, rets


def ppo_episodes(regime, spec, seeds, tm):
    import torch
    label, rest = spec[2:].split("=", 1)
    path, part, heur, combo = rest.split(":")
    assert part == "mode" and combo == "PPPP"
    heur = AE.resolve(regime, "T:" + heur[2:])
    sp, _ = H.from_name(heur, C.MODES)
    sp = tm.wrap("slot", sp)
    mdl = AE.model(path)
    pol = mdl.policy
    rets = []
    for s in seeds:
        e = GE.GymUavSemcomEv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime), seed=s, seeds=[s],
                              core_cls=SV.UavSurveyEnv, slot_pol=sp)
        t_ep = time.perf_counter_ns()
        obs, _ = e.reset()
        done, tot = False, 0.0
        while not done:
            assert e.phase == "mode"
            t0 = time.perf_counter_ns()
            o = e._obs()
            mask = e.action_masks()
            a, _ = mdl.predict(o, deterministic=True, action_masks=mask)
            tm.add("mode", time.perf_counter_ns() - t0)
            t1 = time.perf_counter_ns()
            with torch.no_grad():
                ot = torch.as_tensor(o[None])
                pol.get_distribution(ot, action_masks=mask[None]).get_actions(deterministic=True)
            tm.add("forward", time.perf_counter_ns() - t1)
            obs, r, done, _, info = e.step(np.array(a))
            tot += r
        tm.add("episode", time.perf_counter_ns() - t_ep)
        rets.append(tot)
    return label, rets


def job(j):
    regime, spec, seeds = j
    tm = Timer()
    f = ppo_episodes if spec.startswith("P:") else heur_episodes
    f(regime, spec, seeds[:1], Timer())                      # warm-up episode (caches, model load, JIT paths)
    name, rets = f(regime, spec, seeds, tm)
    st = {}
    for k, v in tm.t.items():
        v = np.asarray(v[WARMUP:] if k != "episode" else v, float) / 1e3          # microseconds
        st[k] = dict(n=int(len(v)), mean_us=float(v.mean()), p50_us=float(np.percentile(v, 50)),
                     p95_us=float(np.percentile(v, 95)), p99_us=float(np.percentile(v, 99)), max_us=float(v.max()))
    return regime, spec, name, rets, st


def init():
    import torch
    torch.set_num_threads(1)
    A55.init(False, None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regimes", default="ref,K5,phy_mcs_blind")
    ap.add_argument("--seeds", default="8000:8020")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    jobs = []
    for r in a.regimes.split(","):
        for sp in ppo_specs(r) + ["T:HSTAR", "T:SC", "T5:best"]:
            jobs.append((r, sp, seeds))
    with Pool(a.procs, initializer=init) as p:
        res = p.map(job, jobs, chunksize=1)
    ref = {}
    for r in a.regimes.split(","):
        for fn in (f"test_ppo_{r}.json", "test_heur.json"):
            d = json.loads((A55_FINAL / fn).read_text())
            for sp, eps in d["results"].get(r, {}).items():
                ref[(r, sp)] = dict(zip(d["seeds"], [e["return"] for e in eps]))
    out = []
    for r, sp, name, rets, st in res:
        a55 = [ref[(r, sp)][s] for s in seeds] if (r, sp) in ref else None
        dev = float(np.max(np.abs(np.array(rets) - np.array(a55)))) if a55 else None
        out.append(dict(regime=r, spec=sp, resolved=name, mean_return=float(np.mean(rets)),
                        max_abs_dev_vs_a55=dev, timing=st))
        print(f"{r:14s} {sp[:28]:28s} ret {np.mean(rets):7.2f} dev {dev}  " +
              "  ".join(f"{k} {v['mean_us']:.0f}/{v['p95_us']:.0f}us" for k, v in st.items() if k != "episode"),
              flush=True)
    json.dump(dict(seeds=seeds, warmup=WARMUP, torch_threads=1, procs=a.procs, cpu=platform.processor() or
                   platform.machine(), python=platform.python_version(), rows=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
