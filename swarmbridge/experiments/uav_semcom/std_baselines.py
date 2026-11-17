"""Literature-style (untuned) baselines for the A5 envs, added after the A-X2 K5 results (user question: the RL papers in
pdf/ compare against simple rules, e.g. SchedCP: Nearest / Round Robin / Max Rate; ours against a TUNED heuristic).

All are causal and use the same information as the A5 heuristics (policies.py): mission movement (next waypoint),
max power, at most one UAV per channel (no co-channel reuse, as in those papers' orthogonal scheduling).
  std_rr       round robin: in slot s the busy UAVs (non-empty queue) are taken in cyclic index order starting at
               s mod K; the first M get channels 1..M; modes by the event-greedy rule (env_ev.greedy_mode)
  std_maxrate  max rate: the M busy UAVs with the highest expected rate (at max power) get the channels; with equal
               power this is also "nearest" (expected gain depends on the UAV position only); greedy modes
  std_rand     random: M busy UAVs drawn uniformly each slot, uniformly random mode per image (the floor); the draws
               are a deterministic function of the episode layout, slot and UAV, so episodes are reproducible
Names plug into mode_heur.from_name (prefix std_), hence into a5_eval (H:std_rr ...) and env_ev.run_episode_ev.
"""
import hashlib

import numpy as np

import env_ev as V
import policies as P


def _busy(env):
    return [k for k, u in enumerate(env.uavs) if u.active and u.queue]


def _acts(env, chosen):
    ch = [0] * len(env.uavs)
    for i, k in enumerate(chosen[:env.c["M"]]):
        ch[k] = i + 1
    pmax = len(env.c["p_levels"]) - 1
    return [(P.mission_move(env, k), ch[k], pmax, (0,) * env.L) for k in range(len(env.uavs))]


def round_robin(env):
    K, busy = len(env.uavs), set(_busy(env))
    order = [(env.slot + j) % K for j in range(K)]
    return _acts(env, [k for k in order if k in busy])


def max_rate(env):
    busy = _busy(env)
    return _acts(env, sorted(busy, key=lambda k: (-P.expected_rate(env, k), k)))


def _rng(env, *key):
    """deterministic per episode (layout of the first UAV's survey), slot / time and key."""
    w = env.uavs[0].wps[0]
    h = hashlib.sha256(repr((round(float(w[0]), 6), round(float(w[1]), 6)) + key).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def random_slot(env):
    busy = _busy(env)
    perm = list(_rng(env, "slot", env.slot).permutation(len(busy))) if busy else []
    return _acts(env, [busy[i] for i in perm])


def random_mode(env, k, task):
    return int(_rng(env, "mode", k, round(float(task.t_cap), 6)).integers(env.n_modes))


round_robin.__name__, max_rate.__name__, random_slot.__name__ = "std_rr", "std_maxrate", "std_rand"
POLICIES = {"std_rr": (round_robin, V.greedy_mode), "std_maxrate": (max_rate, V.greedy_mode),
            "std_rand": (random_slot, random_mode)}


def from_name(name):
    return POLICIES[name]
