"""Review #39: practical feedback-aware reuse back-off ("NACK gate"), usable on top of ANY slot policy (heuristic or
learned) in the link-adaptation env (env_la).

Rule (causal: only ACK / NACK history of the env's link adapters, no hidden parameter):
  * a UAV is FLAGGED for `cool` slots when, among the blocks it sent WITH co-channel interference since its last
    flag, the last <= n of them contain at least n_min blocks and their NACK rate exceeds `theta`;
  * a channel that the base policy gives to >= 2 UAVs becomes exclusive if ANY of its occupants is flagged: the most
    urgent occupant (least slack of its head image; ties by index) keeps it, the others are silenced;
  * silenced UAVs then take channels that nobody uses (most urgent first);
  * after `cool` slots the UAV may share again (probe); a new flag needs n_min fresh shared blocks.
Name suffix for mode_heur.from_name: "BASE+gate{theta}c{cool}" (e.g. "pacedB2+budget1+reuse+gate0.25c10")."""
import math

import numpy as np


def _state(env):
    st = env.__dict__.get("_gate")
    if st is None or st["key"] != id(env.tasks):
        K = len(env.uavs)
        st = env._gate = dict(key=id(env.tasks), until=[-1] * K, since=[-1] * K, flags=0)
    return st


def apply_gate(env, acts, theta, cool, n=10, n_min=3):
    st = _state(env)
    b_now = env.block_of(env.t)
    for k, ad in enumerate(env.la):
        if st["until"][k] > env.slot:
            continue
        sh = [x[1] for x in ad.hist if x[2] and x[0] >= st["since"][k]][-n:]
        if len(sh) >= n_min and float(np.mean(sh)) > theta:
            st["until"][k], st["since"][k] = env.slot + cool, b_now
            st["flags"] += 1
    acts = [list(a) for a in acts]

    def slack(k):
        q = env.uavs[k].queue
        return env.latest_tx_end(q[0]) - env.t if q else math.inf
    occ = {}
    for k, a in enumerate(acts):
        if a[1] > 0:
            occ.setdefault(a[1], []).append(k)
    silenced = []
    for ch, ks in occ.items():
        if len(ks) >= 2 and any(st["until"][k] > env.slot for k in ks):
            keep = min(ks, key=lambda k: (slack(k), k))
            for k in ks:
                if k != keep:
                    acts[k][1] = 0
                    silenced.append(k)
    free = [ch for ch in range(1, env.c["M"] + 1) if all(a[1] != ch for a in acts)]
    for k in sorted(silenced, key=lambda k: (slack(k), k)):
        if not free:
            break
        acts[k][1] = free.pop(0)
    return [tuple(a) for a in acts]


def nack_gate(slot_pol, theta, cool):
    def pol(env):
        return apply_gate(env, slot_pol(env), theta, cool)
    pol.__name__ = f"{slot_pol.__name__}+gate{theta:g}c{cool}"
    return pol


def parse(name):
    """'BASE+gate0.25c10' -> ('BASE', 0.25, 10)"""
    base, g = name.rsplit("+gate", 1)
    th, cool = g.split("c")
    return base, float(th), int(cool)


def gate_family(base):
    return [f"{base}+gate{t:g}c{c}" for t in (0.15, 0.25, 0.4) for c in (3, 10, 30)]
