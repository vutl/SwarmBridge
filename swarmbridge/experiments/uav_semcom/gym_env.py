"""Gymnasium wrapper (env v2): centralised MultiDiscrete action. Per UAV:
[move (11), channel (M+1), power level (P), mode of pending image 1 (n_modes), ..., mode of pending image L].
action_masks() gives the valid entries per sub-action (for maskable policy-gradient methods)."""
import gymnasium as gym
import numpy as np

import env as E


def decode_action(core, action):
    """Flat MultiDiscrete vector -> list of per-UAV (move, channel, power, modes) tuples."""
    c, L = core.c, core.L
    a = np.asarray(action).reshape(c["K"], 3 + L)
    return [(int(r[0]), int(r[1]), int(r[2]), tuple(int(x) for x in r[3:])) for r in a]


class GymUavSemcom(gym.Env):
    metadata = {}

    def __init__(self, table, pool, cfg=None, seed=0, seeds=None):
        self.core = E.UavSemcomEnv(table, pool, cfg, seed)
        c = self.core.c
        self.per = [E.UavSemcomEnv.N_MOVE, c["M"] + 1, len(c["p_levels"])] + [self.core.n_modes] * self.core.L
        self.action_space = gym.spaces.MultiDiscrete(self.per * c["K"])
        obs = self.core.reset(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.seeds = seeds  # optional fixed list of episode seeds (evaluation)
        self._ep = 0

    def reset(self, seed=None, options=None):
        if self.seeds is not None:
            s = self.seeds[self._ep % len(self.seeds)]
            self._ep += 1
        else:
            s = int(self._rng.integers(0, 2 ** 31)) if seed is None else seed
        return self.core.reset(s), {}

    def step(self, action):
        obs, r, done = self.core.step(decode_action(self.core, action))
        info = self.core.metrics() if done else {}
        return obs, float(r), done, False, info

    def action_masks(self):
        return action_masks(self.core, self.per)


def action_masks(core, per):
    """Valid sub-actions: inactive UAV -> only (hover, silent, power 0, mode 0); survey done -> no macro 9;
    mode entries beyond the number of images awaiting a decision -> only mode 0; empty queue -> silent."""
    out = []
    for u in core.uavs:
        n_pend = len(core.awaiting(u))
        for j, n in enumerate(per):
            m = np.ones(n, dtype=bool)
            if not u.active:
                m[1:] = False
            elif j == 0 and u.wp_i >= len(u.wps):
                m[9] = False
            elif j == 1 and not u.queue:
                m[1:] = False
            elif j >= 3 and (j - 3) >= n_pend:
                m[1:] = False
            out.append(m)
    return np.concatenate(out)
