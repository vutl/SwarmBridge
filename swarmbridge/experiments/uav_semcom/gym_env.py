"""Gymnasium wrapper: centralised MultiDiscrete action (per UAV: move 9, channel M+1, power levels, mode)."""
import gymnasium as gym
import numpy as np

import env as E


class GymUavSemcom(gym.Env):
    metadata = {}

    def __init__(self, table, pool, cfg=None, seed=0, seeds=None):
        self.core = E.UavSemcomEnv(table, pool, cfg, seed)
        c = self.core.c
        per = [E.UavSemcomEnv.N_MOVE, c["M"] + 1, len(c["p_levels"]), self.core.n_modes]
        self.action_space = gym.spaces.MultiDiscrete(per * c["K"])
        self._per = len(per)
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
        a = np.asarray(action).reshape(self.core.c["K"], self._per)
        obs, r, done = self.core.step([tuple(int(x) for x in row) for row in a])
        info = self.core.metrics() if done else {}
        return obs, float(r), done, False, info
