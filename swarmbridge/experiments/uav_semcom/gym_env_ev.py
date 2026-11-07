"""Gymnasium wrapper for the two-timescale env (env_ev.py): ONE policy, two kinds of decision steps.

  slot step  (at every slot start): action = (move, channel, power) per UAV; the mode entry is masked to 0.
  mode step  (whenever an image's per-mode info becomes available, possibly mid-slot): action = mode of THAT image;
             all slot entries are masked to 0.
The observation is the env observation at the decision instant + [is_mode_step] + features of the image being
decided (UAV one-hot, slack, quality floor, per-mode (bits, PSNR) as known at that instant, its UAV's encoder backlog
and transmit backlog). Nothing computed after the decision instant enters the observation.
Action space: MultiDiscrete([11, M+1, P] * K + [n_modes]). Reward of a step = reward accrued until the next decision.
info of every step: dt = env time from this RL decision to the next one (or to the episode end), kind = "slot"/"mode";
at the episode end also the experience counters ep_* (RL and heuristic decisions of each kind, env time).
Discounting: stock PPO applies gamma per decision step (A4 batch 1, preliminary); timeppo.TimeMaskablePPO uses dt to
apply gamma and GAE lambda per second (semi-MDP), the objective used from the A4 corrected runs on.
"""
import gymnasium as gym
import numpy as np

import env_ev as V


class GymUavSemcomEv(gym.Env):
    metadata = {}

    def __init__(self, table, pool, cfg=None, seed=0, seeds=None, core_cls=None, slot_pol=None, mode_pol=None):
        """slot_pol / mode_pol (ablations): if given, that kind of decision is taken by the heuristic inside the
        wrapper and the RL policy only sees the other kind (movement-only RL: mode_pol=env_ev.greedy_mode;
        mode-only RL: slot_pol=<v2 heuristic>)."""
        assert not (slot_pol and mode_pol), "at least one decision kind must be left to the RL policy"
        self.slot_pol, self.mode_pol = slot_pol, mode_pol
        self.core = (core_cls or V.UavSemcomEnvEv)(table, pool, cfg, seed)   # env_survey.UavSurveyEnv for A4
        c = self.core.c
        self.K, self.n_modes = c["K"], self.core.n_modes
        self.slot_nvec = [V.E.UavSemcomEnv.N_MOVE, c["M"] + 1, len(c["p_levels"])]
        self.action_space = gym.spaces.MultiDiscrete(self.slot_nvec * self.K + [self.n_modes])
        self._rng = np.random.default_rng(seed)
        self.seeds, self._ep = seeds, 0
        self.phase, self.pending = "slot", None
        self.core.reset(seed)
        o = self._obs()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=o.shape, dtype=np.float32)

    def _obs(self):
        core = self.core
        f = np.zeros(1 + self.K + 4 + 2 * self.n_modes, dtype=np.float32)
        if self.phase == "mode":
            k, tk = self.pending
            u = core.uavs[k]
            b, q = core.mode_info(tk)
            tx_bits = sum(t.rem_bits for t in u.queue if t.mode >= 0)
            f[0] = 1.0
            f[1 + k] = 1.0
            o = 1 + self.K
            f[o:o + 4] = [(core.latest_tx_end(tk) - core.t) / 12, tk.qmin / 40, max(u.enc_free - core.t, 0.0),
                          np.log10(1 + tx_bits) / 7]
            f[o + 4:o + 4 + self.n_modes] = np.log10(b) / 7
            f[o + 4 + self.n_modes:] = np.asarray(q) / 40
        return np.concatenate([core.obs(), f]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if self.seeds is not None:
            s = self.seeds[self._ep % len(self.seeds)]
            self._ep += 1
        else:
            s = int(self._rng.integers(0, 2 ** 31)) if seed is None else seed
        self.core.reset(s)
        self.phase, self.pending = "slot", None
        self._done = False
        self._cnt = dict(ep_rl_slot=0, ep_rl_mode=0, ep_heur_slot=0, ep_heur_mode=0)
        self._run_heuristic_parts()   # reward accrued before the first RL decision is paid out with its first step
        return self._obs(), {}

    def _apply(self, slot_actions=None, mode=None):
        if slot_actions is not None:
            self.core.begin_slot(slot_actions)
        else:
            self.core.decide(mode)
        ev = self.core.advance()
        if ev[0] == "mode":
            self.phase, self.pending = "mode", (ev[1], ev[2])
        else:
            self.phase, self.pending, self._done = "slot", None, ev[1]

    def _run_heuristic_parts(self):
        """Resolve decisions owned by a heuristic until the next RL decision (or the episode end)."""
        while not self._done:
            if self.phase == "slot" and self.slot_pol is not None:
                self._cnt["ep_heur_slot"] += 1
                self._apply(slot_actions=[a[:3] for a in self.slot_pol(self.core)])
            elif self.phase == "mode" and self.mode_pol is not None:
                self._cnt["ep_heur_mode"] += 1
                self._apply(mode=self.mode_pol(self.core, *self.pending))
            else:
                break

    def step(self, action):
        a = np.asarray(action)
        t0, kind = self.core.t, self.phase
        self._cnt["ep_rl_" + kind] += 1
        if self.phase == "slot":
            self._apply(slot_actions=[tuple(a[3 * k:3 * k + 3]) for k in range(self.K)])
        else:
            self._apply(mode=int(a[-1]))
        self._run_heuristic_parts()
        r = self.core.pop_reward()
        info = dict(dt=float(self.core.t - t0), kind=kind)
        if self._done:
            info.update(self.core.metrics(), ep_t=float(self.core.t), **self._cnt)
            info["ep_uav_s"] = info["mission_s"] * self.K
        return self._obs(), float(r), self._done, False, info

    def action_masks(self):
        core, out = self.core, []
        for u in core.uavs:
            for j, n in enumerate(self.slot_nvec):
                m = np.ones(n, dtype=bool)
                if self.phase == "mode" or not u.active:
                    m[1:] = False
                elif j == 0 and u.wp_i >= len(u.wps):
                    m[9] = False
                elif j == 1 and not u.queue:
                    m[1:] = False
                out.append(m)
        mm = np.ones(self.n_modes, dtype=bool)
        if self.phase == "slot":
            mm[1:] = False
        out.append(mm)
        return np.concatenate(out)
