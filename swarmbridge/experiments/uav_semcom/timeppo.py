"""Time-consistent (semi-MDP) discounting for MaskablePPO on the two-timescale env (fix of the A4 batch-1 objective).

Stock (Maskable)PPO discounts per DECISION STEP: delta_t = r_t + gamma V(s_{t+1}) - V(s_t), A_t = delta_t +
gamma*lambda*A_{t+1}. In the two-timescale env a mode step can take ~0 s and a slot step 1 s, so the effective horizon
depended on how many decisions an episode has (joint > mode-only > movement-only RL). Here both factors are per
SECOND of simulated time:

    dt_t   = (time of the next RL decision) - (time of decision t)        [from info["dt"], set by gym_env_ev]
    delta_t = r_t + gamma^dt_t (1 - d_t) V(s_{t+1}) - V(s_t)
    A_t     = delta_t + (gamma*lambda)^dt_t (1 - d_t) A_{t+1}

r_t is the reward accrued in [t, t + dt_t) (lumped at the decision instant, as before). With dt = 1 s everywhere this
is exactly stock GAE (same float32 operations, tested); several decisions at the same instant (dt = 0) add no
discount and no lambda decay. Each env column of the vectorized buffer uses its own dt.

How dt reaches the buffer: sb3-contrib 2.9.2 MaskablePPO.collect_rollouts calls self._update_info_buffer(infos, dones)
right before rollout_buffer.add(...). The override below hands that step's dt to the buffer, and the buffer's add()
asserts it received one, so a change of that call order in another sb3 version fails loudly instead of silently
falling back to per-step discounting. Truncation bootstrapping (which would use the per-step gamma) is asserted
absent: the env only terminates.
"""
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


class TimeRolloutBuffer(MaskableRolloutBuffer):
    def reset(self):
        self.dts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_dt = None
        super().reset()

    def add(self, *args, **kwargs):
        assert self.next_dt is not None, "dt of this step was not provided (infos['dt'] missing or call order changed)"
        self.dts[self.pos] = self.next_dt
        self.next_dt = None
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        dts = self.dts.astype(np.float64)
        g = (self.gamma ** dts).astype(np.float32)                       # gamma^dt
        gl = ((self.gamma * self.gae_lambda) ** dts).astype(np.float32)  # (gamma*lambda)^dt
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + g[step] * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gl[step] * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


class TimeMaskablePPO(MaskablePPO):
    """MaskablePPO whose gamma and gae_lambda are per second of env time (see module docstring)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", TimeRolloutBuffer)
        super().__init__(*args, **kwargs)
        assert isinstance(self.rollout_buffer, TimeRolloutBuffer)

    def _update_info_buffer(self, infos, dones=None):
        assert not any(i.get("TimeLimit.truncated", False) for i in infos), "truncation not supported"
        self.rollout_buffer.next_dt = np.array([i["dt"] for i in infos], dtype=np.float32)
        super()._update_info_buffer(infos, dones)
