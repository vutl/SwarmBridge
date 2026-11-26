"""paperA_ix: training-action histograms (prereg §7). Read-only: the counters look at actions that were already chosen
and never call any RNG, model or environment method, so training is unchanged (tested in test_ix_actlog.py)."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ActionHist(BaseCallback):
    """PPO (SB3) callback: counts the executed mode action of every RL step whose kind is 'mode' (gym_env_ev: info
    kind = the kind of the step just taken; the mode is the last component of the MultiDiscrete action)."""

    def __init__(self, n_modes):
        super().__init__()
        self.counts = np.zeros(n_modes, np.int64)

    def _on_step(self):
        acts = np.asarray(self.locals["actions"])
        for j, info in enumerate(self.locals["infos"]):
            if info.get("kind") == "mode":
                self.counts[int(np.atleast_1d(acts[j])[-1])] += 1
        return True


def count(hist, act):
    """off-policy trainers: add the executed mode actions of one vectorised step to `hist` in place."""
    np.add.at(hist, np.asarray(act, np.int64), 1)
