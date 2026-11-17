"""A-X2 (review #37): learned channel scheduling (C-only) and channel + mode (C+Mo) on the A5 two-timescale env.

GymSchedEv = the A5 wrapper (gym_env_ev.GymUavSemcomEv: same observation, MultiDiscrete action space, masks, reward,
separate slot / mode decision steps, dt for semi-MDP discounting), with movement and transmit power of every slot
step FIXED to the state-dependent heuristic slot rule `sp_heur` (the regime's T:best slot part):
  * the heuristic is evaluated on the CURRENT state at the slot step (never a previous slot's action; cached only
    for the same decision instant, keyed on the episode's task list, env time and slot index);
  * action_masks(): for every active UAV the move and power sub-masks allow exactly the heuristic's value (inactive
    UAVs: everything 0, as in A5); the channel sub-mask is the A5 one (no channel for an empty queue);
  * step(): the move and power entries of the action are overwritten with the heuristic's values before the env
    sees them, so the learned controller cannot influence them even if it ignores the masks;
  * learn_mode=False (C-only): mode decisions are taken inside the wrapper by `mp_heur` (the T:best mode rule; the
    policy never sees a mode step); learn_mode=True (C+Mo): the policy takes the mode steps as in A5 joint.
"""
import numpy as np

import gym_env_ev as GE


class GymSchedEv(GE.GymUavSemcomEv):
    def __init__(self, table, pool, cfg=None, seed=0, seeds=None, core_cls=None, sp_heur=None, mp_heur=None,
                 learn_mode=False):
        assert sp_heur is not None and (learn_mode or mp_heur is not None)
        self.sp_heur, self.learn_mode = sp_heur, learn_mode
        self._hkey, self._h = None, None
        super().__init__(table, pool, cfg, seed, seeds, core_cls, slot_pol=None, mode_pol=None if learn_mode else mp_heur)

    def heur_slot(self):
        """(move, power) per UAV from the heuristic on the current state (inactive UAV: (0, 0))."""
        core = self.core
        key = (id(core.tasks), core.t, core.slot)
        if key != self._hkey:
            h = [tuple(x[:3]) for x in self.sp_heur(core)]
            self._h = [(int(h[k][0]), int(h[k][2])) if u.active else (0, 0) for k, u in enumerate(core.uavs)]
            self._hkey = key
        return self._h

    def action_masks(self):
        m = super().action_masks()
        if self.phase != "slot":
            return m
        n_mv, _, n_pw = self.slot_nvec
        stride = sum(self.slot_nvec)
        for k, (mv, pw) in enumerate(self.heur_slot()):
            o = k * stride
            assert m[o + mv], "heuristic move not allowed by the A5 mask"
            m[o:o + n_mv] = False
            m[o + mv] = True
            o2 = o + n_mv + self.slot_nvec[1]
            assert m[o2 + pw]
            m[o2:o2 + n_pw] = False
            m[o2 + pw] = True
        return m

    def step(self, action):
        a = np.array(action, dtype=np.int64)
        if self.phase == "slot":
            for k, (mv, pw) in enumerate(self.heur_slot()):
                a[3 * k] = mv
                a[3 * k + 2] = pw
        return super().step(a)
