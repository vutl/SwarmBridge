"""Tests for the two-timescale env (env_ev.py, A2 v2.2). Run: python -m pytest -q test_env_ev.py"""
import math

import numpy as np
import pytest

import env as E
import env_ev as V
import policies as P
from test_env import make_table


def ev_env(tmp_path, **cfg):
    return V.UavSemcomEnvEv(make_table(tmp_path), range(8), dict(dict(probe_mode="m0"), **cfg))


def test_mode_decided_exactly_when_probe_finishes(tmp_path):
    env = ev_env(tmp_path, K=1, M=1, mode_info="predicted")
    V.run_episode_ev(env, P.greedy(), V.greedy_mode, 0)
    assert env.decisions
    probe = env.probe_s()
    by = {(tk.wp, tk.tile): tk for tk in env.tasks}
    for t, k, wp, tile, m in env.decisions:
        tk = by[(wp, tile)]
        assert t == pytest.approx(tk.info_t)                 # decided at the info instant, not at the next slot
        assert tk.info_t == pytest.approx(tk.t_cap + (tile + 1) * probe)
        assert tk.enc_start >= t - 1e-12


def test_no_information_before_the_probe(tmp_path):
    """At every decision instant the policy can only see per-mode info of images whose probe is done."""
    env = ev_env(tmp_path, K=2, M=2, mode_info="predicted")
    seen = []

    def spy(env, k, task):
        assert task.info_t <= env.t + 1e-9
        for u in env.uavs:
            for tk in env.awaiting(u):
                assert tk.info_t <= env.t + 1e-9
        o = env.obs()
        seen.append(o)
        return V.greedy_mode(env, k, task)
    V.run_episode_ev(env, P.greedy(), spy, 1)
    assert seen


def test_conservation_and_deadlines(tmp_path):
    for info in ("oracle", "oracle_probe", "predicted", "mean"):
        env = ev_env(tmp_path, mode_info=info, B_total=3e5)
        V.run_episode_ev(env, P.greedy(), V.greedy_mode, 4)
        for tk in env.finished:
            if not tk.dropped:
                assert tk.sent_bits == pytest.approx(tk.bits, rel=1e-9)
                assert tk.tx_start >= tk.ready - 1e-9 >= tk.enc_start - 1e-9
                assert tk.delay <= tk.dmax + 1e-9
        for k in range(len(env.uavs)):
            seg = sorted((g[2], g[3]) for g in env.segments if g[0] == k)
            for (s1, e1), (s2, e2) in zip(seg, seg[1:]):
                assert s2 >= e1 - 1e-9
        assert env.e_tx == pytest.approx(sum(env.c["p_levels"][-1] * (g[3] - g[2]) for g in env.segments))
        assert len(env.finished) == len(env.tasks)


def test_encoder_is_serial(tmp_path):
    env = ev_env(tmp_path, K=1, M=1, mode_info="predicted")
    V.run_episode_ev(env, P.greedy(), V.greedy_mode, 2)
    probe = env.probe_s()
    jobs = []
    for tk in env.tasks:
        jobs.append((tk.info_t - probe, tk.info_t))          # probe interval
        if tk.mode >= 0:
            jobs.append((tk.enc_start, tk.ready))             # mode-encode interval
    jobs.sort()
    for (s1, e1), (s2, e2) in zip(jobs, jobs[1:]):
        assert s2 >= e1 - 1e-9


@pytest.mark.parametrize("cfg", [dict(), dict(w_time=0.3), dict(M=1, K=1)])
def test_oracle_matches_slotted_v21(tmp_path, cfg):
    """With oracle info every image is decidable at the slot start, so the event-driven env with the event-greedy
    mode policy must reproduce env v2.1 with the slotted greedy exactly."""
    tab = make_table(tmp_path)
    a = V.run_episode_ev(V.UavSemcomEnvEv(tab, range(8), cfg), P.greedy(), V.greedy_mode, 9)
    b = P.run_episode(E.UavSemcomEnv(tab, range(8), cfg), P.greedy(), 9)
    for key in ("return", "mean_util", "ok_rate", "mean_delay", "energy_kj", "mission_s"):
        assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-9), key


def test_event_decisions_cut_the_slot_wait(tmp_path):
    """predicted, same fixed mode for every image, one UAV: deciding at the probe instant instead of the next slot
    can only make each image's encode start, encode end and delivery earlier (never later), and strictly earlier
    for the first image of a waypoint."""
    tab = make_table(tmp_path)
    cfg = dict(mode_info="predicted", probe_mode="m0", K=1, M=1)
    ea = V.UavSemcomEnvEv(tab, range(8), cfg)
    V.run_episode_ev(ea, P.fixed(1), V.fixed_mode(1), 5)
    eb = E.UavSemcomEnv(tab, range(8), cfg)
    P.run_episode(eb, P.fixed(1), 5)
    A = {(t.wp, t.tile): t for t in ea.tasks}
    B = {(t.wp, t.tile): t for t in eb.tasks}
    assert A.keys() == B.keys()
    for key in A:
        assert A[key].enc_start <= B[key].enc_start + 1e-9
        if not A[key].dropped and not B[key].dropped:
            assert A[key].delay <= B[key].delay + 1e-9
    assert all(A[(w, 0)].enc_start < B[(w, 0)].enc_start - 0.5 for w in {k[0] for k in A})


def test_oracle_probe_is_exact_but_delayed(tmp_path):
    env = ev_env(tmp_path, K=1, M=1, mode_info="oracle_probe")
    tk = E.Task(0, 0, 0, img=3, t_cap=0, dmax=5, qmin=25)
    assert np.array_equal(env.mode_info(tk)[0], env.tab.bits[3]) and env.probe_s() > 0


# ---------------- RL interface (gym_env_ev) ----------------
def test_gym_two_timescale_steps_and_masks(tmp_path):
    import gym_env_ev as GE
    g = GE.GymUavSemcomEv(make_table(tmp_path), range(8), dict(mode_info="predicted", probe_mode="m0"), seed=0)
    o, _ = g.reset()
    n_mode, n_slot, done, tot = 0, 0, False, 0.0
    rng = np.random.default_rng(0)
    while not done:
        m = g.action_masks()
        if g.phase == "mode":
            k, tk = g.pending
            assert tk.info_t <= g.core.t + 1e-9 and o[-(1 + 3 + 4 + 2 * g.n_modes)] == 1.0
            n_mode += 1
        else:
            n_slot += 1
        a, i = [], 0
        for j, nv in enumerate(g.action_space.nvec):
            ok = np.flatnonzero(m[i:i + nv])
            a.append(9 if (j % 3 == 0 and j < 3 * g.K and 9 in ok) else rng.choice(ok))   # fly the mission
            i += nv
        o, r, done, _, info = g.step(np.array(a))
        tot += r
    assert n_mode == len(g.core.tasks) > 0 and n_slot >= 30
    assert info["captured"] > 0


def test_maskable_ppo_two_timescale_consumes_masks(tmp_path):
    sb3c = pytest.importorskip("sb3_contrib")
    import gym_env_ev as GE

    class Watch(GE.GymUavSemcomEv):
        violations = 0

        def step(self, action):
            m = self.action_masks()
            off = np.concatenate([[0], np.cumsum(self.action_space.nvec)[:-1]])
            if not all(m[o + a] for o, a in zip(off, np.asarray(action))):
                Watch.violations += 1
            return super().step(action)
    env = Watch(make_table(tmp_path), range(8), dict(mode_info="predicted", probe_mode="m0"), seed=0)
    sb3c.MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=1, seed=0, device="cpu").learn(512)
    assert Watch.violations == 0
