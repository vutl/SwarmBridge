"""Tests for the time-consistent (semi-MDP) GAE of timeppo.py. Run: python -m pytest -q test_timeppo.py"""
import numpy as np
import pytest
import torch

sb3c = pytest.importorskip("sb3_contrib")
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

import env_ev as V
import env_survey as SV
import gym_env_ev as GE
import policies as P
import timeppo as TP
from test_env import make_table

OBS, ACT = spaces.Box(-1, 1, (2,)), spaces.MultiDiscrete([3, 2])
GAMMA, LAM = 0.99, 0.95


def fill(buf, rewards, values, starts, dts=None):
    """Fill a buffer with given per-step arrays of shape (T, n_envs)."""
    T, n = rewards.shape
    for t in range(T):
        if dts is not None:
            buf.next_dt = dts[t]
        buf.add(np.zeros((n, 2), np.float32), np.zeros((n, 2)), rewards[t], starts[t],
                torch.as_tensor(values[t]), torch.zeros(n), action_masks=np.ones((n, 5)))


def random_rollout(T=40, n=3, seed=0, p_end=0.1):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(T, n)).astype(np.float32)
    v = rng.normal(size=(T, n)).astype(np.float32)
    starts = (rng.random((T, n)) < p_end).astype(np.float32)
    last_v, last_done = rng.normal(size=n).astype(np.float32), (rng.random(n) < 0.5)
    return r, v, starts, last_v, last_done


def time_gae(r, v, starts, last_v, last_done, dts, gamma=GAMMA, lam=LAM):
    """Plain-python reference: per env, backwards, gamma^dt and (gamma*lam)^dt, no bootstrap across episode ends."""
    T, n = r.shape
    adv = np.zeros((T, n))
    for e in range(n):
        a = 0.0
        for t in reversed(range(T)):
            nnt = (1.0 - float(last_done[e])) if t == T - 1 else 1.0 - starts[t + 1, e]
            nv = last_v[e] if t == T - 1 else v[t + 1, e]
            d = r[t, e] + gamma ** dts[t, e] * nv * nnt - v[t, e]
            a = d + (gamma * lam) ** dts[t, e] * nnt * a
            adv[t, e] = a
    return adv


def time_buffer(T, n):
    return TP.TimeRolloutBuffer(T, OBS, ACT, "cpu", gae_lambda=LAM, gamma=GAMMA, n_envs=n)


def test_dt_one_second_is_exactly_stock_gae():
    r, v, s, lv, ld = random_rollout()
    T, n = r.shape
    stock = MaskableRolloutBuffer(T, OBS, ACT, "cpu", gae_lambda=LAM, gamma=GAMMA, n_envs=n)
    fill(stock, r, v, s)
    stock.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    tb = time_buffer(T, n)
    fill(tb, r, v, s, np.ones((T, n), np.float32))
    tb.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    assert np.array_equal(tb.advantages, stock.advantages)          # bit-identical, not just close
    assert np.array_equal(tb.returns, stock.returns)


def test_matches_reference_with_terminations_and_per_env_dt():
    r, v, s, lv, ld = random_rollout(T=60, n=4, seed=1, p_end=0.15)
    dts = np.random.default_rng(2).choice([0.0, 0.0, 0.07, 0.4, 1.0], size=r.shape).astype(np.float32)
    tb = time_buffer(*r.shape)
    fill(tb, r, v, s, dts)
    tb.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    assert np.allclose(tb.advantages, time_gae(r, v, s, lv, ld, dts), atol=1e-5)
    # vectorized == each env alone (no mixing of env columns)
    for e in range(r.shape[1]):
        one = time_buffer(r.shape[0], 1)
        sl = slice(e, e + 1)
        fill(one, r[:, sl], v[:, sl], s[:, sl], dts[:, sl])
        one.compute_returns_and_advantage(torch.as_tensor(lv[sl]), ld[sl])
        assert np.allclose(one.advantages[:, 0], tb.advantages[:, e], atol=1e-6)


def test_episode_end_cuts_bootstrap():
    """A terminal step's advantage is r - V(s) whatever comes after it (next episode / last value)."""
    T, n = 6, 1
    r = np.arange(1, T + 1, dtype=np.float32)[:, None]
    v = np.full((T, n), 0.5, np.float32)
    s = np.zeros((T, n), np.float32)
    s[3] = 1.0                                       # step 2 ended an episode; step 3 starts a new one
    dts = np.full((T, n), 0.3, np.float32)
    tb = time_buffer(T, n)
    fill(tb, r, v, s, dts)
    tb.compute_returns_and_advantage(torch.as_tensor(np.array([100.0], np.float32)), np.array([True]))
    assert tb.advantages[2, 0] == pytest.approx(r[2, 0] - v[2, 0])
    assert tb.advantages[T - 1, 0] == pytest.approx(r[T - 1, 0] - v[T - 1, 0])     # done at the rollout end too


@pytest.mark.parametrize("lam", [0.95, 1.0])
def test_simultaneous_mode_decisions_add_no_discount(lam):
    """Three decisions at the same instant (dt = 0, then 1 s) give the decision at that instant the same advantage
    as ONE decision collecting their rewards over 1 s: no extra discount, no extra lambda decay, whatever the
    intermediate values are."""
    rng = np.random.default_rng(3)
    r_split = rng.normal(size=(5, 1)).astype(np.float32)          # steps 0,1,2 at t=0 ; 3 at t=1 ; 4 at t=2
    v_split = rng.normal(size=(5, 1)).astype(np.float32)
    dts = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]], np.float32)
    r_merged = np.array([[r_split[:3].sum()], r_split[3], r_split[4]], np.float32)
    v_merged = np.array([v_split[0], v_split[3], v_split[4]], np.float32)
    lv, ld = np.array([0.7], np.float32), np.array([False])
    a = TP.TimeRolloutBuffer(5, OBS, ACT, "cpu", gae_lambda=lam, gamma=GAMMA, n_envs=1)
    fill(a, r_split, v_split, np.zeros((5, 1), np.float32), dts)
    a.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    b = TP.TimeRolloutBuffer(3, OBS, ACT, "cpu", gae_lambda=lam, gamma=GAMMA, n_envs=1)
    fill(b, r_merged, v_merged, np.zeros((3, 1), np.float32), np.ones((3, 1), np.float32))
    b.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    assert a.advantages[0, 0] == pytest.approx(b.advantages[0, 0], abs=1e-5)
    assert a.advantages[3, 0] == pytest.approx(b.advantages[1, 0], abs=1e-6)
    # per-step discounting (stock PPO) would NOT give this: it discounts the 3 simultaneous steps
    st = MaskableRolloutBuffer(5, OBS, ACT, "cpu", gae_lambda=lam, gamma=GAMMA, n_envs=1)
    fill(st, r_split, v_split, np.zeros((5, 1), np.float32))
    st.compute_returns_and_advantage(torch.as_tensor(lv), ld)
    assert abs(st.advantages[0, 0] - b.advantages[0, 0]) > 1e-4


def test_add_without_dt_fails():
    tb = time_buffer(4, 1)
    with pytest.raises(AssertionError):
        fill(tb, np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32))


def survey_gym(tmp_path, seed, **kw):
    return GE.GymUavSemcomEv(make_table(tmp_path), range(8), dict(mode_info="predicted", probe_mode="m0", K=2, M=1,
                                                                   v_max=20.0), seed=seed, core_cls=SV.UavSurveyEnv,
                             **kw)


@pytest.mark.parametrize("part", ["joint", "movement", "mode"])
def test_wrapper_dt_is_env_time(tmp_path, part):
    """info['dt'] = env time between consecutive RL decisions: sums to the episode time after the first RL decision;
    movement-only RL steps are exactly 1 s apart (so time discounting = stock PPO there); joint RL has dt = 0 steps."""
    kw = dict(movement=dict(mode_pol=V.greedy_mode), mode=dict(slot_pol=P.greedy()), joint={})[part]
    g = survey_gym(tmp_path, 0, **kw)
    g.reset(seed=5)
    t_first, done, dts, kinds = g.core.t, False, [], []
    rng = np.random.default_rng(0)
    while not done:
        m, a, i = g.action_masks(), [], 0
        for j, nv in enumerate(g.action_space.nvec):
            ok = np.flatnonzero(m[i:i + nv])
            a.append(9 if (j % 3 == 0 and j < 3 * g.K and 9 in ok) else rng.choice(ok))
            i += nv
        _, _, done, _, info = g.step(np.array(a))
        dts.append(info["dt"])
        kinds.append(info["kind"])
    dts = np.array(dts)
    assert np.all(dts >= 0) and dts.sum() == pytest.approx(info["ep_t"] - t_first)
    assert info["ep_rl_slot"] == kinds.count("slot") and info["ep_rl_mode"] == kinds.count("mode")
    assert info["ep_rl_mode"] + info["ep_heur_mode"] == len(g.core.tasks)
    if part == "movement":
        assert np.allclose(dts[:-1], 1.0) and info["ep_heur_slot"] == 0
    if part == "joint":
        assert (dts < 0.999).sum() > len(dts) / 4 and info["ep_heur_mode"] == info["ep_heur_slot"] == 0
    if part == "mode":
        assert info["ep_rl_slot"] == 0 and info["ep_heur_slot"] > 0


def test_ppo_buffer_receives_the_env_dts(tmp_path):
    """End to end: after a rollout of TimeMaskablePPO on 2 vectorized envs, buffer.dts holds exactly each env's own
    info['dt'] sequence, in order (episode ends and auto-resets included)."""
    class Rec(GE.GymUavSemcomEv):
        def step(self, action):
            out = super().step(action)
            self.rec.append(out[4]["dt"])
            return out

    def mk(s):
        def f():
            e = Rec(make_table(tmp_path), range(8), dict(mode_info="predicted", probe_mode="m0", K=2, M=1,
                                                         v_max=20.0, survey_lines=1, line_len=100, T=20),
                    seed=s, core_cls=SV.UavSurveyEnv)
            e.rec = []
            return e
        return f
    venv = DummyVecEnv([mk(0), mk(1)])
    n_steps = 300
    model = TP.TimeMaskablePPO("MlpPolicy", venv, n_steps=n_steps, batch_size=100, n_epochs=1, seed=0, device="cpu")
    model.learn(n_steps * 2)
    buf = model.rollout_buffer
    assert buf.episode_starts[1:].sum() > 0                        # at least one episode ended inside the rollout
    for e in range(2):
        assert np.allclose(buf.dts[:, e], venv.envs[e].rec[-n_steps:])
