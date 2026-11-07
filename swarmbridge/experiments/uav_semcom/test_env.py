"""Unit tests for the A2 v2 environment. Each test targets one modelling error found in v1 (report section 3.2) or one
accounting invariant. Run: python -m pytest -q test_env.py"""
import json
import math

import numpy as np
import pytest

import env as E
import policies as P


def make_table(tmp_path, n_img=8):
    rows = []
    for i in range(n_img):
        for j, (b, q, enc) in enumerate([(2000, 22.0, 0.01), (20000, 28.0, 0.05), (80000, 34.0, 0.2),
                                         (300000, 40.0, 0.3)]):
            rows.append(dict(image=f"im{i}", mode=f"m{j}", bytes=b * (1 + 0.1 * i), psnr=q + 0.2 * i, lpips=0.1,
                             enc_s=enc, dec_s=0.02))
    p = tmp_path / "profile.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    tab = E.ModeTable([p])
    tab.split(probe_mode="m0")
    return tab


def run(env, pol, seed):
    env.reset(seed)
    done = False
    while not done:
        _, _, done = env.step(pol(env))
    return env


# ---------------- physics / utility (kept from v1) ----------------
def test_propulsion_hover_and_min():
    assert abs(E.propulsion_power(0.0) - (79.86 + 88.63)) < 1e-6
    assert E.propulsion_power(10.0) < E.propulsion_power(0.0) < E.propulsion_power(30.0)


def test_los_gain_decreases_with_distance(tmp_path):
    env = E.UavSemcomEnv(make_table(tmp_path), range(8))
    assert env.expected_gain(np.array([100.0, 0])) > env.expected_gain(np.array([1000.0, 0]))


def test_utility_penalty():
    class T:
        qmin, dmax = 25.0, 5.0
    env = E.UavSemcomEnv.__new__(E.UavSemcomEnv)
    env.c = dict(E.DEFAULT)
    u_ok, ok = env.utility(T, 30.0, 1.0)
    u_bad, bad = env.utility(T, 20.0, 1.0)
    assert ok and not bad and u_ok > 0 > u_bad


def test_deterministic(tmp_path):
    tab = make_table(tmp_path)
    a = P.run_episode(E.UavSemcomEnv(tab, range(8)), P.greedy(), 3)
    b = P.run_episode(E.UavSemcomEnv(tab, range(8)), P.greedy(), 3)
    assert a == b and a["captured"] == 1.0


# ---------------- v1 error 1: one image per UAV per slot ----------------
def test_several_images_per_slot(tmp_path):
    """With a fast link and a small mode, one UAV must deliver more than one image inside a single slot."""
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(B_total=1e8, K=1, M=1))
    run(env, P.fixed(0), 1)
    per_slot = {}
    for tk in env.finished:
        if not tk.dropped:
            per_slot.setdefault(math.floor(tk.tx_start), []).append(tk)
    assert max(len(v) for v in per_slot.values()) >= 2


# ---------------- v1 error 2: one mode decision -> one image ----------------
def test_mode_vector_applies_per_image(tmp_path):
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(K=1, M=1))
    env.reset(0)
    while not env.uavs[0].queue:                       # fly to the first waypoint
        env.step([(9, 0, 2, (0, 0, 0, 0))])
    pend = [tk for tk in env.uavs[0].queue if tk.mode < 0]
    assert len(pend) == env.L
    env.step([(0, 0, 2, (3, 1, 2, 0))])
    assert [tk.mode for tk in pend] == [3, 1, 2, 0]
    # encoder is serial: each image starts encoding when the previous one is done
    for a, b in zip(pend, pend[1:]):
        assert b.enc_start == pytest.approx(a.ready)


def test_mode_vector_length_enforced(tmp_path):
    env = E.UavSemcomEnv(make_table(tmp_path), range(8), dict(K=1, M=1))
    with pytest.raises(AssertionError):
        env.step([(0, 0, 2, (1,))])


# ---------------- accounting invariants: bits, time, energy ----------------
@pytest.mark.parametrize("cfg", [dict(), dict(B_total=2e5), dict(M=1)])
def test_bits_time_energy_conservation(tmp_path, cfg):
    tab = make_table(tmp_path)
    env = run(E.UavSemcomEnv(tab, range(8), cfg), P.greedy(), 5)
    for tk in env.finished:
        if not tk.dropped:
            assert tk.sent_bits == pytest.approx(tk.bits, rel=1e-9)
            assert tk.tx_start >= tk.ready - 1e-9 and tk.ready >= tk.enc_start
            assert tk.delay <= tk.dmax + 1e-9                  # hard expiry: nothing is delivered late
        else:
            assert tk.sent_bits <= tk.bits + 1e-6
    # per UAV: transmission intervals never overlap; per channel-slot nothing exceeds the slot
    for k in range(len(env.uavs)):
        seg = sorted((g[2], g[3]) for g in env.segments if g[0] == k)
        for (s1, e1), (s2, e2) in zip(seg, seg[1:]):
            assert s2 >= e1 - 1e-9
        for s, e in seg:
            assert math.floor(s + 1e-12) == math.floor(e - 1e-12) or e == pytest.approx(math.ceil(s))
    # transmit energy = sum over segments of power x airtime
    assert env.e_tx == pytest.approx(sum(env.c["p_levels"][-1] * (g[3] - g[2]) for g in env.segments))
    # every captured image is accounted exactly once
    assert len(env.finished) == len(env.tasks)


def test_no_energy_after_sortie_end(tmp_path):
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8))
    env.reset(2)
    pol, done, frozen = P.greedy(), False, {}
    while not done:
        _, _, done = env.step(pol(env))
        for k, u in enumerate(env.uavs):
            if not u.active:
                frozen.setdefault(k, u.energy)
                assert u.energy == frozen[k]
    assert frozen   # at least one sortie ended before the horizon


def test_deadline_expiry_spends_no_airtime(tmp_path):
    """Images whose deadline has passed are discarded; no transmission segment of an image extends past its
    latest on-time end."""
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(B_total=5e4, deadline=(1.0, 2.0)))
    env = run(env, P.fixed(3), 4)
    dropped = [tk for tk in env.finished if tk.dropped]
    assert dropped
    for tk in env.tasks:
        if tk.mode >= 0 and not math.isnan(tk.tx_start):
            assert tk.tx_start + tk.airtime <= env.latest_tx_end(tk) + 1e-9


# ---------------- v1 error 3 / v2.1: interference consistency and timing ----------------
def _two_uav_slot(tmp_path, ready1, frac0, frac1):
    """Two UAVs on the same channel, one pre-encoded image each (mode 0): UAV 0 ready at 0, UAV 1 ready at ready1;
    payloads = frac x (bits UAV k could send in 1 s alone). Runs one slot; returns env and the stand-alone rates."""
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(K=2, M=1))
    env.reset(7)
    g = [env.slot_gain(k) for k in range(2)]
    c = env.c
    r_alone = [c["B_total"] / c["M"] * math.log2(1 + c["p_levels"][2] * g[k] / env.noise_w()) for k in range(2)]
    for k, (rdy, fr) in enumerate([(0.0, frac0), (ready1, frac1)]):
        b = fr * r_alone[k]
        env.uavs[k].queue = [E.Task(uav=k, wp=0, tile=0, img=0, t_cap=0.0, dmax=50.0, qmin=0.0, mode=0,
                                    enc_start=0.0, ready=rdy, bits=b, rem_bits=b)]
    env.step([(0, 1, 2, (0,) * env.L), (0, 1, 2, (0,) * env.L)])
    return env, g, r_alone


def test_interference_non_overlapping(tmp_path):
    """UAV 0 sends 0-0.2 s, UAV 1 sends from 0.6 s: no overlap -> neither is interfered (v2 would interfere both
    for the whole slot)."""
    env, g, r = _two_uav_slot(tmp_path, ready1=0.6, frac0=0.2, frac1=0.2)
    segs = env.segments
    assert {s[0] for s in segs} == {0, 1}
    assert all(s[4] == 0.0 for s in segs)
    t0 = [tk for tk in env.finished if tk.uav == 0][0]
    t1 = [tk for tk in env.finished if tk.uav == 1][0]
    assert t0.tx_start + t0.airtime == pytest.approx(0.2)
    assert t1.tx_start == pytest.approx(0.6) and t1.airtime == pytest.approx(0.2)


def test_interference_overlapping(tmp_path):
    """Both ready at 0: while both send, each sees exactly the other's p x gain (same gain draw as the other's own
    link); once one finishes, the other continues interference-free."""
    env, g, r = _two_uav_slot(tmp_path, ready1=0.0, frac0=0.3, frac1=0.6)
    p = env.c["p_levels"][2]
    both = [s for s in env.segments if s[2] == 0.0]
    assert len(both) == 2
    for s in both:
        assert s[4] == pytest.approx(p * g[1 - s[0]])
        assert s[5] < r[s[0]]
    later = [s for s in env.segments if s[2] > 0.0]
    assert later and all(s[4] == 0.0 for s in later)
    # segments of one UAV are contiguous and the bits add up
    for tk in env.finished:
        assert tk.sent_bits == pytest.approx(tk.bits)


# ---------------- v1 error 4: common random numbers across policies ----------------
def test_scenario_independent_of_policy(tmp_path):
    tab = make_table(tmp_path)
    e1 = run(E.UavSemcomEnv(tab, range(8)), P.greedy(), 11)
    e2 = run(E.UavSemcomEnv(tab, range(8)), P.paced(0), 11)
    key = lambda tk: (tk.uav, tk.wp, tk.tile)
    t1 = {key(t): (t.img, t.dmax, t.qmin) for t in e1.tasks}
    t2 = {key(t): (t.img, t.dmax, t.qmin) for t in e2.tasks}
    assert t1 == t2
    assert np.array_equal(e1.f_u, e2.f_u) and np.array_equal(e1.f_e, e2.f_e)


# ---------------- v1 error 5: information budget ----------------
def test_observation_contains_time_and_remaining_path(tmp_path):
    env = E.UavSemcomEnv(make_table(tmp_path), range(8))
    o0 = env.reset(0)
    env.step([(9, 0, 2, (0,) * env.L)] * env.c["K"])
    o1 = env.obs()
    assert o0[0] == 0.0 and o1[0] == pytest.approx(env.c["dt"] / env.c["T"])
    assert o1[1 + 6] < o0[1 + 6]          # UAV 0 remaining path shrinks when flying the mission


@pytest.mark.parametrize("info", ["oracle", "predicted", "mean"])
def test_mode_info_levels(tmp_path, info):
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(mode_info=info, probe_mode="m0"))
    t_a = E.Task(0, 0, 0, img=1, t_cap=0, dmax=5, qmin=25)
    t_b = E.Task(0, 0, 1, img=6, t_cap=0, dmax=5, qmin=25)
    ba, qa = env.mode_info(t_a)
    bb, qb = env.mode_info(t_b)
    if info == "mean":
        assert np.array_equal(ba, bb) and np.array_equal(qa, qb)
    else:
        assert not np.array_equal(ba, bb)
    if info == "oracle":
        assert np.array_equal(ba, tab.bits[1])
    if info == "predicted":
        assert ba[tab.probe_j] == tab.bits[1, tab.probe_j]   # probe mode known exactly
        assert env.probe_s() == tab.enc_s[tab.probe_j]       # and its encode time is charged


def test_action_masks_shape(tmp_path):
    import gym_env as G
    g = G.GymUavSemcom(make_table(tmp_path), range(8), seed=0)
    g.reset()
    m = g.action_masks()
    assert m.shape == (int(sum(g.action_space.nvec)),) and m.dtype == bool


def test_maskable_ppo_consumes_masks(tmp_path):
    """The masks are not just defined: MaskablePPO must never sample a masked sub-action during learning."""
    sb3c = pytest.importorskip("sb3_contrib")
    import gym_env as G

    class Watch(G.GymUavSemcom):
        violations = 0

        def step(self, action):
            m = self.action_masks()
            off = np.concatenate([[0], np.cumsum(self.action_space.nvec)[:-1]])
            if not all(m[o + a] for o, a in zip(off, np.asarray(action))):
                Watch.violations += 1
            return super().step(action)
    env = Watch(make_table(tmp_path), range(8), seed=0)
    model = sb3c.MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(512)
    assert Watch.violations == 0


# ---------------- v2.1: causal 'predicted' information ----------------
def test_predicted_info_is_causal(tmp_path):
    """capture -> probe (serial on the encoder) -> info available -> mode decision at the next epoch -> encode."""
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(K=1, M=1, mode_info="predicted", probe_mode="m0"))
    env.reset(0)
    while not env.uavs[0].queue:
        env.step([(9, 0, 2, (0,) * env.L)])
    u, probe = env.uavs[0], env.probe_s()
    caps = list(u.queue)
    t_cap = caps[0].t_cap
    assert probe > 0
    for i, tk in enumerate(caps):                      # probes run back to back right after capture
        assert tk.info_t == pytest.approx(t_cap + (i + 1) * probe)
    assert env.awaiting(u) == []                        # nothing decidable at the capture instant
    o = env.obs()
    blk = 1 + 11 + env.c["q_obs"] * 6                   # start of UAV 0's pending-decision block
    assert np.all(o[blk:blk + env.L * (3 + 2 * env.n_modes)] == 0)   # no predicted info leaks into the observation
    env.step([(0, 0, 2, (1,) * env.L)])                 # this epoch: modes cannot be assigned yet
    assert all(tk.mode < 0 for tk in caps)
    env.step([(0, 0, 2, (1, 2, 3, 1))])                 # next epoch: probes done -> decided
    assert [tk.mode for tk in caps] == [1, 2, 3, 1]
    for tk in caps:
        assert tk.enc_start >= tk.info_t - 1e-12 and tk.ready == pytest.approx(tk.enc_start + tab.enc_s[tk.mode])


def test_oracle_info_available_at_capture(tmp_path):
    tab = make_table(tmp_path)
    env = E.UavSemcomEnv(tab, range(8), dict(K=1, M=1))
    env.reset(0)
    while not env.uavs[0].queue:
        env.step([(9, 0, 2, (0,) * env.L)])
    assert len(env.awaiting(env.uavs[0])) == env.L


# ---------------- v2.1 regression against the frozen v2 ----------------
@pytest.mark.parametrize("cfg", [dict(), dict(interference=False), dict(M=1, K=1), dict(w_time=0.3)])
def test_regression_vs_v2_without_cochannel(tmp_path, cfg):
    """With oracle info and no co-channel transmission (the heuristics give each UAV its own channel), v2.1 must
    reproduce v2 exactly: same deliveries, delays, energy and return."""
    import legacy_env_v2 as V2
    tab = make_table(tmp_path)
    for m in (1, 3):
        a = P.run_episode(E.UavSemcomEnv(tab, range(8), cfg), P.fixed(m), 9)
        b = P.run_episode(V2.UavSemcomEnv(tab, range(8), cfg), P.fixed(m), 9)
        for key in ("return", "mean_util", "ok_rate", "drop_rate", "mean_delay", "energy_kj", "mission_s"):
            assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-9), key

