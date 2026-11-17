"""Tests for A-X2 (review #37): exact state-dependent heuristic movement / power enforcement in GymSchedEv."""
import numpy as np
import pytest

import a5_common as C
import a5_eval as AE
import ax2_env as X
import env_ev as V
import env_survey as SV
import mode_heur as MH


@pytest.fixture(scope="module", autouse=True)
def init():
    AE.init(False)


def make(regime, seed, learn_mode):
    sp, mp = MH.from_name(AE.tuned(regime, "best"), C.MODES)
    e = X.GymSchedEv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime), seed=seed, seeds=[seed],
                     core_cls=SV.UavSurveyEnv, sp_heur=sp, mp_heur=mp, learn_mode=learn_mode)
    return e, sp, mp


def run(e, chooser, spy=None):
    """chooser(e, mask) -> action; spy(e, slot_actions) records what the env executes at slot steps."""
    if spy is not None:
        orig = e.core.begin_slot

        def wrapped(acts):
            spy(e.core, acts)
            return orig(acts)
        e.core.begin_slot = wrapped
    obs, _ = e.reset()
    done, tot = False, 0.0
    while not done:
        a = chooser(e, e.action_masks())
        obs, r, done, _, info = e.step(a)
        tot += r
    return tot, info


def random_chooser(seed):
    rng = np.random.default_rng(seed)

    def f(e, mask):
        a, o = [], 0
        for n in e.action_space.nvec:
            ok = np.flatnonzero(mask[o:o + n])
            a.append(int(rng.choice(ok)))
            o += n
        return np.array(a)
    return f


@pytest.mark.parametrize("regime,learn_mode", [("K5", False), ("K5", True), ("ref", True)])
def test_masks_and_executed_move_power_follow_current_heuristic(regime, learn_mode):
    e, sp, _ = make(regime, 8800, learn_mode)
    stride = sum(e.slot_nvec)
    seen = {"slots": 0}

    def spy(env, acts):
        h = sp(env)                                  # heuristic on the state at THIS slot start
        for k, u in enumerate(env.uavs):
            if u.active:
                assert acts[k][0] == h[k][0] and acts[k][2] == h[k][2]
        seen["slots"] += 1

    def chooser(env, mask):
        if env.phase == "slot":
            h = sp(env.core)
            for k, u in enumerate(env.core.uavs):
                mv = mask[k * stride:k * stride + 11]
                pw = mask[k * stride + 11 + env.slot_nvec[1]:(k + 1) * stride]
                if u.active:
                    assert mv.sum() == 1 and mv[h[k][0]] and pw.sum() == 1 and pw[h[k][2]]
                else:
                    assert mv.sum() == 1 and mv[0] and pw.sum() == 1 and pw[0]
        else:
            assert learn_mode                        # C-only never exposes a mode step
        return random_chooser(hash((env.core.t, env.phase)) % 2 ** 31)(env, mask)
    run(e, chooser, spy)
    assert seen["slots"] > 20


@pytest.mark.parametrize("learn_mode", [False, True])
def test_policy_cannot_influence_move_power(learn_mode):
    """same channel / mode choices, arbitrary move / power entries in the action -> identical episode."""
    def chooser_factory(garbage):
        rng = np.random.default_rng(5)

        def f(e, mask):
            a = random_chooser(int(rng.integers(1 << 30)))(e, mask)
            if garbage and e.phase == "slot":
                for k in range(e.K):
                    a[3 * k] = 10 - a[3 * k]
                    a[3 * k + 2] = (a[3 * k + 2] + 1) % e.slot_nvec[2]
            return a
        return f
    r1, _ = run(make("K5", 8801, learn_mode)[0], chooser_factory(False))
    r2, _ = run(make("K5", 8801, learn_mode)[0], chooser_factory(True))
    assert abs(r1 - r2) < 1e-9


@pytest.mark.parametrize("regime", ["K5", "ref"])
@pytest.mark.parametrize("learn_mode", [False, True])
def test_tbest_channels_reproduce_tbest(regime, learn_mode):
    """choosing T:best's own channels (and, for C+Mo, T:best's modes) reproduces the T:best episode exactly."""
    e, sp, mp = make(regime, 8802, learn_mode)

    def chooser(env, mask):
        a = np.zeros(len(env.action_space.nvec), dtype=np.int64)
        if env.phase == "slot":
            for k, x in enumerate(sp(env.core)):
                a[3 * k:3 * k + 3] = x[:3]
        else:
            a[-1] = mp(env.core, *env.pending)
        return a
    r, _ = run(e, chooser)
    ref = V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime)), sp, mp, 8802)
    assert abs(r - ref["return"]) < 1e-6


def test_heuristic_is_pure():
    e, sp, _ = make("K5", 8803, False)
    e.reset()
    a1 = [tuple(x[:3]) for x in sp(e.core)]
    a2 = [tuple(x[:3]) for x in sp(e.core)]
    assert a1 == a2


def test_eval_harness_matches_a5_and_pilot():
    """ax2_eval episodes (GymSchedEv, M/P enforced) reproduce T:best, the A5 mode-only path and the A-X1 jc|jm."""
    import ax1_pilot as XP
    import ax2_eval as XE
    sp, mp = MH.from_name(AE.tuned("K5", "best"), C.MODES)
    m, _ = XE.episode("K5", "K5", "tbest|heur", -1, 8803)
    r = V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg("K5")), sp, mp, 8803)
    assert abs(m["return"] - r["return"]) < 1e-6
    for combo, i in (("tbest|ppo", 0), ("jc|jm", 2)):
        m, _ = XE.episode("K5", "K5", combo, i, 8804)
        p, _ = XP.episode("K5", combo, i, 8804)
        assert abs(m["return"] - p["return"]) < 1e-6, combo
    assert m["service"]["n"] > 0 and m["service"]["ratio"] > 0
