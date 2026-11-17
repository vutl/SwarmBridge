"""Tests for the literature-style baselines (std_baselines.py)."""
import pytest

import a5_common as C
import a5_eval as AE
import env_ev as V
import env_survey as SV
import mode_heur as MH
import policies as P


@pytest.fixture(scope="module", autouse=True)
def init():
    AE.init(False)


def env(regime="K5"):
    return SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))


def busy(e):
    return [k for k, u in enumerate(e.uavs) if u.active and u.queue]


@pytest.mark.parametrize("name", ["std_rr", "std_maxrate", "std_rand"])
def test_orthogonal_channels_and_rules(name):
    sp, mp = MH.from_name(name, C.MODES)
    e = env()
    e.reset(8800)
    seen = 0
    while True:
        acts = sp(e)
        ch = [a[1] for a in acts]
        used = [c for c in ch if c > 0]
        b = busy(e)
        assert len(used) == len(set(used)) == min(len(b), e.c["M"])          # orthogonal, all channels used if needed
        assert all(ch[k] == 0 for k in range(len(e.uavs)) if k not in b)
        assert all(a[2] == len(e.c["p_levels"]) - 1 and a[0] == P.mission_move(e, k) for k, a in enumerate(acts))
        if name == "std_rr" and len(b) > e.c["M"]:
            order = [(e.slot + j) % len(e.uavs) for j in range(len(e.uavs))]
            assert sorted(k for k in b if ch[k]) == sorted([k for k in order if k in b][:e.c["M"]])
        if name == "std_maxrate" and len(b) > e.c["M"]:
            r = {k: P.expected_rate(e, k) for k in b}
            assert min(r[k] for k in b if ch[k]) >= max(r[k] for k in b if not ch[k])
        seen += len(b) > e.c["M"]
        e.begin_slot([a[:3] for a in acts])
        done = False
        while True:
            ev = e.advance()
            if ev[0] == "mode":
                m = mp(e, ev[1], ev[2])
                assert 0 <= m < e.n_modes
                e.decide(m)
            else:
                done = ev[1]
                break
        if done:
            break
    assert seen > 5


@pytest.mark.parametrize("name", ["std_rr", "std_maxrate", "std_rand"])
def test_reproducible_and_no_cochannel(name):
    sp, mp = MH.from_name(name, C.MODES)
    a = V.run_episode_ev(env(), sp, mp, 8801)
    sp, mp = MH.from_name(name, C.MODES)
    b = V.run_episode_ev(env(), sp, mp, 8801)
    assert a["return"] == b["return"]
    assert a.get("cochannel_air_frac", 0.0) == 0.0


def test_existing_names_unchanged():
    sp, _ = MH.from_name("greedy", C.MODES)
    assert sp.__name__ == "greedy"
