"""Tests for the A4 survey workload (env_survey.py). Run: python -m pytest -q test_env_survey.py"""
import numpy as np
import pytest

import env_ev as V
import env_survey as S
import policies as P
from test_env import make_table


def run(env, seed, mode_pol=V.greedy_mode, slot_pol=None):
    return V.run_episode_ev(env, slot_pol or P.greedy(), mode_pol, seed)


def test_geometry_and_capture_instants(tmp_path):
    env = S.UavSurveyEnv(make_table(tmp_path), range(8), dict(K=1, M=1, B_total=1e8))
    m = run(env, 0)
    W = len(env.uavs[0].wps)
    assert W == env.c["survey_lines"] * (int(env.c["line_len"] / env.c["frame_spacing"]) + 1)
    assert m["captured"] == 1.0 and len(env.tasks) == W
    tc = np.array(sorted(t.t_cap for t in env.tasks))
    on_line = np.diff(tc)[:int(env.c["line_len"] / env.c["frame_spacing"])]
    assert np.allclose(on_line, env.c["frame_spacing"] / env.c["v_max"])     # one frame every spacing / v
    assert not np.allclose(tc, np.round(tc))                                   # captures fall inside slots


def test_several_captures_in_one_slot(tmp_path):
    env = S.UavSurveyEnv(make_table(tmp_path), range(8), dict(K=1, M=1, v_max=20.0, frame_spacing=8.0,
                                                               B_total=1e8))
    run(env, 1)
    slots = np.floor([t.t_cap - 1e-9 for t in env.tasks])
    assert np.max(np.unique(slots, return_counts=True)[1]) >= 2


@pytest.mark.parametrize("info", ["oracle", "oracle_probe", "predicted", "mean"])
def test_survey_conservation_and_causality(tmp_path, info):
    env = S.UavSurveyEnv(make_table(tmp_path), range(8), dict(mode_info=info, probe_mode="m0", B_total=4e5))
    run(env, 3)
    for tk in env.finished:
        if not tk.dropped:
            assert tk.sent_bits == pytest.approx(tk.bits, rel=1e-9)
            assert tk.tx_start >= tk.ready - 1e-9 >= tk.enc_start - 1e-9 >= tk.info_t - 1e-9 >= tk.t_cap - 1e-9
            assert tk.delay <= tk.dmax + 1e-9
    for t, k, wp, tile, mo in env.decisions:
        tk = [x for x in env.tasks if x.uav == k and x.wp == wp][0]
        assert t == pytest.approx(tk.info_t)
    for k in range(len(env.uavs)):
        seg = sorted((g[2], g[3]) for g in env.segments if g[0] == k)
        for (s1, e1), (s2, e2) in zip(seg, seg[1:]):
            assert s2 >= e1 - 1e-9
    assert len(env.finished) == len(env.tasks)


def test_hover_delays_the_survey(tmp_path):
    tab = make_table(tmp_path)
    a = run(S.UavSurveyEnv(tab, range(8), dict(B_total=4e5)), 2, slot_pol=P.greedy())
    b = run(S.UavSurveyEnv(tab, range(8), dict(B_total=4e5)), 2, slot_pol=P.paced(0))
    assert b["mission_s"] >= a["mission_s"]
