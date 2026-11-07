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


def test_budget_mode_heuristic(tmp_path):
    """mode_heur.budget_mode: with an unbounded budget it is exactly event-greedy (same choice at every decision); a tight
    budget never picks a larger predicted payload than greedy, and every image still gets exactly one mode."""
    import mode_heur as MH
    tab = make_table(tmp_path)
    cfg = dict(mode_info="predicted", probe_mode="m0", B_total=4e5)
    checked = []

    def spy_inf(env, k, task):
        m_b, m_g = MH.budget_mode(1e9)(env, k, task), V.greedy_mode(env, k, task)
        rate = max(P.expected_rate(env, k) * P.channel_share(env), 1.0)
        u, tab_ = env.uavs[k], env.tab
        tx_free = env.t
        for tk in u.queue:
            if tk is task:
                break
            if tk.mode >= 0:
                bb = tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits
                tx_free = max(tx_free, tk.ready) + max(bb, 0.0) / rate
        b_hat, q_hat = env.mode_info(task)

        def value(m):                        # greedy's own criterion (env_ev.greedy_mode / policies.greedy_select)
            end = max(tx_free, max(env.t, u.enc_free) + tab_.enc_s[m]) + b_hat[m] / rate
            if end > task.t_cap + task.dmax - tab_.dec_s[m]:
                return -1.0
            return env.utility(task, q_hat[m], end + tab_.dec_s[m] - task.t_cap)[0]
        checked.append(m_b == m_g)          # tau = inf: exactly event-greedy
        return m_g
    g = S.UavSurveyEnv(tab, range(8), cfg)
    run(g, 5, mode_pol=spy_inf)
    assert checked and all(checked)
    t = S.UavSurveyEnv(tab, range(8), cfg)
    calls = []

    def spy(env, k, task):
        m_t, m_g = MH.budget_mode(0.05)(env, k, task), V.greedy_mode(env, k, task)
        calls.append((env.mode_info(task)[0][m_t], env.mode_info(task)[0][m_g]))
        return m_t
    run(t, 5, mode_pol=spy)
    assert calls and all(bt <= bg + 1e-9 for bt, bg in calls)
    assert len(t.decisions) == len(t.tasks) == len({(d[1], d[2]) for d in t.decisions})
    sp, mp = MH.from_name("pacedB2+budget1.5", None)
    assert mp.__name__ == "budget1.5"


def test_channel_reuse_wrapper(tmp_path):
    """reuse(): keeps the base policy's assignments; every other active UAV with a queue gets a (shared) channel,
    never UAVs without a queue; reuse does happen."""
    import mode_heur as MH
    env = S.UavSurveyEnv(make_table(tmp_path), range(8), dict(K=3, M=2, v_max=20.0, mode_info="predicted"))
    base = P.from_name("pacedB1.5")
    wrapped = MH.reuse(base)
    n_shared = []

    def sp(e):
        b, w = base(e), wrapped(e)
        for k, u in enumerate(e.uavs):
            if b[k][1] > 0:
                assert w[k][:3] == b[k][:3]
            elif u.active and u.queue:
                assert w[k][1] > 0 and w[k][0] == b[k][0] and w[k][2] == b[k][2]
            else:
                assert w[k][1] == 0
        n_shared.append(sum(w[k][1] > 0 for k in range(3)) > 2)
        return w
    run(env, 0, slot_pol=sp)
    assert any(n_shared)
    sp2, mp2 = MH.from_name("pacedB1.5+budget1.5+reuse", env.tab.modes)
    assert sp2.__name__ == "pacedB1.5+reuse" and mp2.__name__ == "budget1.5"
