import json
import math

import numpy as np

import env as E


def make_table(tmp_path, n_img=6):
    rows = []
    for i in range(n_img):
        for j, (b, q) in enumerate([(2000, 22.0), (20000, 28.0), (80000, 34.0)]):
            rows.append(dict(image=f"im{i}", mode=f"m{j}", bytes=b * (1 + 0.1 * i), psnr=q + 0.2 * i, lpips=0.1,
                             enc_s=0.01, dec_s=0.02))
    p = tmp_path / "profile.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return E.ModeTable([p])


def test_propulsion_hover_and_min():
    assert abs(E.propulsion_power(0.0) - (79.86 + 88.63)) < 1e-6
    # power first decreases with speed (induced power) then increases
    assert E.propulsion_power(10.0) < E.propulsion_power(0.0) < E.propulsion_power(30.0)


def test_los_gain_decreases_with_distance(tmp_path):
    env = E.UavSemcomEnv(make_table(tmp_path), range(6))
    assert env.expected_gain(np.array([100.0, 0])) > env.expected_gain(np.array([1000.0, 0]))


def test_episode_runs_and_is_deterministic(tmp_path):
    tab = make_table(tmp_path)

    def run(seed):
        env = E.UavSemcomEnv(tab, range(6), dict(T=80))
        env.reset(seed)
        tot, done = 0.0, False
        while not done:
            acts = []
            for u in env.uavs:
                wp = u.wps[min(u.wp_i, len(u.wps) - 1)]
                d = wp - u.pos
                mv = 0 if u.wp_i >= len(u.wps) else 1 + int(round((math.atan2(d[1], d[0]) % (2 * math.pi)) / (math.pi / 4))) % 8
                acts.append((mv, 1 + len(acts) % env.c["M"], 2, 1))
            _, r, done = env.step(acts)
            tot += r
        return tot, env.metrics()

    a, b = run(3), run(3)
    assert a == b
    assert a[1]["captured"] > 0


def test_utility_penalty():
    class T:
        qmin, dmax = 25.0, 5.0
    env = E.UavSemcomEnv.__new__(E.UavSemcomEnv)
    env.c = dict(E.DEFAULT)
    u_ok, ok = env.utility(T, 30.0, 1.0)
    u_bad, bad = env.utility(T, 20.0, 1.0)
    assert ok and not bad and u_ok > 0 > u_bad
