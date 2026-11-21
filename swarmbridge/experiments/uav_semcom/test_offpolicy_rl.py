"""Paper A E7: unit tests of the off-policy agents (offpolicy_rl.py)."""
import numpy as np
import pytest
import torch

import offpolicy_rl as OP


def test_boot_is_semi_mdp():
    g = 0.99
    dt = torch.tensor([0.0, 1.0, 2.5, 1.0])
    d = torch.tensor([0.0, 0.0, 0.0, 1.0])
    b = OP.boot(g, dt, d)
    assert torch.allclose(b, torch.tensor([1.0, 0.99, 0.99 ** 2.5, 0.0]))


def _bandit(agent, steps=1500, best=3, seed=0):
    """contextual bandit: obs = one-hot context c in {0, 1}; reward 1 for action (best + c) % n, else 0; terminal."""
    rng = np.random.default_rng(seed)
    rep = OP.Replay(5000, 4, seed=seed)
    for _ in range(2000):
        c = rng.integers(0, 2, size=8)
        o = np.zeros((8, 4), np.float32)
        o[np.arange(8), c] = 1
        a = rng.integers(0, agent.n_act, size=8)
        r = (a == (best + c) % agent.n_act).astype(np.float32)
        rep.add(o, a, r, o, np.ones(8, np.float32), np.ones(8, np.float32))
    for _ in range(steps):
        agent.update(rep.sample(128))
    o = np.eye(4, dtype=np.float32)[:2]
    return agent.greedy(o)


@pytest.mark.parametrize("make", [lambda: OP.QRDQN(4, 6, lr=1e-3, seed=0), lambda: OP.DSAC(4, 6, lr=1e-3, ent_ratio=0.3, seed=0)])
def test_agents_learn_contextual_bandit(make):
    agent = make()
    assert list(_bandit(agent)) == [3, 4]


def test_mode_policy_wrapper_and_mask(tmp_path):
    agent = OP.QRDQN(4, 6, lr=1e-3, seed=0)
    _bandit(agent)
    torch.save(agent.state(), tmp_path / "q.pt")
    nvec = [11, 3, 3, 6]
    pol = OP.ModePolicy(str(tmp_path / "q.pt"), nvec)
    a, _ = pol.predict(np.eye(4, dtype=np.float32)[0])
    assert list(a) == [0, 0, 0, 3]
    mask = np.ones(11 + 3 + 3 + 6, dtype=bool)
    mask[-6 + 3] = False                       # forbid the greedy mode 3
    a, _ = pol.predict(np.eye(4, dtype=np.float32)[0], action_masks=mask)
    assert a[-1] != 3 and list(a[:-1]) == [0, 0, 0]
    assert pol.observation_space.shape == (4,) and list(pol.action_space.nvec) == nvec
