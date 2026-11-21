"""Paper A E7 (RL-family check, pre-registered in reports/phases/2026-11-18_paperA-writeup/paperA-e7-rl-families-prereg.md):
off-policy mode-selection agents for the mode-only task of PPO-LA (A-X9), in plain PyTorch.

  QR-DQN        (Dabney et al., AAAI 2018): distributional value-based; 50 quantiles, quantile-Huber loss (kappa 1),
                greedy action = argmax of the mean quantile; target network (Polyak tau 0.005), epsilon-greedy.
  Discrete SAC  (Christodoulou, arXiv:1910.07207, 2019): categorical actor, twin Q heads over all actions, soft
                state value V(s) = sum_a pi(a|s) [min Q_targ(s, a) - alpha log pi(a|s)], learned temperature with
                target entropy = ent_ratio * log(|A|); deterministic evaluation = argmax of the actor logits.

Both use the SAME semi-MDP discounting as TimeMaskablePPO (timeppo.py): the bootstrap factor is gamma^dt with dt
the simulated seconds until the next RL decision (gamma = 0.99 per second), no bootstrap at episode end.
Action masks: in the mode phase every mode is valid (gym_env_ev.action_masks); masks are still applied (invalid
logits / values set to -inf) so the agents stay correct if that changes.
ModePolicy wraps a saved checkpoint with the predict() interface of the SB3 models used by train_ev.evaluate_rl
and ax2_eval (full MultiDiscrete action vector; only the last entry, the mode, is used in the mode phase)."""
import copy
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, o, h=(256, 256)):
    layers, d = [], i
    for w in h:
        layers += [nn.Linear(d, w), nn.ReLU()]
        d = w
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def soft_update(tgt, src, tau):
    with torch.no_grad():
        for pt, ps in zip(tgt.parameters(), src.parameters()):
            pt.mul_(1 - tau).add_(ps, alpha=tau)


class Replay:
    def __init__(self, size, obs_dim, seed=0):
        self.size, self.n, self.i = int(size), 0, 0
        self.obs = np.zeros((self.size, obs_dim), np.float32)
        self.nobs = np.zeros((self.size, obs_dim), np.float32)
        self.act = np.zeros(self.size, np.int64)
        self.rew = np.zeros(self.size, np.float32)
        self.done = np.zeros(self.size, np.float32)
        self.dt = np.zeros(self.size, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, no, d, dt):
        for j in range(len(a)):
            k = self.i
            self.obs[k], self.nobs[k], self.act[k], self.rew[k], self.done[k], self.dt[k] = o[j], no[j], a[j], r[j], d[j], dt[j]
            self.i = (self.i + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def sample(self, b):
        idx = self.rng.integers(0, self.n, size=b)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.obs), t(self.act), t(self.rew), t(self.nobs), t(self.done), t(self.dt)


def boot(gamma, dt, d):
    """semi-MDP bootstrap factor gamma^dt (dt = simulated seconds to the next RL decision), 0 at episode end."""
    return gamma ** dt * (1 - d)


def _masked(x, mask):
    if mask is None:
        return x
    m = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    return x.masked_fill(~m, -1e9)


class QRDQN:
    algo = "qrdqn"

    def __init__(self, obs_dim, n_act, lr=1e-4, n_q=50, gamma=0.99, tau=0.005, kappa=1.0, seed=0):
        torch.manual_seed(seed)
        self.obs_dim, self.n_act, self.n_q, self.gamma, self.tau, self.kappa = obs_dim, n_act, n_q, gamma, tau, kappa
        self.q = mlp(obs_dim, n_act * n_q)
        self.qt = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.taus = (torch.arange(n_q, dtype=torch.float32) + 0.5) / n_q
        self.hp = dict(lr=lr, n_q=n_q, gamma=gamma, tau=tau, kappa=kappa)

    def values(self, obs, net=None):
        z = (net or self.q)(obs).view(-1, self.n_act, self.n_q)
        return z

    def greedy(self, obs, mask=None):
        with torch.no_grad():
            v = self.values(torch.as_tensor(obs, dtype=torch.float32)).mean(-1)
            return _masked(v, mask).argmax(-1).numpy()

    def act(self, obs, rng, eps=0.0, mask=None):
        a = self.greedy(obs, mask)
        for j in range(len(a)):
            if rng.random() < eps:
                ok = np.flatnonzero(mask[j]) if mask is not None else np.arange(self.n_act)
                a[j] = int(rng.choice(ok))
        return a

    def update(self, batch):
        o, a, r, no, d, dt = batch
        with torch.no_grad():
            zn = self.values(no, self.qt)                                   # (B, A, N)
            an = zn.mean(-1).argmax(-1)
            zt = zn[torch.arange(len(a)), an]                               # (B, N)
            y = r[:, None] + boot(self.gamma, dt, d)[:, None] * zt
        z = self.values(o)[torch.arange(len(a)), a]                         # (B, N)
        u = y[:, None, :] - z[:, :, None]                                   # (B, N_cur, N_tgt)
        k = self.kappa
        hub = torch.where(u.abs() <= k, 0.5 * u * u, k * (u.abs() - 0.5 * k))
        loss = ((self.taus[None, :, None] - (u.detach() < 0).float()).abs() * hub / k).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        soft_update(self.qt, self.q, self.tau)
        return float(loss.detach())

    def state(self):
        return dict(algo=self.algo, obs_dim=self.obs_dim, n_act=self.n_act, hp=self.hp, net=self.q.state_dict())


class DSAC:
    algo = "dsac"

    def __init__(self, obs_dim, n_act, lr=3e-4, ent_ratio=0.6, gamma=0.99, tau=0.005, seed=0):
        torch.manual_seed(seed)
        self.obs_dim, self.n_act, self.gamma, self.tau = obs_dim, n_act, gamma, tau
        self.pi = mlp(obs_dim, n_act)
        self.q1, self.q2 = mlp(obs_dim, n_act), mlp(obs_dim, n_act)
        self.q1t, self.q2t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_ent = ent_ratio * math.log(n_act)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=lr)
        self.hp = dict(lr=lr, ent_ratio=ent_ratio, gamma=gamma, tau=tau)

    def greedy(self, obs, mask=None):
        with torch.no_grad():
            return _masked(self.pi(torch.as_tensor(obs, dtype=torch.float32)), mask).argmax(-1).numpy()

    def act(self, obs, rng, eps=0.0, mask=None):
        with torch.no_grad():
            p = F.softmax(_masked(self.pi(torch.as_tensor(obs, dtype=torch.float32)), mask), -1).numpy().astype(np.float64)
        return np.array([int(rng.choice(self.n_act, p=pj / pj.sum())) for pj in p])

    def update(self, batch):
        o, a, r, no, d, dt = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            lpn = F.log_softmax(self.pi(no), -1)
            qn = torch.min(self.q1t(no), self.q2t(no))
            v = (lpn.exp() * (qn - alpha * lpn)).sum(-1)
            y = r + boot(self.gamma, dt, d) * v
        idx = torch.arange(len(a))
        lq = F.mse_loss(self.q1(o)[idx, a], y) + F.mse_loss(self.q2(o)[idx, a], y)
        self.opt_q.zero_grad()
        lq.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.opt_q.step()
        lp = F.log_softmax(self.pi(o), -1)
        p = lp.exp()
        with torch.no_grad():
            qm = torch.min(self.q1(o), self.q2(o))
        lpi = (p * (alpha * lp - qm)).sum(-1).mean()
        self.opt_pi.zero_grad()
        lpi.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), 10.0)
        self.opt_pi.step()
        ent = -(p * lp).sum(-1).detach()
        la = (self.log_alpha * (ent - self.target_ent)).mean()
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        soft_update(self.q1t, self.q1, self.tau)
        soft_update(self.q2t, self.q2, self.tau)
        return float(lq.detach())

    def state(self):
        return dict(algo=self.algo, obs_dim=self.obs_dim, n_act=self.n_act, hp=self.hp, net=self.pi.state_dict(),
                    q1=self.q1.state_dict(), q2=self.q2.state_dict(), log_alpha=float(self.log_alpha.detach()))


class ModePolicy:
    """Evaluation wrapper with the SB3 predict() interface: deterministic greedy mode, other entries 0."""

    def __init__(self, agent_or_path, nvec):
        if isinstance(agent_or_path, (QRDQN, DSAC)):
            st = agent_or_path.state()
        else:
            st = torch.load(agent_or_path, map_location="cpu", weights_only=False)
        self.algo, self.n_act = st["algo"], st["n_act"]
        out = st["n_act"] * st["hp"]["n_q"] if self.algo == "qrdqn" else st["n_act"]
        self.net = mlp(st["obs_dim"], out)
        self.net.load_state_dict(st["net"])
        self.net.eval()
        self.nvec = list(nvec)
        assert self.nvec[-1] == self.n_act
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(st["obs_dim"],), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(self.nvec)

    def predict(self, obs, deterministic=True, action_masks=None):
        with torch.no_grad():
            x = self.net(torch.as_tensor(np.asarray(obs, dtype=np.float32)[None]))
            if self.algo == "qrdqn":
                x = x.view(1, self.n_act, -1).mean(-1)
            if action_masks is not None:
                x = _masked(x, np.asarray(action_masks)[None, -self.n_act:])
            m = int(x.argmax(-1)[0])
        a = np.zeros(len(self.nvec), dtype=np.int64)
        a[-1] = m
        return a, None
