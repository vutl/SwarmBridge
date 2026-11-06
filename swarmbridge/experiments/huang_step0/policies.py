"""Causal policies, a model-based rollout planner and a hindsight best-found search for step 0B.

All causal policies see: task attributes, current port reservations, LEO geometry, EXPECTED link rates.
They never see the realised fading draws or future tasks. The hindsight search sees everything (diagnostic only).
"""
import numpy as np

import model as M
import sim as S

OPTIONS = [(k, n) for k in range(5) for n in range(M.N_UAV)]


def area_uav(sc, i):
    return int(sc.user_area[sc.dst[i]])


def fixed(k):
    def pol(sc, st, i):
        return k, area_uav(sc, i)
    pol.__name__ = f"fixed_{M.MODES[k]}"
    return pol


def greedy(sc, st, i):
    """Maximise the task's own predicted utility over all (mode, UAV)."""
    return max(OPTIONS, key=lambda o: S.outcome(sc, st, i, o[0], o[1], actual=False)[0])


def greedy_area(sc, st, i):
    n = area_uav(sc, i)
    return max(((k, n) for k in range(5)), key=lambda o: S.outcome(sc, st, i, o[0], o[1], actual=False)[0])


def occupancy(sc, st, i, k, n):
    """Port time the option consumes (expected): uplink + downlink + decoder."""
    _, _, _, tm = S.outcome(sc, st, i, k, n, actual=False)
    if tm is None:
        return 1e3
    t1e, t2s, t2e, t3e, _ = tm
    return (t2e - t2s) + sc.bits[k] / sc.leo_rate(M.H_LEO, None) + M.COMPUTE_S[k]


def cheapest_feasible(sc, st, i):
    """Among options predicted to meet both constraints, the one using the least port time."""
    feas = []
    for o in OPTIONS:
        u, ok, _, _ = S.outcome(sc, st, i, o[0], o[1], actual=False)
        if ok:
            feas.append((occupancy(sc, st, i, *o), -u, o))
    if feas:
        return min(feas)[2]
    return greedy(sc, st, i)


def price(lmb):
    """Value minus price (GOSC-style): predicted utility - lmb * port seconds."""
    def pol(sc, st, i):
        return max(OPTIONS, key=lambda o: S.outcome(sc, st, i, o[0], o[1], actual=False)[0]
                   - lmb * occupancy(sc, st, i, *o))
    pol.__name__ = f"price_{lmb}"
    return pol


def rollout(n_scen=3, horizon=6, seed=0):
    """Model-based lookahead: for each option, commit it (expected rates), then continue with greedy_area on
    sampled future tasks drawn from the KNOWN task distribution; pick the best mean total."""
    def pol(sc, st, i):
        rng = np.random.default_rng(seed * 100003 + i)
        futures = []
        for _ in range(n_scen):
            gaps = rng.exponential(1 / sc.lam, horizon)
            futures.append(dict(t=sc.arrival[i] + np.cumsum(gaps), src=rng.integers(0, M.K_LEO, horizon),
                                dst=rng.integers(0, M.M_USER, horizon),
                                dmax=rng.uniform(*M.DMAX_RANGE, horizon), qmin=rng.uniform(*M.QMIN_RANGE, horizon)))
        best, best_v = None, -1e9
        for o in OPTIONS:
            u0 = S.outcome(sc, st, i, o[0], o[1], actual=False)[0]
            tot = 0.0
            for f in futures:
                tot += u0 + _continue(sc, st, i, o, f)
            v = tot / n_scen
            if v > best_v:
                best, best_v = o, v
        return best
    pol.__name__ = f"rollout_s{n_scen}_h{horizon}"
    return pol


class _Ghost:
    """Temporary task view for sampled future tasks (expected rates only)."""

    def __init__(self, sc, f):
        self.__dict__.update(sc.__dict__)
        self.sc = sc
        self.arrival = f["t"]
        self.src, self.dst, self.dmax, self.qmin = f["src"], f["dst"], f["dmax"], f["qmin"]

    def __getattr__(self, name):
        return getattr(self.sc, name)


def _commit_expected(sc, st, i, k, n):
    u, ok, d, tm = S.outcome(sc, st, i, k, n, actual=False)
    if tm is not None:
        t1e, t2s, t2e, t3e, user = tm
        st.rx[n] = max(st.rx[n], t1e)
        st.tx[n] = max(st.tx[n], t2e)
        st.dec[user] = max(st.dec[user], t3e)
        st.txint[n].append((t2s, t2e))
    return u


def _continue(sc, st, i, o, f, base="area"):
    s2 = st.copy()
    _commit_expected(sc, s2, i, *o)
    g = _Ghost(sc, f)
    tot = 0.0
    for j in range(len(f["t"])):
        if base == "area":
            n = int(sc.user_area[f["dst"][j]])
            cand = [(k, n) for k in range(5)]
        else:
            cand = OPTIONS
        best = max(cand, key=lambda c: S.outcome(g, s2, j, c[0], c[1], actual=False)[0])
        tot += _commit_expected(g, s2, j, *best)
    return tot


def rollout2(n_scen=3, horizon=10, base="all", foresight=False, seed=0):
    """Closed-loop lookahead. foresight=False: futures sampled from the known task distribution (causal).
    foresight=True: the TRUE next `horizon` tasks (arrival, source, user, deadline, quality) are used, still with
    expected link rates and re-planned at every decision from the realised state -> reference for the value of
    knowing future tasks (not achievable causally)."""
    def pol(sc, st, i):
        if foresight:
            j = slice(i + 1, min(i + 1 + horizon, M.G_TASKS))
            futures = [dict(t=sc.arrival[j], src=sc.src[j], dst=sc.dst[j], dmax=sc.dmax[j], qmin=sc.qmin[j])]
        else:
            rng = np.random.default_rng(seed * 100003 + i)
            futures = []
            for _ in range(n_scen):
                gaps = rng.exponential(1 / sc.lam, horizon)
                futures.append(dict(t=sc.arrival[i] + np.cumsum(gaps), src=rng.integers(0, M.K_LEO, horizon),
                                    dst=rng.integers(0, M.M_USER, horizon),
                                    dmax=rng.uniform(*M.DMAX_RANGE, horizon),
                                    qmin=rng.uniform(*M.QMIN_RANGE, horizon)))
        best, best_v = None, -1e9
        for o in OPTIONS:
            u0 = S.outcome(sc, st, i, o[0], o[1], actual=False)[0]
            v = u0 + float(np.mean([_continue(sc, st, i, o, f, base) for f in futures])) if len(futures[0]["t"]) else u0
            if v > best_v:
                best, best_v = o, v
        return best
    pol.__name__ = f"{'foresight' if foresight else 'rollout'}_{base}_h{horizon}" + ("" if foresight else f"_s{n_scen}")
    return pol


def hindsight(sc, init_acts, passes=3, know_fading=True):
    """Best-found schedule with knowledge of all future tasks: coordinate descent over per-task (mode, UAV)
    starting from init_acts, exact re-simulation of the suffix for every candidate. know_fading=True also uses the
    realised fading draws (full clairvoyance); False optimises on expected rates (future TASKS only) and the
    returned score is the realised outcome of that schedule."""
    acts = list(init_acts)
    a = know_fading
    st, snaps = S.replay(sc, acts, snapshots=True, actual=a)
    best = st.total
    for _ in range(passes):
        improved = False
        for i in range(M.G_TASKS):
            cur = acts[i]
            changed = False
            for o in OPTIONS:
                if o == cur:
                    continue
                acts[i] = o
                st2, _ = S.replay(sc, acts, start_state=snaps[i], start=i, actual=a)
                if st2.total > best + 1e-9:
                    best, cur, improved, changed = st2.total, o, True, True
            acts[i] = cur
            if changed:
                st, snaps = S.replay(sc, acts, snapshots=True, actual=a)
        if not improved:
            break
    if not a:
        st, _ = S.replay(sc, acts, actual=True)
    return st.total / M.G_TASKS, st.n_ok / M.G_TASKS, acts
