"""A-X1 (review #36): a practical channel-aware slot scheduler family for the multi-UAV uplink.

Only the CHANNEL entries of a base slot policy are replaced; movement (survey / pacing) and power come unchanged from
the base slot policy (the slot part of the regime's tuned T:best heuristic, without its '+reuse' wrapper), so the
family isolates channel scheduling. All inputs are causal (current state, expected link at the current position,
statistical CSI only; no fading realisation).

Member SCH:a{alpha}:s{skip}:{reuse}
  priority      UAVs with a non-empty queue are ranked by  slack_k / r_k^alpha  (ascending; ties by index), with
                r_k the expected interference-free rate at the slot's power and slack_k the time left until the
                latest transmission end of its head image; alpha = 0 is exactly earliest-deadline-first (the A5 base
                rule), alpha = 1 also favours good channels (slack per unit rate).
  skip          1: a UAV whose head image cannot finish in time even alone on a channel (decided payload / r_k >
                slack_k) is ranked after all feasible ones (no airtime spent on a lost image while others wait).
  exclusive     the M best-ranked UAVs get one channel each.
  reuse         none  : the others stay silent;
                least : every other UAV with a queue shares the channel whose occupants have the least expected
                        received power at the BS (= the A5 '+reuse' rule);
                gG    : as least, but only if, with both transmitting at max power, the env's PHY (link_rate with the
                        co-channel interference) keeps the newcomer AND every occupant at >= G x their exclusive rate
                        (interference-aware reuse; MCS-blind link adaptation fails blocks below threshold).
Tuning (as A5): every member on the TRAIN pool, seeds 0..29, with the T:best mode rule; the member with the best mean
train return is SCH* of the regime (chosen before any pilot evaluation)."""
import itertools
import json
from pathlib import Path

import numpy as np

import a5_common as C
import mode_heur as MH
import policies as P

TUNE_DIR = C.REPO / "runs" / "uav_semcom" / "ax1" / "tune"


def _head_bits(env, u):
    tk = u.queue[0]
    if tk.mode < 0:
        return None
    return tk.rem_bits if tk.ready <= env.t else env.mode_info(tk)[0][tk.mode] - tk.sent_bits


def channel_rule(alpha=0.0, skip=0, reuse="none"):
    gamma = float(reuse[1:]) if reuse.startswith("g") else None

    def assign(env, acts):
        c = env.c
        noise = env.noise_w()
        cand = []
        for k, u in enumerate(env.uavs):
            if not (u.active and u.queue):
                continue
            slack = env.latest_tx_end(u.queue[0]) - env.t
            r = max(P.expected_rate(env, k, acts[k][2]), 1.0)
            b = _head_bits(env, u)
            infeasible = bool(skip) and b is not None and b / r > slack
            cand.append((infeasible, slack / r ** alpha, k))
        cand.sort()
        ch = [0] * len(env.uavs)
        for i, (_, _, k) in enumerate(cand[:c["M"]]):
            ch[k] = i + 1
        if reuse != "none":
            g = [env.expected_gain(u.pos) for u in env.uavs]
            pw = [c["p_levels"][a[2]] for a in acts]
            for k in sorted(k for _, _, k in cand[c["M"]:]):      # index order, as the A5 '+reuse' rule
                load = {j: sum(pw[i] * g[i] for i in range(len(env.uavs)) if ch[i] == j) for j in range(1, c["M"] + 1)}
                j = min(load, key=lambda x: (load[x], x))
                if gamma is not None:
                    occ = [i for i in range(len(env.uavs)) if ch[i] == j]
                    ok = True
                    for a_, others in [(k, occ)] + [(i, [k] + [o for o in occ if o != i]) for i in occ]:
                        sig = pw[a_] * g[a_]
                        intf = sum(pw[o] * g[o] for o in others)
                        solo = env.link_rate(sig, noise, 0.0, sig / noise)
                        if env.link_rate(sig, noise, intf, sig / noise) < gamma * solo:
                            ok = False
                            break
                    if not ok:
                        continue
                ch[k] = j
        return [(a[0], ch[k], a[2]) + tuple(a[3:]) for k, a in enumerate(acts)]
    return assign


def base_name(regime):
    import a5_eval as AE
    n = AE.tuned(regime, "best")
    return n[:-len("+reuse")] if n.endswith("+reuse") else n


def from_name(name, regime):
    """'SCH:a1:s1:g0.5' -> (slot policy, mode policy) on the regime's T:best base (movement, power, mode)."""
    _, a, s, reuse = name.split(":")
    sp0, mp = MH.from_name(base_name(regime), C.MODES)
    rule = channel_rule(float(a[1:]), int(s[1:]), reuse)

    def sp(env):
        return rule(env, [tuple(x) for x in sp0(env)])
    sp.__name__ = name
    return sp, mp


def family():
    return [f"SCH:a{a:g}:s{s}:{r}" for a, s, r in itertools.product((0, 1), (0, 1), ("none", "least", "g0.5", "g0.25"))]


def tuned(regime):
    d = json.loads((TUNE_DIR / f"{regime}.json").read_text())
    return d["selected"]


def _tune_job(j):
    regime, name, seed = j
    import a5_eval as AE
    import env_ev as V
    import env_survey as SV
    sp, mp = from_name(name, regime)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["train"], C.regime_cfg(regime))
    return regime, name, V.run_episode_ev(core, sp, mp, seed)["return"]


def _init():
    import a5_eval as AE
    AE.init(False)


def tune(regimes, procs=12):
    from multiprocessing import Pool
    import a5_eval as AE
    jobs = [(r, n, s) for r in regimes for n in family() for s in C.TRAIN_SEEDS]
    with Pool(procs, initializer=_init) as pool:
        res = pool.map(_tune_job, jobs, chunksize=8)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    for r in regimes:
        rows = {n: float(np.mean([x for rr, nn, x in res if rr == r and nn == n])) for n in family()}
        sel = max(rows, key=rows.get)
        Path(TUNE_DIR / f"{r}.json").write_text(json.dumps(dict(regime=r, base=base_name(r), seeds=C.TRAIN_SEEDS,
                                                                train_return=rows, selected=sel), indent=1))
        print(r, "base", base_name(r), "selected", sel, round(rows[sel], 2), "| A5-equivalent rules:",
              {n: round(rows[n], 2) for n in ("SCH:a0:s0:none", "SCH:a0:s0:least")}, flush=True)


if __name__ == "__main__":
    import sys
    tune(sys.argv[1].split(","), int(sys.argv[2]) if len(sys.argv) > 2 else 12)
