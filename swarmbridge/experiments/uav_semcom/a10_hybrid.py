"""A-E4 (review #29): PPO-guided, online-calibrated planning vs PPO-only, MPC variants and the heuristic.

All controllers decide ONLY the mode; the slot policy is the slot part of the regime's A5 T:best heuristic (as for
PPO mode-only). Variants (VARIANT strings):
  heur                                  T:best (budget-heuristic mode)
  ppo:I                                 PPO mode-only, A5 best-val model I (0..2)
  mpc:MODEL:N:H:ROLL[:PERSIST]          MPC with the heuristic mode as incumbent; MODEL env | shannon | cal | cals
                                        (cal: A-E4 airtime-weighted calibrator; cals: slot-weighted, A-E5);
                                        ROLL scenario | causal; PERSIST cold | warm (cal only)
  hyb:I:MODEL:N:H:ROLL:Z[:PERSIST]      PPO model I proposes the mode (incumbent); the MPC switches only if its
                                        paired per-sample value gain exceeds Z standard errors; with MODEL cal the
                                        planner runs only when the ACK-based rate model has data for the UAV's
                                        expected-SNR bin (sample-availability gate), otherwise PPO's mode is used.
  ehyb:I:MODEL:N:H:ROLL:Z:TAU[:PERSIST] A-E5 (review #30) model-error-aware hybrid: as hyb, but the planner runs only
                                        when the RateErrTracker's recent relative bias of MODEL's predicted rate vs
                                        observed throughput, in the UAV's expected-SNR bin, is known (>= 20 s airtime)
                                        and |bias| <= TAU; otherwise PPO's mode is used directly. PERSIST (cold|warm)
                                        applies to both the calibrator (cal) and the tracker.
Every slot boundary first updates the CalRates calibrator (causal: finished segments only).
Per variant: returns, metrics, decision time (incl. PPO inference for ppo / hyb), switch counts, chosen-mode
histogram and mean predicted bits of the chosen modes (diagnostic for shallow MPC).
Usage: python a10_hybrid.py OUT --regimes phy_mcs_blind,ref --variants V1,V2,.. [--seeds 8000:8020 --procs 12]
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import a55_heur as H
import a9_latency as LAT
import a9_mpc as M
import env_ev as V
import env_survey as SV
import gym_env_ev as GE


def ppo_path(regime, i):
    spec = LAT.ppo_specs(regime)[i]
    return spec[2:].split("=", 1)[1].split(":")[0]


def job(j):
    regime, variant, seeds = j
    import torch
    torch.set_num_threads(1)
    heur = AE.tuned(regime, "best")
    sp0, mp_heur = H.from_name(heur, C.MODES)
    p = variant.split(":")
    kind = p[0]
    cal = None
    mpc = None
    trk, tau = None, None
    if kind in ("mpc", "hyb", "ehyb"):
        o = 1 if kind == "mpc" else 2
        model, n, hz, roll = p[o], int(p[o + 1]), int(p[o + 2]), p[o + 3]
        z = float(p[o + 4]) if kind != "mpc" else None
        if kind == "ehyb":
            tau = float(p[o + 5])
            persist = p[o + 6] if len(p) > o + 6 else "cold"
        else:
            persist = (p[o + 5] if kind == "hyb" else (p[o + 4] if len(p) > o + 4 else "cold")) if model in ("cal", "cals") else "cold"
        if model in ("cal", "cals"):                    # cals: slot-weighted calibrator (A-E5), cal: airtime-weighted
            cal = M.CalRates(persistent=(persist == "warm"), slot_weight=(model == "cals"))
            model = "cal"
        mpc = M.MPCMode(model, n, hz, None, fallback=mp_heur, seed=12345, rollout=roll, cal=cal, margin_z=z)
        if kind == "ehyb":
            trk = M.RateErrTracker(mpc, persistent=(persist == "warm"))
    biases = []

    def sp(env):
        if cal is not None:
            cal.update(env)
        if trk is not None:
            trk.update(env)
        return sp0(env)

    times, chosen_bits, hist, gated = [], [], np.zeros(len(C.MODES), int), 0
    rows = []
    for s in seeds:
        if kind in ("heur", "mpc"):
            pol = mp_heur if kind == "heur" else mpc

            def mp(env, k, task):
                t0 = time.perf_counter()
                m = pol(env, k, task)
                times.append(time.perf_counter() - t0)
                hist[m] += 1
                chosen_bits.append(env.mode_info(task)[0][m])
                return m
            core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime))
            m = V.run_episode_ev(core, sp, mp, s)
            rows.append({k: float(m[k]) for k in AE.KEEP})
            continue
        mdl = AE.model(ppo_path(regime, int(p[1])))
        e = GE.GymUavSemcomEv(AE.G["table"], AE.G["pools"]["test"], C.regime_cfg(regime), seed=s, seeds=[s],
                              core_cls=SV.UavSurveyEnv, slot_pol=sp)
        obs, _ = e.reset()
        done, tot = False, 0.0
        while not done:
            t0 = time.perf_counter()
            a, _ = mdl.predict(e._obs(), deterministic=True, action_masks=e.action_masks())
            m_ppo = int(np.asarray(a)[-1])
            k, task = e.pending
            if kind == "ehyb":
                b = trk.bias(e.core, k)
                biases.append(np.nan if b is None else b)
                if b is None or abs(b) > tau:
                    m_sel = m_ppo
                    gated += 1
                else:
                    m_sel = mpc(e.core, k, task, incumbent=m_ppo)
            elif kind == "hyb":
                if cal is not None and cal.sample(e.core, k, (1, 1), np.random.default_rng(0)) is None:
                    m_sel = m_ppo
                    gated += 1
                else:
                    m_sel = mpc(e.core, k, task, incumbent=m_ppo)
            else:
                m_sel = m_ppo
            times.append(time.perf_counter() - t0)
            hist[m_sel] += 1
            chosen_bits.append(e.core.mode_info(task)[0][m_sel])
            act = np.array(a)
            act[-1] = m_sel
            obs, r, done, _, info = e.step(act)
            tot += r
        info["return"] = tot
        rows.append({k: float(info[k]) for k in AE.KEEP})
    t = np.asarray(times) * 1e3
    out = dict(regime=regime, variant=variant, slot_policy_from=heur, per_seed_return=[x["return"] for x in rows],
               decisions=int(len(t)), ms_mean=float(t.mean()), ms_p95=float(np.percentile(t, 95)),
               ms_p99=float(np.percentile(t, 99)), frac_over_50ms=float((t > 50).mean()),
               frac_over_100ms=float((t > 100).mean()), mode_hist=hist.tolist(), mean_chosen_kbits=float(np.mean(chosen_bits)) / 1e3)
    for k in ("return", "ok_rate", "drop_rate", "late_rate", "q_viol_rate", "mean_psnr"):
        out[k] = float(np.nanmean([x[k] for x in rows]))
    if biases:
        bb = np.asarray(biases, float)
        out.update(bias_known_frac=float(np.isfinite(bb).mean()),
                   bias_q=[float(x) for x in np.nanpercentile(bb, [10, 50, 90])] if np.isfinite(bb).any() else None)
    if mpc is not None:
        out.update(switched=int(mpc.switched), planned=int(len(mpc.times)), gated=int(gated),
                   cal_fallbacks=int(mpc.cal_fallbacks))
    return out


def init():
    AE.init(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regimes", default="phy_mcs_blind")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", default="8000:8020")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    jobs = [(r, v, seeds) for r in a.regimes.split(",") for v in a.variants.split(",")]
    t0 = time.time()
    with Pool(a.procs, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=1)
    for r in res:
        print(f"{r['regime']:14s} {r['variant']:34s} ret {r['return']:7.2f} drop {r['drop_rate']:.3f} "
              f"ms {r['ms_mean']:6.1f}/{r['ms_p95']:6.1f} kbit {r['mean_chosen_kbits']:6.1f} "
              f"sw {r.get('switched', '-')} gated {r.get('gated', '-')} calfb {r.get('cal_fallbacks', '-')} bias {r.get('bias_q', '-')}", flush=True)
    json.dump(dict(seeds=seeds, wall_s=time.time() - t0, rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
