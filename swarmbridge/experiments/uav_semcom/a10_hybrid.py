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
  dhyb:I:MODEL:N:H:ROLL:Z:K[:PERSIST]   A-E6 (review #31) decision-aware hybrid: PPO proposes; the MPC runs whenever the
                                        RateErrTracker has >= 20 s of data in the UAV's SNR bin (else PPO directly); a
                                        switch must be resolvable (Z SE) at the nominal model AND when the same sampled
                                        rate paths are scaled to both ends of [min(1, 1+b) - K se, max(1, 1+b) + K se]
                                        (b: tracker bias, se: standard error of the mean observed/predicted ratio): the
                                        model's historical rate error is propagated into the PPO-vs-planner gain.
  shyb:I:MODEL:N:H:ROLL:Z:K:H1:C[:PERSIST]  A-E7 selective two-stage version of dhyb: a cheap screen evaluates every
                                        mode with the short horizon H1; if no mode beats PPO's mode in the screen, PPO's
                                        mode is used (no full planning); otherwise the full MPC (horizon H) evaluates
                                        only PPO's mode and the C best screened modes, then the same robustness check.
  ahyb:I:MODEL:N:H:ROLL:Z:K:H1:PERSIST[:key=val..]  A-E8 (review #33) selective controller with options
                                        cand=all|n2|np (all: no screen = dhyb; n2: top-2 nominal screen = shyb; np: one
                                        stacked short-horizon screen at the nominal model and at the low end lo of the
                                        sensitivity interval, candidates = best nominal + best pessimistic mode),
                                        trig=nom|pess|any (which screen must show a mode beating PPO), stack=0|1
                                        (nominal + endpoint scales in one stacked pass; same decisions, cheaper),
                                        dl=MS (cooperative decision budget incl. PPO inference, a9_mpc.Deadline; PPO's
                                        mode is the fallback), margin=MS (reserved timing margin, default 3),
                                        fbd=D, fbl=P (A-E9: throughput reports reach the calibrator / tracker D slots
                                        late and are lost with probability P, a9_mpc.Feedback; also during the 'pre'
                                        calibration episodes), fbs=SEED (loss draws).
  K may be written as e.g. 2g5: the robustness check then uses 5 scales evenly spaced over the interval instead of
  its 2 endpoints.
PERSIST: cold (reset per episode) | warm (kept across the test episodes of the run: continual adaptation, episodes not
independent) | pre (A-E6: calibrator / tracker warm-started on CAL_EPISODES separate calibration episodes driven by
the heuristic on the val image pool and seeds 9000.., then frozen as the start state of EVERY test episode; online
updates within an episode; test episodes independent).
--interior-diag: for every executed dhyb / shyb switch, the paired gain of the switch is re-evaluated with the
decision's own rate samples at 9 scales evenly spaced over [lo, hi]; reports the fraction of switches with an interior
scale where the gain is not positive (mean <= 0) or not Z-resolvable. Computed outside the timed decision.
--recall-diag: whenever the planner may run (tracker has data), a reference full planner (all modes, same model,
calibrator and endpoint check, no deadline, independent samples) decides as well; reported: how often its switch
target is among the modes the controller evaluated (candidate recall), decision agreement, missed triggers. Computed
outside the timed decision; the controller's own random streams are unaffected.
--rank-diag (phy_mcs_blind, where the 'env' model is the true PHY up to fading): for every executed switch and every
switch rejected by the robustness check, a correct-PHY planner (N=32, same H / rollout, fresh samples) evaluates
the gain of the switch; reported as the fraction it endorses (mean gain > 0). Computed outside the timed decision.
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


CAL_SEEDS, CAL_POOL = range(9000, 9010), "val"


def job(j):
    regime, variant, seeds, rank_diag, interior_diag, recall_diag = j
    import torch
    torch.set_num_threads(1)
    heur = AE.tuned(regime, "best")
    sp0, mp_heur = H.from_name(heur, C.MODES)
    p = variant.split(":")
    kind = p[0]
    cal = None
    mpc = None
    trk, tau = None, None
    kk = None
    n1 = n_cand = None
    grid = 0
    opt = {}
    dl = None
    if kind == "ahyb":
        opt = dict(x.split("=") for x in p[10:])
        p = p[:10]
    if kind in ("mpc", "hyb", "ehyb", "dhyb", "shyb", "ahyb"):
        o = 1 if kind == "mpc" else 2
        model, n, hz, roll = p[o], int(p[o + 1]), int(p[o + 2]), p[o + 3]
        z = float(p[o + 4]) if kind != "mpc" else None
        if kind in ("ehyb", "dhyb", "shyb", "ahyb"):
            ks = p[o + 5]                               # ehyb: residual threshold; dhyb / shyb: K (SE multiplier)
            if "g" in ks:
                ks, grid = ks.split("g")[0], int(ks.split("g")[1])
            tau = float(ks)
            if kind == "shyb":
                n1, n_cand = int(p[o + 6]), int(p[o + 7])
                persist = p[o + 8] if len(p) > o + 8 else "cold"
            elif kind == "ahyb":
                n1, persist = int(p[o + 6]), p[o + 7]
            else:
                persist = p[o + 6] if len(p) > o + 6 else "cold"
        else:
            persist = (p[o + 5] if kind == "hyb" else (p[o + 4] if len(p) > o + 4 else "cold")) if model in ("cal", "cals") else "cold"
        fbd, fbl = int(opt.get("fbd", 0)), float(opt.get("fbl", 0.0))
        mkfb = (lambda: M.Feedback(fbd, fbl, seed=int(opt.get("fbs", 31)))) if (fbd or fbl) else (lambda: None)
        if model in ("cal", "cals"):                    # cals: slot-weighted calibrator (A-E5), cal: airtime-weighted
            cal = M.CalRates(persistent=(persist in ("warm", "pre")), slot_weight=(model == "cals"), fb=mkfb())
            model = "cal"
        mpc = M.MPCMode(model, n, hz, None, fallback=mp_heur, seed=12345, rollout=roll, cal=cal, margin_z=z)
        if kind in ("ehyb", "dhyb", "shyb", "ahyb"):
            trk = M.RateErrTracker(mpc, persistent=(persist in ("warm", "pre")), fb=mkfb())
    biases, scale_w = [], []
    if float(opt.get("dl", 0)) > 0:
        dl = M.Deadline(float(opt["dl"]), margin_ms=float(opt.get("margin", 3.0)))
    cand = opt.get("cand", "n2")
    trig = opt.get("trig", "any" if cand == "np" else "nom")
    stack = opt.get("stack", "0") == "1"
    aborts = {}
    rec = {"full_switch": [], "in_cands": [], "agree": [], "missed_trigger": [], "ctrl_endorsed": []}
    ref_full = M.MPCMode(mpc.model, mpc.n, mpc.h, None, seed=4242, rollout=mpc.rollout, cal=cal,
                         margin_z=mpc.margin_z) if (recall_diag and mpc is not None) else None
    ref_mpc = M.MPCMode("env", 32, mpc.h, None, seed=999, rollout=mpc.rollout) if (rank_diag and mpc is not None) else None
    endorse = {"switch": [], "rejected": []}
    interior = {"nonpos": [], "unresolved": [], "min_rel": []}
    screens = fulls = 0

    def sp(env):
        if cal is not None:
            cal.update(env)
        if trk is not None:
            trk.update(env)
        return sp0(env)

    if mpc is not None and (cal is not None or trk is not None) and persist == "pre":
        for s in CAL_SEEDS:                             # separate calibration episodes (heuristic, val pool)
            V.run_episode_ev(SV.UavSurveyEnv(AE.G["table"], AE.G["pools"][CAL_POOL], C.regime_cfg(regime)), sp, mp_heur, s)
        for o_ in (cal, trk):
            if o_ is not None:
                o_.freeze()
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
            if dl is not None:
                dl.start(t0)
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
            elif kind == "ahyb":
                st = trk.stats(e.core, k)
                if st is None:
                    m_sel = m_ppo
                    gated += 1
                else:
                    b, se = st
                    biases.append(b)
                    lo = max(min(1.0, 1.0 + b) - tau * se, 0.05)
                    hi = max(1.0, 1.0 + b) + tau * se
                    scale_w.append(hi - lo)
                    mpc.scales = list(np.linspace(lo, hi, grid)) if grid else [lo, hi]
                    screens += int(cand != "all")
                    m_sel = mpc.selective(e.core, k, task, m_ppo, lo, cand=cand, trig=trig, h1=n1, n_cand=2,
                                          stack=stack, dl=dl)
                    fulls += int(mpc.last_full)
                    if mpc.last_abort:
                        aborts[mpc.last_abort] = aborts.get(mpc.last_abort, 0) + 1
            elif kind in ("dhyb", "shyb"):
                st = trk.stats(e.core, k)
                if st is None:
                    m_sel = m_ppo
                    gated += 1
                else:
                    b, se = st
                    biases.append(b)
                    lo = max(min(1.0, 1.0 + b) - tau * se, 0.05)
                    hi = max(1.0, 1.0 + b) + tau * se
                    scale_w.append(hi - lo)
                    mpc.scales = list(np.linspace(lo, hi, grid)) if grid else [lo, hi]
                    if kind == "dhyb":
                        m_sel = mpc(e.core, k, task, incumbent=m_ppo)
                    else:
                        screens += 1
                        sv = mpc.screen(e.core, k, task, n1)          # n1 = screen horizon H1
                        order = [m for m in np.argsort(-sv) if m != m_ppo]
                        if sv[order[0]] <= sv[m_ppo]:
                            m_sel = m_ppo                       # screen: nothing beats PPO -> no full planning
                        else:
                            fulls += 1
                            mpc.cands = [int(m) for m in order[:n_cand]]
                            m_sel = mpc(e.core, k, task, incumbent=m_ppo)
                            mpc.cands = None
            elif kind == "hyb":
                if cal is not None and cal.sample(e.core, k, (1, 1), np.random.default_rng(0)) is None:
                    m_sel = m_ppo
                    gated += 1
                else:
                    m_sel = mpc(e.core, k, task, incumbent=m_ppo)
            else:
                m_sel = m_ppo
            times.append(time.perf_counter() - t0)
            if ref_full is not None and kind == "ahyb" and st is not None:
                ref_full.scales = list(mpc.scales)
                m_full = ref_full(e.core, k, task, incumbent=m_ppo)
                if m_full != m_ppo:
                    rec["full_switch"].append(1.0)
                    rec["in_cands"].append(float(m_full in mpc.last_cands))
                    rec["missed_trigger"].append(float(not mpc.last_full))
                else:
                    rec["full_switch"].append(0.0)
                rec["agree"].append(float(m_full == m_sel))
                if m_sel != m_ppo:
                    rec["ctrl_endorsed"].append(float(m_full == m_sel))
            if interior_diag and kind in ("dhyb", "shyb") and m_sel != m_ppo:
                lo_, hi_ = mpc.scales[0], mpc.scales[-1]
                nonpos = unres = False
                rels = []
                for sc in np.linspace(lo_, hi_, 9):
                    dd = mpc.gain_at(e.core, k, task, m_ppo, m_sel, np.maximum(mpc.last_rates * sc, 1.0))
                    nonpos |= bool(dd.mean() <= 0)
                    unres |= not mpc._resolvable(dd)
                    rels.append(float(dd.mean()))
                interior["nonpos"].append(float(nonpos)); interior["unresolved"].append(float(unres))
                interior["min_rel"].append(min(rels) / max(max(rels), 1e-9))
            if ref_mpc is not None and kind != "ppo" and mpc.chosen and len(mpc.times) and planned_now(mpc):
                if m_sel != m_ppo:
                    endorse["switch"].append(float(ref_mpc.gain(e.core, k, task, m_ppo, m_sel).mean() > 0))
                elif getattr(mpc, "last_rejected", False):
                    endorse["rejected"].append(float(ref_mpc.gain(e.core, k, task, m_ppo, mpc.last_cand).mean() > 0))
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
    if interior["nonpos"]:
        out.update(interior_n=len(interior["nonpos"]), interior_nonpos=float(np.mean(interior["nonpos"])),
                   interior_unresolved=float(np.mean(interior["unresolved"])))
    if kind in ("shyb", "ahyb"):
        out.update(screens=int(screens), full_calls=int(fulls), full_frac=float(fulls / max(len(t), 1)))
    if kind == "ahyb":
        out.update(options=dict(cand=cand, trig=trig, stack=stack, dl=float(opt.get("dl", 0)),
                                margin=float(opt.get("margin", 3.0))), aborts=aborts,
                   abort_frac=float(sum(aborts.values()) / max(len(t), 1)))
        if trk is not None and trk.fb is not None:
            out.update(feedback=dict(delay=trk.fb.delay, loss=trk.fb.loss, sent=trk.fb.n_sent, lost=trk.fb.n_lost,
                                     used=trk.fb.n_used))
        if dl is not None:
            out.update(budget_ms=dl.budget, miss_frac=float((t > dl.budget).mean()),
                       miss_ms_max=float(t.max()), unit_est_ms=dl.est)
    if ref_full is not None and rec["full_switch"]:
        out.update(recall=dict(n=len(rec["agree"]), full_switch_frac=float(np.mean(rec["full_switch"])),
                               n_full_switch=len(rec["in_cands"]),
                               cand_recall=float(np.mean(rec["in_cands"])) if rec["in_cands"] else None,
                               missed_trigger=float(np.mean(rec["missed_trigger"])) if rec["missed_trigger"] else None,
                               agree=float(np.mean(rec["agree"])),
                               ctrl_switch_matches_full=float(np.mean(rec["ctrl_endorsed"])) if rec["ctrl_endorsed"] else None))
    if scale_w:
        out.update(scale_width_q=[float(x) for x in np.percentile(scale_w, [10, 50, 90])])
    if ref_mpc is not None:
        out.update(endorse_switch=float(np.mean(endorse["switch"])) if endorse["switch"] else None,
                   n_switch_diag=len(endorse["switch"]),
                   endorse_rejected=float(np.mean(endorse["rejected"])) if endorse["rejected"] else None,
                   n_rejected_diag=len(endorse["rejected"]))
    if biases:
        bb = np.asarray(biases, float)
        out.update(bias_known_frac=float(np.isfinite(bb).mean()),
                   bias_q=[float(x) for x in np.nanpercentile(bb, [10, 50, 90])] if np.isfinite(bb).any() else None)
    if mpc is not None:
        out.update(switched=int(mpc.switched), planned=int(len(mpc.times)), gated=int(gated),
                   cal_fallbacks=int(mpc.cal_fallbacks), robust_checks=int(mpc.robust_checks),
                   robust_rejects=int(mpc.robust_rejects), protocol=persist)
    return out


def planned_now(mpc):
    """True if the MPC ran for the current decision (its call counter advanced since the last check)."""
    n = len(mpc.times)
    ran = n != getattr(mpc, "_seen_calls", 0)
    mpc._seen_calls = n
    return ran


def init():
    AE.init(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regimes", default="phy_mcs_blind")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", default="8000:8020")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--rank-diag", action="store_true")
    ap.add_argument("--interior-diag", action="store_true")
    ap.add_argument("--recall-diag", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    jobs = [(r, v, seeds, a.rank_diag, a.interior_diag, a.recall_diag) for r in a.regimes.split(",") for v in a.variants.split(",")]
    t0 = time.time()
    with Pool(a.procs, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=1)
    for r in res:
        print(f"{r['regime']:14s} {r['variant']:34s} ret {r['return']:7.2f} drop {r['drop_rate']:.3f} "
              f"ms {r['ms_mean']:6.1f}/{r['ms_p95']:6.1f} kbit {r['mean_chosen_kbits']:6.1f} "
              f"sw {r.get('switched', '-')} gated {r.get('gated', '-')} calfb {r.get('cal_fallbacks', '-')} bias {r.get('bias_q', '-')} "
              f"rob {r.get('robust_rejects', '-')}/{r.get('robust_checks', '-')} endorse {r.get('endorse_switch', '-')}/"
              f"{r.get('endorse_rejected', '-')} full {r.get('full_frac', '-')} interior {r.get('interior_nonpos', '-')}/"
              f"{r.get('interior_unresolved', '-')} miss {r.get('miss_frac', '-')} aborts {r.get('aborts', '-')} "
              f"recall {r.get('recall', '-')}", flush=True)
    json.dump(dict(seeds=seeds, wall_s=time.time() - t0, rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
