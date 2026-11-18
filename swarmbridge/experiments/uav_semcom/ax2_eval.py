"""A-X2 evaluation (review #37): trained channel schedulers (C-only, C+Mo) vs fixed-scheduler baselines, with and
without the selective PPO-MPC mode override, on fresh paired environment seeds.

Episode = ax2_env.GymSchedEv with learn_mode=True (both decision kinds exposed; movement + power always the
controller regime's T:best slot rule, state-dependent, enforced), so every combination shares the observation,
masks, reward, task list, fading and movement / power.
Combination  SCHED|MODE[+mpc]
  SCHED  tbest (T:best channel rule) | sch (A-X1 SCH*) | jc (A5 joint PPO channel head, ckpt I)
         | c (A-X2 C-only, ckpt I) | cm (A-X2 C+Mo, ckpt I)
         | ja (review #42: joint-assignment heuristic ja_sched, member given by --ja; channels only)
         | ls (movement / power of the controller regime's T:best, channels = least-slack M UAVs, no reuse)
         | jae (as ja, but movement / power of the ENVIRONMENT regime's T:best, i.e. matched with tbest_env)
  MODE   heur (T:best mode rule) | ppo (A5 mode-only PPO I) | jm (A5 joint PPO mode head I) | cmm (A-X2 C+Mo mode head I)
         | h=NAME (review #43: any mode_heur mode rule, e.g. h=fixed_avif_20, h=greedy, h=greedy+budget1.5)
         | hc=NAME (the same rule with its expected rate replaced by the CALIBRATED rate of the MPC's own online
           calibrator: RateErrTracker.predict with the cold slot-weighted CalRates, x channel share; fixed rules do not
           use a rate and are unaffected)
  +mpc   also allowed after h= / hc= : the heuristic is the incumbent of the selective MPC (heuristic-seeded)
  +mpcnc (review #44) the same selective MPC (same N, H, causal rollout, 50 ms budget, incumbent) WITHOUT online
         calibration: the planner uses the nominal rate model of the environment (env.link_rate; for the LA PHY the
         genie CQI rate x (1 - BLER target)), plans from the first decision on, no bias-derived scales (lo = hi = 1)
  ppola  (review #44) PPO mode-only trained on the LA PHY (checkpoint I from runs/uav_semcom/ax9/ppo_la_ckpts.json)
Scheduler options (review #44): SCHED[~JAMEMBER][@MOVE] with SCHED in {ja, ls}: MOVE = mission (the slot rule of
fixed_avif_20: mission flight) or a paced slot rule name (pacedB1.5, pacedB2, pacedB3); default movement as before.
  +mpc   the mode head proposes, the A-E8 selective controller (cals np-any, N 8, H 4, causal, Z 1, kappa 2, H1 1,
         stacked, 50 ms cooperative budget started before the mode head's inference) may override; calibration
         COLD (per episode, as recommended in review #37: a warm start made under T:best could be stale under a
         learned scheduler).
Checkpoint I pairs the I-th training seed of every learned part (they are separate training runs; I only pairs them
for averaging). Combinations without a learned part run once.
--env-regime: the environment can differ from the controller regime (MCS stress: controllers of K5, environment
K5 with the interference-blind MCS PHY); its T:best is used only by the 'tbest_env' scheduler / 'heur_env' mode.
Diagnostics per episode, besides the A5 metrics:
  * service-rate audit: at every slot start, for each UAV with a non-empty queue, the planner's predicted service
    rate = (cold calibrator / tracker per-channel mean rate, rho-Shannon fallback) x channel share M / N_busy, versus
    the realised bits sent in that slot / dt; ratio of sums and normalised MAE (independent of whether MPC runs);
  * MPC: tracker bias quantiles, gated fraction, time of the first planner run (cold-start fallback time), aborts;
  * decision wall times: scheduler per slot, mode head (+MPC) per mode decision, p99 of each.
--env-extra JSON: cfg overrides of the environment (review #42: e.g. the link-adaptation PHY of A-X6 / A-X7 with
obs_la false, so that controllers trained on K5 see their own observation layout).
Usage: python ax2_eval.py OUT --regime K5 [--env-regime K5_mcs] --combos A,B,.. --seeds 8800:8900 --procs 6"""
import argparse
import gzip
import json
import math
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import a9_mpc as M
import ax1_pilot as XP
import ax1_sched as XS
import ax2_env as X
import env_survey as SV
import mode_heur as MH
import policies as P

AX2 = "/home/uservutl/work/ax2"


def ax2_path(regime, variant, i):
    d = f"{AX2}/{regime}_{variant}_s{i}"
    best = json.load(open(f"{d}/train.json"))["best_steps"]
    return f"{d}/ckpt_{best}.zip"


class ServiceAudit:
    """predicted (calibrated per-channel rate x M / N_busy) vs realised per-slot service rate of busy UAVs."""

    def __init__(self):
        self.cal = M.CalRates(slot_weight=True)
        self.trk = M.RateErrTracker(types_ns(self.cal))
        self.seg_i, self.prev = 0, None
        self.pred, self.real = [], []

    def update(self, env):
        self.cal.update(env)
        self.trk.update(env)
        if self.prev is not None and self.prev[0] != env.t:
            t0, pr = self.prev
            bits = {}
            segs = env.segments
            while self.seg_i < len(segs):
                k, ch, a, b, interf, r = segs[self.seg_i]
                self.seg_i += 1
                bits[k] = bits.get(k, 0.0) + r * max(b - a, 0.0)
            for k, p in pr.items():
                self.pred.append(p)
                self.real.append(bits.get(k, 0.0) / (env.t - t0))
        if self.prev is None or self.prev[0] != env.t:
            share = P.channel_share(env)
            self.prev = (env.t, {k: self.trk.predict(env, k) * share for k, u in enumerate(env.uavs)
                                 if u.active and u.queue})

    def stats(self):
        p, r = np.asarray(self.pred), np.asarray(self.real)
        if not len(p):
            return dict(n=0)
        return dict(n=int(len(p)), ratio=float(r.sum() / max(p.sum(), 1e-9)),
                    nmae=float(np.abs(r - p).mean() / max(p.mean(), 1e-9)), zero_frac=float((r == 0).mean()))


def types_ns(cal):
    import types
    return types.SimpleNamespace(model="cal", cal=cal)


def calibrated(mode_pol, rate_fn):
    """mode rule whose expected rate (policies.expected_rate, used by greedy / budget rules) is rate_fn(env, k)."""
    def f(env, k, task):
        orig = P.expected_rate
        P.expected_rate = lambda env_, k_, p_level=-1: rate_fn(env_, k_)
        try:
            return mode_pol(env, k, task)
        finally:
            P.expected_rate = orig
    return f


MOVE_NAMES = {"mission": "fixed_avif_20"}


def episode(ctrl_regime, env_regime, combo, i, seed, extra=None, ja=None, pool="test"):
    sched, mode = combo.split("|")
    move = None
    if "@" in sched:
        sched, move = sched.split("@")
    if "~" in sched:
        sched, ja = sched.split("~")
    nocal = mode.endswith("+mpcnc")
    use_mpc = mode.endswith("+mpc") or nocal
    mode = mode[:-6] if nocal else (mode[:-4] if use_mpc else mode)
    mp_h, cal_h = None, False
    if mode.startswith(("h=", "hc=")):
        cal_h = mode.startswith("hc=")
        mp_h = MH.from_name(mode.split("=", 1)[1], C.MODES)[1]
    sp_t, mp_t = MH.from_name(AE.tuned(ctrl_regime, "best"), C.MODES)
    sp_e, mp_e = (MH.from_name(AE.tuned(env_regime, "best"), C.MODES) if env_regime != ctrl_regime else (sp_t, mp_t))
    sp_s = XS.from_name(XS.tuned(ctrl_regime), ctrl_regime)[0] if sched == "sch" else None
    models = {}

    def mdl(key, path_fn):
        if key not in models:
            models[key] = AE.model(path_fn())
        return models[key]
    smod = {"jc": lambda: mdl("joint", lambda: XP.joint_path(ctrl_regime, i)),
            "c": lambda: mdl("c", lambda: ax2_path(ctrl_regime, "c", i)),
            "cm": lambda: mdl("cm", lambda: ax2_path(ctrl_regime, "cm", i))}.get(sched)
    mmod = {"ppola": lambda: mdl("ppola", lambda: json.loads(
                (C.REPO / "runs" / "uav_semcom" / "ax9" / "ppo_la_ckpts.json").read_text())[i]),
            "ppo": lambda: mdl("mode", lambda: XP.mode_path(ctrl_regime, i)),
            "jm": lambda: mdl("joint", lambda: XP.joint_path(ctrl_regime, i)),
            "cmm": lambda: mdl("cm", lambda: ax2_path(ctrl_regime, "cm", i))}.get(mode)
    # movement + power: the controller regime's T:best (tbest_env: the env regime's own tuned heuristic, whole slot)
    sp_mp = sp_e if sched in ("tbest_env", "jae") else sp_t
    if move is not None:
        sp_mp = MH.from_name(MOVE_NAMES.get(move, move), C.MODES)[0]
    if sched in ("ja", "jae"):
        import ja_sched as JA
        sp_s = JA.from_name(ja, sp_mp)
    elif sched == "ls":
        def sp_s(env):
            ch = P.assign_channels(env)
            return [(a[0], ch[k], a[2]) for k, a in enumerate(sp_mp(env))]
    e = X.GymSchedEv(AE.G["table"], AE.G["pools"][pool], C.regime_cfg(env_regime, **(extra or {})), seed=seed, seeds=[seed],
                     core_cls=SV.UavSurveyEnv, sp_heur=sp_mp, mp_heur=mp_t, learn_mode=True)
    for f in (smod, mmod):
        if f is not None:
            m_ = f()
            assert m_.observation_space.shape == e.observation_space.shape
            assert list(m_.action_space.nvec) == list(e.action_space.nvec)
    audit = ServiceAudit()
    mpc = trk = cal = dl = None
    if (use_mpc and not nocal) or cal_h:
        assert mmod is not None or mp_h is not None, "+mpc needs an incumbent (learned mode head or h= / hc= rule)"
        cal = M.CalRates(persistent=False, slot_weight=True)
        mpc = M.MPCMode("cal", 8, 4, None, fallback=mp_t, seed=12345, rollout="causal", cal=cal, margin_z=1.0)
        trk = M.RateErrTracker(mpc, persistent=False)
        dl = M.Deadline(50.0, margin_ms=3.0) if use_mpc else None
    elif nocal:
        assert mmod is not None or mp_h is not None
        mpc = M.MPCMode("env", 8, 4, None, fallback=mp_t, seed=12345, rollout="causal", margin_z=1.0)
        dl = M.Deadline(50.0, margin_ms=3.0)
    if mp_h is not None and cal_h:
        mp_h = calibrated(mp_h, lambda env_, k_: trk.predict(env_, k_))
    slot_of_mode = []
    obs, _ = e.reset()
    done, tot = False, 0.0
    t_slot, t_mode, biases, aborts = [], [], [], 0
    gated = planned = 0
    t_first_plan = None
    K = e.K
    while not done:
        core = e.core
        if e.phase == "slot":
            audit.update(core)
            if cal is not None:
                cal.update(core)
                trk.update(core)
            t0 = time.perf_counter()
            a = np.zeros(len(e.action_space.nvec), dtype=np.int64)
            if sched in ("tbest", "tbest_env", "sch", "ja", "jae", "ls"):
                acts = [x[:3] for x in (sp_s if sched in ("sch", "ja", "jae", "ls") else sp_mp)(core)]
                for k in range(K):
                    a[3 * k:3 * k + 3] = acts[k]
            else:
                pa, _ = smod().predict(obs, deterministic=True, action_masks=e.action_masks())
                a[:] = np.asarray(pa)
            t_slot.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            if dl is not None:
                dl.start(t0)
            a = np.zeros(len(e.action_space.nvec), dtype=np.int64)
            k, task = e.pending
            if mode == "heur":
                m_sel = mp_t(core, k, task)
            elif mode == "heur_env":
                m_sel = mp_e(core, k, task)
            else:
                if mp_h is not None:
                    m_sel = int(mp_h(core, k, task))
                else:
                    pa, _ = mmod().predict(obs, deterministic=True, action_masks=e.action_masks())
                    m_sel = int(np.asarray(pa)[-1])
                if use_mpc and nocal:
                    planned += 1
                    if t_first_plan is None:
                        t_first_plan = float(core.t)
                    mpc.scales = [1.0, 1.0]
                    m_sel = mpc.selective(core, k, task, m_sel, 1.0, cand="np", trig="any", h1=1, n_cand=2,
                                          stack=True, dl=dl)
                    aborts += int(mpc.last_abort is not None)
                elif use_mpc:
                    st = trk.stats(core, k)
                    if st is None:
                        gated += 1
                    else:
                        planned += 1
                        if t_first_plan is None:
                            t_first_plan = float(core.t)
                        b, se = st
                        biases.append(b)
                        lo = max(min(1.0, 1.0 + b) - 2.0 * se, 0.05)
                        hi = max(1.0, 1.0 + b) + 2.0 * se
                        mpc.scales = [lo, hi]
                        m_sel = mpc.selective(core, k, task, m_sel, lo, cand="np", trig="any", h1=1, n_cand=2,
                                              stack=True, dl=dl)
                        aborts += int(mpc.last_abort is not None)
            a[-1] = m_sel
            t_mode.append(time.perf_counter() - t0)
            slot_of_mode.append(len(t_slot) - 1)
        obs, r, done, _, info = e.step(a)
        tot += r
    core = e.core
    m = {k: float(info[k]) for k in AE.KEEP if k in info}
    m["return"] = tot
    per_uav = np.asarray([sum(tk.ok for tk in core.finished if tk.uav == k) / (core.L * core.c["n_wp"]) for k in range(K)])
    air = sum(s[3] - s[2] for s in core.segments)
    air_int = sum(s[3] - s[2] for s in core.segments if s[4] > 0)
    ts, tm = np.asarray(t_slot) * 1e3, np.asarray(t_mode) * 1e3
    m.update(jain=float(per_uav.sum() ** 2 / (K * (per_uav ** 2).sum())) if (per_uav ** 2).sum() > 0 else float("nan"),
             min_uav_ok=float(per_uav.min()), cochannel_air_frac=float(air_int / max(air, 1e-9)),
             chan_util=float(air / (core.c["M"] * max(m["makespan_s"], 1e-9))),
             ms_slot_mean=float(ts.mean()), ms_slot_p99=float(np.percentile(ts, 99)),
             ms_mode_mean=float(tm.mean()) if len(tm) else 0.0, ms_mode_p99=float(np.percentile(tm, 99)) if len(tm) else 0.0,
             mode_over50=float((tm > 50).mean()) if len(tm) else 0.0, service=audit.stats(),
             ms_slot_p50=float(np.percentile(ts, 50)), ms_mode_p50=float(np.percentile(tm, 50)) if len(tm) else 0.0)
    # integrated latency per slot: the slot decision plus the mode decisions taken during that slot
    mx, sm = np.zeros(len(ts)), np.zeros(len(ts))
    for s_i, x in zip(slot_of_mode, tm):
        if 0 <= s_i < len(ts):
            mx[s_i] = max(mx[s_i], x)
            sm[s_i] += x
    m.update(ms_int_max_p99=float(np.percentile(ts + mx, 99)), ms_int_sum_p99=float(np.percentile(ts + sm, 99)),
             ms_int_max_p50=float(np.percentile(ts + mx, 50)), int_over50=float(((ts + mx) > 50).mean()))
    if hasattr(core, "la_metrics"):
        lm = core.la_metrics()
        m.update(goodput_kbps=lm["goodput_kbps"], bler=lm["bler"], delivered_mbit=lm["delivered_mbit"])
    if use_mpc:
        m.update(switched=int(mpc.switched), gated=gated, planned=planned, aborts=aborts, t_first_plan=t_first_plan,
                 bias_q=[float(x) for x in np.percentile(biases, [10, 50, 90])] if biases else None)
    return m, AE.deliveries(core)


def job(j):
    ctrl, envr, combo, i, seeds = j[:5]
    extra, ja = (j[5], j[6]) if len(j) > 5 else (None, None)
    pool = j[7] if len(j) > 7 else "test"
    import torch
    torch.set_num_threads(1)
    rows, dl = [], []
    for s in seeds:
        m, d = episode(ctrl, envr, combo, i, s, extra, ja, pool)
        rows.append(m)
        dl.append(d)
    return dict(ctrl_regime=ctrl, env_regime=envr, combo=combo, ckpt=i, seeds=seeds, rows=rows), dl


def needs_ckpt(combo):
    s, m = combo.split("|")
    return s.split("@")[0].split("~")[0] in ("jc", "c", "cm") or         m.replace("+mpcnc", "").replace("+mpc", "") in ("ppo", "ppola", "jm", "cmm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--env-regime", default=None)
    ap.add_argument("--combos", required=True)
    ap.add_argument("--seeds", default="8800:8900")
    ap.add_argument("--procs", type=int, default=6)       # WSL has 6 CPUs: more processes break the 50 ms budget
    ap.add_argument("--env-extra", default=None)
    ap.add_argument("--ja", default=None)
    ap.add_argument("--pool", default="test", choices=("train", "test"))
    a = ap.parse_args()
    extra = json.loads(a.env_extra) if a.env_extra else None
    envr = a.env_regime or a.regime
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    chunks = [seeds[x:x + 25] for x in range(0, len(seeds), 25)]
    jobs = [(a.regime, envr, c, i, ch, extra, a.ja, a.pool) for c in a.combos.split(",")
            for i in (range(3) if needs_ckpt(c) else [-1]) for ch in chunks]
    t0 = time.time()
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    merged, deliv = {}, {}
    for r, d in res:
        key = (r["combo"], r["ckpt"])
        mm = merged.setdefault(key, dict(ctrl_regime=r["ctrl_regime"], env_regime=r["env_regime"], combo=r["combo"],
                                         ckpt=r["ckpt"], seeds=[], rows=[]))
        mm["seeds"] += r["seeds"]; mm["rows"] += r["rows"]
        deliv.setdefault(f"{key[0]}|{key[1]}", []).extend(d)
    out = sorted(merged.values(), key=lambda x: (x["combo"], x["ckpt"]))
    for o in out:
        rr = o["rows"]
        print(f"{o['env_regime']:8s} {o['combo']:16s} ckpt {o['ckpt']:2d} ret {np.mean([x['return'] for x in rr]):7.2f} "
              f"ok {np.mean([x['ok_rate'] for x in rr]):.3f} cochan {np.mean([x['cochannel_air_frac'] for x in rr]):.3f} "
              f"svc {np.mean([x['service'].get('ratio', math.nan) for x in rr]):.2f} "
              f"mode p99 {np.mean([x['ms_mode_p99'] for x in rr]):.1f}", flush=True)
    ck = {}
    for v in ("c", "cm"):
        try:
            ck[v] = [ax2_path(a.regime, v, i) for i in range(3)]
        except OSError:
            pass
    meta = dict(wall_s=time.time() - t0, ctrl_regime=a.regime, env_regime=envr, ax2_ckpts=ck, env_extra=extra, ja=a.ja,
                pool=a.pool)
    json.dump(dict(meta, results=out), open(a.out, "w"), indent=1)
    json.dump(deliv, gzip.open(a.out.replace(".json", ".deliv.json.gz"), "wt"))


if __name__ == "__main__":
    main()
