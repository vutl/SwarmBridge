"""A-X1 pilot (review #36): does learned channel scheduling add value beyond a practical heuristic, together with
adaptive mode selection? No training: the learned scheduler is the CHANNEL head of the A5 joint PPO (trained jointly
with movement / power / mode, best-val checkpoints), with movement and power from the T:best slot rule, i.e. the A5
post-hoc hypothesis (section 4.4: heuristic M+P with PPO C(+Mo) at K5), now re-tested on fresh seeds.

Controls share everything except the decision maker under test: same environment (A5 UavSurveyEnv, regime config),
same test image pool and environment seeds (task list, deadlines, quality floors and fading are drawn from the seed,
independent of the policy), same reward, same movement / power rule (T:best slot part), same causal information
(env observation / statistical CSI).

Combination  SCHED|MODE
  SCHED  tbest           T:best slot rule (its channel rule = EDF, + reuse where T:best uses it)
         sch             SCH* of the regime (ax1_sched, tuned on the train pool before this pilot)
         jc              A5 joint PPO channel head (checkpoint I), movement + power from T:best
         joint           A5 joint PPO movement + channel + power (reference, = A5 'PPPP' slot part)
  MODE   heur            T:best mode rule (budget heuristic)
         ppo             A5 mode-only PPO (checkpoint I; trained with the T:best slot rule)
         jm              A5 joint PPO mode head (checkpoint I; only with jc / joint)
Checkpoint I (0..2) pairs the joint and mode-only seeds; combinations without a PPO part are run once.
Per episode: return and A5 metrics, per-UAV on-time rate (Jain fairness), airtime with co-channel interference,
channel utilisation (airtime / (M x makespan)), decision wall time per slot / mode decision, and the delivered
(image, mode, ok) list for the detector-recall join (a5_fidelity per_image_val.jsonl).
Usage: python ax1_pilot.py OUT --regimes K5,ref --combos tbest|heur,sch|ppo,jc|jm,.. --seeds 8700:8800 [--procs 6]"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import a9_latency as LAT
import ax1_sched as XS
import env_survey as SV
import gym_env_ev as GE
import mode_heur as MH

A5_TRAIN = "/home/uservutl/work/a5"


def joint_path(regime, i):
    spec = [s for s in json.load(open(LAT.A55_FINAL / f"test_ppo_{regime}.json"))["results"][regime]
            if s.startswith(f"P:joint_best_s{i}=")][0]
    return spec[2:].split("=", 1)[1].split(":")[0]


def mode_path(regime, i):
    spec = [s for s in LAT.ppo_specs(regime) if s.startswith(f"P:mode_best_s{i}=")][0]
    return spec[2:].split("=", 1)[1].split(":")[0]


def episode(regime, combo, i, seed):
    sched, mode = combo.split("|")
    cfg = C.regime_cfg(regime)
    sp_t, mp_t = MH.from_name(AE.tuned(regime, "best"), C.MODES)
    sp_s = XS.from_name(XS.tuned(regime), regime)[0] if sched == "sch" else None
    jmod = AE.model(joint_path(regime, i)) if (sched in ("jc", "joint") or mode == "jm") else None
    mmod = AE.model(mode_path(regime, i)) if mode == "ppo" else None
    e = GE.GymUavSemcomEv(AE.G["table"], AE.G["pools"]["test"], cfg, seed=seed, seeds=[seed], core_cls=SV.UavSurveyEnv)
    for mdl in (jmod, mmod):
        if mdl is not None:
            assert mdl.observation_space.shape == e.observation_space.shape
            assert list(mdl.action_space.nvec) == list(e.action_space.nvec)
    obs, _ = e.reset()
    done, tot = False, 0.0
    t_slot, t_mode = [], []
    K = e.K
    while not done:
        t0 = time.perf_counter()
        a = np.zeros(len(e.action_space.nvec), dtype=np.int64)
        if e.phase == "slot":
            if sched == "tbest":
                acts = [x[:3] for x in sp_t(e.core)]
            elif sched == "sch":
                acts = [x[:3] for x in sp_s(e.core)]
            else:
                pa, _ = jmod.predict(obs, deterministic=True, action_masks=e.action_masks())
                pa = np.asarray(pa)
                acts = [tuple(pa[3 * k:3 * k + 3]) for k in range(K)]
                if sched == "jc":
                    h = sp_t(e.core)
                    acts = [(h[k][0], acts[k][1], h[k][2]) for k in range(K)]
            for k in range(K):
                a[3 * k:3 * k + 3] = acts[k]
            t_slot.append(time.perf_counter() - t0)
        else:
            if mode == "heur":
                a[-1] = mp_t(e.core, *e.pending)
            else:
                mdl = mmod if mode == "ppo" else jmod
                pa, _ = mdl.predict(obs, deterministic=True, action_masks=e.action_masks())
                a[-1] = int(np.asarray(pa)[-1])
            t_mode.append(time.perf_counter() - t0)
        obs, r, done, _, info = e.step(a)
        tot += r
    core = e.core
    m = {k: float(info[k]) for k in AE.KEEP if k in info}
    m["return"] = tot
    per_uav = []
    for k in range(K):
        n_k = core.L * core.c["n_wp"]
        per_uav.append(sum(tk.ok for tk in core.finished if tk.uav == k) / n_k)
    x = np.asarray(per_uav)
    air = sum(s[3] - s[2] for s in core.segments)
    air_int = sum(s[3] - s[2] for s in core.segments if s[4] > 0)
    mk = max(m["makespan_s"], 1e-9)
    m.update(jain=float(x.sum() ** 2 / (K * (x ** 2).sum())) if (x ** 2).sum() > 0 else float("nan"),
             min_uav_ok=float(x.min()), cochannel_air_frac=float(air_int / max(air, 1e-9)),
             chan_util=float(air / (core.c["M"] * mk)), ms_slot_mean=float(np.mean(t_slot) * 1e3),
             ms_mode_mean=float(np.mean(t_mode) * 1e3) if t_mode else 0.0,
             ms_decision_p99=float(np.percentile(np.r_[t_slot, t_mode], 99) * 1e3))
    return m, AE.deliveries(core)


def job(j):
    regime, combo, i, seeds = j
    import torch
    torch.set_num_threads(1)
    rows, dl = [], []
    for s in seeds:
        m, d = episode(regime, combo, i, s)
        rows.append(m)
        dl.append(d)
    return dict(regime=regime, combo=combo, ckpt=i, seeds=seeds, rows=rows), dl


def needs_ckpt(combo):
    return combo.split("|")[0] in ("jc", "joint") or combo.split("|")[1] in ("ppo", "jm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--regimes", default="K5,ref")
    ap.add_argument("--combos", required=True)
    ap.add_argument("--seeds", default="8700:8800")
    ap.add_argument("--procs", type=int, default=6)       # WSL has 6 CPUs
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    chunks = [seeds[x:x + 25] for x in range(0, len(seeds), 25)]
    jobs = [(r, c, i, ch) for r in a.regimes.split(",") for c in a.combos.split(",")
            for i in (range(3) if needs_ckpt(c) else [-1]) for ch in chunks]
    t0 = time.time()
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    merged, deliv = {}, {}
    for (r, d) in res:
        key = (r["regime"], r["combo"], r["ckpt"])
        mm = merged.setdefault(key, dict(regime=r["regime"], combo=r["combo"], ckpt=r["ckpt"], seeds=[], rows=[]))
        mm["seeds"] += r["seeds"]; mm["rows"] += r["rows"]
        deliv.setdefault(f"{key[0]}|{key[1]}|{key[2]}", []).extend(d)
    out = sorted(merged.values(), key=lambda x: (x["regime"], x["combo"], x["ckpt"]))
    for o in out:
        print(f"{o['regime']:6s} {o['combo']:12s} ckpt {o['ckpt']:2d} ret {np.mean([x['return'] for x in o['rows']]):7.2f} "
              f"ok {np.mean([x['ok_rate'] for x in o['rows']]):.3f} jain {np.nanmean([x['jain'] for x in o['rows']]):.3f} "
              f"cochan {np.mean([x['cochannel_air_frac'] for x in o['rows']]):.3f} util {np.mean([x['chan_util'] for x in o['rows']]):.3f}",
              flush=True)
    json.dump(dict(wall_s=time.time() - t0, sch={r: XS.tuned(r) for r in a.regimes.split(",")}, results=out),
              open(a.out, "w"), indent=1)
    json.dump(deliv, gzip.open(a.out.replace(".json", ".deliv.json.gz"), "wt"))


if __name__ == "__main__":
    main()
