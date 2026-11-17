"""Review #39 step B: evaluator for the link-adaptation env (env_la).

  gate-tune REGIME : the NACK gate (la_heur) on top of the regime's tuned reuse heuristic, 9 members (theta x cool),
                     TRAIN pool seeds 0..29; writes runs/uav_semcom/ax5/gate_REGIME.json (selected member + its
                     theta / cool, which are reused for every gated policy evaluated with --gate-regime REGIME).
  eval OUT         : policies x env regimes on the TEST pool, fresh seeds (default 8800..8899), paired.
Policy specs:
  H:NAME             heuristic (slot + mode rule) by mode_heur name
  T:FAM@REG          tuned member FAM (best | reuse | ...) of regime REG
  C:REG              A-X2 C-only channel scheduler trained on REG (3 checkpoints, best val), with REG's T:best
                     movement / power / modes; observations truncated to the model's input size (older controllers
                     trained without the link-adaptation features see the unchanged observation prefix)
  any spec + "+gate" the NACK gate with the --gate-regime's tuned theta / cool applied to the slot decisions
Per episode: A5 metrics, link metrics (BLER, BLER of shared blocks, shared-block fraction, failed-airtime fraction,
goodput, delivered Mbit), co-channel airtime, chan util, fairness, gate flags, slot decision latency, and the HIDDEN
link-adaptation parameters of the episode (only for the adaptation analysis; never observed by any policy).
Usage: python ax5_eval.py gate-tune K5_la_rand
       python ax5_eval.py eval runs/uav_semcom/ax5/eval.json --env-regimes K5_la,K5_la_good --policies ... --procs 6
"""
import argparse
import gzip
import json
import time
from multiprocessing import Pool

import numpy as np

import a5_common as C
import a5_eval as AE
import ax2_env as X
import ax2_eval as XE
import env_ev as V
import env_survey as SV
import la_heur as LH
import mode_heur as MH

OUT = C.REPO / "runs" / "uav_semcom" / "ax5"


def gate_params(regime):
    d = json.loads((OUT / f"gate_{regime}.json").read_text())
    return d["theta"], d["cool"]


def _metrics(core, info_ret, t_slot):
    m = core.metrics()
    m["return"] = info_ret
    m.update(core.la_metrics())
    K = len(core.uavs)
    per_uav = np.asarray([sum(tk.ok for tk in core.finished if tk.uav == k) / (core.L * core.c["n_wp"]) for k in range(K)])
    air = sum(s[3] - s[2] for s in core.segments)
    m.update(jain=float(per_uav.sum() ** 2 / (K * (per_uav ** 2).sum())) if (per_uav ** 2).sum() > 0 else float("nan"),
             min_uav_ok=float(per_uav.min()),
             cochannel_air_frac=float(sum(s[3] - s[2] for s in core.segments if s[4] > 0) / max(air, 1e-9)),
             chan_util=float(air / (core.c["M"] * max(m["makespan_s"], 1e-9))),
             gate_flags=int(core.__dict__.get("_gate", {}).get("flags", 0)))
    if t_slot:
        ts = np.asarray(t_slot) * 1e3
        m.update(ms_slot_mean=float(ts.mean()), ms_slot_p99=float(np.percentile(ts, 99)))
    return m


def episode(spec, env_regime, i, seed, gate_regime):
    gate = spec.endswith("+gate")
    base = spec[:-5] if gate else spec
    th, cool = gate_params(gate_regime) if gate else (None, None)
    cfg = C.regime_cfg(env_regime)
    if base.startswith(("H:", "T:")):
        if base.startswith("H:"):
            name = base[2:]
        else:
            fam, reg = base[2:].split("@")
            name = AE.tuned(reg, fam)
        sp, mp = MH.from_name(name, C.MODES)
        if gate:
            sp = LH.nack_gate(sp, th, cool)
        core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["test"], cfg)
        r = V.run_episode_ev(core, sp, mp, seed)
        return _metrics(core, r["return"], None), AE.deliveries(core)
    assert base.startswith("C:")
    reg = base[2:]
    sp, mp = MH.from_name(AE.tuned(reg, "best"), C.MODES)
    e = X.GymSchedEv(AE.G["table"], AE.G["pools"]["test"], cfg, seed=seed, seeds=[seed], core_cls=SV.UavSurveyEnv,
                     sp_heur=sp, mp_heur=mp, learn_mode=False)
    model = AE.model(XE.ax2_path(reg, "c", i))
    n_obs = model.observation_space.shape[0]
    assert n_obs <= e.observation_space.shape[0] and list(model.action_space.nvec) == list(e.action_space.nvec)
    obs, _ = e.reset()
    done, tot, t_slot = False, 0.0, []
    while not done:
        t0 = time.perf_counter()
        a, _ = model.predict(obs[:n_obs], deterministic=True, action_masks=e.action_masks())
        a = np.asarray(a, dtype=np.int64).copy()
        if gate:
            acts = LH.apply_gate(e.core, [tuple(a[3 * k:3 * k + 3]) for k in range(e.K)], th, cool)
            for k in range(e.K):
                a[3 * k + 1] = acts[k][1]
        t_slot.append(time.perf_counter() - t0)
        obs, r, done, _, info = e.step(a)
        tot += r
    return _metrics(e.core, tot, t_slot), AE.deliveries(e.core)


def job(j):
    spec, envr, i, seeds, gate_regime = j
    import torch
    torch.set_num_threads(1)
    rows, dl = [], []
    for s in seeds:
        m, d = episode(spec, envr, i, s, gate_regime)
        rows.append(m)
        dl.append(d)
    return dict(spec=spec, env_regime=envr, ckpt=i, seeds=seeds, rows=rows), dl


def _tune_job(j):
    regime, name, seed = j
    sp, mp = MH.from_name(name, C.MODES)
    core = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["train"], C.regime_cfg(regime))
    r = V.run_episode_ev(core, sp, mp, seed)
    return name, r["return"], core.la_metrics()["shared_block_frac"]


def gate_tune(regime, procs):
    base = AE.tuned(regime, "reuse")
    names = [base] + LH.gate_family(base) + [AE.tuned(regime, "best")]
    with Pool(procs, initializer=AE.init, initargs=(False,)) as p:
        res = p.map(_tune_job, [(regime, n, s) for n in names for s in C.TRAIN_SEEDS], chunksize=4)
    rows = {n: dict(ret=float(np.mean([r for nn, r, _ in res if nn == n])),
                    shared=float(np.mean([s for nn, _, s in res if nn == n]))) for n in names}
    fam = LH.gate_family(base)
    sel = max(fam, key=lambda n: rows[n]["ret"])
    _, th, cool = LH.parse(sel)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate_{regime}.json").write_text(json.dumps(dict(regime=regime, base=base, seeds=C.TRAIN_SEEDS, rows=rows,
                                                             selected=sel, theta=th, cool=cool), indent=1))
    for n in names:
        print(f"{n:48s} ret {rows[n]['ret']:8.2f} shared {rows[n]['shared']:.3f}" + ("  <- selected" if n == sel else ""))


def run_eval(a):
    lo, hi = map(int, a.seeds.split(":"))
    seeds = list(range(lo, hi))
    chunks = [seeds[x:x + 25] for x in range(0, len(seeds), 25)]
    specs = a.policies.split(",")
    jobs = [(sp, envr, i, ch, a.gate_regime) for envr in a.env_regimes.split(",") for sp in specs
            for i in (range(3) if sp.startswith("C:") else [-1]) for ch in chunks]
    t0 = time.time()
    with Pool(a.procs, initializer=AE.init, initargs=(False,)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    merged, deliv = {}, {}
    for r, d in res:
        key = (r["env_regime"], r["spec"], r["ckpt"])
        mm = merged.setdefault(key, dict(env_regime=key[0], spec=key[1], ckpt=key[2], seeds=[], rows=[]))
        mm["seeds"] += r["seeds"]
        mm["rows"] += r["rows"]
        deliv.setdefault("|".join(map(str, key)), []).extend(d)
    out = sorted(merged.values(), key=lambda x: (x["env_regime"], x["spec"], x["ckpt"]))
    for o in out:
        rr = o["rows"]
        f = lambda k: np.mean([x[k] for x in rr])
        print(f"{o['env_regime']:12s} {o['spec']:28s} ckpt {o['ckpt']:2d} ret {f('return'):8.2f} ok {f('ok_rate'):.3f} "
              f"bler {f('bler'):.3f} shbler {f('shared_bler'):.3f} cochan {f('cochannel_air_frac'):.3f} "
              f"goodput {f('goodput_kbps'):6.1f}", flush=True)
    json.dump(dict(wall_s=time.time() - t0, gate_regime=a.gate_regime,
                   gate=(gate_params(a.gate_regime) if (OUT / f"gate_{a.gate_regime}.json").exists() else None),
                   results=out), open(a.out, "w"), indent=1)
    json.dump(deliv, gzip.open(a.out.replace(".json", ".deliv.json.gz"), "wt"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("gate-tune")
    t.add_argument("regime")
    t.add_argument("--procs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("out")
    e.add_argument("--env-regimes", required=True)
    e.add_argument("--policies", required=True)
    e.add_argument("--seeds", default="8800:8900")
    e.add_argument("--gate-regime", default="K5_la_rand")
    e.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "gate-tune":
        gate_tune(a.regime, a.procs)
    else:
        run_eval(a)


if __name__ == "__main__":
    main()
