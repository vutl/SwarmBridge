"""Review #39: PHY validation of the link-adaptation model (env_la) before any RL experiment.

Part 1 (standalone link, no env): BLER and goodput / genie of the OLLA loop for true SINR x CQI error x CQI bias x
  feedback delay, (a) stationary SINR, (b) block fading that redraws the SINR every 10 blocks (= the env's 1 s slot
  fading; Rician K = 10 dB LoS or Rayleigh NLoS around the mean), to isolate the cost of stale reports at slot edges.
Part 2 (env): the same heuristics on K5_mcsg (genie MCS) and on the link-adaptation conditions, TRAIN pool seeds
  0..19 (no test seeds): return, ok rate, link goodput (delivered bits / airtime), BLER, BLER of shared blocks, failed
  airtime fraction, co-channel airtime.
Writes runs/uav_semcom/ax5/phy_validation.json.  Usage: python ax5_phy_validate.py [--procs 6]
"""
import argparse
import json
import math
from multiprocessing import Pool

import numpy as np

import a5_common as C
import env_la as LA

OUT = C.REPO / "runs" / "uav_semcom" / "ax5"


def link(gamma_db, sigma, bias, delay, step=0.5, fading=None, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    ad = LA.LinkAdapter(0.1, step, delay, 10.0)
    nack, good, genie = 0, 0.0, 0.0
    g = gamma_db
    for b in range(n):
        if fading is not None and b % 10 == 0:
            if fading == "rayleigh":
                g = gamma_db + 10 * math.log10(max(rng.exponential(), 1e-6))
            else:
                kf = 10.0
                h = math.sqrt(kf / (kf + 1)) + math.sqrt(1 / (kf + 1)) * complex(*rng.normal(size=2)) / math.sqrt(2)
                g = gamma_db + 10 * math.log10(abs(h) ** 2)
        rep = ad.report(b)
        est = (rep[1] if rep else gamma_db) - ad.off
        i = LA.cqi_for(est)
        ok = rng.random() < 1 - LA.bler(g, i)
        ad.feedback(not ok)
        ad.meas.append((b, g + bias + sigma * rng.normal(), 0.0))
        nack += not ok
        good += ok * LA.EFF[i]
        genie += max(LA.EFF[j] * (1 - LA.bler(g, j)) for j in range(len(LA.THR)))
    return dict(bler=nack / n, eff_vs_genie=good / genie)


def part1():
    rows = []
    for fading in (None, "rician", "rayleigh"):
        for g in (2.0, 8.0, 15.0):
            for sigma in (0.0, 1.0, 3.0):
                for bias in (-2.0, 0.0, 4.0):
                    for delay in (1, 10, 20):
                        r = link(g, sigma, bias, delay, fading=fading, n=6000)
                        rows.append(dict(fading=fading or "none", gamma_db=g, sigma=sigma, bias=bias, delay=delay, **r))
    return rows


def _env_job(j):
    regime, name, seed, kw = j
    import a5_eval as AE
    import env_ev as V
    import env_survey as SV
    import mode_heur as MH
    e = SV.UavSurveyEnv(AE.G["table"], AE.G["pools"]["train"], C.regime_cfg(regime, **kw))
    sp, mp = MH.from_name(name, C.MODES)
    m = V.run_episode_ev(e, sp, mp, seed)
    air = sum(s[3] - s[2] for s in e.segments)
    out = dict(regime=regime, name=name, seed=seed, kw=kw, ret=m["return"], ok=m["ok_rate"],
               goodput_kbps=sum((s[3] - s[2]) * s[5] for s in e.segments) / max(air, 1e-9) / 1e3,
               cochan=sum(s[3] - s[2] for s in e.segments if s[4] > 0) / max(air, 1e-9), airtime=air)
    if hasattr(e, "la_metrics"):
        lm = e.la_metrics()
        out.update(bler=lm["bler"], shared_bler=lm["shared_bler"], fail_air=lm["fail_air_frac"])
    return out


def part2(procs):
    import a5_eval as AE
    regs = [("K5_mcsg", {}), ("K5_la", {}), ("K5_la_good", {}), ("K5_la_slow", {}), ("K5_la_noisy", {}),
            ("K5_la_bad", {}), ("K5_la", dict(la_fb_delay=1, la_cqi_sigma=0.0)), ("K5_la", dict(la_olla_step=1e-6)),
            ("K5_la_l50", {})]
    names = ("fixed_mbt2018-mean_q3", "fixed_mbt2018-mean_q1", "pacedB2+budget1+reuse")
    jobs = [(r, n, s, kw) for r, kw in regs for n in names for s in range(20)]
    with Pool(procs, initializer=AE.init, initargs=(False,)) as p:
        res = p.map(_env_job, jobs, chunksize=4)
    agg = []
    for r, kw in regs:
        for n in names:
            xs = [x for x in res if x["regime"] == r and x["name"] == n and x["kw"] == kw]
            a = {k: float(np.mean([x[k] for x in xs])) for k in ("ret", "ok", "goodput_kbps", "cochan")}
            for k in ("bler", "shared_bler", "fail_air"):
                if k in xs[0]:
                    a[k] = float(np.mean([x[k] for x in xs]))
            agg.append(dict(regime=r, kw=kw, name=n, **a))
            print(f"{r:11s} {str(kw):40s} {n:24s} ret {a['ret']:8.2f} ok {a['ok']:.3f} goodput {a['goodput_kbps']:7.1f} "
                  f"cochan {a['cochan']:.2f}" + (f" bler {a['bler']:.3f} shared {a['shared_bler']:.3f} failair "
                                                 f"{a['fail_air']:.3f}" if "bler" in a else ""), flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    p1 = part1()
    for f in ("none", "rician", "rayleigh"):
        xs = [r for r in p1 if r["fading"] == f]
        print(f"part1 fading {f:8s} BLER range {min(r['bler'] for r in xs):.3f}..{max(r['bler'] for r in xs):.3f}  "
              f"eff/genie median {np.median([r['eff_vs_genie'] for r in xs]):.3f} "
              f"(delay 1: {np.median([r['eff_vs_genie'] for r in xs if r['delay'] == 1]):.3f}, "
              f"delay 20: {np.median([r['eff_vs_genie'] for r in xs if r['delay'] == 20]):.3f})", flush=True)
    p2 = part2(a.procs)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(part1=p1, part2=p2), open(OUT / "phy_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
