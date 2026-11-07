"""A4 step 2 (heuristics only, no learning): operating regime and heuristic baselines of the photogrammetry survey
workload (env_survey.UavSurveyEnv) on the NATIVE-resolution mode table, at survey speeds v_max = 10 and 20 m/s.

  regime : per-UAV frame rate lambda = v / frame_spacing; expected per-channel rate at the capture points (max power,
           no interference; median / p10 / p90 over test seeds); per mode on the test images: mean payload, PSNR,
           encode time, rho_link = K * lambda * bits / (min(K, M) * R_median), rho_enc = lambda * enc_s;
  baselines : slot heuristics x event-greedy mode policy and every fixed mode, for every information level
           (oracle / oracle_probe / predicted / mean), same 30 test seeds and test images.
Nothing here is tuned; this is the input to A4 training.
Usage: python check_a4.py OUT_JSON --profiles ... --cpu-file ... --modes ... [--cfg JSON] [--wt 0.3]
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import env as E
import env_ev as V
import env_survey as SV
import policies as P

G = {}
SLOT_POLS = ["greedy", "paced_q4", "pacedB1", "pacedB1.5", "pacedB2", "stag_near"]
INFOS = ["oracle", "oracle_probe", "predicted", "mean"]
SEEDS = list(range(1000, 1030))


def init(profiles, modes, cpu_file):
    G["table"] = E.ModeTable(profiles, modes, cpu_file)
    G["tr"], G["te"] = G["table"].split()
    G["modes"] = modes


def job(a):
    cfg, name = a
    if name.startswith("fixed_"):
        j = G["modes"].index(name[6:])
        sp, mp = P.fixed(j), V.fixed_mode(j)
    else:
        sp, mp = P.from_name(name, G["modes"]), V.greedy_mode
    return a, [V.run_episode_ev(SV.UavSurveyEnv(G["table"], G["te"], cfg), sp, mp, s) for s in SEEDS]


def regime(cfg, table, te):
    env = SV.UavSurveyEnv(table, te, cfg)
    c = env.c
    rates = []
    for s in SEEDS:
        env.reset(s)
        for u in env.uavs:
            for w in u.wps:
                g = env.expected_gain(w)
                rates.append(c["B_total"] / c["M"] * np.log2(1 + max(c["p_levels"]) * g / env.noise_w()))
    rates = np.array(rates)
    R = float(np.median(rates))
    lam = c["v_max"] / c["frame_spacing"]
    rows = {}
    for j, m in enumerate(table.modes):
        b = float(table.bits[te, j].mean())
        rows[m] = dict(bits=b, psnr=float(table.psnr[te, j].mean()), enc_s=float(table.enc_s[j]),
                       airtime_s=b / R, rho_link=c["K"] * lam * b / (min(c["K"], c["M"]) * R),
                       rho_enc=lam * float(table.enc_s[j]))
    return dict(v_max=c["v_max"], frames_per_uav=len(env.uavs[0].wps), lambda_uav=lam,
                rate_median=R, rate_p10=float(np.percentile(rates, 10)), rate_p90=float(np.percentile(rates, 90)),
                probe_s=float(table.enc_s[table.probe_j]), modes=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--wt", type=float, default=0.3)
    ap.add_argument("--speeds", default="10,20")
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()
    modes = args.modes.split(",")
    base = json.loads(args.cfg)
    table = E.ModeTable(args.profiles, modes, args.cpu_file)
    _, te = table.split()
    speeds = [float(v) for v in args.speeds.split(",")]
    out = dict(cfg=base, w_time=args.wt, modes=modes, seeds=SEEDS, regime={}, baselines={})
    jobs = []
    for v in speeds:
        cfg = dict(base, v_max=v, w_time=args.wt)
        out["regime"][f"v{v:g}"] = regime(cfg, table, te)
        for info in INFOS:
            for name in SLOT_POLS + ["fixed_" + m for m in modes]:
                jobs.append((dict(cfg, mode_info=info), name))
    with Pool(args.procs, initializer=init, initargs=(args.profiles, modes, args.cpu_file)) as pool:
        res = pool.map(job, jobs)
    for (cfg, name), eps in res:
        key = f"v{cfg['v_max']:g}"
        out["baselines"].setdefault(key, {}).setdefault(cfg["mode_info"], {})[name] = dict(
            ret=float(np.mean([e["return"] for e in eps])), util=float(np.mean([e["mean_util"] for e in eps])),
            ok=float(np.mean([e["ok_rate"] for e in eps])), drop=float(np.mean([e["drop_rate"] for e in eps])),
            q_viol=float(np.mean([e["q_viol_rate"] for e in eps])),
            delay=float(np.nanmean([e["mean_delay"] for e in eps])),
            psnr=float(np.nanmean([e["mean_psnr"] for e in eps])), mission=float(np.mean([e["mission_s"] for e in eps])),
            energy_kj=float(np.mean([e["energy_kj"] for e in eps])), ret_all=[e["return"] for e in eps])
    json.dump(out, open(args.out, "w"), indent=1)
    for key, rg in out["regime"].items():
        print(f"\n== {key}: lambda/UAV {rg['lambda_uav']:.2f}/s, frames/UAV {rg['frames_per_uav']}, R median "
              f"{rg['rate_median'] / 1e6:.2f} Mb/s (p10 {rg['rate_p10'] / 1e6:.2f}, p90 {rg['rate_p90'] / 1e6:.2f}), "
              f"probe {rg['probe_s']:.3f}s")
        for m, r in rg["modes"].items():
            print(f"  {m:24s} {r['bits'] / 1e6:7.3f} Mb  PSNR {r['psnr']:5.2f}  enc {r['enc_s']:6.3f}s  "
                  f"rho_link {r['rho_link']:6.2f}  rho_enc {r['rho_enc']:6.2f}")
        b = out["baselines"][key]
        print(f"  {'policy':28s} " + " ".join(f"{i:>22s}" for i in INFOS))
        for name in b["oracle"]:
            print(f"  {name:28s} " + " ".join(f"{b[i][name]['ret']:+8.2f} ok{b[i][name]['ok']:.2f} {b[i][name]['delay']:.1f}s"
                                              .rjust(22) for i in INFOS))


if __name__ == "__main__":
    main()
