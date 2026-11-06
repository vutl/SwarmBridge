"""Step 0A: is per-task mode selection already solved by a greedy rule? (no simulation of queues)

With independent tasks (no shared transmitter, receiver or decoder), the paper's reward is a sum of per-task terms,
each depending only on that task's own mode and link rates, so choosing the per-task best feasible mode is optimal.
This script asks the remaining question: for the paper's own mode table and channel parameters, WHICH mode wins and
over what range of link rates is the choice non-trivial?

Usage: python step0a.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

import model as M


def best_mode(rate_bps, qmin, dmax, bits):
    """Per-task optimum for a single end-to-end bottleneck rate (two store-and-forward hops at the same rate)."""
    d = M.COMPUTE_S + 2 * bits / rate_bps
    u = np.array([M.utility(M.PSNR[k], d[k], qmin, dmax)[0] for k in range(5)])
    return int(np.argmax(u)), u


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    # 1. link budget implied by the stated parameters
    elev_hi, d_hi = M.leo_elev_dist(0.0)
    elev_lo, d_lo = M.leo_elev_dist(M.PHI_VIS)
    lb = {}
    for noise in ("paper", "thermal"):
        n = M.noise_power(noise)
        lb[noise] = dict(
            noise_dbm=10 * np.log10(n) + 30,
            snr_db_zenith=10 * np.log10(M.leo_uav_snr_mean(d_hi, n)),
            snr_db_40deg=10 * np.log10(M.leo_uav_snr_mean(d_lo, n)),
            mean_rate_mbps_zenith=M.expected_log_rate(M.leo_uav_snr_mean(d_hi, n)) / 1e6,
            mean_rate_mbps_40deg=M.expected_log_rate(M.leo_uav_snr_mean(d_lo, n)) / 1e6)
    res["leo_uav_link"] = lb
    res["outdated_csi_delta_zenith"] = M.outdated_csi_delta(d_hi)
    res["access_time_s"] = M.access_time()
    res["slant_range_km"] = dict(zenith=d_hi / 1e3, elev40=d_lo / 1e3)
    res["isl_rate_mbps"] = {f"{km}km": M.isl_rate(km * 1e3) / 1e6 for km in (250, 375, 500)}
    # UAV-user link: user at horizontal distance x, interferers at 500 m / 1000 m (adjacent areas)
    def uu(x, inter):
        dist = np.hypot(x, M.UAV_ALT)
        ig = [M.uav_user_gain(np.hypot(xi, M.UAV_ALT), 0.0) for xi in inter]
        return M.uav_user_rate(dist, 0.0, ig, M.noise_power("paper")) / 1e6
    res["uav_user_rate_mbps"] = {f"x{x}m_{tag}": uu(x, inter) for x in (50, 200, 400)
                                 for tag, inter in (("no_interf", []), ("2interf", [500 + x, 1000 + x]))}
    # 2. dominance: rate above which M1 beats every generative mode in BOTH quality and delay
    dom = {}
    for payload in ("fig5", "original"):
        bits = M.mode_bits(payload)
        # M1 delay <= Mk delay  <=>  2*(b1-bk)/R <= ck - c1
        thr = max(2 * (bits[0] - bits[k]) / (M.COMPUTE_S[k] - M.COMPUTE_S[0]) for k in range(1, 5))
        dom[payload] = dict(m1_bits=float(bits[0]), m1_dominates_above_mbps=thr / 1e6)
    res["m1_dominance"] = dom
    # 3. mode map over rate for the paper's task distribution (Qmin U[10,30], Dmax U[2,11])
    rng = np.random.default_rng(0)
    qmin = rng.uniform(*M.QMIN_RANGE, 20000)
    dmax = rng.uniform(*M.DMAX_RANGE, 20000)
    rates = np.geomspace(1e4, 1e8, 41)
    maps = {}
    for payload in ("fig5", "original"):
        bits = M.mode_bits(payload)
        rows = []
        for r in rates:
            ch = np.zeros(5)
            ub, uf, sat = [], [], 0
            for q, d in zip(qmin[:4000], dmax[:4000]):
                k, u = best_mode(r, q, d, bits)
                ch[k] += 1
                ub.append(u[k])
                uf.append(u.max())
                sat += M.utility(M.PSNR[k], M.COMPUTE_S[k] + 2 * bits[k] / r, q, d)[1]
            fixed = []
            for k in range(5):
                dk = M.COMPUTE_S[k] + 2 * bits[k] / r
                fixed.append(float(np.mean([M.utility(M.PSNR[k], dk, q, d)[0] for q, d in zip(qmin[:4000], dmax[:4000])])))
            rows.append(dict(rate_mbps=r / 1e6, choice_frac=(ch / ch.sum()).round(3).tolist(),
                             greedy_utility=float(np.mean(ub)), best_fixed_mode=M.MODES[int(np.argmax(fixed))],
                             best_fixed_utility=max(fixed), satisfied=sat / 4000))
        maps[payload] = rows
    res["mode_map"] = maps
    (out / "step0a.json").write_text(json.dumps(res, indent=1))
    # short text summary
    lines = ["# Step 0A (Huang-inspired, per-task, no queues)", ""]
    for k, v in lb.items():
        lines.append(f"- LEO->UAV, noise {k} ({v['noise_dbm']:.1f} dBm): SNR {v['snr_db_zenith']:.1f} dB zenith / "
                     f"{v['snr_db_40deg']:.1f} dB at 40 deg; mean rate {v['mean_rate_mbps_zenith']:.1f} / "
                     f"{v['mean_rate_mbps_40deg']:.1f} Mbit/s")
    for k, v in dom.items():
        lines.append(f"- M1 payload {k} ({v['m1_bits']/1e3:.0f} kbit): M1 has higher PSNR and lower delay than every "
                     f"generative mode when the rate exceeds {v['m1_dominates_above_mbps']:.3f} Mbit/s")
    lines.append(f"- outdated-CSI delta at zenith {res['outdated_csi_delta_zenith']:.4f}; pass {res['access_time_s']:.0f} s")
    lines += ["", "| payload | rate Mbit/s | greedy utility | best fixed | best fixed utility | choice M1/M2/M3_1/M3_2/M3_3 |",
              "|---|---|---|---|---|---|"]
    for payload, rows in maps.items():
        for r in rows[::4]:
            lines.append(f"| {payload} | {r['rate_mbps']:.3g} | {r['greedy_utility']:.3f} | {r['best_fixed_mode']} | "
                         f"{r['best_fixed_utility']:.3f} | {r['choice_frac']} |")
    (out / "step0a.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
