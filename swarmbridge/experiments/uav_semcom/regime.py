"""Operating-regime analysis for the A2 v2 environment: offered traffic vs service capacity, from measured payloads.

For a configuration it reports
  * link: expected per-channel rate at the survey waypoints (max power, no interference) - median / p10 / p90;
  * per mode: mean payload of the test images, airtime at the median rate, encode / decode time, PSNR;
  * load: capture rate lambda (images/s) when every UAV flies its survey at v_max, and for each mode
      rho_link = lambda_total * bits / (min(K, M) * R_median)   (channels are the shared server)
      rho_enc  = lambda_uav * enc_s                              (each UAV's serial encoder)
    rho < 1 means the mode can be sustained at full survey speed; rho << 1 means payload size barely matters;
  * the same for bandwidths used in the closest papers, and for native-resolution frames (probe file);
  * quality of the 'predicted' mode-information model on the test images.
Usage: python regime.py OUT_MD --profiles ... --cpu-file ... --modes ... [--native PROBE.jsonl] [--cfg JSON]
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

import env as E


def rates(cfg, table, pool, n_ep=200):
    env = E.UavSemcomEnv(table, pool, cfg)
    c = env.c
    out = []
    for s in range(n_ep):
        env.reset(s)
        for u in env.uavs:
            for w in u.wps:
                g = env.expected_gain(w)
                out.append(c["B_total"] / c["M"] * math.log2(1 + c["p_levels"][-1] * g / env.noise_w()))
    return np.array(out), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--cpu-file")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--native")
    ap.add_argument("--cfg", default="{}")
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    modes = args.modes.split(",")
    tab = E.ModeTable(args.profiles, modes, args.cpu_file)
    tr, te = tab.split()
    R, c = rates(cfg, tab, te)
    Rm = float(np.median(R))
    K, M, L = c["K"], c["M"], c["tiles_per_wp"]
    t_wp = c["wp_spacing"] / c["v_max"]                    # seconds between waypoints at full speed
    lam_uav = L / t_wp
    lam = K * lam_uav
    servers = min(K, M)
    lines = ["# Operating regime (env v2)", "",
             f"Config: K={K} UAVs, M={M} channels, B_total={c['B_total']/1e6:g} MHz, p_max={c['p_levels'][-1]} W, "
             f"H={c['H']} m, survey radius {c['sector_radius']} m, {L} images per waypoint, waypoint spacing "
             f"{c['wp_spacing']} m, v_max={c['v_max']} m/s, deadline U{c['deadline']} s.", "",
             f"Per-channel rate at waypoints (max power, no interference): median {Rm/1e6:.2f} Mb/s, "
             f"p10 {np.percentile(R,10)/1e6:.2f}, p90 {np.percentile(R,90)/1e6:.2f}.", "",
             f"Capture rate at full survey speed: {lam_uav:.2f} images/s per UAV, {lam:.2f} images/s in total.", "",
             "| mode | kbit (test mean) | airtime @ median (s) | enc (s) | dec (s) | PSNR (dB) | rho_link | rho_enc |",
             "|---|---|---|---|---|---|---|---|"]
    for j, m in enumerate(modes):
        b = tab.bits[te, j].mean()
        lines.append(f"| {m} | {b/1e3:.1f} | {b/Rm:.3f} | {tab.enc_s[j]:.3f} | {tab.dec_s[j]:.3f} | "
                     f"{tab.psnr[te, j].mean():.1f} | {lam*b/(servers*Rm):.2f} | {lam_uav*tab.enc_s[j]:.2f} |")
    # bandwidths of the closest papers (noise scales with bandwidth)
    top = int(np.argmax(tab.psnr[te].mean(0)))
    lines += ["", f"Highest-quality mode ({modes[top]}) under the bandwidths of the closest papers "
              "(same geometry, same per-channel split):", "",
              "| B_total | source | median rate / channel (Mb/s) | airtime (s) | rho_link |", "|---|---|---|---|---|"]
    for B, src in ((1e6, "this env (assumed)"), (5e6, "Si et al. 2023"), (10e6, "Huang et al. 2025"),
                   (20e6, "Li et al. 2026")):
        Rb, _ = rates(dict(cfg, B_total=B), tab, te, n_ep=50)
        rb = float(np.median(Rb))
        b = tab.bits[te, top].mean()
        lines.append(f"| {B/1e6:g} MHz | {src} | {rb/1e6:.2f} | {b/rb:.3f} | {lam*b/(servers*rb):.2f} |")
    # how many UAVs per channel would saturate the link with the top mode at full survey speed
    b = tab.bits[te, top].mean()
    k_sat = M * Rm / (lam_uav * b)
    lines += ["", f"UAVs per {M} channels at which {modes[top]} saturates the link at full survey speed "
              f"(rho_link = 1): K ~ {k_sat:.1f}."]
    if args.native:
        rows = [json.loads(l) for l in open(args.native)]
        by = defaultdict(list)
        for r in rows:
            by[r["mode"]].append(r)
        sizes = defaultdict(int)
        for r in by[next(iter(by))]:
            sizes[f"{r['w']}x{r['h']}"] += 1
        lines += ["", f"Native-resolution probe ({len(by[next(iter(by))])} VisDrone-DET-val images, sizes "
                  f"{dict(sizes)}; CPU encode on the A1 machine), one frame per waypoint instead of {L} 512x512 crops:",
                  "", "| mode | kbit / frame | PSNR | enc (s) | airtime @ median (s) | rho_link (1 frame / waypoint) |",
                  "|---|---|---|---|---|---|"]
        for m, v in by.items():
            kb = np.mean([r["bytes"] for r in v]) * 8
            lines.append(f"| {m} | {kb/1e3:.0f} | {np.mean([r['psnr'] for r in v]):.1f} | "
                         f"{np.mean([r['enc_s'] for r in v]):.2f} | {kb/Rm:.3f} | {K*(1/t_wp)*kb/(servers*Rm):.2f} |")
    # predictor quality ('predicted' mode information)
    if tab.probe_j is not None:
        lines += ["", f"'predicted' mode information (per-mode linear fit on log bits of the probe mode "
                  f"{modes[tab.probe_j]}, fitted on train images), test images:", "",
                  "| mode | R2 log bits | R2 PSNR | MAE PSNR (dB) |", "|---|---|---|---|"]
        P = np.array([tab.predict(i)[0] for i in te])
        Q = np.array([tab.predict(i)[1] for i in te])
        for j, m in enumerate(modes):
            if j == tab.probe_j:
                continue
            yb, yq = np.log(tab.bits[te, j]), tab.psnr[te, j]
            r2b = 1 - np.sum((yb - np.log(P[:, j])) ** 2) / np.sum((yb - yb.mean()) ** 2)
            r2q = 1 - np.sum((yq - Q[:, j]) ** 2) / np.sum((yq - yq.mean()) ** 2)
            lines.append(f"| {m} | {r2b:.2f} | {r2q:.2f} | {np.mean(np.abs(yq - Q[:, j])):.2f} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
