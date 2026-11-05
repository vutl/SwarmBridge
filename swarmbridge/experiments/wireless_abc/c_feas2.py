"""C v2 (audit fixes 05/11): link management with sweep airtime charged to every sweeping policy, real per-frame
payloads, completion times (for arrival-time-correct tracking), and scheduling variants incl. a diagnostic planner.

Same radio model as c_feas.py (uplink, one RF chain, 100 MHz, UAV 23 dBm, SSB measurement SNR from the SSB budget,
10 ms control steps, per-path Doppler inside a step). Changes:
  * every sweep (broadcast SSB burst, 0.57 ms) is charged in the step where it happens, for EVERY sweeping policy
    (ssb20, ssb20avg every 20 ms, ssb200 every 200 ms, stale once at t = 0); oracle / gps pay nothing;
  * payload per (UAV, frame) = real KF-PV packet size from c_track2.py payload (mapped frames), else the mean;
  * output: completion time tau (s after the frame start) per (UAV, frame), NaN if not complete within the window;
  * scheduling of the single RF chain inside each 200 ms window:
      rr     round-robin over UAVs with pending payload (v1 behaviour)
      greedy serve the pending UAV with the highest ESTIMATED current rate (causal: from the last sweep for ssb
             policies; true rate for the oracle beam)
      plan   diagnostic planner with oracle knowledge of all future rates in the window (least-laxity-first: serve
             the pending UAV whose remaining payload is closest to its remaining future capacity); not a policy
Combinations: beam in {oracle, ssb20, ssb20avg, ssb200, gps, stale} with rr; {oracle, ssb20avg} with greedy and plan.
Output OUT/c_compl.npz (keys '<beam>-<sched>__<ep>', arrays (U, n_frames) of tau) and OUT/c_feas2.json.
Usage: python c_feas2.py TRACE_DIR PAYLOAD_JSON OUT [--first_ep 96] [--step 0.01]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import c_feas as C1
import linear as Lc

N, FRAME, DT_RT = C1.N, C1.FRAME, C1.DT_RT
COMBOS = [(b, "rr") for b in ("oracle", "ssb20", "ssb20avg", "ssb200", "gps", "stale")] + \
         [("oracle", "greedy"), ("oracle", "plan"), ("ssb20avg", "greedy"), ("ssb20avg", "plan")]
PERIOD = {"ssb20": 0.02, "ssb20avg": 0.02, "ssb200": 0.2, "stale": 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("payload"); ap.add_argument("out")
    ap.add_argument("--first_ep", type=int, default=96); ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pay = json.loads(Path(args.payload).read_text())
    meta = json.loads((Path(args.trace) / "meta.json").read_text())
    bs = np.array(meta["bs"])
    F = Lc.dft_codebook(N)
    # steering-sign calibration (as v1)
    agree = {}
    for sg in (1.0, -1.0):
        C1.SIGN, hit, n = sg, 0, 0
        for f in [f for f in sorted(Path(args.trace).glob("ep*.npz")) if int(f.stem[2:]) < 20]:
            d = np.load(f)
            hs, ps, ls = d["a"].sum(2), d["pos"], d["los"].any(-1)
            for u in range(hs.shape[0]):
                for t in range(0, hs.shape[1], 10):
                    if ls[u, t]:
                        n += 1
                        hit += C1.bearing_beam(ps[u, t], bs) == int(np.argmax(np.abs(hs[u, t] @ F.conj())))
        agree[sg] = hit / max(n, 1)
    C1.SIGN = max(agree, key=agree.get)
    rng = np.random.default_rng(0)
    dt = args.step
    spf = int(round(FRAME / dt))
    compl, stats = {}, {f"{b}-{s}": dict(frames=0, delivered=0, nlos_frames=0, nlos_deliv=0, airtime_sweep=0.0)
                         for b, s in COMBOS}
    files = [f for f in sorted(Path(args.trace).glob("ep*.npz")) if int(f.stem[2:]) >= args.first_ep]
    for f in files:
        d = np.load(f)
        a, dop, pos, los = d["a"].astype(np.complex128), d["dop"].astype(np.float64), d["pos"], d["los"].any(-1)
        U, T = a.shape[:2]
        steps = int(round((T - 1) * DT_RT / dt))
        nfr = steps // spf
        sub = np.arange(0, 1.0, 0.1)
        tt = (np.arange(steps)[:, None] + sub[None]) * dt
        kk = np.minimum((tt // DT_RT).astype(int), T - 1)
        dd = tt - kk * DT_RT
        rate = np.zeros((U, steps, N))
        for u in range(U):
            ph = np.exp(2j * np.pi * dop[u][kk] * dd[..., None])
            H = np.einsum("sxk,sxkn->sxn", ph, a[u][kk])
            rate[u] = C1.EFF * C1.BW * np.log2(1 + C1.snr_lin(np.abs(H @ F.conj()) ** 2)).mean(1)
        ep = f.stem
        bits = np.full((U, nfr), float(pay["default_bits"]))
        if ep in pay["episodes"]:
            e = pay["episodes"][ep]
            for u in range(U):
                bits[u, e["offset"]:e["offset"] + len(e["bits"][u])] = e["bits"][u]
        # beams and rate estimates per beam policy
        beams, est, sweep_at = {}, {}, {}
        for bpol in {b for b, _ in COMBOS}:
            beam = np.zeros((U, steps), int)
            erate = np.zeros((U, steps))
            sweeps = np.zeros(steps, bool)
            for u in range(U):
                last = None
                for s in range(steps):
                    t0 = s * dt
                    if bpol == "oracle":
                        beam[u, s] = int(np.argmax(rate[u, s])); erate[u, s] = rate[u, s, beam[u, s]]
                        continue
                    if bpol == "gps":
                        beam[u, s] = C1.bearing_beam(pos[u, min(int(t0 // DT_RT), T - 1)], bs)
                        erate[u, s] = rate[u, s, beam[u, s]]
                        continue
                    per = PERIOD[bpol]
                    if last is None or (t0 + 1e-9) // per != last[0]:
                        h = C1.channel_at(a[u], dop[u], None, t0)
                        g = np.abs(h @ F.conj()) ** 2
                        msnr = C1.snr_lin(g) * 10 ** (C1.MEAS_GAIN_DB / 10)
                        meas = np.abs(np.sqrt(msnr) * np.exp(1j * rng.uniform(0, 2 * np.pi, N)) + Lc.crandn(rng, N)) ** 2
                        avg = meas if (last is None or bpol != "ssb20avg") else 0.3 * meas + 0.7 * last[2]
                        b_ = int(np.argmax(avg))
                        est_snr = max(avg[b_] - 1, 1e-6) / 10 ** (C1.MEAS_GAIN_DB / 10)
                        last = ((t0 + 1e-9) // per, b_, avg, C1.EFF * C1.BW * np.log2(1 + est_snr))
                        sweeps[s] = True
                    beam[u, s], erate[u, s] = last[1], last[3]
            beams[bpol], est[bpol], sweep_at[bpol] = beam, erate, sweeps
        for bpol, sched in COMBOS:
            key = f"{bpol}-{sched}"
            beam, erate, sweeps = beams[bpol], est[bpol], sweep_at[bpol]
            tau = np.full((U, nfr), np.nan)
            for fr in range(nfr):
                pend = {u: bits[u, fr] for u in range(U)}
                for s in range(fr * spf, (fr + 1) * spf):
                    act = [u for u in pend if pend[u] > 0]
                    if not act:
                        break
                    avail = dt - (C1.SSB_S if sweeps[s] else 0.0)
                    stats[key]["airtime_sweep"] += dt - avail
                    if sched == "rr":
                        u = act[s % len(act)]
                    elif sched == "greedy":
                        u = max(act, key=lambda x: erate[x, s])
                    else:                                                     # oracle least-laxity planner
                        def lax(x):
                            fut = sum(rate[x, q, beam[x, q]] * dt for q in range(s, (fr + 1) * spf))
                            return fut - pend[x] if rate[x, s, beam[x, s]] > 0 else 1e18
                        cand = [x for x in act if lax(x) >= 0]
                        u = min(cand, key=lax) if cand else max(act, key=lambda x: rate[x, s, beam[x, s]])
                    r = rate[u, s, beam[u, s]]
                    need = pend[u]
                    pend[u] -= r * avail
                    if pend[u] <= 0:
                        frac = need / max(r, 1e-9)
                        tau[u, fr] = (s - fr * spf) * dt + (dt - avail) + frac
                for u in range(U):
                    k = min(int(fr * FRAME // DT_RT), T - 1)
                    ok = np.isfinite(tau[u, fr])
                    st = stats[key]
                    st["frames"] += 1; st["delivered"] += int(ok)
                    if not los[u, k]:
                        st["nlos_frames"] += 1; st["nlos_deliv"] += int(ok)
            compl[f"{key}__{ep}"] = tau
        print(ep, {k: round(v["delivered"] / max(v["frames"], 1), 3) for k, v in stats.items()}, flush=True)
    np.savez_compressed(out / "c_compl.npz", **compl)
    res = dict(sign_calibration={str(k): v for k, v in agree.items()}, step=dt, payload=args.payload,
               combos={k: dict(v, deliv_frac=round(v["delivered"] / max(v["frames"], 1), 4),
                               nlos_frac=round(v["nlos_deliv"] / max(v["nlos_frames"], 1), 4)) for k, v in stats.items()})
    (out / "c_feas2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["combos"], indent=1))


if __name__ == "__main__":
    main()
