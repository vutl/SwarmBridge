"""C feasibility (gate before any RL): does link management change message delivery for UAV3D-sized payloads? (CPU)

Uplink UAV -> BS (TDD reciprocity, ideal calibration declared), BS 16-ant ULA with ONE RF chain, 28 GHz, 100 MHz.
Link budget: UAV Pt 23 dBm, noise -174 dBm/Hz + 80 dB + NF 7 dB = -87 dBm; SNR = Pt |w^H h|^2 / N0 with h the raw
ray-traced channel (per-antenna gains, array gain through w, unit norm).
Time: control step 10 ms (sensitivity 5 ms); perception frame 200 ms; inside a step the channel evolves from the nearest
RT sample by per-path Doppler phase rotation (h(t+d) = sum_p a_p exp(j 2 pi f_p d)), evaluated every 1 ms and the step
rate is the mean over the step (fast fading is not frozen for 10 ms).
Payload per helper per frame: PAYLOAD_BITS (UAV3D KF-PV full message ~ 41 kbit / 4 helpers ~ 10.3 kbit), coding rate
0.75 on top of Shannon rate x 0.6 implementation loss. Delivered if the cumulative bits within the 200 ms frame reach the
payload; otherwise the frame's message is lost (the receiver keeps its cached older message).
Policies (beam for each UAV at each step; one RF chain: serving one UAV per step, round-robin among UAVs with pending
payload; a beam sweep is a broadcast SSB burst of 16 beams x 4 OFDM symbols (120 kHz SCS, 8.93 us) = 0.57 ms):
  oracle      best beam at every 1 ms with true CSI (upper reference)
  ssb20       full sweep every 20 ms (noisy measurement at the sweep time), keep the best measured beam until next
  ssb200      full sweep every 200 ms
  gps         geometric LoS beam from the true UAV and BS positions (no pilots)
  ssb20avg    full sweep every 20 ms, beam = argmax of an exponential average (weight 0.3 on the newest sweep) of
              the measured beam powers (smooths fast fading; strongest simple heuristic)
  stale       one sweep at the start of the episode, never updated
Per-frame delivery for every (episode, UAV, frame) and policy is saved to OUT/c_deliv.npz for the tracking stage.
Output OUT/c_feas.json: delivered fraction per policy (all / LoS-only / NLoS-only frames), beam coherence statistics.
Usage: python c_feas.py TRACE_DIR OUT [--first_ep 96]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import linear as Lc

N, FC, BW = 16, 28e9, 100e6
PT_DBM, N0_DBM = 23.0, -174 + 80 + 7
PAYLOAD_BITS = 10300
EFF = 0.75 * 0.6
FRAME, DT_RT = 0.2, 0.1
SSB_S = 16 * 4 * 8.93e-6


def snr_lin(gain):
    return 10 ** ((PT_DBM - N0_DBM) / 10) * gain


def channel_at(a, dop, t_rt, dt):
    """a (T, K, N), dop (T, K): channel at absolute time t from the nearest RT sample at or before t."""
    k = min(int(dt // DT_RT), a.shape[0] - 1)
    d = dt - k * DT_RT
    return (a[k] * np.exp(2j * np.pi * dop[k] * d)[:, None]).sum(0)


SIGN = 1.0                                             # steering sign convention, calibrated on train LoS samples


def bearing_beam(pos, bs):
    v = pos - bs
    # ULA along x (Sionna PlanarArray columns are along the y axis of the device frame; device orientation 0 ->
    # array along y). Use the angle relative to the array axis (y): sin(theta) = v_y / |v|.
    s = SIGN * v[1] / np.linalg.norm(v)
    F = Lc.dft_codebook(N)
    return int(np.argmax(np.abs(Lc.ula_steering(N, s) @ F.conj()) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--first_ep", type=int, default=96)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(args.trace) / "meta.json").read_text())
    bs = np.array(meta["bs"])
    F = Lc.dft_codebook(N)
    rng = np.random.default_rng(0)
    # calibrate the steering sign: on train-episode LoS samples, which sign makes the geometric beam the best beam?
    global SIGN
    agree = {}
    for sg in (1.0, -1.0):
        SIGN, hit, n = sg, 0, 0
        for f in [f for f in sorted(Path(args.trace).glob("ep*.npz")) if int(f.stem[2:]) < 20]:
            d = np.load(f)
            hs, ps, ls = d["a"].sum(2), d["pos"], d["los"].any(-1)
            for u in range(hs.shape[0]):
                for t in range(0, hs.shape[1], 10):
                    if ls[u, t]:
                        n += 1
                        hit += bearing_beam(ps[u, t], bs) == int(np.argmax(np.abs(hs[u, t] @ F.conj())))
        agree[sg] = hit / max(n, 1)
    SIGN = max(agree, key=agree.get)
    print("steering sign calibration (LoS agreement):", agree, "->", SIGN, flush=True)
    pols = ["oracle", "ssb20", "ssb20avg", "ssb200", "gps", "stale"]
    deliv = {}
    stats = {p: dict(frames=0, delivered=0, los_frames=0, los_deliv=0, nlos_frames=0, nlos_deliv=0) for p in pols}
    coh, gps_ok = [], []
    files = [f for f in sorted(Path(args.trace).glob("ep*.npz")) if int(f.stem[2:]) >= args.first_ep]
    for f in files:
        d = np.load(f)
        a, dop, pos, los = d["a"].astype(np.complex128), d["dop"].astype(np.float64), d["pos"], d["los"].any(-1)
        U, T = a.shape[:2]
        dur = (T - 1) * DT_RT
        steps = int(round(dur / args.step))
        sub = np.arange(0, 1.0, 0.1)                       # 10 sub-samples inside each step
        # pre-compute per UAV, per step: gain of every beam at sub-samples -> mean rate per beam
        rate = np.zeros((U, steps, N))
        best_inst = np.zeros((U, steps), int)
        tt = (np.arange(steps)[:, None] + sub[None]) * args.step                  # (steps, 10)
        kk = np.minimum((tt // DT_RT).astype(int), T - 1)
        dd = tt - kk * DT_RT
        for u in range(U):
            ph = np.exp(2j * np.pi * dop[u][kk] * dd[..., None])                    # (steps, 10, K)
            H = np.einsum("sxk,sxkn->sxn", ph, a[u][kk])                            # (steps, 10, N)
            g = np.abs(H @ F.conj()) ** 2                                           # (steps, 10, B)
            rate[u] = EFF * BW * np.log2(1 + snr_lin(g)).mean(1)
            best_inst[u] = np.argmax(g[:, 0], 1)
        # beam coherence: run lengths of the best beam (within 3 dB of best) at the step grid
        for u in range(U):
            r = rate[u]
            cur, run = best_inst[u, 0], 0
            for s in range(steps):
                if r[s, cur] >= r[s].max() * 0.5:
                    run += 1
                else:
                    coh.append(run * args.step); cur, run = best_inst[u, s], 1
            coh.append(run * args.step)
        # policies
        for p in pols:
            beam = np.zeros((U, steps), int)
            for u in range(U):
                last = None
                for s in range(steps):
                    t0 = s * args.step
                    if p == "oracle":
                        beam[u, s] = int(np.argmax(rate[u, s]))
                        continue
                    if p == "gps":
                        beam[u, s] = bearing_beam(pos[u, min(int(t0 // DT_RT), T - 1)], bs)
                        continue
                    period = {"ssb20": 0.02, "ssb20avg": 0.02, "ssb200": 0.2, "stale": 1e9}[p]
                    if last is None or (t0 + 1e-9) // period != last[0]:
                        h = channel_at(a[u], dop[u], None, t0)
                        g = np.abs(h @ F.conj()) ** 2
                        meas = np.abs(np.sqrt(snr_lin(g)) * np.exp(1j * rng.uniform(0, 2 * np.pi, N)) + Lc.crandn(rng, N)) ** 2
                        avg = meas if (last is None or p != "ssb20avg") else 0.3 * meas + 0.7 * last[2]
                        last = ((t0 + 1e-9) // period, int(np.argmax(avg)), avg)
                    beam[u, s] = last[1]
            # delivery: frames of 200 ms; one RF chain, round-robin over UAVs with pending payload
            spf = int(round(FRAME / args.step))
            dv = deliv.setdefault(p, {}).setdefault(f.stem, np.zeros((U, steps // spf), bool))
            for fr in range(steps // spf):
                pend = {u: PAYLOAD_BITS for u in range(U)}
                for s in range(fr * spf, (fr + 1) * spf):
                    act = [u for u in pend if pend[u] > 0]
                    if not act:
                        break
                    u = act[s % len(act)]
                    overhead = SSB_S if p in ("ssb20",) and (s * args.step) % 0.02 < 1e-9 else 0.0
                    pend[u] -= rate[u, s, beam[u, s]] * max(args.step - overhead, 0)
                for u in range(U):
                    k = min(int(fr * FRAME // DT_RT), T - 1)
                    st = stats[p]
                    ok = pend[u] <= 0
                    dv[u, fr] = ok
                    st["frames"] += 1; st["delivered"] += ok
                    if los[u, k]:
                        st["los_frames"] += 1; st["los_deliv"] += ok
                    else:
                        st["nlos_frames"] += 1; st["nlos_deliv"] += ok
        print(f.name, {p: round(stats[p]["delivered"] / max(stats[p]["frames"], 1), 3) for p in pols}, flush=True)
    res = dict(sign_calibration={str(k): v for k, v in agree.items()}, payload_bits=PAYLOAD_BITS, eff=EFF, pt_dbm=PT_DBM, n0_dbm=N0_DBM, step=args.step, episodes=len(files),
               policies={p: dict(s, deliv_frac=round(s["delivered"] / max(s["frames"], 1), 4),
                                 los_frac=round(s["los_deliv"] / max(s["los_frames"], 1), 4),
                                 nlos_frac=round(s["nlos_deliv"] / max(s["nlos_frames"], 1), 4)) for p, s in stats.items()},
               beam_coherence_s=dict(median=float(np.median(coh)), p10=float(np.percentile(coh, 10)),
                                     p90=float(np.percentile(coh, 90)), n=len(coh)))
    (out / "c_feas.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "c_deliv.npz", **{f"{p}__{ep}": v for p, dd in deliv.items() for ep, v in dd.items()})
    print(json.dumps(res["policies"], indent=1), res["beam_coherence_s"])


if __name__ == "__main__":
    main()
