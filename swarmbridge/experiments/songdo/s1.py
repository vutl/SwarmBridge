"""Songdo S1 (review #25): is structured traffic-state communication cheap enough to make scheduling moot?

CPU only, no learning. For one session CSV (one intersection) at 1 Hz decision steps:
  * evaluation masks: an intersection is OBSERVED at second s if its CSV has rows at s (some drone over it). Metrics
    are computed only on observed seconds (communication suppression simulated there) and at REVISIT instants
    (first observed second after a gap, BS prediction made before any new message). Unobserved seconds carry no
    direct ground truth and are never scored. T3 horizons are scored only if s+h is observed in the same segment.
  * sender observation (causal): raw positions (geo-trax positions are not smoothed or interpolated), class, lane
    label; velocity = 1-s backward difference of raw positions. The CSV's smoothed speed/acceleration and vehicle
    dimensions are NOT available to the sender (non-causal); they are not used.
  * M1 codec: dead-reckoning updates sent when the BS-predicted position (exactly simulated by the sender replica,
    confirmed only through ACKs) is off by more than EPS metres; births (new vehicle), updates and deaths (vehicle
    unseen for 2 steps) are real records written to a bitstream with signed exp-Golomb codes and decoded back
    (round-trip asserted). Packets: application header + records, split at the MTU; IP/UDP overhead reported.
    Losses: Bernoulli per packet; ACK (DL, 12-bit sequence number) arrives one step later; a lost record is not
    applied to the replica, so the next step resends whatever is still needed (state resynchronisation).
  * predictors (identical at sender replica and BS): hold-last; constant velocity (velocity sent); lane-aware
    (scalar speed along a static lane heading table built from a DIFFERENT session).
  * M0: per-road-section vehicle counts per 10-s window, residual-coded against the previous window.
  * full-state reference: every observed vehicle every step.
Segment boundaries (drone leaves): the BS keeps its state, is scored at the revisit instant, then drops that
intersection's tracks (vehicle ids are not shared across drones); the arriving drone re-announces vehicles (births).
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

Q_POS, Q_VEL = 0.25, 0.25            # m, m/s
HDR_BITS = 4 + 12 + 17 + 10 + 3 + 3  # drone, seq, second-of-day, n records, k_pos, k_vel
MTU_BITS, IPUDP_BITS, ACK_BITS = 1400 * 8, 28 * 8, 12
DEATH_AFTER, WINDOW_S, REVISIT_GATE_M = 2, 10, 5.0
K_ID, K_ABS, K_LANE, K_CNT = 3, 8, 2, 1


# ---------------------------------------------------------------- bit I/O (exp-Golomb)
class BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, v, n):
        self.bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]

    def ue(self, v, k):
        x = v + (1 << k)
        n = x.bit_length() - 1
        self.bits += [0] * (n - k)
        self.u(x, n + 1)

    def se(self, v, k):
        self.ue(2 * v if v >= 0 else -2 * v - 1, k)

    def __len__(self):
        return len(self.bits)


class BitReader:
    def __init__(self, bits):
        self.b, self.i = bits, 0

    def u(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.b[self.i]; self.i += 1
        return v

    def ue(self, k):
        z = 0
        while self.b[self.i] == 0:
            z += 1; self.i += 1
        n = z + k
        return self.u(n + 1) - (1 << k)

    def se(self, k):
        z = self.ue(k)
        return z // 2 if z % 2 == 0 else -(z + 1) // 2


def best_k(values, kmax=7):
    """exp-Golomb order minimising the coded length of signed residuals (sent in the header)."""
    if not values:
        return 0
    z = np.abs(np.asarray(values)) * 2
    return int(min(range(kmax + 1), key=lambda k: sum(2 * int(math.log2(v + (1 << k))) - k + 1 for v in z)))


# ---------------------------------------------------------------- records and packets
def encode_packet(records, kind, k_pos, k_vel, header):
    """records: dicts with type (0 birth, 1 update, 2 death), id, and quantised fields. -> list of bits."""
    w = BitWriter()
    w.u(header["drone"], 4); w.u(header["seq"] % 4096, 12); w.u(header["sec"] % (1 << 17), 17)
    w.u(len(records), 10); w.u(k_pos, 3); w.u(k_vel, 3)
    prev = 0
    for r in sorted(records, key=lambda r: r["id"]):
        w.u(r["type"], 2); w.ue(r["id"] - prev, K_ID); prev = r["id"]
        if r["type"] == 0:
            w.u(r["cls"], 2); w.ue(r["lane"], K_LANE); w.se(r["qx"], K_ABS); w.se(r["qy"], K_ABS)
            if kind == "cv":
                w.se(r["qvx"], k_vel); w.se(r["qvy"], k_vel)
            elif kind == "lane":
                w.se(r["qs"], k_vel)
        elif r["type"] == 1:
            w.se(r["dx"], k_pos); w.se(r["dy"], k_pos)
            w.u(int(r["lane_changed"]), 1)
            if r["lane_changed"]:
                w.ue(r["lane"], K_LANE)
            if kind == "cv":
                w.se(r["dvx"], k_vel); w.se(r["dvy"], k_vel)
            elif kind == "lane":
                w.se(r["ds"], k_vel)
    return w.bits


def decode_packet(bits, kind):
    rd = BitReader(bits)
    hdr = dict(drone=rd.u(4), seq=rd.u(12), sec=rd.u(17))
    n, k_pos, k_vel = rd.u(10), rd.u(3), rd.u(3)
    out, prev = [], 0
    for _ in range(n):
        r = dict(type=rd.u(2)); r["id"] = prev + rd.ue(K_ID); prev = r["id"]
        if r["type"] == 0:
            r["cls"] = rd.u(2); r["lane"] = rd.ue(K_LANE); r["qx"] = rd.se(K_ABS); r["qy"] = rd.se(K_ABS)
            if kind == "cv":
                r["qvx"] = rd.se(k_vel); r["qvy"] = rd.se(k_vel)
            elif kind == "lane":
                r["qs"] = rd.se(k_vel)
        elif r["type"] == 1:
            r["dx"] = rd.se(k_pos); r["dy"] = rd.se(k_pos); r["lane_changed"] = bool(rd.u(1))
            if r["lane_changed"]:
                r["lane"] = rd.ue(K_LANE)
            if kind == "cv":
                r["dvx"] = rd.se(k_vel); r["dvy"] = rd.se(k_vel)
            elif kind == "lane":
                r["ds"] = rd.se(k_vel)
        out.append(r)
    return hdr, out


def packetize(records, kind, header):
    """split records into MTU-sized packets; returns list of (bits, records)."""
    if not records:
        return []
    kp = best_k([v for r in records if r["type"] == 1 for v in (r["dx"], r["dy"])])
    kv = best_k([v for r in records for key in ("qvx", "qvy", "qs", "dvx", "dvy", "ds") if key in r for v in (r[key],)])
    recs = sorted(records, key=lambda r: r["id"])
    n = 1
    while True:                                   # split into n near-equal chunks until every packet fits the MTU
        chunks = [recs[i * len(recs) // n:(i + 1) * len(recs) // n] for i in range(n)]
        out = [(encode_packet(c, kind, kp, kv, dict(header, seq=header["seq"] + i)), c) for i, c in enumerate(chunks)]
        if all(len(b) <= MTU_BITS for b, _ in out) or n >= len(recs):
            return out
        n += 1


def same_records(dec, enc):
    """decoded records equal the encoded ones on every field the format carries."""
    enc = sorted(enc, key=lambda r: r["id"])
    return len(dec) == len(enc) and all(all(e.get(k) == v for k, v in r.items()) for r, e in zip(dec, enc))


# ---------------------------------------------------------------- data preparation
def load_session(path):
    cols = ["Vehicle_ID", "Local_Time", "Drone_ID", "Local_X", "Local_Y", "Vehicle_Class", "Vehicle_Speed",
            "Road_Section", "Lane_Number"]
    parts = []
    for c in pd.read_csv(path, usecols=cols, chunksize=1_000_000):   # chunked: 30 Hz CSVs do not fit 10x in RAM
        h, m, s = c["Local_Time"].str.split(":", expand=True).T.values
        c["t"] = h.astype(float) * 3600 + m.astype(float) * 60 + s.astype(float)
        c["sec"] = np.floor(c["t"]).astype(int)
        parts.append(c.drop(columns=["Local_Time"]).sort_values(["Vehicle_ID", "t"]).drop_duplicates(["Vehicle_ID", "sec"]))
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values(["Vehicle_ID", "t"]).drop_duplicates(["Vehicle_ID", "sec"])    # first sample in each second
    d["lane_key"] = d["Road_Section"].astype(str) + "/" + d["Lane_Number"].astype(str)
    return d


def causal_velocity(d):
    """1-s backward difference of raw positions (NaN if the vehicle was not observed one second earlier)."""
    d = d.sort_values(["Vehicle_ID", "sec"]).copy()
    g = d.groupby("Vehicle_ID")
    dt = d["sec"] - g["sec"].shift(1)
    d["vx"] = np.where(dt == 1, d["Local_X"] - g["Local_X"].shift(1), np.nan)
    d["vy"] = np.where(dt == 1, d["Local_Y"] - g["Local_Y"].shift(1), np.nan)
    return d


def lane_table(d):
    """static lane headings (unit vectors) from moving vehicles of ANOTHER session; index 0 = unknown lane."""
    d = causal_velocity(d)
    sp = np.hypot(d["vx"], d["vy"])
    mv = d[sp > 2.0]
    keys, heads = ["?"], [None]
    for k, g in mv.groupby("lane_key"):
        if k.startswith("nan") or len(g) < 20:
            continue
        u = np.c_[g["vx"], g["vy"]] / np.hypot(g["vx"], g["vy"]).values[:, None]
        m = u.mean(0)
        if np.hypot(*m) > 0.5:
            keys.append(k); heads.append(m / np.hypot(*m))
    return dict(keys=keys, index={k: i for i, k in enumerate(keys)}, heads=heads)


def segments(secs, gap=1):
    secs = np.sort(np.unique(secs))
    br = np.where(np.diff(secs) > gap)[0]
    return [np.arange(a, b + 1) for a, b in zip(np.r_[secs[0], secs[br + 1]], np.r_[secs[br], secs[-1]])]


# ---------------------------------------------------------------- predictors (BS == sender replica)
def predict(st, s, kind, lt):
    dt = s - st["t"]
    if kind == "hold":
        return st["x"], st["y"]
    if kind == "cv":
        return st["x"] + st["vx"] * dt, st["y"] + st["vy"] * dt
    h = lt["heads"][st["lane"]] if st["lane"] < len(lt["heads"]) else None
    if h is None:
        return st["x"], st["y"]
    return st["x"] + st["s"] * h[0] * dt, st["y"] + st["s"] * h[1] * dt


def apply(state, r, s, kind, origin, lt):
    """apply a decoded record to a BS (or replica) state dict."""
    if r["type"] == 2:
        state.pop(r["id"], None)
        return
    if r["type"] == 0:
        st = dict(x=origin[0] + r["qx"] * Q_POS, y=origin[1] + r["qy"] * Q_POS, t=s, lane=r["lane"], cls=r["cls"],
                  vx=0.0, vy=0.0, s=0.0)
        if kind == "cv":
            st["vx"], st["vy"] = r["qvx"] * Q_VEL, r["qvy"] * Q_VEL
        elif kind == "lane":
            st["s"] = r["qs"] * Q_VEL
        state[r["id"]] = st
        return
    st = state.get(r["id"])
    if st is None:
        return
    px, py = predict(st, s, kind, lt)
    st.update(x=px + r["dx"] * Q_POS, y=py + r["dy"] * Q_POS, t=s)
    if r["lane_changed"]:
        st["lane"] = r["lane"]
    if kind == "cv":
        st["vx"] += r["dvx"] * Q_VEL; st["vy"] += r["dvy"] * Q_VEL
    elif kind == "lane":
        st["s"] += r["ds"] * Q_VEL


# ---------------------------------------------------------------- simulation
def simulate(d, kind, eps, lt, p_loss=0.0, seed=0, horizons=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    d = causal_velocity(d)
    origin = (float(d["Local_X"].median()), float(d["Local_Y"].median()))
    by_sec = dict(tuple(d.groupby("sec")))
    segs = segments(d["sec"].values)
    observed = set(int(s) for seg in segs for s in seg)
    sections = sorted(d["Road_Section"].dropna().astype(str).unique())
    bs, rep = {}, {}                          # BS state and sender replica (confirmed by ACK)
    pending = []                              # (records) sent at s, ACK known at s+1
    seq, ul, ul_ip, dl, n_pk, n_lost = 0, 0, 0, 0, 0, 0
    ul_sec = {}                               # application UL bits per observed second (fleet-load analysis)
    err, miss, false, n_obs, t3 = [], 0, 0, 0, {h: [] for h in horizons}
    cnt_true, cnt_bs, revisit = [], [], []
    unseen = {}
    for si, seg in enumerate(segs):
        if si > 0:                            # revisit scoring before any message of the new segment
            g = by_sec.get(int(seg[0]))
            if g is not None and bs:
                P = np.array([predict(st, int(seg[0]), kind, lt) for st in bs.values()])
                Tm = g[["Local_X", "Local_Y"]].values
                D = np.linalg.norm(P[:, None, :] - Tm[None, :, :], axis=2)
                r_, c_ = linear_sum_assignment(np.minimum(D, 1e3))
                ok = D[r_, c_] <= REVISIT_GATE_M
                revisit.append(dict(gap_s=int(seg[0] - segs[si - 1][-1]), n_true=len(Tm), n_bs=len(P),
                                    matched=int(ok.sum()), mean_err=float(D[r_, c_][ok].mean()) if ok.any() else None))
            bs.clear(); rep.clear(); unseen.clear(); pending = []
        for s in seg:
            s = int(s)
            # ACKs of the previous step: apply confirmed records to the replica
            for recs, lost in pending:
                if not lost:
                    for r in recs:
                        apply(rep, r, s - 1, kind, origin, lt)
            pending = []
            g = by_sec.get(s)
            if g is None:
                continue
            ids = set(int(i) for i in g["Vehicle_ID"].values)
            recs = []
            for _, row in g.iterrows():
                vid = int(row["Vehicle_ID"]); unseen.pop(vid, None)
                lane = lt["index"].get(row["lane_key"], 0)
                vx = 0.0 if np.isnan(row["vx"]) else row["vx"]; vy = 0.0 if np.isnan(row["vy"]) else row["vy"]
                h = lt["heads"][lane] if lane else None
                spd = (vx * h[0] + vy * h[1]) if h is not None else 0.0
                if vid not in rep:
                    r = dict(type=0, id=vid, cls=int(row["Vehicle_Class"]) & 3, lane=lane,
                             qx=int(round((row["Local_X"] - origin[0]) / Q_POS)), qy=int(round((row["Local_Y"] - origin[1]) / Q_POS)))
                    if kind == "cv":
                        r.update(qvx=int(round(vx / Q_VEL)), qvy=int(round(vy / Q_VEL)))
                    elif kind == "lane":
                        r.update(qs=int(round(spd / Q_VEL)))
                    recs.append(r)
                    continue
                st = rep[vid]
                px, py = predict(st, s, kind, lt)
                if math.hypot(row["Local_X"] - px, row["Local_Y"] - py) > eps or lane != st["lane"]:
                    r = dict(type=1, id=vid, dx=int(round((row["Local_X"] - px) / Q_POS)),
                             dy=int(round((row["Local_Y"] - py) / Q_POS)), lane_changed=lane != st["lane"], lane=lane)
                    if kind == "cv":
                        r.update(dvx=int(round((vx - st["vx"]) / Q_VEL)), dvy=int(round((vy - st["vy"]) / Q_VEL)))
                    elif kind == "lane":
                        r.update(ds=int(round((spd - st["s"]) / Q_VEL)))
                    recs.append(r)
            for vid in list(rep):
                if vid not in ids:
                    unseen[vid] = unseen.get(vid, 0) + 1
                    if unseen[vid] >= DEATH_AFTER:
                        recs.append(dict(type=2, id=vid))
            for bits, prs in packetize(recs, kind, dict(drone=int(g["Drone_ID"].iloc[0]) & 15, seq=seq, sec=s)):
                hdr, dec = decode_packet(bits, kind)
                assert same_records(dec, prs), "codec round trip failed"
                seq += 1; n_pk += 1; ul += len(bits); ul_ip += len(bits) + IPUDP_BITS; dl += ACK_BITS
                ul_sec[s] = ul_sec.get(s, 0) + len(bits) + IPUDP_BITS
                lost = rng.random() < p_loss
                n_lost += lost
                if not lost:
                    for r in dec:
                        apply(bs, r, s, kind, origin, lt)
                pending.append((dec, lost))
            # T2 scoring on this observed second (after the update)
            for _, row in g.iterrows():
                vid = int(row["Vehicle_ID"]); n_obs += 1
                if vid in bs:
                    px, py = predict(bs[vid], s, kind, lt)
                    err.append(math.hypot(row["Local_X"] - px, row["Local_Y"] - py))
                else:
                    miss += 1
            false += sum(1 for vid in bs if vid not in ids)
            # T3: forecasts from the BS state, scored only where s+h is observed and the vehicle present
            for hz in horizons:
                g2 = by_sec.get(s + hz)
                if g2 is None or (s + hz) not in observed or (s + hz) > seg[-1]:
                    continue
                fut = dict(zip(g2["Vehicle_ID"].astype(int), zip(g2["Local_X"], g2["Local_Y"])))
                for vid, st in bs.items():
                    if vid in fut:
                        px, py = predict(st, s + hz, kind, lt)
                        t3[hz].append(math.hypot(fut[vid][0] - px, fut[vid][1] - py))
            # T1 counts per section (truth vs BS lane-label counts)
            if (s - seg[0]) % WINDOW_S == WINDOW_S - 1:
                tc = g["Road_Section"].astype(str).value_counts()
                lane_sec = {i: k.split("/")[0] for i, k in enumerate(lt["keys"])}
                bc = pd.Series([lane_sec.get(st["lane"], "?") for st in bs.values()]).value_counts() if bs else pd.Series(dtype=int)
                cnt_true.append([int(tc.get(x, 0)) for x in sections]); cnt_bs.append([int(bc.get(x, 0)) for x in sections])
    nsec = len(observed)
    ct, cb = np.array(cnt_true), np.array(cnt_bs)
    return dict(kind=kind, eps=eps, p_loss=p_loss, observed_s=nsec, packets=n_pk, lost=n_lost,
                ul_bps=ul / nsec, ul_ip_bps=ul_ip / nsec, dl_bps=dl / nsec,
                t2_mean_err_m=float(np.mean(err)) if err else None, t2_p95_err_m=float(np.percentile(err, 95)) if err else None,
                t2_miss_rate=miss / max(n_obs, 1), t2_false_per_obs=false / max(n_obs, 1),
                t3_ade_m={h: float(np.mean(v)) if v else None for h, v in t3.items()},
                t1_count_mae=float(np.abs(ct - cb).mean()) if len(ct) else None,
                revisit=revisit, ul_ip_bits_by_sec=ul_sec)


def m0_bits(d):
    """M0: per-section counts per 10-s window, residual vs previous window, exp-Golomb, + header per window."""
    sections = sorted(d["Road_Section"].dropna().astype(str).unique())
    bits, n_win, prev = 0, 0, None
    for seg in segments(d["sec"].values):
        prev = None
        for a in range(int(seg[0]), int(seg[-1]) + 1, WINDOW_S):
            g = d[(d["sec"] >= a) & (d["sec"] < a + WINDOW_S)]
            c = g.groupby("sec")["Road_Section"].apply(lambda x: x.astype(str).value_counts())
            cur = [int(round(c.xs(x, level=1).mean())) if x in c.index.get_level_values(1) else 0 for x in sections]
            w = BitWriter(); w.u(0, HDR_BITS)
            for i, v in enumerate(cur):
                w.se(v - (prev[i] if prev else 0), K_CNT)
            bits += len(w); n_win += 1; prev = cur
    nsec = len(set(d["sec"]))
    return dict(m0_ul_bps=bits / nsec, windows=n_win, sections=len(sections))


def full_state_bits(d):
    """reference: every observed vehicle every second as a birth-style record (absolute position + velocity)."""
    d = causal_velocity(d)
    origin = (float(d["Local_X"].median()), float(d["Local_Y"].median()))
    total = 0
    for s, g in d.groupby("sec"):
        recs = [dict(type=0, id=int(r.Vehicle_ID), cls=int(r.Vehicle_Class) & 3, lane=0,
                     qx=int(round((r.Local_X - origin[0]) / Q_POS)), qy=int(round((r.Local_Y - origin[1]) / Q_POS)),
                     qvx=0 if np.isnan(r.vx) else int(round(r.vx / Q_VEL)), qvy=0 if np.isnan(r.vy) else int(round(r.vy / Q_VEL)))
                for r in g.itertuples()]
        for bits, _ in packetize(recs, "cv", dict(drone=0, seq=0, sec=int(s))):
            total += len(bits)
    return dict(full_ul_bps=total / d["sec"].nunique())
