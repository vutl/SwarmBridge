"""Synthetic tests for Songdo S1 (s1.py). No dataset record is used."""
import numpy as np
import pandas as pd
import pytest

import s1 as S


def test_exp_golomb_round_trip():
    rng = np.random.default_rng(0)
    vals = list(rng.integers(-5000, 5000, 300)) + [0, 1, -1, 2 ** 15, -(2 ** 15)]
    for k in (0, 2, 5, 8):
        w = S.BitWriter()
        for v in vals:
            w.se(int(v), k)
        r = S.BitReader(w.bits)
        assert [r.se(k) for _ in vals] == [int(v) for v in vals]


def _recs():
    return [dict(type=0, id=7, cls=1, lane=3, qx=-120, qy=55, qvx=4, qvy=-2, qs=5),
            dict(type=1, id=9, dx=3, dy=-1, lane_changed=True, lane=2, dvx=1, dvy=0, ds=-1),
            dict(type=1, id=12, dx=0, dy=8, lane_changed=False, lane=1, dvx=0, dvy=0, ds=0),
            dict(type=2, id=15)]


@pytest.mark.parametrize("kind", ["hold", "cv", "lane"])
def test_packet_round_trip_all_kinds(kind):
    for bits, recs in S.packetize(_recs(), kind, dict(drone=3, seq=41, sec=63000)):
        hdr, dec = S.decode_packet(bits, kind)
        assert hdr["drone"] == 3 and hdr["sec"] == 63000 % (1 << 17)
        assert S.same_records(dec, recs)


def test_mtu_split_keeps_every_record():
    recs = [dict(type=0, id=i, cls=0, lane=0, qx=1000 + i, qy=-900 - i, qvx=3, qvy=-3) for i in range(1, 400)]
    out = S.packetize(recs, "cv", dict(drone=1, seq=0, sec=1))
    assert len(out) > 1 and all(len(b) <= S.MTU_BITS for b, _ in out)
    assert sorted(r["id"] for _, c in out for r in c) == list(range(1, 400))


def _session(n_sec=40, gap=(15, 25), speed=2.0):
    """two vehicles moving along +x at constant speed, one stationary; intersection unobserved during `gap`."""
    rows = []
    for s in range(n_sec):
        if gap[0] <= s < gap[1]:
            continue
        for vid, (x0, v) in {1: (0.0, speed), 2: (50.0, speed), 3: (20.0, 0.0)}.items():
            rows.append(dict(Vehicle_ID=vid + (100 if s >= gap[1] else 0), Local_Time=f"17:00:{s:02d}.000" if s < 60 else "",
                             Drone_ID=1, Local_X=x0 + v * s, Local_Y=5.0, Vehicle_Class=0, Vehicle_Speed=v * 3.6,
                             Road_Section="1_1", Lane_Number=1))
    d = pd.DataFrame(rows)
    h, m, x = d["Local_Time"].str.split(":", expand=True).T.values
    d["t"] = h.astype(float) * 3600 + m.astype(float) * 60 + x.astype(float)
    d["sec"] = np.floor(d["t"]).astype(int)
    d["lane_key"] = "1_1/1"
    return d.drop(columns=["Local_Time"])


def _lt():
    return dict(keys=["?", "1_1/1"], index={"?": 0, "1_1/1": 1}, heads=[None, np.array([1.0, 0.0])])


def test_masks_segments_and_revisit():
    d = _session()
    segs = S.segments(d["sec"].values)
    assert len(segs) == 2 and segs[0][-1] - segs[0][0] == 14
    r = S.simulate(d, "cv", 1.0, _lt())
    assert r["observed_s"] == 30
    assert len(r["revisit"]) == 1 and r["revisit"][0]["gap_s"] == 11


def test_constant_velocity_needs_fewer_bits_than_hold_for_moving_traffic():
    d = _session(gap=(100, 100))
    hold = S.simulate(d, "hold", 1.0, _lt())
    cv = S.simulate(d, "cv", 1.0, _lt())
    lane = S.simulate(d, "lane", 1.0, _lt())
    assert cv["ul_bps"] < hold["ul_bps"] and lane["ul_bps"] < hold["ul_bps"]
    assert cv["t2_mean_err_m"] <= 1.0 + S.Q_POS and hold["t2_mean_err_m"] <= 1.0 + S.Q_POS


def test_loss_is_recovered_by_replica_resynchronisation():
    d = _session(gap=(100, 100))
    clean = S.simulate(d, "hold", 1.0, _lt(), p_loss=0.0)        # hold-last updates every second -> many packets
    lossy = S.simulate(d, "hold", 1.0, _lt(), p_loss=0.3, seed=1)
    assert lossy["lost"] > 0 and lossy["packets"] >= clean["packets"]
    assert lossy["t2_miss_rate"] < 0.2            # births are re-sent until acknowledged
    assert lossy["t2_mean_err_m"] > clean["t2_mean_err_m"]   # lost updates cost accuracy until resent


def test_sender_never_uses_smoothed_speed():
    d = _session(gap=(100, 100))
    d2 = d.copy(); d2["Vehicle_Speed"] = 999.0          # corrupt the non-causal column
    a = S.simulate(d, "cv", 1.0, _lt()); b = S.simulate(d2, "cv", 1.0, _lt())
    assert a["ul_bps"] == b["ul_bps"] and a["t2_mean_err_m"] == b["t2_mean_err_m"]
