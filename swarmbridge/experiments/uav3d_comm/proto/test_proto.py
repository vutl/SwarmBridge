"""Unit tests for the protocol v1 building blocks. Run:  python -I test_proto.py [DATA_ROOT]
(no pytest needed; with DATA_ROOT it also round-trips real UAV3D detections)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boxcodec as BC  # noqa: E402
import ledger as L  # noqa: E402

PASSED = []


def check(name, cond, info=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {info}")
    PASSED.append(name)


# ------------------------------------------------------------------ ledger
def test_ledger():
    frames = {"sA": [f"a{i}" for i in range(20)], "sB": [f"b{i}" for i in range(20)]}
    led = L.build(frames, "sync", 0)
    lk = L.lookup(led)
    check("sync uses own frame, age 0", all(lk[f"a{t}"][j] == (f"a{t}", 0) for t in range(20) for j in L.SENDERS))

    led = L.build(frames, "fix2", 0)
    lk = L.lookup(led)
    check("fix2 cold start: no message before first arrival",
          all(lk[f"a{t}"][j] == (None, None) for t in (0, 1) for j in L.SENDERS))
    check("fix2 age exactly 2 afterwards", all(lk[f"a{t}"][j] == (f"a{t - 2}", 2) for t in range(2, 20)
                                              for j in L.SENDERS))

    # out-of-order: packet 0 arrives at 3, packet 1 arrives at 1 -> newest send frame wins, old late packet ignored
    pk = [{"sender": 1, "send_frame": 0, "drop": False, "arrival_frame": 3},
          {"sender": 1, "send_frame": 1, "drop": False, "arrival_frame": 1}]
    for j in (2, 3, 4):
        pk.append({"sender": j, "send_frame": 0, "drop": True, "arrival_frame": None})
    u = L.usage(pk, 6, "latest", max_age=3)[1]
    check("out-of-order: t=0 nothing", u[0] == (None, None))
    check("out-of-order: t=1 uses s=1", u[1] == (1, 0))
    check("out-of-order: t=3 keeps newer s=1 (age 2)", u[3] == (1, 2))
    check("hold expires after max_age", u[5] == (None, None) and u[4] == (1, 3))
    ua = L.usage(pk, 6, "arrival_only")[1]
    check("arrival_only uses packet only on its arrival frame", ua == [(None, None), (1, 0), (None, None), (0, 3),
                                                                       (None, None), (None, None)], ua)

    # causality on random scenarios: the used packet always arrived by t and was not dropped
    for sc in ("rand", "rand_loss0.2", "loss0.5"):
        for seed in (0, 1, 2):
            led = L.build(frames, sc, seed)
            for s in led["scenes"].values():
                arr = {(p["sender"], p["send_frame"]): p for p in s["packets"]}
                for t, row in enumerate(s["use"]):
                    for j, u in row.items():
                        if u["send_frame"] is None:
                            continue
                        p = arr[(int(j), u["send_frame"])]
                        assert not p["drop"] and p["arrival_frame"] <= t and u["age"] == t - u["send_frame"] <= 3
    check("random scenarios are causal (no future, no dropped packet used)", True)

    a, b = L.build(frames, "rand_loss0.2", 1), L.build(frames, "rand_loss0.2", 1)
    check("deterministic given seed", a["sha256"] == b["sha256"])
    d1 = [p["delay"] for p in L.build(frames, "rand", 1)["scenes"]["sA"]["packets"]]
    d2 = [p["delay"] for p in L.build(frames, "rand_loss0.2", 1)["scenes"]["sA"]["packets"]]
    check("loss stream does not change delay stream", d1 == d2)
    c = L.build(frames, "rand_loss0.2", 2)
    check("different seed -> different ledger", c["sha256"] != a["sha256"])
    a["scenes"]["sA"]["packets"][0]["drop"] = not a["scenes"]["sA"]["packets"][0]["drop"]
    check("tampering changes the hash", L.digest(a) != a["sha256"])

    led = L.build(frames, "loss0.5", 0)
    tx = L.tx_packets(led)
    check("every frame of every link is transmitted (bits spent even when lost)", len(tx) == 2 * 4 * 20)
    lossrate = np.mean(list(tx.values()))
    check("loss rate near p", 0.35 < lossrate < 0.65, lossrate)


# ------------------------------------------------------------------ box codec
def rand_boxes(rng, n, pos):
    return {"xy": pos[:2] + rng.uniform(-100, 100, (n, 2)), "z": rng.uniform(0.5, 3.5, n),
            "size": rng.uniform([1.5, 3.5, 1.2], [2.5, 5.5, 2.2], (n, 3)), "yaw": rng.uniform(-math.pi, math.pi, n),
            "score": rng.uniform(0, 1, n), "vel": rng.uniform(-10, 10, (n, 2))}


def test_codec():
    rng = np.random.default_rng(0)
    pos = np.array([1234.567, -987.654, 61.9])
    for n in (0, 1, 7, 189, 500):
        b = rand_boxes(rng, n, pos)
        data, st = BC.encode(3, 17, pos, b)
        check(f"bits n={n}", st["bits"] == BC.HEADER_BITS + n * BC.BOX_BITS == BC.packet_bits(n))
        check(f"bytes n={n}", len(data) == st["bytes"] == BC.packet_bytes(n) == math.ceil(st["bits"] / 8))
        h, d = BC.decode(data)
        check(f"header n={n}", (h["sender"], h["send_frame"], h["count"]) == (3, 17, n))
        if n:
            tol = 1e-6
            check(f"xy err <= 0.025 n={n}", np.abs(d["xy"] - b["xy"]).max() <= 0.025 + 1e-3, np.abs(d["xy"] - b["xy"]).max())
            check(f"z err <= 0.05 n={n}", np.abs(d["z"] - b["z"]).max() <= 0.05 + tol)
            check(f"size err <= 0.025 n={n}", np.abs(d["size"] - b["size"]).max() <= 0.025 + tol)
            dyaw = np.abs((d["yaw"] - b["yaw"] + math.pi) % (2 * math.pi) - math.pi)
            check(f"yaw err <= pi/256 n={n}", dyaw.max() <= math.pi / 256 + tol)
            check(f"score err <= 1/254 n={n}", np.abs(d["score"] - b["score"]).max() <= 1 / 254 + tol)
            check(f"vel err <= 0.05 n={n}", np.abs(d["vel"] - b["vel"]).max() <= 0.05 + tol)
            check(f"no saturation in range n={n}", not st["saturated"], st["saturated"])
    b = rand_boxes(rng, 3, pos)
    b["xy"][0] = pos[:2] + [250.0, 0.0]
    b["vel"][1] = [20.0, -20.0]
    b["yaw"][2] = 3 * math.pi
    data, st = BC.encode(1, 0, pos, b)
    h, d = BC.decode(data)
    check("out-of-range dx saturates and is counted", st["saturated"].get("dx") == 1 and d["xy"][0, 0] < pos[0] + 204.8)
    check("velocity saturates and is counted", st["saturated"].get("vx") == 1 and st["saturated"].get("vy") == 1)
    check("yaw wraps (3pi == pi == -pi)", abs(abs(d["yaw"][2]) - math.pi) < 0.03 and "yaw" not in st["saturated"])
    b = rand_boxes(rng, 5, pos)
    b["tid"] = np.array([0, 1, 1023, 1024, 77])
    b["logvar"] = rng.uniform(-6, 2, (5, 3))
    data, st = BC.encode(2, 5, pos, b, track=True)
    h, d = BC.decode(data, track=True)
    check("track ext bits", st["bits"] == BC.packet_bits(5, track=True) == BC.HEADER_BITS + 5 * 115)
    check("track id mod 1024", list(d["tid"]) == [0, 1, 1023, 0, 77])
    check("logvar err <= 0.094", np.abs(d["logvar"] - b["logvar"]).max() <= 0.09375 + 1e-6)


def test_kalman():
    import kalman as K
    rng = np.random.default_rng(1)
    v = np.array([[5.0, -2.0], [0.0, 0.0], [-8.0, 3.0]])
    p0 = np.array([[0.0, 0.0], [30.0, 30.0], [-40.0, 10.0]])
    tr = K.Tracker()
    outs = []
    for k in range(15):
        xy = p0 + v * k * K.DT + rng.normal(0, 0.3, (3, 2))
        det = {"xy": xy, "z": np.ones(3), "size": np.ones((3, 3)), "yaw": np.zeros(3), "score": np.full(3, 0.9),
               "vel": np.zeros((3, 2))}
        outs.append(tr.step(det))
    check("kalman: first frame emits nothing (MIN_HITS=2)", len(outs[0]["score"]) == 0)
    last = outs[-1]
    order = np.argsort(last["tid"])
    check("kalman: 3 stable ids", list(last["tid"][order]) == [0, 1, 2], last["tid"])
    verr = np.abs(last["vel"][order] - v).max()
    check("kalman: velocity error < 1 m/s after 15 frames", verr < 1.0, verr)
    true_next = p0 + v * 16 * K.DT
    f = K.forecast(last, 2)
    ferr = np.linalg.norm(f["xy"][order] - true_next, axis=1).max()
    stale = np.linalg.norm(last["xy"][order] - true_next, axis=1).max()
    check("kalman: 2-frame forecast beats stale position for movers", ferr < stale, (ferr, stale))


def test_kalman_real(root):
    """Velocity quality of the sender-side tracker on real detections; GT is used ONLY to score here."""
    import json
    import kalman as K
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fusion import load_variant
    from evaldet import MiniDB
    from ledger import scene_frames_from_db
    repo = Path(__file__).resolve().parents[4]
    db = MiniDB(root)
    det = load_variant(repo / "runs/uav3d_comm/agents_ft_cap500/agent1/results_nusc.json")
    errs_fc, errs_stale = [], []
    for sc, toks in scene_frames_from_db(db, ["scene_11", "scene_13", "scene_15"]).items():
        outs = K.run_sequence([det[t] for t in toks])
        for k in range(len(toks) - 2):
            gt2 = np.array([a["translation"][:2] for a in db.gt[toks[k + 2]]])
            gt0 = np.array([a["translation"][:2] for a in db.gt[toks[k]]])
            gv = np.array([[a["vx"], a["vy"]] for a in db.gt[toks[k]]])
            o = outs[k]
            if not len(o["score"]) or not len(gt0):
                continue
            moving = np.linalg.norm(gv, axis=1) > 2
            for i in range(len(o["score"])):
                d = np.linalg.norm(gt0 - o["xy"][i], axis=1)
                j = int(np.argmin(d))
                if d[j] > 1.5 or not moving[j]:
                    continue
                inst = db.gt[toks[k]][j]["instance_token"]
                nxt = [a for a in db.gt[toks[k + 2]] if a["instance_token"] == inst]
                if not nxt:
                    continue
                target = np.array(nxt[0]["translation"][:2])
                errs_fc.append(np.linalg.norm(o["xy"][i] + o["vel"][i] * 2 * K.DT - target))
                errs_stale.append(np.linalg.norm(o["xy"][i] - target))
    fc, st = np.median(errs_fc), np.median(errs_stale)
    print(f"  kalman on real agent1 dets, moving cars, 2-frame-old message: median error forecast {fc:.2f} m "
          f"vs stale {st:.2f} m (n={len(errs_fc)})")
    check("kalman forecast helps on real moving cars", fc < st, (fc, st))


def test_real(root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fusion import load_variant
    repo = Path(__file__).resolve().parents[4]
    det = load_variant(repo / "runs/uav3d_comm/agents_ft_cap500/agent1/results_nusc.json")
    import json
    meta = Path(root) / "v1.0-mini"
    cals = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    pos = {}
    for d in json.loads((meta / "sample_data.json").read_text()):
        if "CAMERA_BOTTOM_id_1/" in d["filename"]:
            pos[d["sample_token"]] = np.array(cals[d["calibrated_sensor_token"]]["translation"])
    errs, nbytes, sat = [], [], 0
    for tok, b in det.items():
        data, st = BC.encode(1, 0, pos[tok], b)
        _, d = BC.decode(data)
        errs.append(np.abs(d["xy"] - b["xy"]).max())
        nbytes.append(st["bytes"])
        sat += sum(st["saturated"].values())
    check("real detections: xy error <= 0.026 m", max(errs) <= 0.026, max(errs))
    print(f"  real agent1 cap500: {len(det)} frames, mean {np.mean(nbytes):.0f} B/packet "
          f"({np.mean(nbytes) * 8 / 1000:.1f} kbit), saturated fields {sat}")


if __name__ == "__main__":
    test_ledger()
    test_codec()
    test_kalman()
    if len(sys.argv) > 1:
        test_real(sys.argv[1])
        test_kalman_real(sys.argv[1])
    print(f"ALL {len(PASSED)} CHECKS PASSED")
