"""Constant-velocity Kalman tracker run by each SENDER on its own (decoded-quality) detections, used for the
"Kalman forecast" baseline inspired by CERF (protocol v1 section 7, Codex review point 5). Not an official CERF
reproduction: CERF's released code has no Kalman/forecast; this follows the paper's description in spirit only.

Everything here is causal and uses only the sender's own detections up to the current frame (no GT, no future):
  state  [x, y, vx, vy] (global, m and m/s), dt = 0.2 s (simulated time per frame)
  model  x' = F x + w, white-noise acceleration, sigma_a = SIGMA_A m/s^2
  meas.  z = [x, y] + v, sigma_r = SIGMA_R m
  init   from an unmatched detection with score >= NEW_SCORE: velocity 0, P = diag(R, R, V0^2, V0^2)
  assoc. greedy by detection score: each detection (score >= IN_SCORE) takes the nearest unmatched predicted track
         within GATE m (Euclidean on predicted position)
  life   a track is "confirmed" after MIN_HITS updates; deleted after MAX_MISS consecutive misses
Output per frame: confirmed tracks that were updated this frame (so a message never reports a coasted track as an
observation) with position, velocity, the last matched detection's z/size/yaw/score, track id and log-variances.

forecast(boxes, age): receiver-side constant-velocity extrapolation of a decoded message by age * dt.
"""
import numpy as np

DT = 0.2
SIGMA_A = 3.0
SIGMA_R = 0.5
V0 = 6.0
GATE = 3.0
IN_SCORE = 0.2
NEW_SCORE = 0.3
MIN_HITS = 2
MAX_MISS = 3


def _F(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def _Q(dt, sa):
    q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sa ** 2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q
    Q[np.ix_([1, 3], [1, 3])] = q
    return Q


H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


class Tracker:
    def __init__(self, dt=DT, sigma_a=SIGMA_A, sigma_r=SIGMA_R, gate=GATE):
        self.F, self.Q, self.R = _F(dt), _Q(dt, sigma_a), np.eye(2) * sigma_r ** 2
        self.gate = gate
        self.tracks = []
        self.next_id = 0

    def step(self, det):
        """det: dict xy, z, size, yaw, score, vel (one frame). Returns the frame's output boxes (dict, + tid, logvar)."""
        for t in self.tracks:
            t["x"] = self.F @ t["x"]
            t["P"] = self.F @ t["P"] @ self.F.T + self.Q
            t["updated"] = False
        keep = np.where(det["score"] >= IN_SCORE)[0]
        order = keep[np.argsort(-det["score"][keep])]
        free = set(range(len(self.tracks)))
        for i in order:
            z = det["xy"][i]
            best, bd = None, self.gate
            for k in free:
                d = np.linalg.norm(self.tracks[k]["x"][:2] - z)
                if d < bd:
                    best, bd = k, d
            if best is None:
                if det["score"][i] >= NEW_SCORE:
                    P = np.diag([self.R[0, 0], self.R[1, 1], V0 ** 2, V0 ** 2])
                    self.tracks.append({"x": np.r_[z, 0.0, 0.0], "P": P, "id": self.next_id, "hits": 1,
                                        "miss": 0, "updated": True, "det": i, "frame_det": det})
                    self.next_id += 1
                continue
            free.discard(best)
            t = self.tracks[best]
            S = H @ t["P"] @ H.T + self.R
            K = t["P"] @ H.T @ np.linalg.inv(S)
            t["x"] = t["x"] + K @ (z - H @ t["x"])
            t["P"] = (np.eye(4) - K @ H) @ t["P"]
            t["hits"] += 1
            t["miss"] = 0
            t["updated"] = True
            t["det"], t["frame_det"] = i, det
        for t in self.tracks:
            if not t["updated"]:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] < MAX_MISS]
        out = [t for t in self.tracks if t["updated"] and t["hits"] >= MIN_HITS]
        return _boxes(out)


def _boxes(tr):
    if not tr:
        return {"xy": np.zeros((0, 2)), "z": np.zeros(0), "size": np.zeros((0, 3)), "yaw": np.zeros(0),
                "score": np.zeros(0), "vel": np.zeros((0, 2)), "tid": np.zeros(0, int), "logvar": np.zeros((0, 3))}
    g = lambda key: np.array([t["frame_det"][key][t["det"]] for t in tr])
    return {"xy": np.array([t["x"][:2] for t in tr]), "vel": np.array([t["x"][2:] for t in tr]),
            "z": g("z"), "size": g("size"), "yaw": g("yaw"), "score": g("score"),
            "tid": np.array([t["id"] for t in tr]),
            "logvar": np.log(np.array([[t["P"][0, 0], t["P"][1, 1], 1e-2] for t in tr]))}


def run_sequence(frames):
    """frames: list of per-frame det dicts in time order (one scene). Returns list of output box dicts."""
    tr = Tracker()
    return [tr.step(d) for d in frames]


def forecast(boxes, age, dt=DT):
    out = dict(boxes)
    out["xy"] = boxes["xy"] + boxes["vel"] * (age * dt)
    return out
