"""Output-level (box) communication on UAV3D: per-UAV detections -> fused detections at the receiver (UAV0).

Box message bit accounting (explicit, per box) -- see BOX_BITS.
"""
import json
from pathlib import Path
import numpy as np

# Per-box field budget for a compact message, quantised in the receiver's local frame (+-102.4 m BEV range):
#   x, y: 0.05 m over 204.8 m -> 12 bit each; z: 6 bit; l, w, h: 0.05 m steps -> 3 x 8 bit;
#   yaw: 1.4 deg -> 8 bit; score: 7 bit; velocity vx, vy: 0.1 m/s over +-12.8 m/s -> 2 x 8 bit.
BOX_FIELDS = {"x": 12, "y": 12, "z": 6, "lwh": 24, "yaw": 8, "score": 7, "vel": 16}
BOX_BITS = sum(BOX_FIELDS.values())          # 85 bit per box (no ID, no covariance)
TRACK_EXTRA = {"id": 10, "cov": 3 * 6}       # optional: track id + 3 log-variances (x, y, yaw) at 6 bit
HEADER_BITS = 64 + 32                         # sender pose (x, y, yaw quantised) + timestamp + count


def load_variant(path):
    """results_nusc.json -> {token: dict(xy, z, size, yaw, vel, score)} (global frame)."""
    R = json.loads(Path(path).read_text())["results"]
    out = {}
    for t, boxes in R.items():
        if not boxes:
            out[t] = empty()
            continue
        q = np.array([b["rotation"] for b in boxes])
        out[t] = {
            "xy": np.array([b["translation"][:2] for b in boxes]),
            "z": np.array([b["translation"][2] for b in boxes]),
            "size": np.array([b["size"] for b in boxes]),
            "yaw": 2 * np.arctan2(q[:, 3], q[:, 0]),
            "vel": np.array([b["velocity"][:2] for b in boxes]),
            "score": np.array([b["detection_score"] for b in boxes]),
        }
    return out


def empty():
    return {"xy": np.zeros((0, 2)), "z": np.zeros(0), "size": np.zeros((0, 3)), "yaw": np.zeros(0),
            "vel": np.zeros((0, 2)), "score": np.zeros(0)}


def cat(dets):
    return {k: np.concatenate([d[k] for d in dets]) for k in dets[0]}


def select(d, mask):
    return {k: v[mask] for k, v in d.items()}


def center_nms(d, radius=2.0):
    """Greedy NMS on xy centre distance (cars rarely closer than ~2 m centre to centre)."""
    order = np.argsort(-d["score"])
    keep, taken = [], np.zeros(len(order), bool)
    xy = d["xy"]
    for oi, i in enumerate(order):
        if taken[oi]:
            continue
        keep.append(i)
        rest = order[oi + 1:]
        close = np.linalg.norm(xy[rest] - xy[i], axis=1) < radius
        taken[oi + 1:] |= close
    return select(d, np.array(keep, int))


def late_fuse(ego, others, radius=2.0):
    """Union of ego + received boxes, then centre-NMS (keeps the most confident copy)."""
    return center_nms(cat([ego] + list(others)), radius)


def to_preds(dets):
    return {t: (d["xy"], d["score"]) for t, d in dets.items()}
