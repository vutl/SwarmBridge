"""Real bitstream codec for box / track-state messages (protocol v1, section 3; Codex review point 3).

Packet = header + count x box record, bit-packed MSB-first, zero-padded to a whole byte.

Header (132 bit):
  version 4 | sender 4 | send_frame 16 | count 12 | sender position x, y, z as IEEE float32 (3 x 32)
  The sender position is the sender's OWN position at the send frame (known to it at send time). Box positions are
  coded relative to it, in global (world) axes, so the encoder never needs the receiver's pose or any future pose.

Box record (87 bit, unsigned uniform quantisers, round-to-nearest, values outside the range saturate and are counted):
  dx, dy   13 bit each  range [-204.8, 204.8) m, step 0.05 m      (relative to the sender position; 12 bit / +-102.4 m
                        was the earlier fusion.py assumption, but UAV3D helpers detect inside UAV0's +-102.4 m grid, up
                        to ~125 m from themselves, so 12 bit saturated on real data)
  z         6 bit       range [0, 6.4) m, step 0.1 m              (global z of the box centre)
  w, l, h   8 bit each  range [0, 12.8) m, step 0.05 m
  yaw       8 bit       range [-pi, pi), step 2pi/256 (wrapped)
  score     7 bit       range [0, 1], q = round(score * 127)
  vx, vy    8 bit each  range [-12.8, 12.8) m/s, step 0.1 m/s    (sender's own estimate, e.g. its tracker; never GT)
Optional track extension (TRACK=True, +28 bit): track id 10 bit (mod 1024) | log-variance of x, y, yaw 3 x 6 bit
  (range [-8, 4) in ln(m^2) / ln(rad^2), step 0.1875).

Receivers must use decode() output only, never the float input of encode().
"""
import math
import struct

import numpy as np

VERSION = 1
HEADER_BITS = 4 + 4 + 16 + 12 + 3 * 32
BOX_SPEC = [  # (field, bits, lo, step)
    ("dx", 13, -204.8, 0.05), ("dy", 13, -204.8, 0.05), ("z", 6, 0.0, 0.1),
    ("w", 8, 0.0, 0.05), ("l", 8, 0.0, 0.05), ("h", 8, 0.0, 0.05),
    ("yaw", 8, -math.pi, 2 * math.pi / 256), ("score", 7, 0.0, 1.0 / 127),
    ("vx", 8, -12.8, 0.1), ("vy", 8, -12.8, 0.1),
]
# Diagnostic only (Codex review of box results): same codec with a wider velocity range, to measure the effect of
# velocity saturation. vx, vy 9 bit each over [-25.6, 25.6) m/s, step 0.1 -> 89 bit/box. Not a protocol message.
WIDE_VEL_SPEC = [f if f[0] not in ("vx", "vy") else (f[0], 9, -25.6, 0.1) for f in BOX_SPEC]
TRACK_SPEC = [("tid", 10, 0.0, 1.0), ("lvx", 6, -8.0, 0.1875), ("lvy", 6, -8.0, 0.1875), ("lvyaw", 6, -8.0, 0.1875)]
BOX_BITS = sum(b for _, b, _, _ in BOX_SPEC)              # 87
TRACK_BITS = sum(b for _, b, _, _ in TRACK_SPEC)          # 28
MAX_COUNT = (1 << 12) - 1


def _box_spec(wide_vel):
    return WIDE_VEL_SPEC if wide_vel else BOX_SPEC


def packet_bits(n, track=False, wide_vel=False):
    box = sum(b for _, b, _, _ in _box_spec(wide_vel))
    return HEADER_BITS + n * (box + (TRACK_BITS if track else 0))


def packet_bytes(n, track=False, wide_vel=False):
    return (packet_bits(n, track, wide_vel) + 7) // 8


class _Writer:
    def __init__(self):
        self.v, self.n = 0, 0

    def put(self, x, bits):
        assert 0 <= x < (1 << bits)
        self.v = (self.v << bits) | int(x)
        self.n += bits

    def bytes(self):
        pad = (-self.n) % 8
        return (self.v << pad).to_bytes((self.n + pad) // 8, "big")


class _Reader:
    def __init__(self, data):
        self.v, self.left = int.from_bytes(data, "big"), len(data) * 8

    def get(self, bits):
        self.left -= bits
        assert self.left >= 0, "truncated packet"
        return (self.v >> self.left) & ((1 << bits) - 1)


def _f32_bits(x):
    return struct.unpack(">I", struct.pack(">f", float(x)))[0]


def _bits_f32(u):
    return struct.unpack(">f", struct.pack(">I", u))[0]


def _quant(x, bits, lo, step, wrap=False):
    q = np.round((np.asarray(x, float) - lo) / step).astype(np.int64)
    top = (1 << bits) - 1
    if wrap:
        return np.mod(q, 1 << bits), np.zeros(q.shape, bool)
    sat = (q < 0) | (q > top)
    return np.clip(q, 0, top), sat


def encode(sender, send_frame, sender_pos, boxes, track=False, wide_vel=False):
    """boxes: dict of arrays xy (N,2) global, z (N,), size (N,3) [w,l,h], yaw (N,), score (N,), vel (N,2)
    [+ tid (N,), logvar (N,3) if track]. Returns (bytes, stats) with stats = {'bits', 'bytes', 'n', 'saturated'}."""
    n = len(boxes["score"])
    assert n <= MAX_COUNT, "too many boxes for one packet"
    w = _Writer()
    w.put(VERSION, 4); w.put(sender, 4); w.put(send_frame, 16); w.put(n, 12)
    for c in sender_pos[:3]:
        w.put(_f32_bits(c), 32)
    pos32 = np.array([_bits_f32(_f32_bits(c)) for c in sender_pos[:2]])  # receiver sees the float32 value
    vals = {"dx": boxes["xy"][:, 0] - pos32[0], "dy": boxes["xy"][:, 1] - pos32[1], "z": boxes["z"],
            "w": boxes["size"][:, 0], "l": boxes["size"][:, 1], "h": boxes["size"][:, 2],
            "yaw": boxes["yaw"], "score": boxes["score"], "vx": boxes["vel"][:, 0], "vy": boxes["vel"][:, 1]}
    spec = list(_box_spec(wide_vel))
    if track:
        vals.update({"tid": np.mod(boxes["tid"], 1024), "lvx": boxes["logvar"][:, 0], "lvy": boxes["logvar"][:, 1],
                     "lvyaw": boxes["logvar"][:, 2]})
        spec += TRACK_SPEC
    q, sat = {}, {}
    for f, bits, lo, step in spec:
        q[f], sat[f] = _quant(vals[f], bits, lo, step, wrap=(f == "yaw"))
    for i in range(n):
        for f, bits, _, _ in spec:
            w.put(q[f][i], bits)
    data = w.bytes()
    return data, {"bits": w.n, "bytes": len(data), "n": n,
                  "saturated": {f: int(s.sum()) for f, s in sat.items() if s.any()}}


def decode(data, track=False, wide_vel=False):
    """-> (header dict, boxes dict in the same layout as encode's input, global xy)."""
    r = _Reader(data)
    ver, sender, send_frame, n = r.get(4), r.get(4), r.get(16), r.get(12)
    assert ver == VERSION
    pos = [_bits_f32(r.get(32)) for _ in range(3)]
    spec = list(_box_spec(wide_vel)) + (TRACK_SPEC if track else [])
    q = {f: np.zeros(n, np.int64) for f, _, _, _ in spec}
    for i in range(n):
        for f, bits, _, _ in spec:
            q[f][i] = r.get(bits)
    v = {f: lo + q[f] * step for f, _, lo, step in spec}
    out = {"xy": np.stack([v["dx"] + pos[0], v["dy"] + pos[1]], 1).reshape(-1, 2), "z": v["z"],
           "size": np.stack([v["w"], v["l"], v["h"]], 1).reshape(-1, 3), "yaw": v["yaw"], "score": v["score"],
           "vel": np.stack([v["vx"], v["vy"]], 1).reshape(-1, 2)}
    if track:
        out["tid"] = q["tid"]
        out["logvar"] = np.stack([v["lvx"], v["lvy"], v["lvyaw"]], 1).reshape(-1, 3)
    return {"version": ver, "sender": sender, "send_frame": send_frame, "count": n, "sender_pos": pos}, out
