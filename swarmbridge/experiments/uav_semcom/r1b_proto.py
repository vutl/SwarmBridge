"""R1b-0a state-synchronisation protocol (design v3 sec 4, amended 10/11): integer canonical state, CRC32 at t_state,
RFC 1982 serial arithmetic, delta / snapshot messages, UAV-side replica for conditions C1 (versioned) and C0 (best effort).

Canonical track (all Python ints, identical on both sides):
    (tid, cls, cx, cy, w, h, vx, vy, conf)   positions in 1/4096 of frame width (x) / height (y), velocity in the same
    units per tick, conf in 0..7.  Prediction to tick t is exact integer arithmetic: cx + vx*dt, cy + vy*dt (no clamp),
    so UAV and BS obtain the same bits for any dt (test 4).
Wire sizes (bits), amended from v3 (12-bit coordinates: 8-bit / (1/256 frame) quantisation is ~5 px at 1344 px, too
coarse for VisDrone small objects; 10-bit signed velocity covers +-168 px / tick):
    op add/update: tid 10 | kind 2 | cls 4 | box 4x12 | vel 2x10 | conf 3 = 87;  op delete: tid 10 | kind 2 = 12
    C1 header: msg_id 16 | uav 4 | base 16 | version 16 | t_state 16 | frame_processed 16 | n_ops 8 | crc 32 = 124
    C0 header: msg_id 16 | uav 4 | t_state 16 | frame_processed 16 | n_ops 8 = 60
    uplink sync header (C1 only): applied_version 16 | crc(applied) 32 = 48
"""
import struct
import zlib

MOD = 1 << 16
HALF = 1 << 15
Q = 4096                     # coordinate quantisation per frame dimension
VMAX = 511                   # signed 10-bit velocity
OP_BITS, DEL_BITS = 87, 12
C1_HDR, C0_HDR, UL_SYNC = 124, 60, 48
ADD, UPD, DEL = 1, 2, 3


# ---------------- serial arithmetic (RFC 1982, 16 bit)
def s_add(a, n):
    return (a + n) % MOD


def s_newer(a, b):
    """a strictly newer than b."""
    d = (a - b) % MOD
    return 0 < d < HALF


def s_diff(a, b):
    """signed distance a - b in the serial window."""
    d = (a - b) % MOD
    return d - MOD if d >= HALF else d


# ---------------- quantisation and canonical state
def quantise(box_xyxy, vel_xy, cls, conf, W, H):
    x1, y1, x2, y2 = box_xyxy
    cx = round((x1 + x2) / 2 * Q / W); cy = round((y1 + y2) / 2 * Q / H)
    w = round((x2 - x1) * Q / W); h = round((y2 - y1) * Q / H)
    cx, cy = min(max(cx, 0), Q - 1), min(max(cy, 0), Q - 1)
    w, h = min(max(w, 1), Q - 1), min(max(h, 1), Q - 1)
    vx = min(max(round(vel_xy[0] * Q / W), -VMAX - 1), VMAX); vy = min(max(round(vel_xy[1] * Q / H), -VMAX - 1), VMAX)
    return (int(cls) & 15, cx, cy, w, h, vx, vy, min(max(int(conf * 8), 0), 7))


def dequant_box(t, W, H):
    _, cx, cy, w, h = t[:5]
    return ((cx - w / 2) * W / Q, (cy - h / 2) * H / Q, (cx + w / 2) * W / Q, (cy + h / 2) * H / Q)


def predict(state, dt):
    """state: {tid: (cls, cx, cy, w, h, vx, vy, conf)} -> same, advanced by dt ticks (exact ints)."""
    if dt == 0:
        return dict(state)
    return {k: (c, cx + vx * dt, cy + vy * dt, w, h, vx, vy, cf) for k, (c, cx, cy, w, h, vx, vy, cf) in state.items()}


def crc(state, t_state):
    b = bytearray(struct.pack("<H", t_state % MOD))
    for k in sorted(state):
        b += struct.pack("<H8i", k, *state[k])
    return zlib.crc32(bytes(b))


def delta(old, new):
    ops = []
    for k in sorted(new):
        if k not in old:
            ops.append((ADD, k, new[k]))
        elif old[k] != new[k]:
            ops.append((UPD, k, new[k]))
    ops += [(DEL, k, None) for k in sorted(old) if k not in new]
    return ops


def apply_ops(state, ops):
    s = dict(state)
    for kind, k, v in ops:
        if kind == DEL:
            s.pop(k, None)
        else:
            s[k] = v
    return s


def ops_bits(ops):
    return sum(DEL_BITS if kind == DEL else OP_BITS for kind, _, _ in ops)


# ---------------- messages
class Msg:
    __slots__ = ("msg_id", "base", "version", "t_state", "frame_processed", "ops", "h", "snapshot", "bits")

    def __init__(self, msg_id, base, version, t_state, frame_processed, ops, h, snapshot, bits):
        self.msg_id, self.base, self.version, self.t_state = msg_id, base, version, t_state
        self.frame_processed, self.ops, self.h, self.snapshot, self.bits = frame_processed, ops, h, snapshot, bits


class Publisher:
    """BS side. Holds canonical published states by version; builds delta / snapshot messages.
    `versioned=False` gives C0 (no base / version / hash on the wire, but same content)."""

    def __init__(self, versioned=True, n_hist=64):
        self.versioned, self.n_hist = versioned, n_hist
        self.msg_id = 0
        self.version = 0                       # 0 = empty initial state, known to both sides
        self.t_state = 0
        self.state = {}
        self.hist = {0: (0, {}, crc({}, 0))}   # version -> (t_state, state, crc)

    def current_at(self, t):
        return predict(self.state, t - self.t_state)

    def make(self, new_state, t_state, frame_processed, snapshot=False):
        """new_state: canonical state at t_state. Returns Msg (and advances the published version)."""
        old = self.current_at(t_state)
        ops = delta({} if snapshot else old, new_state)
        if not ops and not snapshot:
            return None
        base = 0 if snapshot else self.version
        self.version = s_add(self.version, 1)
        if self.version == 0:                  # 0 is reserved for the empty initial state
            self.version = 1
        self.msg_id = s_add(self.msg_id, 1)
        self.state, self.t_state = dict(new_state), t_state
        h = crc(self.state, t_state)
        self.hist[self.version] = (t_state, dict(self.state), h)
        if len(self.hist) > self.n_hist:
            for v in sorted(self.hist, key=lambda v: s_diff(v, self.version))[:len(self.hist) - self.n_hist]:
                if v != self.version:
                    del self.hist[v]
        bits = (C1_HDR if self.versioned else C0_HDR) + ops_bits(ops)
        return Msg(self.msg_id, base, self.version, t_state, frame_processed, ops, h, snapshot, bits)

    def check_uplink(self, applied, h):
        """C1: uplink header (applied version, crc). True -> BS should send a snapshot."""
        if applied not in self.hist:
            return True
        return self.hist[applied][2] != h


class Replica:
    """UAV side, condition C1 (versioned=True) or C0 (versioned=False)."""

    def __init__(self, versioned=True, w_hold=10):
        self.versioned, self.w_hold = versioned, w_hold
        self.applied, self.t_state, self.state = 0, 0, {}
        self.h = crc({}, 0)
        self.desync = False
        self.buf = {}                          # base -> (arrival tick, msg)
        self.last_t = None                     # C0: newest t_state applied

    def view(self, t):
        return predict(self.state, t - self.t_state)

    def header(self):
        return self.applied, (self.h if not self.desync else self.h ^ 0xFFFFFFFF)

    def receive(self, m, now):
        if not self.versioned:
            return self._recv_c0(m)
        if m.snapshot:
            if s_newer(m.version, self.applied) or self.desync:
                self._set(m.version, m.t_state, apply_ops({}, m.ops), m.h)
                self._drain(now)
            return
        if not s_newer(m.version, self.applied):
            return                              # duplicate / stale
        if m.base == self.applied and not self.desync:
            self._apply(m)
            self._drain(now)
        elif s_newer(m.base, self.applied):
            self.buf[m.base] = (now, m)         # gap: hold
        # base older than applied but version newer: cannot happen with a chain; ignore

    def tick(self, now):
        """Expire held messages; a gap older than w_hold ticks marks desync (until the next snapshot)."""
        for b, (t0, _) in list(self.buf.items()):
            if now - t0 > self.w_hold:
                self.desync = True
                del self.buf[b]

    def _apply(self, m):
        s = apply_ops(predict(self.state, m.t_state - self.t_state), m.ops)
        if crc(s, m.t_state) == m.h:
            self._set(m.version, m.t_state, s, m.h)
        else:
            self.desync = True
            self.state, self.t_state = s, m.t_state     # best effort until snapshot

    def _set(self, v, t, s, h):
        self.applied, self.t_state, self.state, self.h, self.desync = v, t, s, h, False

    def _drain(self, now):
        while self.applied in self.buf and not self.desync:
            _, m = self.buf.pop(self.applied)
            if s_newer(m.version, self.applied):
                self._apply(m)

    def _recv_c0(self, m):
        if self.last_t is not None and s_diff(m.t_state, self.last_t) < 0:
            return                              # older than what we hold
        base = {} if m.snapshot else predict(self.state, m.t_state - self.t_state)
        self.state, self.t_state, self.last_t = apply_ops(base, m.ops), m.t_state, m.t_state
