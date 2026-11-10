"""R1b-0a feedback protocol (design v3 sec 4, amended): integer canonical state, CRC32 at t_state, RFC 1982 serial
arithmetic, delta / snapshot messages, UAV replica for C1 (versioned) and C0 (best effort), and the BIT-EXACT wire
format of every downlink message (C1, C0, hint B / B+, ACK) plus the C1 uplink sync header.

Canonical track (Python ints, identical on both sides):
    (cls, cx, cy, w, h, vx, vy, conf) keyed by tid;  positions in 1/4096 of frame width (x) / height (y), velocity in
    the same units per tick.  Prediction to tick t is exact integer arithmetic (cx + vx*dt), no clamp.
    A published canonical state only contains REPRESENTABLE tracks (all fields inside their wire ranges below); the BS
    deletes a track (DEL op) as soon as its integer prediction leaves the range, so every snapshot can be serialised.
Wire format (bits, MSB first, payload padded to whole bytes):
    C1 header : msg_id 16 | uav 4 | base 16 | version 16 | t_state 16 | frame_processed 16 | n_ops 11 | crc 32 = 127
    C0 header : msg_id 16 | uav 4 | snap 1 | t_state 16 | frame_processed 16 | n_ops 11                =  64
    op        : tid 10 | kind 2 | [cls 4 | cx 12 | cy 12 | w 12 | h 12 | vx 10s | vy 10s | conf 3]     = 87 (DEL: 12)
    hint B    : frame_processed 16 | n 11 | n x (cx 12 | cy 12 | w 12 | h 12)                          = 27 + 48 n
    hint B+   : as B, each box + cls 4 + conf 3                                                         = 27 + 55 n
    ACK       : n 11 | n x capture_tick 16                                                              = 11 + 16 n
    UL sync   : applied_version 16 | crc 32                                                             = 48
n_ops is 11 bits because tids are 10 bits: a delta touches at most |old U new| <= 1024 ids (8 bits would cap at 255).
"""
import struct
import zlib

MOD = 1 << 16
HALF = 1 << 15
Q = 4096
VMAX = 511
ADD, UPD, DEL = 1, 2, 3
C1_HDR, C0_HDR, UL_SYNC = 127, 64, 48
OP_BITS, DEL_BITS = 87, 12
HINT_HDR, HINT_BOX, HINT_BOX_PLUS = 27, 48, 55
ACK_HDR, ACK_ITEM = 11, 16
MAX_OPS = (1 << 11) - 1
UAV_ID = 0


# ---------------- serial arithmetic (RFC 1982, 16 bit)
def s_add(a, n):
    return (a + n) % MOD


def s_newer(a, b):
    d = (a - b) % MOD
    return 0 < d < HALF


def s_diff(a, b):
    d = (a - b) % MOD
    return d - MOD if d >= HALF else d


def unwrap(c16, now):
    """Full tick of a 16-bit tick value c16 that is not later than `now` (and within 2^16 ticks)."""
    return now - ((now - c16) % MOD)


# ---------------- quantisation and canonical state
def quantise(box_xyxy, vel_xy, cls, conf, W, H):
    x1, y1, x2, y2 = box_xyxy
    cx = round((x1 + x2) / 2 * Q / W); cy = round((y1 + y2) / 2 * Q / H)
    w = round((x2 - x1) * Q / W); h = round((y2 - y1) * Q / H)
    cx, cy = min(max(cx, 0), Q - 1), min(max(cy, 0), Q - 1)
    w, h = min(max(w, 1), Q - 1), min(max(h, 1), Q - 1)
    vx = min(max(round(vel_xy[0] * Q / W), -VMAX - 1), VMAX); vy = min(max(round(vel_xy[1] * Q / H), -VMAX - 1), VMAX)
    return (int(cls) & 15, cx, cy, w, h, vx, vy, min(max(int(conf * 8), 0), 7))


def representable(v):
    c, cx, cy, w, h, vx, vy, cf = v
    return (0 <= c < 16 and 0 <= cx < Q and 0 <= cy < Q and 1 <= w < Q and 1 <= h < Q and -VMAX - 1 <= vx <= VMAX
            and -VMAX - 1 <= vy <= VMAX and 0 <= cf < 8)


def dequant_box(t, W, H):
    _, cx, cy, w, h = t[:5]
    return ((cx - w / 2) * W / Q, (cy - h / 2) * H / Q, (cx + w / 2) * W / Q, (cy + h / 2) * H / Q)


def predict(state, dt):
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


# ---------------- bit IO
class BitWriter:
    def __init__(self):
        self.v, self.n = 0, 0

    def u(self, x, bits):
        if not (isinstance(x, int) and 0 <= x < (1 << bits)):
            raise ValueError(f"value {x} does not fit in {bits} unsigned bits")
        self.v = (self.v << bits) | x; self.n += bits

    def s(self, x, bits):
        if not (isinstance(x, int) and -(1 << (bits - 1)) <= x < (1 << (bits - 1))):
            raise ValueError(f"value {x} does not fit in {bits} signed bits")
        self.u(x & ((1 << bits) - 1), bits)

    def bytes(self):
        pad = (-self.n) % 8
        return (self.v << pad).to_bytes((self.n + pad) // 8, "big") if self.n else b""


class BitReader:
    def __init__(self, data):
        self.v, self.n, self.pos = int.from_bytes(data, "big"), 8 * len(data), 0

    def u(self, bits):
        if self.pos + bits > self.n:
            raise ValueError("read past end of message")
        self.pos += bits
        return (self.v >> (self.n - self.pos)) & ((1 << bits) - 1)

    def s(self, bits):
        x = self.u(bits)
        return x - (1 << bits) if x >= (1 << (bits - 1)) else x


def _w_track(w, v):
    c, cx, cy, ww, hh, vx, vy, cf = v
    w.u(c, 4); w.u(cx, 12); w.u(cy, 12); w.u(ww, 12); w.u(hh, 12); w.s(vx, 10); w.s(vy, 10); w.u(cf, 3)


def _r_track(r):
    return (r.u(4), r.u(12), r.u(12), r.u(12), r.u(12), r.s(10), r.s(10), r.u(3))


# ---------------- messages
class Msg:
    __slots__ = ("msg_id", "base", "version", "t_state", "frame_processed", "ops", "h", "snapshot", "bits", "wire")

    def __init__(self, msg_id, base, version, t_state, frame_processed, ops, h, snapshot, bits=0, wire=b""):
        self.msg_id, self.base, self.version, self.t_state = msg_id, base, version, t_state
        self.frame_processed, self.ops, self.h, self.snapshot, self.bits, self.wire = (frame_processed, ops, h,
                                                                                       snapshot, bits, wire)


def pack_state_msg(m, versioned):
    if len(m.ops) > MAX_OPS:
        raise ValueError(f"{len(m.ops)} ops > {MAX_OPS}")
    w = BitWriter()
    w.u(m.msg_id, 16); w.u(UAV_ID, 4)
    if versioned:
        w.u(m.base, 16); w.u(m.version, 16)
    else:
        w.u(int(m.snapshot), 1)
    w.u(m.t_state % MOD, 16); w.u(m.frame_processed % MOD, 16); w.u(len(m.ops), 11)
    if versioned:
        w.u(m.h, 32)
    for kind, k, v in m.ops:
        w.u(k, 10); w.u(kind, 2)
        if kind != DEL:
            if not representable(v):
                raise ValueError(f"track {k} not representable: {v}")
            _w_track(w, v)
    return w.n, w.bytes()


def unpack_state_msg(data, versioned):
    r = BitReader(data)
    msg_id = r.u(16); r.u(4)
    if versioned:
        base, version = r.u(16), r.u(16)
    else:
        base, version, snap = 0, 0, r.u(1)
    t_state, fp, n = r.u(16), r.u(16), r.u(11)
    h = r.u(32) if versioned else 0
    ops = []
    for _ in range(n):
        k, kind = r.u(10), r.u(2)
        if kind not in (ADD, UPD, DEL):
            raise ValueError(f"bad op kind {kind}")
        ops.append((kind, k, None if kind == DEL else _r_track(r)))
    snapshot = (base == 0) if versioned else bool(snap)
    return Msg(msg_id, base, version, t_state, fp, ops, h, snapshot, r.pos, data)


def pack_hint(fp, items, plus):
    """items: list of (cls, cx, cy, w, h, conf) quantised; B drops cls / conf."""
    if len(items) > MAX_OPS:
        raise ValueError("too many boxes")
    w = BitWriter(); w.u(fp % MOD, 16); w.u(len(items), 11)
    for c, cx, cy, ww, hh, cf in items:
        w.u(cx, 12); w.u(cy, 12); w.u(ww, 12); w.u(hh, 12)
        if plus:
            w.u(c, 4); w.u(cf, 3)
    return w.n, w.bytes()


def unpack_hint(data, plus):
    r = BitReader(data); fp, n = r.u(16), r.u(11)
    out = []
    for _ in range(n):
        cx, cy, ww, hh = r.u(12), r.u(12), r.u(12), r.u(12)
        c, cf = (r.u(4), r.u(3)) if plus else (0, 0)
        out.append((c, cx, cy, ww, hh, cf))
    return fp, out


def pack_ack(ticks):
    w = BitWriter(); w.u(len(ticks), 11)
    for t in ticks:
        w.u(t % MOD, 16)
    return w.n, w.bytes()


def unpack_ack(data):
    r = BitReader(data); n = r.u(11)
    return [r.u(16) for _ in range(n)]


def pack_ul_sync(applied, h):
    w = BitWriter(); w.u(applied, 16); w.u(h, 32)
    return w.n, w.bytes()


def unpack_ul_sync(data):
    r = BitReader(data)
    return r.u(16), r.u(32)


class Publisher:
    """BS side: canonical published states by version; builds wire messages. versioned=False -> C0."""

    def __init__(self, versioned=True, n_hist=64):
        self.versioned, self.n_hist = versioned, n_hist
        self.msg_id = 0
        self.version = 0
        self.t_full = 0                         # BS-internal full tick of the published state
        self.state = {}
        self.hist = {0: (0, {}, crc({}, 0))}   # version -> (t_state mod 2^16, state, crc)

    def current_at(self, t):
        return predict(self.state, t - self.t_full)

    def make(self, new_state, t_state, frame_processed, snapshot=False):
        bad = [k for k, v in new_state.items() if not representable(v)]
        if bad:
            raise ValueError(f"non-representable tracks {bad[:5]}")
        old = self.current_at(t_state)
        ops = delta({} if snapshot else old, new_state)
        if not ops and not snapshot:
            return None
        base = 0 if snapshot else self.version
        self.version = s_add(self.version, 1) or 1          # 0 is reserved for the empty initial state
        self.msg_id = s_add(self.msg_id, 1)
        self.state, self.t_full = dict(new_state), t_state
        h = crc(self.state, t_state)
        self.hist[self.version] = (t_state % MOD, dict(self.state), h)
        if len(self.hist) > self.n_hist:
            for v in sorted(self.hist, key=lambda v: s_diff(v, self.version))[:len(self.hist) - self.n_hist]:
                if v != self.version:
                    del self.hist[v]
        # base 0 = the empty initial state, so a delta on it IS a snapshot (same on the wire for C1)
        m = Msg(self.msg_id, base, self.version, t_state % MOD, frame_processed % MOD, ops, h, snapshot or base == 0)
        m.bits, m.wire = pack_state_msg(m, self.versioned)
        return m

    def check_uplink(self, applied, h):
        if applied not in self.hist:
            return True
        return self.hist[applied][2] != h


class Replica:
    """UAV side, C1 (versioned) or C0. Works only on decoded wire messages (16-bit ticks)."""

    def __init__(self, versioned=True, w_hold=10):
        self.versioned, self.w_hold = versioned, w_hold
        self.applied, self.t_state, self.state = 0, 0, {}
        self.h = crc({}, 0)
        self.desync = False
        self.buf = {}
        self.has = False

    def view(self, t):
        return predict(self.state, s_diff(t % MOD, self.t_state))

    def header(self):
        return self.applied, (self.h if not self.desync else self.h ^ 0xFFFFFFFF)

    def receive(self, m, now):
        if not self.versioned:
            return self._recv_c0(m)
        if m.snapshot:
            if s_newer(m.version, self.applied) or self.desync:
                s = apply_ops({}, m.ops)
                if crc(s, m.t_state) == m.h:
                    self._set(m.version, m.t_state, s, m.h)
                    self._drain(now)
                else:
                    self.desync = True
            return
        if not s_newer(m.version, self.applied):
            return
        if m.base == self.applied and not self.desync:
            self._apply(m)
            self._drain(now)
        elif s_newer(m.base, self.applied):
            self.buf[m.base] = (now, m)

    def tick(self, now):
        for b, (t0, _) in list(self.buf.items()):
            if now - t0 > self.w_hold:
                self.desync = True
                del self.buf[b]

    def _apply(self, m):
        s = apply_ops(predict(self.state, s_diff(m.t_state, self.t_state)), m.ops)
        if crc(s, m.t_state) == m.h:
            self._set(m.version, m.t_state, s, m.h)
        else:
            self.desync = True
            self.state, self.t_state = s, m.t_state

    def _set(self, v, t, s, h):
        self.applied, self.t_state, self.state, self.h, self.desync = v, t, s, h, False

    def _drain(self, now):
        while self.applied in self.buf and not self.desync:
            _, m = self.buf.pop(self.applied)
            if s_newer(m.version, self.applied):
                self._apply(m)

    def _recv_c0(self, m):
        if self.has and s_diff(m.t_state, self.t_state) < 0:
            return
        base = {} if m.snapshot else predict(self.state, s_diff(m.t_state, self.t_state) if self.has else 0)
        self.state, self.t_state, self.has = apply_ops(base, m.ops), m.t_state, True
