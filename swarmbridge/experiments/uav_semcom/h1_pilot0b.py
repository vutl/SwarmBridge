"""H1-CROP pilot 0B-TUNE: causal UAV logic, budget packing, BS fusion, scorers and screening criteria.

Protocol: reports/swarmbridge-h1-0b-tune-pilot-protocol.md (sec 3-6), approved for implementation by review #8.
Everything here is pure (numpy / scipy) and runs on precomputed per-frame records written by
h1_pilot0b_precompute.py, so the expensive parts (YOLO11n, AVIF, YOLO11s) run once.

Record of one (epoch, view), produced WITHOUT any label:
    W, H                       image size
    det                        onboard YOLO11n detections (dict b (N,4) xyxy, c (N,) class index, s (N,) conf)
    det_ms                     onboard YOLO11n wall time (CPU, 1 thread)
    cands[P]  (P in P1, P2)    list of dict win (x0,y0,x1,y1) int, conf, bits{q}, enc_ms{q}, bs{q} (YOLO11s dets on
                               the DECODED crop, already in global image coordinates)
    frame                      dict bits, enc_ms, bs (YOLO11s on the decoded full frame, AVIF q FRAME_Q)
GT ({track_id: (x1,y1,x2,y2,cls)}) only enters the scorers and the MARKED T2 oracle.

Conditions (same joint budget B for the 2 UAVs, UAV1 gets alpha*B, UAV2 (1-alpha)*B):
    T0      strong tokens (conf >= TAU_HI)
    T0+     T0, then weak tokens (TAU_LO <= conf < TAU_HI) instead of crops
    T1      T0, then P1 crops (uncertain detection windows), conf-descending
    T1P2    T0, then P2 tiles (R1a 320 px grid tiles holding an uncertain detection)
    F       full AVIF frame if it fits the UAV share, else T0 (fallback counted)
    T2      GT-ASSISTED SUPPRESSION HEURISTIC with costed perfect association (uses GT ids, never deployable; NOT an
            F1-optimal oracle and NOT an upper bound, so T2 - T1 <= 0 does not mean "no cooperative headroom").
            Sequential (review #9): UAV1 schedules first (T1 order, its manifest of SENT items inside its share);
            UAV2 then drops each P1 candidate whose GT object UAV1 actually SENT (a sent crop covering it, or a sent
            strong token matching it). Candidates matching no GT are never dropped. If UAV2 has no candidate, no
            coordination happens (T2 == T1, zero cost).
    T2v     as T2, but an object only counts as delivered by UAV1 if the BS decoding of UAV1's sent items matches it
            (class-aware, conf >= TAU_LO): "sent" vs "sent and recognised".
Resource model (review #9): B is the JOINT UPLINK budget of the two UAVs. Manifests are uplink and charged to the
sending UAV's share (UAV1: 24 bit per sent item; UAV2: 24 bit per P1 candidate). The DL bitmap (one bit per UAV2
candidate + DL header) is a separate link: main setting rho = 0 (DL not charged to B, reported in bits / airtime,
optional DL cap B_dl: if the bitmap does not fit, no suppression happens but manifests are still paid);
sensitivity rho > 0 = shared-time budget where rho = R_UL / R_DL converts DL bits to uplink-equivalent bits.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import h1_gate0a as G

TAU_HI, TAU_LO = 0.25, 0.05
CROP_SCALE, CROP_MIN, MERGE_IOU = 1.5, 64, 0.5
TILE, STRIDE = 320, 288
TOK_BITS = 55                 # R1b hint B+ box: cls 4 | cx 12 | cy 12 | w 12 | h 12 | conf 3
QGRID = 4096
MAN_BITS = 24                 # manifest entry per item (T2 oracle)
NMS_IOU, MATCH_IOU, PROP_IOU = 0.6, 0.5, 0.3
FRAME_Q = 35
NCLS = 4                      # mdmt_adapter.CLASSES: person, car, bicycle, bus
# VisDrone 1..10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor) -> MDMT.
# Fixed before any MDMT image is seen; tricycles have no MDMT class. Class-agnostic scores are reported as a check.
VISDRONE_TO_MDMT = {1: 0, 2: 0, 3: 2, 4: 1, 5: 1, 6: 1, 9: 3, 10: 2}
CONDS = ("T0", "T0+", "T1", "T1P2", "F", "T2", "T2v")
CROP_CONDS = ("T1", "T1P2", "T2", "T2v")
COORD_CONDS = ("T2", "T2v")


# ---------------------------------------------------------------- detections
def dets(b=None, c=None, s=None):
    b = np.zeros((0, 4)) if b is None else np.asarray(b, float).reshape(-1, 4)
    return dict(b=b, c=np.asarray([] if c is None else c, int).reshape(-1),
                s=np.asarray([] if s is None else s, float).reshape(-1))


def cat(ds):
    ds = [d for d in ds if len(d["s"])]
    if not ds:
        return dets()
    return dict(b=np.concatenate([d["b"] for d in ds]), c=np.concatenate([d["c"] for d in ds]),
                s=np.concatenate([d["s"] for d in ds]))


def take(d, idx):
    idx = np.asarray(idx, int)
    return dict(b=d["b"][idx].reshape(-1, 4), c=d["c"][idx], s=d["s"][idx])


def map_visdrone(xyxy, cls1, conf):
    """detector output (classes 1..10) -> MDMT classes; unmapped classes dropped."""
    keep = [k for k, c in enumerate(cls1) if int(c) in VISDRONE_TO_MDMT]
    return dets(np.asarray(xyxy, float).reshape(-1, 4)[keep], [VISDRONE_TO_MDMT[int(cls1[k])] for k in keep],
                np.asarray(conf, float)[keep])


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def nms_classwise(d, thr=NMS_IOU):
    keep = []
    for c in np.unique(d["c"]):
        idx = np.where(d["c"] == c)[0]
        idx = idx[np.argsort(-d["s"][idx], kind="stable")]
        M = iou_matrix(d["b"][idx], d["b"][idx])
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if alive[i]:
                keep.append(idx[i])
                alive[i + 1:] &= M[i, i + 1:] <= thr
    return take(d, sorted(keep))


# ---------------------------------------------------------------- crop candidates (causal, no GT)
def crop_window(box, W, H, scale=CROP_SCALE, min_side=CROP_MIN):
    """box * scale around its centre, each side >= min_side (and <= image side), shifted inside the image."""
    x1, y1, x2, y2 = box
    w = min(max((x2 - x1) * scale, min_side), W); h = min(max((y2 - y1) * scale, min_side), H)
    x0 = min(max((x1 + x2) / 2 - w / 2, 0.0), W - w); y0 = min(max((y1 + y2) / 2 - h / 2, 0.0), H - h)
    return (int(math.floor(x0)), int(math.floor(y0)), min(int(math.ceil(x0 + w)), W), min(int(math.ceil(y0 + h)), H))


def merge_windows(wins, confs, thr=MERGE_IOU):
    """repeatedly replace the most-overlapping pair (IoU > thr) by its union (conf = max) until none is left."""
    wins = [tuple(w) for w in wins]; confs = list(confs)
    while len(wins) > 1:
        M = iou_matrix(wins, wins); np.fill_diagonal(M, -1)
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] <= thr:
            break
        a, b = wins[i], wins[j]
        u = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        c = max(confs[i], confs[j])
        for k in sorted((i, j), reverse=True):
            del wins[k]; del confs[k]
        wins.append(u); confs.append(c)
    return wins, confs


def _sorted(wins, confs):
    order = sorted(range(len(wins)), key=lambda k: (-confs[k], wins[k]))
    return [dict(win=tuple(int(v) for v in wins[k]), conf=float(confs[k])) for k in order]


def uncertain(d):
    return np.where((d["s"] >= TAU_LO) & (d["s"] < TAU_HI))[0]


def p1_candidates(d, W, H):
    u = uncertain(d)
    wins = [crop_window(d["b"][k], W, H) for k in u]
    return _sorted(*merge_windows(wins, [d["s"][k] for k in u]))


def tiles(W, H, tile=TILE, stride=STRIDE):
    """R1a grid: 320 px tiles, stride 288, last tile aligned to the border."""
    xs = list(range(0, max(W - tile, 0) + 1, stride)); ys = list(range(0, max(H - tile, 0) + 1, stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def p2_candidates(d, W, H):
    u = uncertain(d)
    cx = (d["b"][u, 0] + d["b"][u, 2]) / 2; cy = (d["b"][u, 1] + d["b"][u, 3]) / 2
    wins, confs = [], []
    for t in tiles(W, H):
        inside = (cx >= t[0]) & (cx < t[2]) & (cy >= t[1]) & (cy < t[3])
        if inside.any():
            wins.append(t); confs.append(float(d["s"][u][inside].max()))
    return _sorted(wins, confs)


def to_global(d, win):
    """detections on a crop (crop pixel coords) -> image coords."""
    b = d["b"].copy(); b[:, [0, 2]] += win[0]; b[:, [1, 3]] += win[1]
    return dict(b=b, c=d["c"].copy(), s=d["s"].copy())


# ---------------------------------------------------------------- tokens
def quant_tokens(d, W, H):
    """wire quantisation of the token payload: box on the R1b 4096 grid, conf on 3 bits (decoded mid-bin)."""
    if not len(d["s"]):
        return dets()
    b = d["b"]
    cx = np.clip(np.round((b[:, 0] + b[:, 2]) / 2 * QGRID / W), 0, QGRID - 1)
    cy = np.clip(np.round((b[:, 1] + b[:, 3]) / 2 * QGRID / H), 0, QGRID - 1)
    w = np.clip(np.round((b[:, 2] - b[:, 0]) * QGRID / W), 1, QGRID - 1)
    h = np.clip(np.round((b[:, 3] - b[:, 1]) * QGRID / H), 1, QGRID - 1)
    qb = np.stack([(cx - w / 2) * W / QGRID, (cy - h / 2) * H / QGRID, (cx + w / 2) * W / QGRID,
                   (cy + h / 2) * H / QGRID], 1)
    k = np.clip(np.floor(d["s"] * 8), 0, 7)
    return dict(b=qb, c=d["c"].copy(), s=(k + 0.5) / 8)


def strong_idx(d):
    i = np.where(d["s"] >= TAU_HI)[0]
    return i[np.argsort(-d["s"][i], kind="stable")]


def weak_idx(d):
    i = uncertain(d)
    return i[np.argsort(-d["s"][i], kind="stable")]


# ---------------------------------------------------------------- budget packing
def msg_cost(payload):
    return payload + G.msg_overhead(payload)


def pack(bits, budget):
    """greedy in the given order; an item that does not fit is skipped and later (smaller) ones are still tried.
    Cost of a UAV message = payload + per-MTU headers (h1_gate0a.msg_overhead). -> (selected positions, payload)."""
    sel, pay = [], 0
    for k, b in enumerate(bits):
        if msg_cost(pay + b) <= budget:
            sel.append(k); pay += b
    return sel, pay


def uav_items(rec, cond, q, drop=frozenset()):
    """ordered (kind, index) list with bits for one UAV under a condition; F handled separately."""
    d = rec["det"]
    items = [("tok", int(k), TOK_BITS) for k in strong_idx(d)]
    if cond == "T0+":
        items += [("tok", int(k), TOK_BITS) for k in weak_idx(d)]
    elif cond in ("T1",) + COORD_CONDS:
        items += [("P1", k, c["bits"][q]) for k, c in enumerate(rec["cands"]["P1"]) if k not in drop]
    elif cond == "T1P2":
        items += [("P2", k, c["bits"][q]) for k, c in enumerate(rec["cands"]["P2"])]
    return items


def send(rec, cond, q, budget, drop=frozenset()):
    """UAV side -> dict(items sent, payload bits, total UL bits incl. headers, frame flag)."""
    if cond == "F":
        fb = rec["frame"]["bits"]
        if msg_cost(fb) <= budget:
            return dict(items=[("frame", 0, fb)], payload=fb, ul=msg_cost(fb), frame=True)
        cond = "T0"
    items = uav_items(rec, cond, q, drop)
    sel, pay = pack([it[2] for it in items], budget)
    return dict(items=[items[k] for k in sel], payload=pay, ul=msg_cost(pay), frame=False)


def bs_decode(rec, sent, q):
    """BS view of one UAV: quantised tokens + YOLO11s detections on decoded crops / frame, class-wise NMS."""
    W, H, d = rec["W"], rec["H"], rec["det"]
    tok = [k for kind, k, _ in sent["items"] if kind == "tok"]
    parts = [quant_tokens(take(d, tok), W, H)]
    for kind, k, _ in sent["items"]:
        if kind in ("P1", "P2"):
            parts.append(rec["cands"][kind][k]["bs"][q])
        elif kind == "frame":
            parts.append(rec["frame"]["bs"])
    return nms_classwise(cat(parts))


# ---------------------------------------------------------------- GT helpers (scorer / marked oracle only)
def gt_arrays(gt):
    ids = sorted(gt)
    b = np.array([gt[i][:4] for i in ids], float).reshape(-1, 4)
    return ids, b, np.array([gt[i][4] for i in ids], int)


def match(d, gt, thr=MATCH_IOU, class_aware=True):
    """Hungarian per class (max #matches with IoU >= thr, then max IoU). -> list of (det index, gt id)."""
    ids, gb, gc = gt_arrays(gt)
    out = []
    classes = np.unique(np.concatenate([d["c"], gc])) if class_aware else [None]
    for c in classes:
        di = np.where(d["c"] == c)[0] if class_aware else np.arange(len(d["s"]))
        gi = np.where(gc == c)[0] if class_aware else np.arange(len(ids))
        if not len(di) or not len(gi):
            continue
        M = iou_matrix(d["b"][di], gb[gi])
        Wt = np.where(M >= thr, 1.0 + M, 0.0)
        r, k = linear_sum_assignment(-Wt)
        out += [(int(di[a]), ids[gi[b]]) for a, b in zip(r, k) if Wt[a, b] > 0]
    return out


def cover(wins, gt):
    """-> (ids, IoU matrix, bool matrix): window covers a GT box if it contains the GT centre or IoU >= PROP_IOU."""
    ids, gb, _ = gt_arrays(gt)
    w = np.asarray(wins, float).reshape(-1, 4)
    M = iou_matrix(w, gb)
    cx, cy = (gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2
    inside = ((w[:, None, 0] <= cx[None]) & (cx[None] < w[:, None, 2]) & (w[:, None, 1] <= cy[None])
              & (cy[None] < w[:, None, 3]))
    return ids, M, inside | (M >= PROP_IOU)


def cand_gt_all(wins, gt):
    """per window: the covered GT id with the best IoU (ties: smallest id), or None."""
    ids, M, C = cover(wins, gt)
    out = []
    for k in range(len(C)):
        j = np.where(C[k])[0]
        out.append(ids[j[np.argmax(M[k, j])]] if len(j) else None)   # ids sorted -> argmax picks smallest on ties
    return out


def cand_gt(win, gt):
    return cand_gt_all([win], gt)[0]


def covered_ids(win, gt):
    """all GT ids whose centre is in the window or IoU >= PROP_IOU (proposal recall, C2)."""
    ids, _, C = cover([win], gt)
    return {ids[j] for j in np.where(C[0])[0]}


def false_flags(rec, gt, kind):
    """per candidate of `kind`: True if it covers no GT object (false candidate)."""
    c = rec["cands"][kind]
    if not c:
        return []
    return (~cover([x["win"] for x in c], gt)[2].any(1)).tolist()


def t2_context(rec1, rec2, gt1, gt2):
    """budget-independent GT lookups for the T2 heuristics (cacheable per epoch)."""
    d1 = rec1["det"]
    tq = quant_tokens(d1, rec1["W"], rec1["H"])                 # what the BS would receive for each token
    si = strong_idx(d1)
    tok_gt = {int(si[p]): g for p, g in match(take(tq, si), gt1)}
    return dict(c1=cand_gt_all([c["win"] for c in rec1["cands"]["P1"]], gt1),
                c2=cand_gt_all([c["win"] for c in rec2["cands"]["P1"]], gt2), tok1=tok_gt)


def delivered_ids(rec1, s1, gt1, ctx, verified, q):
    """GT objects UAV1 actually sent (T2) or sent AND the BS recognised from what it sent (T2v)."""
    if verified:
        d = bs_decode(rec1, s1, q)
        return {g for _, g in match(take(d, np.where(d["s"] >= TAU_LO)[0]), gt1)}
    out = set()
    for kind, k, _ in s1["items"]:
        g = ctx["c1"][k] if kind == "P1" else ctx["tok1"].get(k) if kind == "tok" else None
        if g is not None:
            out.add(g)
    return out


def send_with_manifest(rec, cond, q, budget, man_bits=MAN_BITS):
    """schedule, then drop trailing items until payload message + manifest message (man_bits per SENT item) fit."""
    s = send(rec, cond, q, budget)
    items = list(s["items"])
    while items:
        pay = sum(b for _, _, b in items)
        if msg_cost(pay) + msg_cost(man_bits * len(items)) <= budget:
            break
        items.pop()
    pay = sum(b for _, _, b in items)
    return dict(items=items, payload=pay, ul=msg_cost(pay), frame=False), (msg_cost(man_bits * len(items))
                                                                           if items else 0)


def run_coord(rec1, rec2, gt1, gt2, cond, q, B, alpha, rho=0.0, B_dl=math.inf, ctx=None):
    """T2 / T2v (GT-assisted suppression heuristic, sequential, costed)."""
    nB = len(rec2["cands"]["P1"])
    if nB == 0:                                                 # nothing to coordinate
        e = run_epoch(rec1, rec2, gt1, gt2, "T1", q, B, alpha)
        e.update(man_ul=0, coord_ok=False)
        return e
    ctx = ctx or t2_context(rec1, rec2, gt1, gt2)
    dl = nB + G.DL_HDR
    Bu = B - rho * dl                                           # shared-time sensitivity only (rho > 0)
    man2 = msg_cost(MAN_BITS * nB)
    s1, man1 = send_with_manifest(rec1, "T1", q, alpha * Bu) if alpha * Bu > 0 else (
        dict(items=[], payload=0, ul=0, frame=False), 0)
    coord_ok = dl <= B_dl and (1 - alpha) * Bu - man2 > 0
    drop = frozenset()
    if coord_ok:
        have = delivered_ids(rec1, s1, gt1, ctx, cond == "T2v", q)
        drop = frozenset(k for k, g in enumerate(ctx["c2"]) if g is not None and g in have)
    b2 = (1 - alpha) * Bu - man2
    s2 = send(rec2, "T1", q, b2, drop) if b2 > 0 else dict(items=[], payload=0, ul=0, frame=False)
    return dict(decl=(bs_decode(rec1, s1, q), bs_decode(rec2, s2, q)), ul=s1["ul"] + man1 + s2["ul"] + man2,
                dl=dl, man_ul=man1 + man2, sent=(s1, s2), drop=drop, coord_ok=coord_ok)


# ---------------------------------------------------------------- one epoch, one condition
def run_epoch(rec1, rec2, gt1, gt2, cond, q, B, alpha, rho=0.0, B_dl=math.inf, ctx=None):
    """-> dict(decl=(d1, d2), ul, dl, sent=(s1, s2)). GT only used by the T2 / T2v heuristics."""
    if cond in COORD_CONDS:
        return run_coord(rec1, rec2, gt1, gt2, cond, q, B, alpha, rho, B_dl, ctx)
    s1 = send(rec1, cond, q, alpha * B)
    s2 = send(rec2, cond, q, (1 - alpha) * B)
    return dict(decl=(bs_decode(rec1, s1, q), bs_decode(rec2, s2, q)), ul=s1["ul"] + s2["ul"], dl=0, man_ul=0,
                sent=(s1, s2), drop=frozenset())


# ---------------------------------------------------------------- scoring
def view_counts(decl, gt, tau, class_aware=True):
    d = take(decl, np.where(decl["s"] >= tau)[0])
    m = match(d, gt, class_aware=class_aware)
    return dict(tp=len(m), fp=len(d["s"]) - len(m), fn=len(gt) - len(m), ids={g for _, g in m})


def unique_counts(v1, v2, gt1, gt2):
    """ideal association at the BS for every condition: a GT object found in either view counts once; every
    unmatched declaration is FP_wrong (no FP_dup / FP_assoc under ideal association)."""
    U = set(gt1) | set(gt2)
    tp = len((v1["ids"] | v2["ids"]) & U)
    return dict(tp=tp, fp=v1["fp"] + v2["fp"], fn=len(U) - tp)


def f1(c):
    den = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / den if den else 1.0


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
    return p, r, f1(c)


def add(acc, c):
    for k in ("tp", "fp", "fn"):
        acc[k] = acc.get(k, 0) + c[k]
    return acc


# ---------------------------------------------------------------- screening measurements (C1, C2, C4, C5)
def c1_c2(rec, gt, q=FRAME_Q):
    """per view: GT missed by strong onboard detections (class-aware), how many of them a P1 candidate covers,
    false candidates (cover no GT) and their bits."""
    d = rec["det"]
    hit = {g for _, g in match(take(d, strong_idx(d)), gt)}
    missed = set(gt) - hit
    cov, false_n, false_bits = set(), 0, 0
    cs = rec["cands"]["P1"]
    if cs:
        ids, _, C = cover([c["win"] for c in cs], gt)
        cov = {ids[j] for j in np.where(C.any(0))[0]}
        for c, hit in zip(cs, C.any(1)):
            if not hit:
                false_n += 1; false_bits += c["bits"][q]
    return dict(n_gt=len(gt), missed=len(missed), missed_covered=len(missed & cov), n_cand=len(rec["cands"]["P1"]),
                false_cand=false_n, false_bits=false_bits)


def c4_redundancy(rec1, rec2, gt1, gt2):
    """PROPOSAL-LEVEL redundancy (before any budget): candidates matched to a GT object, and how many of those the
    other view also proposes or already detects with a strong token (both directions). Says how much overlap
    exists, not how much a scheduler can save."""
    n = red = 0
    for ra, rb, ga, gb in ((rec1, rec2, gt1, gt2), (rec2, rec1, gt2, gt1)):
        other = set(cand_gt_all([c["win"] for c in rb["cands"]["P1"]], gb)) - {None}
        other |= {g for _, g in match(take(rb["det"], strong_idx(rb["det"])), gb)}
        for g in cand_gt_all([c["win"] for c in ra["cands"]["P1"]], ga):
            if g is not None:
                n += 1; red += g in other
    return dict(matched=n, redundant=red)


def cpu_ms_all(rec, cond, q):
    """PRIMARY onboard CPU per epoch (review #9): exact bit lengths are only known after encoding, so the UAV
    encodes EVERY candidate the condition may send before selecting: YOLO11n + all encodes of that candidate set."""
    if cond in ("T0", "T0+"):
        return rec["det_ms"]
    if cond == "F":
        return rec["det_ms"] + rec["frame"]["enc_ms"]
    kind = "P2" if cond == "T1P2" else "P1"
    return rec["det_ms"] + sum(c["enc_ms"][q] for c in rec["cands"][kind])


def cpu_ms(rec, q, sent=None):
    """onboard CPU per epoch: YOLO11n + AVIF encode of all P1 candidates (C5), or, SUPPLEMENTARY and optimistic,
    only of the items actually sent (valid only if bit lengths could be predicted without encoding)."""
    if sent is None:
        return cpu_ms_all(rec, "T1", q)
    return rec["det_ms"] + sum(rec["cands"][kind][k]["enc_ms"][q] for kind, k, _ in sent["items"]
                               if kind in ("P1", "P2")) + sum(rec["frame"]["enc_ms"] for kind, _, _ in sent["items"]
                                                               if kind == "frame")


def budget_levels(pairs, q=FRAME_Q, fracs=(0.25, 0.5, 1.0)):
    """registered rule (protocol sec 4): B = fracs x mean per-epoch stream 'strong tokens + every P1 crop' of both
    UAVs (headers included), plus one level that always fits two full frames (alpha = 0.5)."""
    s = [sum(msg_cost(TOK_BITS * len(strong_idx(r["det"])) + sum(c["bits"][q] for c in r["cands"]["P1"]))
             for r in (r1, r2)) for r1, r2, _, _ in pairs]
    m = float(np.mean(s)) if s else 0.0
    bF = 2 * max((msg_cost(r["frame"]["bits"]) for r1, r2, _, _ in pairs for r in (r1, r2)), default=0)
    return dict(stream_mean=m, levels=[round(f * m) for f in fracs], frame_level=int(bF))
