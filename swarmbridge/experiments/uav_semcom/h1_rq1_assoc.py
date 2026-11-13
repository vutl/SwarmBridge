"""RQ1 feasibility test (reports/swarmbridge-rq1-association-preregistration.md): how much unique-object F1 is lost
when the BS must ASSOCIATE detections across the two UAV views itself, and how much of that loss an oracle recovers
by "sending a crop" only for ambiguous detections. Offline, on the existing pilot 0B records; labels are used for
scoring, for the calibration phase (first half) and for the marked oracle only.

Per epoch each view declares detections at the BS (TOKEN: quantised strong tokens; FRAME: YOLO11s on the AVIF frame),
kept at score >= TAU. Association conditions:
  ideal   pairs = same GT id (B1 scoring; FP_dup = FP_assoc = 0)
  fresh   homography view 1 -> view 2 fitted on the GT boxes of the SAME epoch (bottom-centres, RANSAC), then
          label-free matching: same class, Hungarian on reprojection distance, gate MERGE_PX ("perfect pose" bound)
  stale   as fresh, but the homography is fitted once on the LAST first-half epoch with >= MIN_COMMON common ids and
          reused for the whole second half ("calibration then drift"; the realistic condition, primary)
  none    no association at all (every detection is its own object): lower reference
Ambiguous detection: a same-class candidate in the other view within AMB_HI px AND (>= 2 such candidates, or the best
one farther than AMB_LO px). Oracle: every ambiguous detection gets the GT-correct association; cost = one crop per
ambiguous detection (median P1 crop bits at q50 of the sequence's first half; bits only).
Fused unique-object counting: each fused object (a merged pair or a single detection) may claim at most one GT id
(maximum matching); an object whose ids are all claimed by other objects is FP_dup; an object with no id is FP_wrong;
a merged pair of two different ids is counted as FP_assoc (diagnostic: one of the two objects is lost).
Usage: python h1_rq1_assoc.py PRECOMP_DIR ID_CHECK_JSON B1_JSON OUT_JSON --zip MDMT_ZIP --data-use-basis "..."
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import h1_pilot0b as P

TAU, Q, CROP_Q = 0.3, 35, 50
MERGE_PX, AMB_LO, AMB_HI, MIN_COMMON, RANSAC_PX = 40.0, 20.0, 80.0, 8, 40.0
MODES = ("TT", "FV")                         # both TOKEN; FRAME on the B1 NDET view + TOKEN on the other
ASSOC = ("ideal", "fresh", "stale", "none")
GO_DELTA, NOGO_DELTA, GO_RECOVER, GO_BITS_SHARE, GO_SEQ_SHARE = 0.03, 0.01, 0.5, 0.10, 6 / 8


def foot(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)


def declared(rec, mode):
    """BS-side declarations of one view in mode 'T' or 'F' (score >= TAU) and the UL bits."""
    snd = P.send(rec, "T0" if mode == "T" else "F", Q, 1e12)
    d = P.bs_decode(rec, snd, Q)
    return P.take(d, np.where(d["s"] >= TAU)[0]), snd["ul"]


def homography(g1, g2):
    """view 1 -> view 2 homography from the GT boxes of the ids present in both views (None if too few)."""
    import cv2
    ids = sorted(set(g1) & set(g2))
    if len(ids) < MIN_COMMON:
        return None
    p1 = foot([g1[i][:4] for i in ids]).astype(np.float32)
    p2 = foot([g2[i][:4] for i in ids]).astype(np.float32)
    H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    return H


def project(H, pts):
    if not len(pts):
        return pts
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return q[:, :2] / np.where(np.abs(q[:, 2:]) < 1e-9, 1e-9, q[:, 2:])


def distances(d1, d2, H):
    """reprojection distance (px, in view 2) for every same-class pair; inf for other classes / no homography."""
    D = np.full((len(d1["s"]), len(d2["s"])), np.inf)
    if H is None or not D.size:
        return D
    a, b = project(H, foot(d1["b"])), foot(d2["b"])
    E = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return np.where(d1["c"][:, None] == d2["c"][None, :], E, np.inf)


def associate(D, gate=MERGE_PX):
    """label-free cross-view pairs: Hungarian on distance, gated."""
    if not D.size or not np.isfinite(D).any():
        return []
    C = np.where(D <= gate, D, 1e6)
    r, k = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(r, k) if D[i, j] <= gate]


def ambiguous(D):
    """(set of view-1 indices, set of view-2 indices) whose association is ambiguous."""
    out = []
    for M in (D, D.T):
        s = set()
        for i in range(M.shape[0]):
            c = np.sort(M[i][M[i] <= AMB_HI])
            if len(c) >= 2 or (len(c) == 1 and c[0] > AMB_LO):
                s.add(i)
        out.append(s)
    return out[0], out[1]


def gt_ids(d, g):
    return dict(P.match(d, g))


def ideal_pairs(d1, d2, g1, g2):
    a, b = gt_ids(d1, g1), gt_ids(d2, g2)
    inv = {v: k for k, v in b.items()}
    return [(i, inv[v]) for i, v in a.items() if v in inv]


def oracle_pairs(pairs, ideal, A1, A2):
    """geometric pairs that touch no ambiguous detection + GT-correct pairs for the ambiguous ones."""
    keep = [(i, j) for i, j in pairs if i not in A1 and j not in A2]
    u1, u2 = {i for i, _ in keep}, {j for _, j in keep}
    for i, j in ideal:
        if (i in A1 or j in A2) and i not in u1 and j not in u2:
            keep.append((i, j)); u1.add(i); u2.add(j)
    return keep


def fused_counts(d1, d2, g1, g2, pairs):
    """unique-object counts after fusing the declarations with the given cross-view pairs."""
    a, b = gt_ids(d1, g1), gt_ids(d2, g2)
    p1, p2 = {i for i, _ in pairs}, {j for _, j in pairs}
    objs = [({a[i]} if i in a else set()) | ({b[j]} if j in b else set()) for i, j in pairs]
    objs += [{a[i]} if i in a else set() for i in range(len(d1["s"])) if i not in p1]
    objs += [{b[j]} if j in b else set() for j in range(len(d2["s"])) if j not in p2]
    U = sorted(set(g1) | set(g2))
    col = {u: k for k, u in enumerate(U)}
    tp = 0
    claimed = set()
    if objs and U:
        W = np.zeros((len(objs), len(U)))
        for k, ids in enumerate(objs):
            for u in ids:
                if u in col:
                    W[k, col[u]] = 1.0
        r, c = linear_sum_assignment(-W)
        claimed = {int(k) for k, x in zip(r, c) if W[k, x] > 0}
        tp = len(claimed)
    fp_wrong = sum(1 for ids in objs if not ids)
    fp_dup = sum(1 for k, ids in enumerate(objs) if ids and k not in claimed)
    fp_assoc = sum(1 for (i, j) in pairs if i in a and j in b and a[i] != b[j])
    return dict(tp=tp, fp=fp_wrong + fp_dup, fn=len(U) - tp, fp_wrong=fp_wrong, fp_dup=fp_dup, fp_assoc=fp_assoc)


def epoch(r1, r2, g1, g2, modes, H_stale):
    d1, b1 = declared(r1, modes[0])
    d2, b2 = declared(r2, modes[1])
    ideal = ideal_pairs(d1, d2, g1, g2)
    out = dict(bits=b1 + b2)
    for name, H in (("fresh", homography(g1, g2)), ("stale", H_stale)):
        D = distances(d1, d2, H)
        pairs = associate(D)
        A1, A2 = ambiguous(D)
        out[name] = fused_counts(d1, d2, g1, g2, pairs)
        out[name + "_oracle"] = fused_counts(d1, d2, g1, g2, oracle_pairs(pairs, ideal, A1, A2))
        out[name + "_n_amb"] = len(A1) + len(A2)
        out[name + "_amb_ok"] = int(H is not None)
    out["ideal"] = fused_counts(d1, d2, g1, g2, ideal)
    out["none"] = fused_counts(d1, d2, g1, g2, [])
    return out


def crop_bits(recs):
    b = [c["bits"][CROP_Q] for r in recs for c in r["cands"]["P1"]]
    return float(np.median(b)) if b else 0.0


def run_sequence(pairs, frame_view):
    """frame_view = the view that sends the FRAME in mode FV (B1 NDET choice, label-free)."""
    h = len(pairs) // 2
    H_stale = None
    for r1, r2, g1, g2 in pairs[:h]:
        H = homography(g1, g2)
        if H is not None:
            H_stale = H                                   # the LAST usable first-half epoch
    cb = crop_bits([r for p in pairs[:h] for r in p[:2]]) or crop_bits([r for p in pairs for r in p[:2]])
    if not cb:
        cb = float("inf")                                 # no crop size available: the oracle cannot be "free"
    fb = float(np.median([p[v]["frame"]["bits"] for p in pairs[:h] for v in (0, 1)])) if h else 0.0
    res = dict(n_eval=len(pairs) - h, stale_available=H_stale is not None, crop_bits=cb, frame_bits=fb, modes={})
    for mode in MODES:
        modes = ("T", "T") if mode == "TT" else (("F", "T") if frame_view == 0 else ("T", "F"))
        acc, n_amb, bits = {}, {"fresh": 0, "stale": 0}, []
        for r1, r2, g1, g2 in pairs[h:]:
            e = epoch(r1, r2, g1, g2, modes, H_stale)
            for k in ("ideal", "fresh", "stale", "none", "fresh_oracle", "stale_oracle"):
                acc.setdefault(k, {})
                for x in ("tp", "fp", "fn", "fp_wrong", "fp_dup", "fp_assoc"):
                    acc[k][x] = acc[k].get(x, 0) + e[k][x]
            for k in n_amb:
                n_amb[k] += e[k + "_n_amb"]
            bits.append(e["bits"])
        m = {k: dict(f1=P.prf(v)[2], p=P.prf(v)[0], r=P.prf(v)[1], **v) for k, v in acc.items()}
        n = max(len(pairs) - h, 1)
        for k in n_amb:
            m[k + "_oracle"]["extra_bits_per_epoch"] = cb * n_amb[k] / n if n_amb[k] else 0.0
            m[k + "_oracle"]["amb_per_epoch"] = n_amb[k] / n
        m["bits_per_epoch"] = float(np.mean(bits)) if bits else 0.0
        res["modes"][mode] = m
    return res


def gate(per_seq, mode="TT", assoc="stale"):
    """pre-registered decision on (mode, assoc); every other combination is reported only."""
    seqs = sorted(per_seq)
    f = lambda s, k: per_seq[s]["modes"][mode][k]["f1"]
    dA = [f(s, "ideal") - f(s, assoc) for s in seqs]
    rec = [f(s, assoc + "_oracle") - f(s, assoc) for s in seqs]
    mean_dA = float(np.mean(dA))
    recovery = float(np.mean(rec)) / mean_dA if mean_dA > 1e-12 else 0.0
    extra = float(np.mean([per_seq[s]["modes"][mode][assoc + "_oracle"]["extra_bits_per_epoch"] for s in seqs]))
    fbits = float(np.mean([per_seq[s]["frame_bits"] for s in seqs]))
    share_pos = float(np.mean([x > 0 for x in dA]))
    if mean_dA < NOGO_DELTA:
        verdict = "NO-GO"
    elif (mean_dA >= GO_DELTA and recovery >= GO_RECOVER and extra <= GO_BITS_SHARE * fbits
          and share_pos >= GO_SEQ_SHARE - 1e-9):
        verdict = "GO"
    else:
        verdict = "INCONCLUSIVE"
    return dict(mode=mode, assoc=assoc, delta_A=mean_dA, delta_A_per_seq=dict(zip(map(str, seqs), dA)),
                share_seq_positive=share_pos, recovery=recovery, oracle_extra_bits=extra, frame_bits=fbits,
                extra_over_frame=extra / fbits if fbits else float("inf"), verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("precomp"); ap.add_argument("idcheck"); ap.add_argument("b1"); ap.add_argument("out")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data-use-basis", required=True,
                    help="provenance note only: records the basis, does not verify or create an authorisation")
    a = ap.parse_args()
    import h1_pilot0b_eval as E
    ok = {int(s) for s, v in json.loads(Path(a.idcheck).read_text())["sequences"].items() if v["pass"]}
    b1 = json.loads(Path(a.b1).read_text())["per_seq"]
    pairs, _ = E.load_pairs(a.precomp, a.zip, ("test",))
    per = {s: run_sequence(pairs[s], int(b1[str(s)]["pref"]["NDET"])) for s in sorted(pairs) if s in ok}
    res = dict(status="EXPLORATORY feasibility test (EXPL labels; GT used for calibration, scoring and the marked oracle)",
               sequences=sorted(per), per_seq={str(s): v for s, v in per.items()},
               gate=gate(per), secondary={f"{m}/{x}": gate(per, m, x) for m in MODES for x in ("fresh", "stale")},
               params=dict(TAU=TAU, MERGE_PX=MERGE_PX, AMB_LO=AMB_LO, AMB_HI=AMB_HI, MIN_COMMON=MIN_COMMON,
                           CROP_Q=CROP_Q), data_use_basis=a.data_use_basis)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(dict(gate=res["gate"], secondary=res["secondary"]), indent=1, default=str))


if __name__ == "__main__":
    main()
