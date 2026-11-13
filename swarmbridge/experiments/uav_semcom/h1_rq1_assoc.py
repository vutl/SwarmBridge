"""RQ1 v2 (reports/swarmbridge-rq1-association-preregistration.md, review #23): cross-view association as a reason to
transmit. Offline on the existing pilot 0B records: no new image, detector run, AVIF encode, feature or training.

  phase0   first-half epochs only: can a GT-FREE cross-view association be estimated from the recorded geometry /
           classes, and do the cached crops cover the ambiguous detections? GT is a diagnostic ruler only.
  phase12  (only if phase0 allows) second half: RQ1-A association error budget (ideal vs GT-free vs none, TT and a
           one-FRAME schedule under the B1 cap) and RQ1-B (baseline / REAL-CROP with cached YOLO11s crop detections /
           marked GT-ORACLE).

The GT-free estimator never sees a label: similarity RANSAC over all same-class pairs (or the previous epoch's valid
homography), two homography refinements on mutual nearest neighbours, validity = finite, |det| in [0.1, 10] and
>= MIN_PAIRS mutual nearest neighbours within MERGE_PX. Invalid epoch -> no association (never a GT fallback).
Usage:
  python h1_rq1_assoc.py phase0  PRECOMP_DIR ID_CHECK_JSON OUT_JSON --zip MDMT_ZIP --data-use-basis "..."
  python h1_rq1_assoc.py phase12 PRECOMP_DIR ID_CHECK_JSON B1_JSON PHASE0_JSON OUT_JSON --zip ... --data-use-basis ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import h1_pilot0b as P

TAU, Q, CROP_Q, REQ_BITS = 0.3, 35, 50, 24
MERGE_PX, AMB_LO, AMB_HI, MIN_PAIRS, RANSAC_PX = 40.0, 20.0, 80.0, 8, 40.0
SIM_ITERS, DET_LO, DET_HI, SCALE_LO, SCALE_HI, MIN_BASE_PX = 20000, 0.1, 10.0, 1 / 3, 3.0, 50.0
NO_CAP = 1e18
# phase 0 criteria
P0_VALID_SHARE, P0_DIAG_PX, P0_SEQ_PASS, P0_COVER = 0.5, 40.0, 6, 0.25
# phase 1 / 2 criteria
NOGO_DELTA, GO_DELTA, GO_SEQ_SHARE, MAX_FAIL, GO_RECOVER, GO_BITS_SHARE = 0.01, 0.03, 6 / 8, 0.5, 0.5, 0.10


# ---------------------------------------------------------------- geometry and association (no labels)
def foot(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)


def centre(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def project(H, pts):
    if not len(pts):
        return np.zeros((0, 2))
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return q[:, :2] / np.where(np.abs(q[:, 2:]) < 1e-9, 1e-9, q[:, 2:])


def distances(d1, d2, H):
    """reprojection distance (px, in view 2) of every same-class pair; inf for other classes / no homography."""
    D = np.full((len(d1["s"]), len(d2["s"])), np.inf)
    if H is None or not D.size:
        return D
    a, b = project(H, foot(d1["b"])), foot(d2["b"])
    E = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return np.where(d1["c"][:, None] == d2["c"][None, :], E, np.inf)


def mutual_nn(D, gate):
    if not D.size:
        return []
    r, c = D.argmin(1), D.argmin(0)
    return [(i, int(r[i])) for i in range(D.shape[0]) if D[i, r[i]] <= gate and c[r[i]] == i]


def similarity_ransac(d1, d2, iters=SIM_ITERS, seed=0):
    """2-point similarity RANSAC over all same-class pairs. Scale restricted to [SCALE_LO, SCALE_HI]; the score is
    the number of DISTINCT view-2 detections that are the nearest same-class neighbour (within MERGE_PX) of some
    projected view-1 detection, so a degenerate model that collapses everything onto one spot cannot win."""
    p1, p2 = foot(d1["b"]), foot(d2["b"])
    I, J = np.nonzero(d1["c"][:, None] == d2["c"][None, :])
    if len(I) < 2:
        return None, 0
    rng = np.random.default_rng(seed)
    same = d1["c"][:, None] == d2["c"][None, :]
    best, best_s = None, 0
    a, b = rng.integers(0, len(I), iters), rng.integers(0, len(I), iters)
    for x, y in zip(a, b):
        i1, i2, j1, j2 = I[x], I[y], J[x], J[y]
        if i1 == i2 or j1 == j2:
            continue
        u, v = p1[i2] - p1[i1], p2[j2] - p2[j1]
        nu = np.hypot(*u)
        if nu < MIN_BASE_PX:
            continue
        s = np.hypot(*v) / nu
        if not SCALE_LO <= s <= SCALE_HI:
            continue
        ang = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
        Rm = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        t = p2[j1] - Rm @ p1[i1]
        q = p1 @ Rm.T + t
        E = np.linalg.norm(q[:, None, :] - p2[None, :, :], axis=2)
        E = np.where(same, E, np.inf)
        nn = E.argmin(1)
        score = len(set(nn[E[np.arange(len(nn)), nn] <= MERGE_PX].tolist()))
        if score > best_s:
            best_s, best = score, np.vstack([np.c_[Rm, t], [0.0, 0.0, 1.0]])
    return best, best_s


def register_free(d1, d2, H_prev=None):
    """GT-free homography view 1 -> view 2 from the declared detections only. -> (H or None, info)."""
    import cv2
    p1, p2 = foot(d1["b"]), foot(d2["b"])
    info = dict(init="prev" if H_prev is not None else "sim", n_pairs=0, valid=False, why="")
    H = None if H_prev is None else np.asarray(H_prev, float)
    if H is None:
        if not (d1["c"][:, None] == d2["c"][None, :]).sum() >= 2:
            info["why"] = "too few same-class candidates"
            return None, info
        H, _ = similarity_ransac(d1, d2)
        if H is None:
            info["why"] = "similarity RANSAC failed"
            return None, info
    for _ in range(2):
        m = mutual_nn(distances(d1, d2, H), AMB_HI)
        if len(m) < MIN_PAIRS:
            break
        i, j = map(list, zip(*m))
        Hn, _ = cv2.findHomography(p1[i].astype(np.float32), p2[j].astype(np.float32), cv2.RANSAC, RANSAC_PX)
        if Hn is None:
            break
        H = Hn
    n = len(mutual_nn(distances(d1, d2, H), MERGE_PX))
    info["n_pairs"] = n
    Hn = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    det = abs(np.linalg.det(Hn[:2, :2])) if np.all(np.isfinite(Hn)) else np.nan
    if not np.isfinite(det) or not DET_LO <= det <= DET_HI:
        info["why"] = "degenerate homography"
    elif n < MIN_PAIRS:
        info["why"] = f"only {n} mutual pairs within {MERGE_PX:g} px"
    else:
        info["valid"] = True
        return Hn, info
    return None, info


def associate(D, gate=MERGE_PX):
    if not D.size or not np.isfinite(D).any():
        return []
    C = np.where(D <= gate, D, 1e6)
    r, k = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(r, k) if D[i, j] <= gate]


def ambiguous(D):
    out = []
    for M in (D, D.T):
        s = set()
        for i in range(M.shape[0]):
            c = np.sort(M[i][M[i] <= AMB_HI])
            if len(c) >= 2 or (len(c) == 1 and c[0] > AMB_LO):
                s.add(i)
        out.append(s)
    return out[0], out[1]


def covering_crop(rec, box, kinds=("P1",)):
    """index (kind, k) of the first cached crop window of `rec` containing the box centre, or None."""
    cx, cy = centre(box)[0]
    for kind in kinds:
        for k, c in enumerate(rec["cands"][kind]):
            x0, y0, x1, y1 = c["win"]
            if x0 <= cx < x1 and y0 <= cy < y1:
                return kind, k
    return None


# ---------------------------------------------------------------- what the BS receives
def send_under_cap(recs, req, B=None):
    """B1 sending rule (h1_b1_view.realise): TOKEN views first, then each FRAME if it fits, else tokens."""
    left = NO_CAP if B is None else float(B)
    snd = [None, None]
    tokbits = [P.send(recs[v], "T0", Q, NO_CAP)["ul"] for v in (0, 1)]
    tok = [v for v in (0, 1) if req[v] == "T"]
    fits = sum(tokbits[v] for v in tok) <= left
    for v in tok:
        snd[v] = P.send(recs[v], "T0", Q, NO_CAP if fits else left / len(tok))
    left -= sum(snd[v]["ul"] for v in tok)
    for v in sorted((v for v in (0, 1) if req[v] == "F"), key=lambda v: recs[v]["frame"]["bits"]):
        snd[v] = P.send(recs[v], "F", Q, left)
        left -= snd[v]["ul"]
    return snd


def declare(rec, snd, extra=()):
    d = P.bs_decode(rec, snd, Q)
    if extra:
        d = P.nms_classwise(P.cat([d] + list(extra)))
    return P.take(d, np.where(d["s"] >= TAU)[0])


def budget_ref(first):
    """B1 B_ref from first-half bits only."""
    if not first:
        return None
    t = lambda r: P.send(r, "T0", Q, NO_CAP)["ul"]
    f = lambda r: P.send(r, "F", Q, NO_CAP)["ul"]
    return float(max(max(f(r1) + t(r2), t(r1) + f(r2)) for r1, r2, *_ in first))


# ---------------------------------------------------------------- scoring (labels; scorer and marked oracle only)
def gt_ids(d, g):
    return dict(P.match(d, g))


def ideal_pairs(d1, d2, g1, g2):
    a, b = gt_ids(d1, g1), gt_ids(d2, g2)
    inv = {v: k for k, v in b.items()}
    return [(i, inv[v]) for i, v in a.items() if v in inv]


def oracle_pairs(pairs, ideal, A1, A2):
    keep = [(i, j) for i, j in pairs if i not in A1 and j not in A2]
    u1, u2 = {i for i, _ in keep}, {j for _, j in keep}
    for i, j in ideal:
        if (i in A1 or j in A2) and i not in u1 and j not in u2:
            keep.append((i, j)); u1.add(i); u2.add(j)
    return keep


def fused_counts(d1, d2, g1, g2, pairs):
    """unique-object counts after fusing with the given cross-view pairs (each fused object claims <= 1 GT id)."""
    a, b = gt_ids(d1, g1), gt_ids(d2, g2)
    p1, p2 = {i for i, _ in pairs}, {j for _, j in pairs}
    objs = [({a[i]} if i in a else set()) | ({b[j]} if j in b else set()) for i, j in pairs]
    objs += [{a[i]} if i in a else set() for i in range(len(d1["s"])) if i not in p1]
    objs += [{b[j]} if j in b else set() for j in range(len(d2["s"])) if j not in p2]
    U = sorted(set(g1) | set(g2))
    col = {u: k for k, u in enumerate(U)}
    claimed = set()
    if objs and U:
        W = np.zeros((len(objs), len(U)))
        for k, ids in enumerate(objs):
            for u in ids:
                if u in col:
                    W[k, col[u]] = 1.0
        r, c = linear_sum_assignment(-W)
        claimed = {int(k) for k, x in zip(r, c) if W[k, x] > 0}
    fp_wrong = sum(1 for ids in objs if not ids)
    fp_dup = sum(1 for k, ids in enumerate(objs) if ids and k not in claimed)
    fp_assoc = sum(1 for (i, j) in pairs if i in a and j in b and a[i] != b[j])
    return dict(tp=len(claimed), fp=fp_wrong + fp_dup, fn=len(U) - len(claimed), fp_wrong=fp_wrong, fp_dup=fp_dup,
                fp_assoc=fp_assoc)


def diag_error(H, g1, g2):
    """diagnostic only: median reprojection error of the GT bottom-centres of the ids present in both views."""
    ids = sorted(set(g1) & set(g2))
    if H is None or not ids:
        return None
    a = project(H, foot([g1[i][:4] for i in ids]))
    return float(np.median(np.linalg.norm(a - foot([g2[i][:4] for i in ids]), axis=1)))


def crop_bits(recs):
    b = [c["bits"][CROP_Q] for r in recs for c in r["cands"]["P1"]]
    return float(np.median(b)) if b else float("inf")


# ---------------------------------------------------------------- phase 0
def phase0_sequence(pairs):
    h = len(pairs) // 2
    rows, H_prev = [], None
    for r1, r2, g1, g2 in pairs[:h]:
        snd = send_under_cap((r1, r2), ("T", "T"))
        d1, d2 = declare(r1, snd[0]), declare(r2, snd[1])
        H, info = register_free(d1, d2, H_prev)
        H_prev = H
        row = dict(info, n1=len(d1["s"]), n2=len(d2["s"]), diag_px=diag_error(H, g1, g2), n_amb=0, cov_p1=0, cov_any=0)
        if H is not None:
            A1, A2 = ambiguous(distances(d1, d2, H))
            for A, d, rec in ((A1, d1, r1), (A2, d2, r2)):
                for i in A:
                    row["n_amb"] += 1
                    row["cov_p1"] += covering_crop(rec, d["b"][i], ("P1",)) is not None
                    row["cov_any"] += covering_crop(rec, d["b"][i], ("P1", "P2")) is not None
        rows.append(row)
    valid = [r for r in rows if r["valid"]]
    diag = [r["diag_px"] for r in valid if r["diag_px"] is not None]
    s = dict(n_epochs=len(rows), valid_share=len(valid) / max(len(rows), 1),
             diag_median_px=float(np.median(diag)) if diag else None,
             n_amb=sum(r["n_amb"] for r in rows), cov_p1=sum(r["cov_p1"] for r in rows),
             cov_any=sum(r["cov_any"] for r in rows),
             failures={w: sum(1 for r in rows if r["why"] == w) for w in {r["why"] for r in rows if r["why"]}})
    s["registrable"] = bool(s["valid_share"] >= P0_VALID_SHARE and s["diag_median_px"] is not None
                            and s["diag_median_px"] <= P0_DIAG_PX)
    return s, rows


def phase0_decision(per):
    n_ok = sum(v["registrable"] for v in per.values())
    amb = sum(v["n_amb"] for v in per.values())
    cov = sum(v["cov_any"] for v in per.values()) / amb if amb else 0.0
    a_ok = n_ok >= P0_SEQ_PASS
    b_ok = a_ok and cov >= P0_COVER
    return dict(n_registrable=n_ok, n_seq=len(per), ambiguous_total=amb, crop_cover_share=cov,
                crop_cover_share_p1=sum(v["cov_p1"] for v in per.values()) / amb if amb else 0.0,
                RQ1A="IDENTIFIABLE" if a_ok else "NOT IDENTIFIABLE FROM EXISTING RECORDS",
                RQ1B="IDENTIFIABLE" if b_ok else "NOT IDENTIFIABLE FROM EXISTING RECORDS")


# ---------------------------------------------------------------- phase 1 / 2
def phase12_sequence(pairs, frame_view):
    h = len(pairs) // 2
    first, second = pairs[:h], pairs[h:]
    B = budget_ref(first)
    cb = crop_bits([r for p in first for r in p[:2]])
    fb = float(np.median([p[v]["frame"]["bits"] for p in first for v in (0, 1)])) if first else 0.0
    H0 = None                                                     # last GT-free state of the first half (TT tokens)
    for r1, r2, *_ in first:
        snd = send_under_cap((r1, r2), ("T", "T"))
        H0, _ = register_free(declare(r1, snd[0]), declare(r2, snd[1]), H0)
    keys = ("ideal", "free", "none", "real_crop", "gt_oracle")
    out = dict(B_ref=B, crop_bits=cb, frame_bits=fb, modes={})
    for mode, req in (("TT", ("T", "T")), ("FV", ("F", "T") if frame_view == 0 else ("T", "F"))):
        acc = {k: {} for k in keys}
        H_prev, fails, npairs, bits, rc_bits, or_bits, n_amb = H0, 0, [], [], 0.0, 0.0, 0
        for r1, r2, g1, g2 in second:
            snd = send_under_cap((r1, r2), req, B)
            d1, d2 = declare(r1, snd[0]), declare(r2, snd[1])
            H, info = register_free(d1, d2, H_prev)
            H_prev = H
            fails += not info["valid"]
            npairs.append(info["n_pairs"])
            bits.append(snd[0]["ul"] + snd[1]["ul"])
            D = distances(d1, d2, H)
            pairs_f = associate(D)
            ideal = ideal_pairs(d1, d2, g1, g2)
            res = dict(ideal=fused_counts(d1, d2, g1, g2, ideal), free=fused_counts(d1, d2, g1, g2, pairs_f),
                       none=fused_counts(d1, d2, g1, g2, []))
            A1, A2 = ambiguous(D) if H is not None else (set(), set())
            n_amb += len(A1) + len(A2)
            res["gt_oracle"] = fused_counts(d1, d2, g1, g2, oracle_pairs(pairs_f, ideal, A1, A2))
            or_bits += (cb + REQ_BITS) * (len(A1) + len(A2)) if (A1 or A2) else 0.0
            # REAL-CROP: label-free choice of cached crops covering the ambiguous detections
            extra, recs, ds = [[], []], (r1, r2), (d1, d2)
            for v, A in ((0, A1), (1, A2)):
                ks = {covering_crop(recs[v], ds[v]["b"][i], ("P1", "P2")) for i in A} - {None}
                for kind, k in sorted(ks):
                    c = recs[v]["cands"][kind][k]
                    extra[v].append(c["bs"][CROP_Q]); rc_bits += c["bits"][CROP_Q] + REQ_BITS
            if extra[0] or extra[1]:
                e1, e2 = declare(r1, snd[0], extra[0]), declare(r2, snd[1], extra[1])
                res["real_crop"] = fused_counts(e1, e2, g1, g2, associate(distances(e1, e2, H)))
            else:
                res["real_crop"] = res["free"]
            for k in keys:
                for x, val in res[k].items():
                    acc[k][x] = acc[k].get(x, 0) + val
        n = max(len(second), 1)
        m = {k: dict(f1=P.prf(v)[2], p=P.prf(v)[0], r=P.prf(v)[1], **v) for k, v in acc.items()}
        m.update(fail_share=fails / n, n_fail=fails, n_epochs=len(second), mean_pairs=float(np.mean(npairs)) if npairs else 0.0,
                 bits_per_epoch=float(np.mean(bits)) if bits else 0.0, real_crop_bits=rc_bits / n,
                 gt_oracle_bits=or_bits / n, amb_per_epoch=n_amb / n)
        out["modes"][mode] = m
    return out


def rq1a_decision(per, mode="TT"):
    seqs = sorted(per)
    f = lambda s, k: per[s]["modes"][mode][k]["f1"]
    dA = [f(s, "ideal") - f(s, "free") for s in seqs]
    fails = sum(per[s]["modes"][mode]["n_fail"] for s in seqs)
    eps = sum(per[s]["modes"][mode]["n_epochs"] for s in seqs)
    mean, share = float(np.mean(dA)), float(np.mean([x > 0 for x in dA]))
    if eps and fails / eps > MAX_FAIL:
        v = "INCONCLUSIVE (registration failed in most evaluation epochs)"
    elif mean < NOGO_DELTA:
        v = "NO-GO"
    elif mean >= GO_DELTA and share >= GO_SEQ_SHARE - 1e-9:
        v = "GO (opportunity)"
    else:
        v = "INCONCLUSIVE"
    return dict(mode=mode, delta_A=mean, delta_A_per_seq=dict(zip(map(str, seqs), dA)), share_positive=share,
                fail_share=fails / eps if eps else None, verdict=v)


def rq1b_decision(per, a):
    seqs = sorted(per)
    m = lambda s, k: per[s]["modes"]["TT"][k]
    dA = a["delta_A"]
    rec = lambda k: float(np.mean([m(s, k)["f1"] - m(s, "free")["f1"] for s in seqs])) / dA if dA > 1e-12 else 0.0
    fb = float(np.mean([per[s]["frame_bits"] for s in seqs]))
    cost = lambda k: float(np.mean([m(s, k + "_bits") for s in seqs]))
    oracle_ok = dA >= GO_DELTA and rec("gt_oracle") >= GO_RECOVER and cost("gt_oracle") <= GO_BITS_SHARE * fb
    real_ok = dA >= GO_DELTA and rec("real_crop") >= GO_RECOVER and cost("real_crop") <= GO_BITS_SHARE * fb
    if real_ok:
        v = "REAL-CROP USEFUL (better detections from cached crops reduce geometric association loss; not identity)"
    elif oracle_ok:
        v = "OPPORTUNITY ONLY (GT-ORACLE recovers it; the cached transmitted information does not)"
    else:
        v = "NO IDEAL OPPORTUNITY under the registered cap"
    return dict(recovery_gt_oracle=rec("gt_oracle"), recovery_real_crop=rec("real_crop"),
                bits_gt_oracle=cost("gt_oracle"), bits_real_crop=cost("real_crop"), frame_bits=fb,
                cap_bits=GO_BITS_SHARE * fb, verdict=v)


# ---------------------------------------------------------------- CLI
def _load(a):
    import h1_pilot0b_eval as E
    ok = {int(s) for s, v in json.loads(Path(a.idcheck).read_text())["sequences"].items() if v["pass"]}
    pairs, _ = E.load_pairs(a.precomp, a.zip, ("test",))
    return {s: pairs[s] for s in sorted(pairs) if s in ok}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("phase0", "phase12"):
        s = sub.add_parser(name)
        s.add_argument("precomp"); s.add_argument("idcheck")
        if name == "phase12":
            s.add_argument("b1"); s.add_argument("phase0")
        s.add_argument("out")
        s.add_argument("--zip", required=True)
        s.add_argument("--data-use-basis", required=True,
                       help="provenance note only: records the basis, does not verify or create an authorisation")
    a = ap.parse_args()
    seqs = _load(a)
    if a.cmd == "phase0":
        per = {s: phase0_sequence(p) for s, p in seqs.items()}
        res = dict(status="RQ1 v2 phase 0 (first-half epochs only; GT used as a diagnostic ruler only)",
                   per_seq={str(s): v[0] for s, v in per.items()}, epochs={str(s): v[1] for s, v in per.items()},
                   decision=phase0_decision({s: v[0] for s, v in per.items()}), data_use_basis=a.data_use_basis)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(dict(per_seq=res["per_seq"], decision=res["decision"]), indent=1, default=str))
        return
    p0 = json.loads(Path(a.phase0).read_text())["decision"]
    if p0["RQ1A"] != "IDENTIFIABLE":
        raise SystemExit("phase 0 says RQ1-A is not identifiable: phase 1 / 2 must not run")
    b1 = json.loads(Path(a.b1).read_text())["per_seq"]
    per = {s: phase12_sequence(p, int(b1[str(s)]["pref"]["NDET"])) for s, p in seqs.items()}
    A = rq1a_decision(per)
    res = dict(status="RQ1 v2 phases 1-2, EXPLORATORY (EXPL labels)", per_seq={str(s): v for s, v in per.items()},
               RQ1A=A, RQ1A_FV=rq1a_decision(per, "FV"),
               RQ1B=rq1b_decision(per, A) if p0["RQ1B"] == "IDENTIFIABLE" else dict(verdict=p0["RQ1B"]),
               data_use_basis=a.data_use_basis)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("RQ1A", "RQ1A_FV", "RQ1B")}, indent=1, default=str))


if __name__ == "__main__":
    main()
