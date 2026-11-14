"""RQ1-E (reports/swarmbridge-rq1e-preregistration.md, review #24): cross-view association from zero-shot DINOv2-S
appearance features on the real MDMT frames.

  extract  crops around the declared detections of the existing pilot 0B records (all epochs, both views), DINOv2-S
           embeddings (no training), throughput timing. Output stays OUTSIDE git (identity-linked per-frame data).
  e0       first-half epochs only: FREE-APPEARANCE registration feasibility (full 384-d and compressed widths).
GT is used only by the scorer / diagnostics, never for matching, registration or thresholds.
Usage:
  python h1_rq1e.py extract PRECOMP_DIR ID_CHECK_JSON OUT_DIR --zip MDMT_ZIP --model MODEL_DIR --data-use-basis "..."
  python h1_rq1e.py e0 PRECOMP_DIR ID_CHECK_JSON EMB_DIR OUT_JSON --zip MDMT_ZIP --data-use-basis "..."
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import h1_pilot0b as P
import h1_rq1_assoc as A

MODEL_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
CROP_SCALE, CROP_MIN, INPUT = 1.5, 32, 224
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
RATIO, MIN_CAND, RANSAC_PX, MIN_INLIER_SHARE = 0.8, 8, 40.0, 0.5
DET_LO, DET_HI, MERGE_PX = 0.1, 10.0, 40.0
WIDTHS = (64, 128, 256, 3072)                  # bits: sign random projections; 3072 = 384 dims x 8 bits
# E0 criteria
E0_VALID, E0_DIAG, E0_FALSE, E0_PREC, E0_RECALL, E0_SEQS, FALSE_PX = 0.5, 40.0, 0.10, 0.8, 0.25, 6, 100.0


# ---------------------------------------------------------------- crops and features
def crop_window(b, W, H):
    """square window centred on box b, side max(1.5 x long side, 32), shifted inside the image."""
    x0, y0, x1, y1 = map(float, b)
    s = max(CROP_SCALE * max(x1 - x0, y1 - y0), CROP_MIN)
    s = min(s, W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - s / 2, 0.0), W - s)
    top = min(max(cy - s / 2, 0.0), H - s)
    return int(round(left)), int(round(top)), int(round(left + s)), int(round(top + s))


def preprocess(img, win):
    from PIL import Image
    x0, y0, x1, y1 = win
    c = Image.fromarray(img[y0:y1, x0:x1]).resize((INPUT, INPUT), Image.BICUBIC)
    a = np.asarray(c, np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def declared_with_source(rec):
    """BS-side TOKEN declarations (as in RQ1 phase 0) + index of the original YOLO11n box each one comes from."""
    snd = A.send_under_cap((rec, rec), ("T", "T"))[0]
    d = A.declare(rec, snd)
    det = rec["det"]
    strong = P.strong_idx(det)
    if not len(d["s"]):
        return d, np.zeros(0, int)
    M = P.iou_matrix(d["b"], det["b"][strong])
    return d, np.asarray(strong)[M.argmax(1)]


def compress(E, m, seed=0):
    """sign random projection (m bits) or 8-bit-per-dim quantisation (m = 3072). Deterministic, no fitting."""
    E = np.asarray(E, np.float32)
    if m == 3072:
        return np.clip(np.round(E * 127), -127, 127).astype(np.int8)
    R = np.random.default_rng(seed).standard_normal((m, E.shape[1])).astype(np.float32)
    return E @ R.T > 0


def distance(a, b, m=None):
    """1 - cosine for float / int8 features; normalised Hamming for sign bits."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    if a.dtype == bool:
        return (a[:, None, :] != b[None, :, :]).mean(2)
    a = a.astype(np.float32); b = b.astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return 1.0 - a @ b.T


# ---------------------------------------------------------------- E0 association (no labels)
def candidates(D, c1, c2):
    """mutual nearest neighbours of the same class passing the ratio test both ways."""
    out = []
    for c in np.intersect1d(c1, c2):
        I, J = np.where(c1 == c)[0], np.where(c2 == c)[0]
        S = D[np.ix_(I, J)]

        def ok(row, k):
            if len(row) < 2:
                return True
            s = np.sort(row)
            return s[0] <= RATIO * s[1] + 1e-12 and row[k] == s[0]
        for a in range(len(I)):
            b = int(S[a].argmin())
            if int(S[:, b].argmin()) == a and ok(S[a], b) and ok(S[:, b], a):
                out.append((int(I[a]), int(J[b])))
    return out


def register(d1, d2, cand):
    """homography from the appearance candidates. -> (H or None, info)."""
    import cv2
    info = dict(n_cand=len(cand), n_inlier=0, valid=False, why="")
    if len(cand) < MIN_CAND:
        info["why"] = "too few candidates"
        return None, info
    i, j = map(list, zip(*cand))
    p1, p2 = A.foot(d1["b"][i]).astype(np.float32), A.foot(d2["b"][j]).astype(np.float32)
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    if H is None:
        info["why"] = "RANSAC failed"
        return None, info
    n = int(mask.sum())
    info["n_inlier"] = n
    Hn = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    det = abs(np.linalg.det(Hn[:2, :2])) if np.all(np.isfinite(Hn)) else np.nan
    if not np.isfinite(det) or not DET_LO <= det <= DET_HI:
        info["why"] = "degenerate homography"
    elif n < MIN_CAND:
        info["why"] = "too few inliers"
    elif n / len(cand) < MIN_INLIER_SHARE:
        info["why"] = "low inlier share"
    else:
        info["valid"] = True
        return Hn, info
    return None, info


# ---------------------------------------------------------------- E0 scoring (labels: diagnostics only)
def epoch_metrics(d1, d2, g1, g2, cand, H):
    a, b = A.gt_ids(d1, g1), A.gt_ids(d2, g2)
    ideal = set(A.ideal_pairs(d1, d2, g1, g2))
    verif = [(i, j) for i, j in cand if i in a and j in b]
    correct = sum(a[i] == b[j] for i, j in verif)
    final = A.associate(A.distances(d1, d2, H)) if H is not None else []
    fin_ok = sum((i, j) in ideal for i, j in final)
    return dict(n_cand=len(cand), n_verif=len(verif), n_correct=int(correct), n_ideal=len(ideal),
                n_final=len(final), n_final_ok=int(fin_ok), diag=A.diag_error(H, g1, g2))


def summarize(rows):
    v = [r for r in rows if r["valid"]]
    diag = [r["diag"] for r in v if r["diag"] is not None]
    S = lambda k: sum(r[k] for r in rows)
    s = dict(n_epochs=len(rows), valid_share=len(v) / max(len(rows), 1),
             diag_median_px=float(np.median(diag)) if diag else None,
             false_valid_share=float(np.mean([x > FALSE_PX for x in diag])) if diag else None,
             cand_precision=S("n_correct") / S("n_verif") if S("n_verif") else 0.0,
             cand_unverifiable=1 - S("n_verif") / S("n_cand") if S("n_cand") else None,
             cand_recall=S("n_correct") / S("n_ideal") if S("n_ideal") else 0.0,
             final_precision=S("n_final_ok") / S("n_final") if S("n_final") else 0.0,
             final_recall=S("n_final_ok") / S("n_ideal") if S("n_ideal") else 0.0,
             failures={w: sum(1 for r in rows if r["why"] == w) for w in {r["why"] for r in rows if r["why"]}})
    s["pass"] = bool(s["valid_share"] >= E0_VALID and s["diag_median_px"] is not None and s["diag_median_px"] <= E0_DIAG
                     and s["false_valid_share"] is not None and s["false_valid_share"] <= E0_FALSE
                     and s["cand_precision"] >= E0_PREC and s["cand_recall"] >= E0_RECALL)
    return s


def e0_sequence(pairs, emb, width=None):
    """first half only. emb[i] = (E_view1, E_view2) aligned with the declared detections of epoch i."""
    h = len(pairs) // 2
    rows = []
    for k, (r1, r2, g1, g2) in enumerate(pairs[:h]):
        d1, _ = declared_with_source(r1)
        d2, _ = declared_with_source(r2)
        E1, E2 = emb[k]
        if width is not None:
            E1, E2 = compress(E1, width), compress(E2, width)
        cand = candidates(distance(E1, E2), d1["c"], d2["c"])
        H, info = register(d1, d2, cand)
        rows.append(dict(info, **epoch_metrics(d1, d2, g1, g2, cand, H)))
    return summarize(rows)


def e0_decision(per):
    n = sum(v["pass"] for v in per.values())
    return dict(n_pass=n, n_seq=len(per), verdict="PASS" if n >= E0_SEQS else "FAIL")


def primary_width(per_width):
    """smallest compressed width that also passes E0 on >= E0_SEQS sequences, else None."""
    for m in WIDTHS:
        if sum(v["pass"] for v in per_width[m].values()) >= E0_SEQS:
            return m
    return None


# ---------------------------------------------------------------- extraction (pixels; outside git)
def load_model(model_dir):
    import torch
    from transformers import Dinov2Model
    m = Dinov2Model.from_pretrained(model_dir, use_safetensors=True).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), dev


def embed(model, dev, X, batch=256):
    import torch
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            x = torch.from_numpy(np.stack(X[k:k + batch])).to(dev)
            e = model(pixel_values=x).pooler_output
            out.append(torch.nn.functional.normalize(e, dim=1).float().cpu().numpy())
    if dev == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def extract(precomp, seqs, zip_path, rights, model_dir, out_dir):
    import mdmt_adapter as M
    ds = M.MDMT(zip_path, allowed=("test",), rights_note=rights)
    model, dev = load_model(model_dir)
    t = dict(decode=0.0, crop=0.0, model=0.0, n_frames=0, n_crops=0)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        R = pickle.loads((Path(precomp) / f"seq_{s}.pkl").read_bytes())
        res = []
        for k, f in enumerate(R["frames"]):
            pair = []
            for v in (1, 2):
                rec = R["recs"][v][k]
                d, src = declared_with_source(rec)
                t0 = time.perf_counter(); img = ds.image(s, v, f); t1 = time.perf_counter()
                X = [preprocess(img, crop_window(rec["det"]["b"][i], img.shape[1], img.shape[0])) for i in src]
                t2 = time.perf_counter()
                E = embed(model, dev, X)
                t3 = time.perf_counter()
                t["decode"] += t1 - t0; t["crop"] += t2 - t1; t["model"] += t3 - t2
                t["n_frames"] += 1; t["n_crops"] += len(X)
                pair.append(E.astype(np.float32))
            res.append(tuple(pair))
        (out_dir / f"emb_{s}.pkl").write_bytes(pickle.dumps(dict(seq=s, frames=R["frames"], emb=res, model_rev=MODEL_REV)))
        print(s, "epochs", len(res), "crops so far", t["n_crops"], flush=True)
    t["device"] = dev
    t["crops_per_s_model"] = t["n_crops"] / t["model"] if t["model"] else None
    t["total_s"] = t["decode"] + t["crop"] + t["model"]
    return t


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "e0"):
        s = sub.add_parser(name)
        s.add_argument("precomp"); s.add_argument("idcheck")
        if name == "extract":
            s.add_argument("out_dir"); s.add_argument("--model", required=True)
        else:
            s.add_argument("emb_dir"); s.add_argument("out")
        s.add_argument("--zip", required=True)
        s.add_argument("--data-use-basis", required=True,
                       help="provenance note only: records the basis, does not verify or create an authorisation")
    a = ap.parse_args()
    seqs = sorted(int(s) for s, v in json.loads(Path(a.idcheck).read_text())["sequences"].items() if v["pass"])
    if a.cmd == "extract":
        t = extract(a.precomp, seqs, a.zip, a.data_use_basis, a.model, a.out_dir)
        (Path(a.out_dir) / "timing.json").write_text(json.dumps(t, indent=1))
        print(json.dumps(t, indent=1))
        return
    import h1_pilot0b_eval as E
    pairs, _ = E.load_pairs(a.precomp, a.zip, ("test",))
    emb = {s: pickle.loads((Path(a.emb_dir) / f"emb_{s}.pkl").read_bytes())["emb"] for s in seqs}
    full = {s: e0_sequence(pairs[s], emb[s]) for s in seqs}
    comp = {m: {s: e0_sequence(pairs[s], emb[s], m) for s in seqs} for m in WIDTHS}
    res = dict(status="RQ1-E E0, FREE-APPEARANCE diagnostic (first-half epochs; all embeddings used, no comm cost; "
                      "GT only for diagnostics). Not a deployable policy.",
               full_384={str(s): v for s, v in full.items()}, decision=e0_decision(full),
               compressed={str(m): {str(s): v for s, v in comp[m].items()} for m in WIDTHS},
               compressed_pass={str(m): sum(v["pass"] for v in comp[m].values()) for m in WIDTHS},
               primary_width=primary_width(comp) if e0_decision(full)["verdict"] == "PASS" else None,
               model_rev=MODEL_REV, data_use_basis=a.data_use_basis)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("full_384", "decision", "compressed_pass", "primary_width")}, indent=1, default=str))


if __name__ == "__main__":
    main()
