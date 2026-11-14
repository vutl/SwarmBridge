"""R1b-0a data split v1 (data-integrity amendment of 10/11, reviewed): gate G7 = one clip per independent source.

Rules (reports/archive/r1-visdrone/visdrone-vid-data-audit.md sec 4 + amendment 5.4/5.5):
  * eligible source: no DET-train source id AND no confirmed DET-train pHash match in any of its clips;
  * G7 = the 7 smallest eligible source ids, one clip per source (smallest clip name of that source);
  * other clips of a G7 source -> QUARANTINE (neither gate nor tuning);
  * TUNE = every other VID-train clip; AUX = VID-val clips with no confirmed DET-train match and no DET-train source id;
  * excluded VID-val clips are listed with the reason.
Then a VID-internal near-duplicate check (pHash Hamming <= thr on every frame, NCC >= 0.9 to confirm) between
G7 / AUX / QUARANTINE and TUNE, and between different G7 sources, so that no gate scene appears elsewhere.
Usage: python r1b_split.py AUDIT_JSON VID_ROOT OUT_JSON [--procs 6] [--thr 10]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from vid_audit import POP, hash_job, ncc


def src(c):
    return int(re.match(r"uav(\d+)_", c).group(1))


def build_split(rep):
    L = rep["leakage"]
    det_tr_ids = set(L["id_overlap_det_train"]["overlap"])
    conf_tr = {k.split("/", 1)[1] for k in L["phash_det_train"]["confirmed_per_clip"]}
    train = sorted(rep["splits"]["train"]["clips"])
    val = sorted(rep["splits"]["val"]["clips"])
    by_src = defaultdict(list)
    for c in train:
        by_src[src(c)].append(c)
    eligible = sorted(s for s, cs in by_src.items() if s not in det_tr_ids and not any(c in conf_tr for c in cs))
    gate_src = eligible[:7]
    gate = [sorted(by_src[s])[0] for s in gate_src]
    quarantine = sorted(c for s in gate_src for c in by_src[s] if c not in gate)
    tune = sorted(c for c in train if src(c) not in gate_src)
    aux, val_excl = [], {}
    for c in val:
        if c in conf_tr:
            val_excl[c] = "confirmed DET-train frame match"
        elif src(c) in det_tr_ids:
            val_excl[c] = "DET-train source id"
        else:
            aux.append(c)
    return dict(version="r1b-split-v1", gate=gate, gate_sources=gate_src, quarantine=quarantine, tune=tune,
                aux_val=aux, val_excluded=val_excl, eligible_sources=eligible,
                eligible_not_used=[s for s in eligible if s not in gate_src])


def check_split(S, train_clips, val_clips):
    """Structural invariants; returns list of violations (empty = OK)."""
    err = []
    g, q, t, a = set(S["gate"]), set(S["quarantine"]), set(S["tune"]), set(S["aux_val"])
    if len(S["gate"]) != 7: err.append("gate size != 7")
    if len({src(c) for c in g}) != len(g): err.append("two gate clips share a source")
    if g & t or g & q or q & t: err.append("gate / quarantine / tune overlap")
    if g | q | t != set(train_clips): err.append("train clips not partitioned")
    if {src(c) for c in g} & {src(c) for c in t}: err.append("gate source also in tune")
    if {src(c) for c in a} & ({src(c) for c in g} | {src(c) for c in t}): err.append("aux source also in train sets")
    if not a <= set(val_clips): err.append("aux not in val")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit"); ap.add_argument("vid_root"); ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6); ap.add_argument("--thr", type=int, default=10)
    a = ap.parse_args()
    rep = json.loads(Path(a.audit).read_text())
    S = build_split(rep)
    train = sorted(rep["splits"]["train"]["clips"]); val = sorted(rep["splits"]["val"]["clips"])
    S["structural_violations"] = check_split(S, train, val)
    root = Path(a.vid_root)
    role = {c: "gate" for c in S["gate"]} | {c: "quarantine" for c in S["quarantine"]} | {c: "tune" for c in S["tune"]} \
        | {c: "aux" for c in S["aux_val"]}
    frames = []
    for c, r in role.items():
        sp = "val" if r == "aux" else "train"
        frames += [(c, str(f)) for f in sorted((root / f"VisDrone2019-VID-{sp}" / "sequences" / c).glob("*.jpg"))]
    H, TH = {}, {}
    with Pool(a.procs) as pool:
        for k, (p, h, th, _) in enumerate(pool.imap_unordered(hash_job, [f for _, f in frames], chunksize=32)):
            if h is not None:
                H[p], TH[p] = h, th
            if k % 5000 == 0:
                print(f"hashed {k}/{len(frames)}", flush=True)
    clip_of = {f: c for c, f in frames}
    tune_f = [f for c, f in frames if role[c] == "tune" and f in H]
    TU = np.stack([H[f] for f in tune_f])
    res = {}
    for r in ("gate", "aux", "quarantine"):
        hits = Counter(); conf = Counter(); best = {}
        for c, f in frames:
            if role[c] != r or f not in H:
                continue
            d = POP[np.bitwise_xor(TU, H[f][None, :])].sum(1)
            j = int(np.argmin(d))
            best[c] = min(best.get(c, 99), int(d[j]))
            if d[j] <= a.thr:
                hits[(c, clip_of[tune_f[j]])] += 1
                if ncc(TH[f], TH[tune_f[j]]) >= 0.9:
                    conf[(c, clip_of[tune_f[j]])] += 1
        res[f"{r}_vs_tune"] = dict(min_hamming_per_clip=best, suspects={f"{k[0]}~{k[1]}": v for k, v in hits.items()},
                                   confirmed={f"{k[0]}~{k[1]}": v for k, v in conf.items()})
    # gate source vs other gate sources
    gx = Counter(); gc = Counter()
    gate_f = [(c, f) for c, f in frames if role[c] == "gate" and f in H]
    GA = np.stack([H[f] for _, f in gate_f]); GS = np.array([src(c) for c, _ in gate_f])
    for c, f in gate_f:
        d = POP[np.bitwise_xor(GA, H[f][None, :])].sum(1).astype(int)
        d[GS == src(c)] = 99
        j = int(np.argmin(d))
        if d[j] <= a.thr:
            gx[(c, gate_f[j][0])] += 1
            if ncc(TH[f], TH[gate_f[j][1]]) >= 0.9:
                gc[(c, gate_f[j][0])] += 1
    res["gate_vs_gate"] = dict(suspects={f"{k[0]}~{k[1]}": v for k, v in gx.items()},
                               confirmed={f"{k[0]}~{k[1]}": v for k, v in gc.items()})
    S["vid_internal_check"] = dict(thr=a.thr, n_frames=len(H), **res)
    S["vid_internal_confirmed_total"] = sum(sum(v["confirmed"].values()) for v in res.values())
    S["frames"] = {r: sum(1 for c, _ in frames if role[c] == r) for r in ("gate", "quarantine", "tune", "aux")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "vid_internal_check"}, indent=1))
    print("internal confirmed", S["vid_internal_confirmed_total"], "suspects",
          {k: len(v["suspects"]) for k, v in res.items()})
    return 0 if not S["structural_violations"] else 1


if __name__ == "__main__":
    sys.exit(main())
