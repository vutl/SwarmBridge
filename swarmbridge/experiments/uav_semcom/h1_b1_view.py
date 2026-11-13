"""B1 tier 1 (reports/swarmbridge-a8-b1-preregistration.md): view-aware cooperative OPPORTUNITY, offline on the
existing pilot 0B records. GT-ASSISTED FEASIBILITY SCREENING: the preferred view is chosen with GT (first half of
each sequence) and the BS association is ideal (GT ids) -> not a deployable mechanism, no cooperative-gain claim.

Per epoch each view sends TOKEN (strong tokens) or FRAME (AVIF q35 frame, YOLO11s at the BS). Allocations:
  TT, FF            both TOKEN / both FRAME (FF uses about twice the bits: reference only, never used by the gate)
  F1T, TF2          FRAME fixed on UAV 1 / UAV 2
  ALT               FRAME alternates between the views by epoch (symmetric, one FRAME per epoch)
  NDET              FRAME on the view with more onboard YOLO11n detections in the FIRST half (label-free choice)
  VIEW              FRAME on the view with the higher first-half unique F1 (GT-assisted choice)
Time split: the first half of the epochs of a sequence selects, the second half is scored. Unique-object counts use
ideal association: an object found in either view counts once (ids are shared across views: condition 1, checked by
scripts/uav_semcom/b1_id_check.py); every unmatched declaration is FP_wrong; FP_dup = FP_assoc = 0 by construction.
Gate (pre-registered): VIEW - best equal-bit baseline of {TT, ALT, NDET} >= 2 unique-F1 points (mean over sequences)
and positive in >= 73 % of the sequences that passed condition 1 (8 / 11 when all pass).
Usage: python h1_b1_view.py PRECOMP_DIR ID_CHECK_JSON OUT_JSON --zip MDMT_ZIP --data-use-basis "..."
"""
import argparse
import json
from pathlib import Path

import numpy as np

import h1_pilot0b as P

TAU, Q = 0.3, 35
ALLOCS = ("TT", "FF", "F1T", "TF2", "ALT", "NDET", "VIEW")
EQUAL_BIT_BASELINES = ("TT", "ALT", "NDET")
GO_F1, GO_SHARE = 0.02, 8 / 11


def epoch_table(pairs):
    """per epoch, per view (0, 1), per mode (T, F): view_counts at TAU, UL bits; plus onboard detection counts."""
    out = []
    for r1, r2, g1, g2 in pairs:
        e = dict(gt=(g1, g2), cnt=[{}, {}], bits=[{}, {}], ndet=[len(r1["det"]["s"]), len(r2["det"]["s"])])
        for v, (rec, g) in enumerate(((r1, g1), (r2, g2))):
            for m, cond in (("T", "T0"), ("F", "F")):
                snd = P.send(rec, cond, Q, 1e12)
                e["cnt"][v][m] = P.view_counts(P.bs_decode(rec, snd, Q), g, TAU)
                e["bits"][v][m] = snd["ul"]
        out.append(e)
    return out


def modes_of(alloc, i, pref):
    """(mode of view 0, mode of view 1) at epoch index i; pref = dict(NDET=view, VIEW=view)."""
    if alloc == "TT":
        return "T", "T"
    if alloc == "FF":
        return "F", "F"
    if alloc == "F1T":
        return "F", "T"
    if alloc == "TF2":
        return "T", "F"
    if alloc == "ALT":
        return ("F", "T") if i % 2 == 0 else ("T", "F")
    return ("F", "T") if pref[alloc] == 0 else ("T", "F")


def score(epochs, alloc, pref, i0=0):
    """unique-object counts and mean UL bits of an allocation over `epochs` (i0 = index of the first one)."""
    acc, bits = {}, []
    for k, e in enumerate(epochs):
        m = modes_of(alloc, i0 + k, pref)
        P.add(acc, P.unique_counts(e["cnt"][0][m[0]], e["cnt"][1][m[1]], *e["gt"]))
        bits.append(e["bits"][0][m[0]] + e["bits"][1][m[1]])
    if not acc:
        acc = dict(tp=0, fp=0, fn=0)
    p, r, f = P.prf(acc)
    return dict(f1=f, p=p, r=r, bits=float(np.mean(bits)) if bits else 0.0, fp_wrong=acc["fp"], fp_dup=0, fp_assoc=0,
                **acc)


def select(first):
    """choices made on the FIRST half only. VIEW uses GT (unique F1); NDET uses onboard detection counts only."""
    nd = np.sum([e["ndet"] for e in first], axis=0) if first else np.zeros(2)
    f = [score(first, a, {})["f1"] for a in ("F1T", "TF2")]
    return dict(NDET=int(nd[1] > nd[0]), VIEW=int(f[1] > f[0]))


def run_sequence(pairs):
    T = epoch_table(pairs)
    h = len(T) // 2
    pref = select(T[:h])
    return dict(pref=pref, n_select=h, n_eval=len(T) - h, alloc={a: score(T[h:], a, pref, i0=h) for a in ALLOCS})


def gate(per_seq):
    seqs = sorted(per_seq)
    mean = {a: float(np.mean([per_seq[s]["alloc"][a]["f1"] for s in seqs])) for a in ALLOCS}
    base = max(EQUAL_BIT_BASELINES, key=lambda a: mean[a])          # best baseline in hindsight: favours the baseline
    d = [per_seq[s]["alloc"]["VIEW"]["f1"] - per_seq[s]["alloc"][base]["f1"] for s in seqs]
    share = float(np.mean([x > 0 for x in d]))
    return dict(mean_f1=mean, mean_bits={a: float(np.mean([per_seq[s]["alloc"][a]["bits"] for s in seqs]))
                                         for a in ALLOCS},
                baseline=base, view_minus_baseline=float(np.mean(d)), per_seq_diff=dict(zip(map(str, seqs), d)),
                share_positive=share, n_seq=len(seqs),
                view_minus_each={a: float(np.mean([per_seq[s]["alloc"]["VIEW"]["f1"] - per_seq[s]["alloc"][a]["f1"]
                                                   for s in seqs])) for a in ALLOCS if a != "VIEW"},
                GO=bool(np.mean(d) >= GO_F1 and share >= GO_SHARE - 1e-9),
                note="GT-assisted view selection + ideal association: opportunity screening only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("precomp"); ap.add_argument("idcheck"); ap.add_argument("out")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data-use-basis", required=True,
                    help="who confirmed that this use of the dataset is acceptable (author / lab / university)")
    a = ap.parse_args()
    import h1_pilot0b_eval as E
    ok = {int(s) for s, v in json.loads(Path(a.idcheck).read_text())["sequences"].items() if v["pass"]}
    pairs, _ = E.load_pairs(a.precomp, a.zip, ("test",))
    if len(ok) < 8:
        res = dict(status="INCONCLUSIVE: fewer than 8 sequences passed the cross-view id check", passed=sorted(ok))
    else:
        per = {s: run_sequence(pairs[s]) for s in sorted(pairs) if s in ok}
        res = dict(status="EXPLORATORY, GT-assisted feasibility screening (EXPL labels)", sequences=sorted(per),
                   excluded_by_id_check=sorted(set(pairs) - ok), per_seq={str(s): v for s, v in per.items()},
                   gate=gate(per), tau_decl=TAU, data_use_basis=a.data_use_basis)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res.get("gate", res), indent=1, default=str))


if __name__ == "__main__":
    main()
