"""B1 tier 1 (reports/swarmbridge-a8-b1-preregistration.md + amendment 1): view-aware cooperative OPPORTUNITY,
offline on the existing pilot 0B records. GT-ASSISTED FEASIBILITY SCREENING: the preferred view is chosen with GT
(first half of each sequence) and the BS association is ideal (GT ids) -> not a deployable mechanism, no
cooperative-gain claim.

Per epoch each view REQUESTS TOKEN (strong tokens) or FRAME (AVIF q35 frame, YOLO11s at the BS). Allocations:
  TT, FF            both TOKEN / both FRAME (FF is reported uncapped as a reference only, never used by the gate)
  F1T, TF2          FRAME fixed on UAV 1 / UAV 2
  ALT               FRAME alternates between the views by epoch (symmetric)
  NDET              FRAME on the view with more onboard YOLO11n detections in the FIRST half (label-free)
  BIG               FRAME on the view with the larger mean FRAME size in the FIRST half (label-free; "spend more" control)
  VIEW              FRAME on the view with the higher first-half unique F1 under the same cap (GT-assisted choice)

Amendment 1 (review #19, before any run on dataset records): SAME JOINT UL CAP B FOR EVERY ALLOCATION, PER EPOCH.
  B_ref  = max over the FIRST-half epochs of the cost of one FRAME + the other view's TOKENs (either view); bits only,
           no label. Levels B = kappa * B_ref, kappa in KAPPAS; the gate uses kappa = 1, the others are the F1-bit curve.
  realise(): TOKEN views are served first (if their tokens do not fit, the cap is split equally and tokens are packed
           by the pilot's rule); then each requested FRAME is sent only if it fits in what is left (smaller FRAME
           first), otherwise that view sends TOKENs within what is left. Invariant: bits sent <= B in every epoch.
  Reported per allocation: achieved bits (mean, max), share of epochs sent as requested (feasible), F1 / P / R.
Time split: the first half of the epochs of a sequence selects (and fixes B_ref), the second half is scored.
Unique-object counts use ideal association: an object found in either view counts once (ids shared across views:
condition 1, screened by scripts/uav_semcom/b1_id_check.py); every unmatched declaration is FP_wrong; FP_dup =
FP_assoc = 0 by construction.
Gate (thresholds as pre-registered): at kappa = 1, VIEW - best of {TT, ALT, NDET, BIG} >= 2 unique-F1 points (mean
over sequences) and positive in >= 73 % of the sequences that passed condition 1.
Usage: python h1_b1_view.py PRECOMP_DIR ID_CHECK_JSON OUT_JSON --zip MDMT_ZIP --data-use-basis "..."
"""
import argparse
import json
from pathlib import Path

import numpy as np

import h1_pilot0b as P

TAU, Q = 0.3, 35
ALLOCS = ("TT", "FF", "F1T", "TF2", "ALT", "NDET", "BIG", "VIEW")
VIEW_AGNOSTIC = ("TT", "ALT")
LABEL_FREE_VIEW_AWARE = ("NDET", "BIG")
EQUAL_BIT_BASELINES = VIEW_AGNOSTIC + LABEL_FREE_VIEW_AWARE
GO_F1, GO_SHARE = 0.02, 8 / 11
KAPPAS, PRIMARY = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0), 1.0
NO_CAP = 1e18


def key(kappa):
    return f"{kappa:g}"


def epoch_table(pairs):
    """per epoch, per view (0, 1), per mode (T, F): uncapped view_counts at TAU and UL bits; onboard detection
    counts; the records themselves (needed when a cap truncates the tokens)."""
    out = []
    for r1, r2, g1, g2 in pairs:
        e = dict(gt=(g1, g2), rec=(r1, r2), cnt=[{}, {}], bits=[{}, {}],
                 ndet=[len(r1["det"]["s"]), len(r2["det"]["s"])])
        for v, (rec, g) in enumerate(((r1, g1), (r2, g2))):
            for m, cond in (("T", "T0"), ("F", "F")):
                snd = P.send(rec, cond, Q, NO_CAP)
                e["cnt"][v][m] = P.view_counts(P.bs_decode(rec, snd, Q), g, TAU)
                e["bits"][v][m] = snd["ul"]
        out.append(e)
    return out


def modes_of(alloc, i, pref):
    """(mode of view 0, mode of view 1) REQUESTED at epoch index i; pref = dict(NDET=view, BIG=view, VIEW=view)."""
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


def _counts(e, v, snd):
    if snd["frame"]:
        return e["cnt"][v]["F"]
    if snd["ul"] == e["bits"][v]["T"]:                       # equal-size tokens packed in order -> the full token set
        return e["cnt"][v]["T"]
    return P.view_counts(P.bs_decode(e["rec"][v], snd, Q), e["gt"][v], TAU)


def realise(e, req, B=None):
    """what is actually sent for the requested modes under the joint UL cap B (None = no cap).
    -> (counts per view, UL bits per view, sent exactly as requested?). Never uses a label to decide what is sent."""
    left = NO_CAP if B is None else float(B)
    snd = [None, None]
    tok = [v for v in (0, 1) if req[v] == "T"]
    fits = sum(e["bits"][v]["T"] for v in tok) <= left
    for v in tok:
        snd[v] = P.send(e["rec"][v], "T0", Q, NO_CAP if fits else left / len(tok))
    left -= sum(snd[v]["ul"] for v in tok)
    for v in sorted((v for v in (0, 1) if req[v] == "F"), key=lambda v: e["bits"][v]["F"]):
        snd[v] = P.send(e["rec"][v], "F", Q, left)             # FRAME if it fits, else tokens within `left`
        left -= snd[v]["ul"]
    ok = all(snd[v]["frame"] if req[v] == "F" else snd[v]["ul"] == e["bits"][v]["T"] for v in (0, 1))
    return [_counts(e, v, snd[v]) for v in (0, 1)], [snd[v]["ul"] for v in (0, 1)], ok


def score(epochs, alloc, pref, i0=0, B=None):
    """unique-object counts, achieved UL bits and feasibility of an allocation over `epochs` (i0 = index of the
    first one) under the per-epoch joint cap B."""
    acc, bits, ok = {}, [], []
    for k, e in enumerate(epochs):
        cnt, b, as_req = realise(e, modes_of(alloc, i0 + k, pref), B)
        if B is not None and sum(b) > B + 1e-6:
            raise AssertionError(f"UL cap violated: {sum(b)} > {B} ({alloc}, epoch {i0 + k})")
        P.add(acc, P.unique_counts(cnt[0], cnt[1], *e["gt"]))
        bits.append(sum(b)); ok.append(as_req)
    if not acc:
        acc = dict(tp=0, fp=0, fn=0)
    p, r, f = P.prf(acc)
    return dict(f1=f, p=p, r=r, bits=float(np.mean(bits)) if bits else 0.0, bits_max=float(max(bits, default=0.0)),
                feasible=float(np.mean(ok)) if ok else 1.0, fp_wrong=acc["fp"], fp_dup=0, fp_assoc=0, **acc)


def budget_ref(first):
    """B_ref from the FIRST half, bits only: the smallest cap under which one FRAME (either view) + the other view's
    TOKENs fitted in every first-half epoch. None if there is no first half."""
    if not first:
        return None
    return float(max(max(e["bits"][0]["F"] + e["bits"][1]["T"], e["bits"][0]["T"] + e["bits"][1]["F"])
                     for e in first))


def select(first, B=None):
    """choices made on the FIRST half only. VIEW uses GT (unique F1 under the same cap B); NDET and BIG are label-free
    (onboard detection counts; FRAME sizes)."""
    nd = np.sum([e["ndet"] for e in first], axis=0) if first else np.zeros(2)
    fb = np.sum([[e["bits"][v]["F"] for v in (0, 1)] for e in first], axis=0) if first else np.zeros(2)
    f = [score(first, a, {}, 0, B)["f1"] for a in ("F1T", "TF2")]
    return dict(NDET=int(nd[1] > nd[0]), BIG=int(fb[1] > fb[0]), VIEW=int(f[1] > f[0]))


def run_sequence(pairs):
    T = epoch_table(pairs)
    h = len(T) // 2
    Bref = budget_ref(T[:h])
    levels = {}
    for kappa in KAPPAS:
        B = None if Bref is None else kappa * Bref
        pref = select(T[:h], B)
        levels[key(kappa)] = dict(B=B, pref=pref, alloc={a: score(T[h:], a, pref, h, B) for a in ALLOCS})
    prim = levels[key(PRIMARY)]
    return dict(B_ref=Bref, n_select=h, n_eval=len(T) - h, levels=levels, pref=prim["pref"], alloc=prim["alloc"],
                ff_uncapped=score(T[h:], "FF", {}, h, None))


def _diff(per_seq, seqs, lev, base):
    return [per_seq[s]["levels"][lev]["alloc"]["VIEW"]["f1"] - per_seq[s]["levels"][lev]["alloc"][base]["f1"]
            for s in seqs]


def _level(per_seq, seqs, lev):
    A = {a: [per_seq[s]["levels"][lev]["alloc"][a] for s in seqs] for a in ALLOCS}
    mean = {a: float(np.mean([x["f1"] for x in A[a]])) for a in ALLOCS}
    base = max(EQUAL_BIT_BASELINES, key=lambda a: mean[a])          # best baseline in hindsight: favours the baseline
    d = _diff(per_seq, seqs, lev, base)
    return dict(mean_f1=mean, mean_bits={a: float(np.mean([x["bits"] for x in A[a]])) for a in ALLOCS},
                feasible={a: float(np.mean([x["feasible"] for x in A[a]])) for a in ALLOCS},
                baseline=base, view_minus_baseline=float(np.mean(d)), per_seq_diff=dict(zip(map(str, seqs), d)),
                share_positive=float(np.mean([x > 0 for x in d])),
                view_minus_each={a: float(np.mean(_diff(per_seq, seqs, lev, a))) for a in ALLOCS if a != "VIEW"})


def _passes(per_seq, seqs, lev, baselines, mean):
    base = max(baselines, key=lambda a: mean[a])
    d = _diff(per_seq, seqs, lev, base)
    return bool(np.mean(d) >= GO_F1 and np.mean([x > 0 for x in d]) >= GO_SHARE - 1e-9), base, float(np.mean(d))


def gate(per_seq):
    """decision at kappa = PRIMARY; every other level is reported as the F1-bit trade-off (no decision)."""
    seqs = sorted(per_seq)
    lev = key(PRIMARY)
    g = _level(per_seq, seqs, lev)
    go, _, _ = _passes(per_seq, seqs, lev, EQUAL_BIT_BASELINES, g["mean_f1"])
    agn, agn_base, agn_d = _passes(per_seq, seqs, lev, VIEW_AGNOSTIC, g["mean_f1"])
    g.update(n_seq=len(seqs), kappa=PRIMARY, GO=go,
             bits_ratio_view_over_baseline=g["mean_bits"]["VIEW"] / max(g["mean_bits"][g["baseline"]], 1e-9),
             view_agnostic=dict(baseline=agn_base, view_minus_baseline=agn_d, passes=agn),
             # pre-registered reading of a NO-GO that is only due to NDET / BIG: the view-aware opportunity exists but
             # does not need GT (a label-free rule already captures it) -> reported as such, still not a GO for VIEW
             label_free_rule_sufficient=bool(agn and not go),
             tradeoff={k: {x: v[x] for x in ("mean_f1", "mean_bits", "feasible", "baseline", "view_minus_baseline",
                                             "share_positive")}
                       for k, v in ((key(kp), _level(per_seq, seqs, key(kp))) for kp in KAPPAS)},
             ff_uncapped=dict(f1=float(np.mean([per_seq[s]["ff_uncapped"]["f1"] for s in seqs])),
                              bits=float(np.mean([per_seq[s]["ff_uncapped"]["bits"] for s in seqs]))),
             note="GT-assisted view selection + ideal association: opportunity screening only; "
                  "applies to the sequences that passed the cross-view id screening")
    return g


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
