"""Budgeted message selection, step 1 (protocol v1.1 section 10; dev scenes only; CPU only).

Question: under a shared bit budget smaller than "send everything", which subset of each helper's Kalman-filtered
detections should be sent, and how much tracking quality is left on the table compared with a diagnostic that knows
the future?  Channel delay/loss is EXOGENOUS here (ledger.py, independent of payload and actions), so this tests
content selection at exogenous message age, not scheduling with queues (that is step 2).

Message content: KF-PV (box_proto kf_nor): each helper's confirmed, just-updated CV-Kalman tracks with score >=
TAU_SEND, coded with boxcodec (87 bit/box, 132 bit header, byte-padded). Every helper sends one packet every frame,
possibly with 0 boxes (header only = heartbeat); the receiver forecasts each used message by its age and fuses with
noisy-OR, cap 500 (identical to kf_nor).

Budget B (kbit per frame, all 4 helpers together, counted as 8 * packet bytes at send time, lost packets included).
Allocation:
  local_*    each helper gets B/4 and uses only its own message candidates
  central_*  one scheduler sees all 4 helpers' candidates (no receiver state) and fills B jointly
Policies (greedy fill in priority order; a box is added if the packet sizes still fit the budget):
  local_rand, central_rand   random order (seeded by seed|scene|frame)
  local_score, central_score highest score first
  central_novh               score order, skip a box within 2 m of a box already selected from another helper
  central_nove               score order, skip a box within 2 m of the receiver's own confident detections (score >=
                             0.3) at the PREVIOUS frame (needs 1-frame-old feedback from UAV0; feedback bits reported
                             separately as 26 bit per fed-back position, not charged to B)
  central_novhe              both skips
  central_strong{tau}        score order, skip a box within 2 m of a receiver detection with score >= tau at the
                             previous frame (receiver already SURE: no confirmation needed; weak receiver detections are
                             still confirmed). Feedback = 26 bit per fed-back position (not charged to B)
  central_novh_strong{tau}   central_strong{tau} + the novh cross-helper skip
  central_demote{tau}        score order, but boxes within 2 m of a receiver detection with score >= tau at the
                             previous frame are moved to the END instead of skipped (sent only if budget remains).
                             Same feedback as central_strong{tau}
Diagnostics (NOT policies; they use ground truth and are "best-found", not upper bounds):
  oracle_task   knows GT and UAV0's own detections at the SEND frame (assumes age 0, does not know the channel):
                boxes whose position is within 2 m of a GT car that UAV0 misses (no own confident detection within
                2 m) and not yet covered by another selected box come first (by score), then the rest by score
  oracle_full   same, but also knows the realised channel: evaluates at the actual arrival frame with the forecast
                position, and spends no payload on packets that will be lost or arrive after the episode
Both diagnostics are one-step/myopic (value judged at one frame), ignoring later reuse from the receiver's cache.

Usage: python -I budget_proto.py DATA_ROOT AGENT_DIR OUT [--only P1,P2,...]
  --only: run just these policies and merge their rows into an existing OUT/budget_proto.json
Writes OUT/dets/<policy>_B<kbit>/<ledger>/results_nusc.json and OUT/budget_proto.json (mAP, sha256, bit accounting,
boxes sent, feedback bits).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import boxcodec as BC  # noqa: E402
import box_proto as BP  # noqa: E402
import kalman as K  # noqa: E402
import ledger as L  # noqa: E402
from evaldet import MiniDB, evaluate  # noqa: E402
from fusion import load_variant, select  # noqa: E402
from soft_fusion import fuse as soft_fuse  # noqa: E402

BUDGETS = [5, 10, 20]                       # kbit / frame, all helpers
LEDGERS = [("sync", 0), ("rand", 0), ("rand", 1), ("rand", 2)]
STRONG = [0.6, 0.8]
POLICIES = (["local_rand", "local_score", "central_rand", "central_score", "central_novh", "central_nove",
             "central_novhe"] + [f"central_strong{x}" for x in STRONG] + [f"central_novh_strong{x}" for x in STRONG]
            + ["oracle_task", "oracle_full"] + [f"central_demote{x}" for x in STRONG])
R_DUP = 2.0
EGO_CONF = 0.3
FEEDBACK_BITS_PER_POS = 26                  # dx, dy at 13 bit each (same quantiser as the box codec)
HELPERS = (1, 2, 3, 4)
if "--smoke" in sys.argv:                     # quick end-to-end check on one ledger / budget
    sys.argv.remove("--smoke")
    BUDGETS, LEDGERS = [10], [("rand", 0)]
ONLY = None
if "--only" in sys.argv:
    _k = sys.argv.index("--only")
    ONLY = sys.argv[_k + 1].split(",")
    del sys.argv[_k:_k + 2]


def pkt_bits(n):
    return 8 * BC.packet_bytes(n)


def near(xy, ref, r=R_DUP):
    if len(ref) == 0 or len(xy) == 0:
        return np.zeros(len(xy), bool)
    return (np.linalg.norm(xy[:, None, :] - np.asarray(ref)[None, :, :], axis=2) < r).any(1)


def fill(cands, order, budget_bits, per_helper, alive=None, skip=None):
    """cands: {j: box dict}; order: list of (j, i) in priority order. Greedy fill.
    per_helper: budget is split equally per helper. alive: {j: bool} helpers whose packet is worth payload.
    skip(j, i, chosen_xy) -> True to skip a candidate given the boxes already chosen [(j, xy)]."""
    n = {j: 0 for j in HELPERS}
    chosen = {j: [] for j in HELPERS}
    chosen_xy = []
    cap = {j: budget_bits / 4 for j in HELPERS}
    for j, i in order:
        if alive is not None and not alive[j]:
            continue
        if per_helper:
            if pkt_bits(n[j] + 1) > cap[j]:
                continue
        else:
            if sum(pkt_bits(n[k] + (k == j)) for k in HELPERS) > budget_bits:
                continue
        if skip is not None and skip(j, i, chosen_xy):   # after the budget check: skip may record state
            continue
        n[j] += 1
        chosen[j].append(i)
        chosen_xy.append((j, cands[j]["xy"][i]))
    return chosen


def main():
    root, adir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    db = MiniDB(root)
    frames = L.scene_frames_from_db(db, BP.DEV)
    fidx = {t: k for toks in frames.values() for k, t in enumerate(toks)}
    scene_of = {t: sc for sc, toks in frames.items() for t in toks}
    toks_all = [t for toks in frames.values() for t in toks]
    late = [t for t in toks_all if fidx[t] >= 3]
    pos = BP.uav_positions(root)
    A = {j: load_variant(adir / f"agent{j}" / "results_nusc.json") for j in range(5)}

    # candidates: every helper's KF-PV message content (same as box_proto kf family)
    cand = {}
    for j in HELPERS:
        for sc, toks in frames.items():
            trk = K.run_sequence([A[j][t] for t in toks])
            for k, t in enumerate(toks):
                cand[(j, t)] = select(BP.base(trk[k]), trk[k]["score"] >= BP.TAU_SEND)
    full_kbit = np.mean([sum(pkt_bits(len(cand[(j, t)]["score"])) for j in HELPERS) for t in toks_all]) / 1000
    print(f"full KF-PV message: {full_kbit:.1f} kbit/frame", flush=True)

    ego_conf = {t: BP.base(A[0][t])["xy"][A[0][t]["score"] >= EGO_CONF] for t in toks_all}
    gt = {t: db.gt_xy(t) for t in toks_all}

    def missed_gt(t):
        g = gt[t]
        return g[~near(g, ego_conf[t])]

    ledgers = {}
    for sc_name, seed in LEDGERS:
        ledgers[f"{sc_name}_s{seed}"] = L.build(frames, sc_name, seed)
    pk = {name: {(p["sender"], S["tokens"][p["send_frame"]]): p for S in led["scenes"].values() for p in S["packets"]}
          for name, led in ledgers.items()}

    def select_all(policy, budget_kbit, ledger_name, rng_seed=0):
        """-> {(j, t): chosen index list}, feedback bits per frame list"""
        B = budget_kbit * 1000
        sel, fb = {}, []
        for sc, toks in frames.items():
            for k, t in enumerate(toks):
                c = {j: cand[(j, t)] for j in HELPERS}
                allidx = [(j, i) for j in HELPERS for i in range(len(c[j]["score"]))]
                by_score = sorted(allidx, key=lambda ji: -c[ji[0]]["score"][ji[1]])
                per_helper = policy.startswith("local")
                skip, alive, order = None, None, by_score
                if policy.endswith("_rand"):
                    key = hashlib.sha256(f"{rng_seed}|{sc}|{k}".encode()).digest()
                    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
                    order = [allidx[i] for i in rng.permutation(len(allidx))]
                prev_ego = ego_conf[toks[k - 1]] if k > 0 else np.zeros((0, 2))
                if "strong" in policy or "demote" in policy:
                    tau = float(policy.split("strong" if "strong" in policy else "demote")[1])
                    if k > 0:
                        pe = A[0][toks[k - 1]]
                        prev_ego = BP.base(pe)["xy"][pe["score"] >= tau]
                if "demote" in policy:
                    sure = set(ji for ji in by_score
                               if len(prev_ego) and np.linalg.norm(prev_ego - c[ji[0]]["xy"][ji[1]], axis=1).min() < R_DUP)
                    order = [ji for ji in by_score if ji not in sure] + [ji for ji in by_score if ji in sure]
                if policy in ("central_nove", "central_novhe") or "strong" in policy or "demote" in policy:
                    fb.append(FEEDBACK_BITS_PER_POS * len(prev_ego) / 1000)
                if policy in ("central_novh", "central_nove", "central_novhe") or "strong" in policy:
                    useh = policy in ("central_novh", "central_novhe") or policy.startswith("central_novh_strong")
                    usee = policy != "central_novh"

                    def skip(j, i, chosen, c=c, prev_ego=prev_ego, useh=useh, usee=usee):
                        xy = c[j]["xy"][i]
                        if usee and len(prev_ego) and np.linalg.norm(prev_ego - xy, axis=1).min() < R_DUP:
                            return True
                        return useh and any(jj != j and np.linalg.norm(x - xy) < R_DUP for jj, x in chosen)
                if policy.startswith("oracle"):
                    full = policy == "oracle_full"
                    vals = {}
                    alive = {j: True for j in HELPERS}
                    for j in HELPERS:
                        p = pk[ledger_name][(j, t)] if full else None
                        if full and (p["drop"] or p["arrival_frame"] >= len(toks)):
                            alive[j] = False
                            continue
                        age = p["delay"] if full else 0
                        ta = toks[k + age]
                        xy = c[j]["xy"] + c[j]["vel"] * (age * K.DT)
                        vals[j] = (xy, near(xy, missed_gt(ta)))
                    order_cov = [ji for ji in by_score if ji[0] in vals and vals[ji[0]][1][ji[1]]]
                    cov_set = set(order_cov)
                    rest = [ji for ji in by_score if ji not in cov_set]
                    order = order_cov + rest
                    covered = []

                    def skip(j, i, chosen, vals=vals, covset=cov_set, covered=covered):
                        # among GT-covering boxes, skip a second box for an already covered GT car
                        if (j, i) not in covset:
                            return False
                        xy = vals[j][0][i]
                        if any(np.linalg.norm(xy - x) < R_DUP for x in covered):
                            return True
                        covered.append(xy)
                        return False
                ch = fill(c, order, B, per_helper, alive, skip)
                for j in HELPERS:
                    sel[(j, t)] = sorted(ch[j])
        return sel, fb

    rows = []
    for name, led in ledgers.items():
        use = L.lookup(led)
        for budget in BUDGETS:
            for policy in (ONLY or POLICIES):
                sel, fb = select_all(policy, budget, name)
                msgs, tx = {}, []
                for t in toks_all:
                    bits = 0
                    for j in HELPERS:
                        m = select(cand[(j, t)], np.array(sel[(j, t)], int))
                        data, st = BC.encode(j, fidx[t], pos[(t, j)], m)
                        _, dec = BC.decode(data)
                        msgs[(j, t)] = (dec, st)
                        bits += 8 * st["bytes"]
                    assert bits <= budget * 1000, (policy, budget, bits)
                    tx.append(bits / 1000)
                preds = {}
                for t in toks_all:
                    own = BP.base(A[0][t])
                    sets = [(0, own["xy"], own["score"], BP.records(own, t))]
                    for j in HELPERS:
                        src, age = use[t][j]
                        if src is None:
                            continue
                        m = K.forecast(msgs[(j, src)][0], age)
                        sets.append((j, m["xy"], m["score"], BP.records(m, t)))
                    recs = soft_fuse(sets, "noisyor", return_records=True)
                    recs.sort(key=lambda r: -r["detection_score"])
                    preds[t] = [dict(r, sample_token=t) for r in recs[:BP.CAP]]
                tag = f"{policy}_B{budget}"
                path = out / "dets" / tag / name / "results_nusc.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                blob = json.dumps({"meta": BP.META, "results": preds}).encode()
                path.write_bytes(blob)
                pr = {t: (np.array([r["translation"][:2] for r in v]).reshape(-1, 2),
                          np.array([r["detection_score"] for r in v])) for t, v in preds.items()}
                r_all, r_late = evaluate(db, pr, toks_all), evaluate(db, pr, late)
                nbox = np.mean([sum(len(sel[(j, t)]) for j in HELPERS) for t in toks_all])
                row = {"policy": policy, "budget_kbit": budget, "ledger": name, "mAP": round(r_all["mAP"], 4),
                       "mAP_f3": round(r_late["mAP"], 4), "kbit_tx_mean": round(float(np.mean(tx)), 3),
                       "kbit_tx_max": round(float(np.max(tx)), 3), "boxes_sent_mean": round(float(nbox), 1),
                       "feedback_kbit_mean": round(float(np.mean(fb)), 3) if fb else 0.0,
                       "pred": str(path.relative_to(out)), "sha256": hashlib.sha256(blob).hexdigest()}
                rows.append(row)
                print(f"{tag:22s} {name:9s} mAP {row['mAP']:.3f} (f>=3 {row['mAP_f3']:.3f}) tx {row['kbit_tx_mean']:5.2f}"
                      f" kbit boxes {nbox:5.1f} fb {row['feedback_kbit_mean']:.2f}", flush=True)
    if ONLY:
        old = json.loads((out / "budget_proto.json").read_text())["rows"]
        rows = [r for r in old if r["policy"] not in ONLY] + rows
    (out / "budget_proto.json").write_text(json.dumps({"budgets": BUDGETS, "full_kbit": round(float(full_kbit), 2),
                                                       "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
