"""A7-S0 (reports/swarmbridge-a7s-semantic-family-design.md, pre-registered): the semantic representation family
TOKEN / CROP / FRAME measured in pilot 0B, placed in the A5 task environment (a6l1_task.TaskSurveyEnv). Heuristics
and an information upper bound only -- NO training.

  build PRECOMP_DIR OUT_NPZ --zip MDMT_ZIP     per-record table (bits, TP / FP at tau 0.3, CPU, onboard features) from
                                               the existing pilot records + scorer GT. Stays OUTSIDE git.
  run NPZ OUT_JSON [--regimes ref,K5,phy_mcs_blind] [--procs 5]
                                               both folds: tune on train (seeds 0..29), evaluate on test (11000..11099)
Policies: FIX (best fixed mode x slot policy), AGN (task-valued heuristic, mode-mean value, predicted bits),
CAU (value predicted from onboard features), ORA (true bits and true value: information upper bound, uses GT).
"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MODES = ["TOKEN", "CROP", "FRAME"]
TAU, Q, BETA = 0.3, 50, 0.5
BS_DEC_S = (0.0, 0.05, 0.05)                      # ASSUMED (BS inference time not measured in the pilot)
FOLD_A_TRAIN = (56, 34, 52, 68, 48, 62)           # first six of the registered pilot hash order
TUNE_SEEDS, TEST_SEEDS = list(range(30)), list(range(11000, 11100))
SLOTS = ("greedy", "pacedB1.5", "pacedB2", "pacedB3")
G = {}


# ---------------------------------------------------------------- build (pilot libs only here)
def build(args):
    import h1_pilot0b as P
    import h1_pilot0b_eval as E
    pairs, _ = E.load_pairs(args.precomp, args.zip, ("test",))
    rows = dict(seq=[], frame=[], view=[], bits=[], tp=[], fp=[], ngt=[], feat=[], enc_ms=[], det_ms=[])
    for s in sorted(pairs):
        for i, (r1, r2, g1, g2) in enumerate(pairs[s]):
            for v, (rec, g) in enumerate(((r1, g1), (r2, g2)), 1):
                b, tp, fp = [], [], []
                for cond in ("T0", "T1", "F"):
                    snd = P.send(rec, cond, Q, 1e12)
                    c = P.view_counts(P.bs_decode(rec, snd, Q), g, TAU)
                    b.append(max(snd["ul"], P.msg_cost(0) or 144)); tp.append(c["tp"]); fp.append(c["fp"])
                d = rec["det"]["s"]
                ns, nw = int((d >= P.TAU_HI).sum()), int(((d >= P.TAU_LO) & (d < P.TAU_HI)).sum())
                rows["seq"].append(s); rows["frame"].append(i); rows["view"].append(v)
                rows["bits"].append(b); rows["tp"].append(tp); rows["fp"].append(fp); rows["ngt"].append(len(g))
                rows["feat"].append([ns, nw, nw / max(ns + nw, 1), len(rec["cands"]["P1"])])
                rows["det_ms"].append(rec["det_ms"])
                rows["enc_ms"].append([0.0, sum(c["enc_ms"][Q] for c in rec["cands"]["P1"]), rec["frame"]["enc_ms"]])
    np.savez(args.out, **{k: np.array(v) for k, v in rows.items()})
    print("built", len(rows["seq"]), "records")


# ---------------------------------------------------------------- table / task value (duck-typed for the A5 env)
class SemTable:
    """ModeTable look-alike. probe = TOKEN (YOLO11n runs before every mode decision)."""

    def __init__(self, D, train_idx):
        self.modes = list(MODES)
        self.images = [f"{s}_{f}_{v}" for s, f, v in zip(D["seq"], D["frame"], D["view"])]
        self.bits = D["bits"].astype(float)
        self.psnr = np.full(self.bits.shape, 99.0)            # no PSNR floor in the semantic setting (declared)
        self.lpips = np.zeros(self.bits.shape)
        tr = np.asarray(train_idx)
        self.enc_s = np.array([D["det_ms"][tr].mean(), D["enc_ms"][tr, 1].mean(), D["enc_ms"][tr, 2].mean()]) / 1e3
        self.dec_s = np.array(BS_DEC_S)
        self.probe_j = 0
        self.mean_bits, self.mean_psnr = self.bits[tr].mean(0), self.psnr[tr].mean(0)
        self.feat = D["feat"].astype(float)
        X = np.stack([np.ones(len(tr)), self.bits[tr, 0], self.feat[tr, 3]], 1)   # 1, token bits, n P1 windows
        self.coef_crop = np.linalg.lstsq(X, self.bits[tr, 1], rcond=None)[0]
        self.fitted = True

    def predict(self, img):
        b = np.array([self.bits[img, 0],
                      max(np.array([1.0, self.bits[img, 0], self.feat[img, 3]]) @ self.coef_crop, self.bits[img, 0]),
                      self.mean_bits[2]])
        return b, self.psnr[img]


class SemTask:
    """a6l1_task.TaskTable look-alike: v = (TP - BETA FP) / Vbar, Vbar = train mean of FRAME."""

    def __init__(self, D, table, train_idx):
        tr = np.asarray(train_idx)
        self.tp, self.fp = D["tp"].astype(float), D["fp"].astype(float)
        self.tps = np.zeros(self.tp.shape)
        self.tp0, self.fp0 = self.tp[:, 2].copy(), self.fp[:, 2].copy()       # reference = FRAME
        self.ngt, self.nsmall = D["ngt"].astype(float), np.ones(len(self.tp))
        self.beta = BETA
        self.vbar = float(np.mean(self.tp0[tr] - BETA * self.fp0[tr]))
        self.v = (self.tp - BETA * self.fp) / self.vbar
        self.v_mean = self.v[tr].mean(0)
        f = table.feat
        self.X = np.stack([np.ones(len(f)), f[:, 0], f[:, 1], f[:, 2]], 1)   # 1, n strong, n weak, weak fraction
        self.coef = np.linalg.lstsq(self.X[tr], self.v[tr], rcond=None)[0]
        self.bits, self.probe_j = table.bits, 0
        # A7-S1 diagnostics: previous record of the SAME sequence and view (MDMT time order), and a within-group
        # derangement (control). -1 = no past (first epoch) -> the mode-mean value is used.
        key = {(int(s), int(v), int(f)): i for i, (s, v, f) in enumerate(zip(D["seq"], D["view"], D["frame"]))}
        self.prev = np.array([key.get((int(s), int(v), int(f) - 1), -1)
                              for s, v, f in zip(D["seq"], D["view"], D["frame"])])
        self.shuf = np.full(len(self.prev), -1)
        rng = np.random.default_rng(7)
        for g in sorted({(int(s), int(v)) for s, v in zip(D["seq"], D["view"])}):
            idx = np.where((D["seq"] == g[0]) & (D["view"] == g[1]))[0]
            order = rng.permutation(idx)
            self.shuf[order] = np.roll(order, 1)                 # each record gets another record of its group

    def vhat(self, img):
        return self.X[img] @ self.coef

    def covered(self, idx):
        return True


def make_envs():
    import a6l1_task as TK

    class SemEnv(TK.TaskSurveyEnv):
        VALUE = "hat"

        def task_value_hat(self, img):
            if self.VALUE == "mean":
                return self.tt.v_mean
            if self.VALUE == "true":
                return self.tt.v[img]
            if self.VALUE in ("lag", "shuf"):                   # decision information only; reward stays true v
                j = (self.tt.prev if self.VALUE == "lag" else self.tt.shuf)[img]
                return self.tt.v[j] if j >= 0 else self.tt.v_mean
            return self.tt.vhat(img)

    return {v: type(f"SemEnv_{v}", (SemEnv,), dict(VALUE=v)) for v in ("hat", "mean", "true", "lag", "shuf")}


# ---------------------------------------------------------------- policies
def tv_names():
    import a55_heur as H
    return H.family("tv")


def fix_names():
    return [f"fix:{s}{r}:{m}" for m in MODES for s in SLOTS + ("fixed",) for r in ("", "+reuse")]


def policy(name):
    import a55_heur as H
    import env_ev as V
    import mode_heur as MH
    import policies as P
    if name.startswith("fix:"):
        _, slot, m = name.split(":")
        j = MODES.index(m)
        reuse = slot.endswith("+reuse")
        slot = slot[:-6] if reuse else slot
        sp = P.fixed(j) if slot == "fixed" else P.from_name(slot, MODES)
        return (MH.reuse(sp) if reuse else sp), V.fixed_mode(j)
    return H.from_name(name, MODES)


COND = dict(FIX=("mean", "predicted"), AGN=("mean", "predicted"), CAU=("hat", "predicted"),
            ORA=("true", "oracle_probe"),
            # A7-S1 (review #15): diagnostics, none deployable
            TV_PB=("true", "predicted"), HV_TB=("hat", "oracle_probe"),
            LAG_TB=("lag", "oracle_probe"), LAG_PB=("lag", "predicted"),
            SHUF_TB=("shuf", "oracle_probe"), SHUF_PB=("shuf", "predicted"))


def init(npz):
    import torch
    torch.set_num_threads(1)
    import a6l1_task as TK
    D = dict(np.load(npz))
    G["D"], G["envs"], G["fold"] = D, make_envs(), {}
    for fold, train_seqs in (("A", FOLD_A_TRAIN), ("B", None)):
        is_tr = np.isin(D["seq"], FOLD_A_TRAIN) if fold == "A" else ~np.isin(D["seq"], FOLD_A_TRAIN)
        tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
        tab = SemTable(D, tr)
        tt = SemTask(D, tab, tr)
        TK._CACHE[id(tab)] = (tab, tt)
        G["fold"][fold] = dict(tab=tab, tt=tt, pools=dict(train=tr, test=te))


def episode(job):
    import a5_common as C
    import env_ev as V
    fold, regime, cond, name, pool, seed = job
    F = G["fold"][fold]
    value, info = COND[cond]
    core = G["envs"][value](F["tab"], F["pools"][pool], C.regime_cfg(regime, mode_info=info))
    sp, mp = policy(name)
    m = V.run_episode_ev(core, sp, mp, seed)
    ok = [tk for tk in core.finished if tk.ok]
    share = [sum(tk.mode == j for tk in ok) / max(len(ok), 1) for j in range(3)]
    keep = {k: float(m[k]) for k in ("return", "recall", "precision", "task_score", "ok_rate", "bits_sent", "n_ok",
                                    "n_planned", "fp")}
    keep.update(share_TOKEN=share[0], share_CROP=share[1], share_FRAME=share[2])
    by = {}
    for tk in core.finished:                                    # per-sequence task utility (cluster view)
        s = int(G["D"]["seq"][tk.img])
        a = by.setdefault(s, [0.0, 0, 0.0, 0])
        a[0] += tk.util; a[1] += 1
        if tk.ok:
            a[2] += float(F["tt"].v[tk.img, tk.mode]); a[3] += tk.mode == 2
    keep["by_seq"] = by
    return job, keep


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def run(args):
    regimes = args.regimes.split(",")
    tv, fx = tv_names(), fix_names()
    cand = {c: (fx if c == "FIX" else tv) for c in COND}
    jobs = [(f, r, c, n, "train", s) for f in "AB" for r in regimes for c in cand for n in cand[c] for s in TUNE_SEEDS]
    print("tune jobs", len(jobs), flush=True)
    with Pool(args.procs, initializer=init, initargs=(args.npz,)) as pool:
        res = pool.map(episode, jobs, chunksize=16)
        tune = {}
        for (f, r, c, n, _, _), m in res:
            tune.setdefault((f, r, c, n), []).append(m)
        best, fixed_best = {}, {}
        for f in "AB":
            for r in regimes:
                for c in cand:
                    best[(f, r, c)] = max(cand[c], key=lambda n: mean(tune[(f, r, c, n)], "return"))
                for m in MODES:      # per fixed mode: its best slot policy (reported, not used by the gate)
                    fixed_best[(f, r, m)] = max((n for n in fx if n.endswith(":" + m)),
                                                key=lambda n: mean(tune[(f, r, "FIX", n)], "return"))
        ev = [(f, r, c, best[(f, r, c)], "test", s) for f in "AB" for r in regimes for c in cand for s in TEST_SEEDS]
        ev += [(f, r, "FIX", fixed_best[(f, r, m)], "test", s) for f in "AB" for r in regimes for m in MODES
               for s in TEST_SEEDS]
        ev = list(dict.fromkeys(ev))
        print("eval jobs", len(ev), flush=True)
        res = pool.map(episode, ev, chunksize=8)
    test = {}
    for (f, r, c, n, _, _), m in res:
        test.setdefault((f, r, c, n), []).append(m)
    keys = ("return", "task_score", "recall", "precision", "ok_rate", "bits_sent", "share_TOKEN", "share_CROP",
            "share_FRAME")
    out = dict(design="reports/swarmbridge-a7s-semantic-family-design.md", status="EXPLORATORY; no training",
               modes=MODES, tau_decl=TAU, q=Q, bs_dec_s_assumed=BS_DEC_S, tune_seeds=[0, 29],
               test_seeds=[TEST_SEEDS[0], TEST_SEEDS[-1]], cells=[])
    for f in "AB":
        F = None
        for r in regimes:
            cell = dict(fold=f, regime=r, policies={}, fixed_modes={})
            for c in cand:
                n = best[(f, r, c)]
                rows = test[(f, r, c, n)]
                bs = {}
                for x in rows:
                    for s, a in x["by_seq"].items():
                        b = bs.setdefault(s, [0.0, 0, 0.0, 0])
                        for i in range(4):
                            b[i] += a[i]
                cell["policies"][c] = dict(name=n, train_return=mean(tune[(f, r, c, n)], "return"),
                                           by_seq={str(s): dict(util_per_task=b[0] / max(b[1], 1),
                                                                v_per_task=b[2] / max(b[1], 1),
                                                                frame_share=b[3] / max(b[1], 1))
                                                   for s, b in sorted(bs.items())},
                                           **{k: mean(rows, k) for k in keys},
                                           per_seed_task_score=[x["task_score"] for x in rows],
                                           per_seed_return=[x["return"] for x in rows])
            for m in MODES:
                n = fixed_best[(f, r, m)]
                cell["fixed_modes"][m] = dict(name=n, **{k: mean(test[(f, r, "FIX", n)], k) for k in keys})
            p = cell["policies"]
            head = p["ORA"]["task_score"] - p["FIX"]["task_score"]
            gap = p["ORA"]["return"] - p["FIX"]["return"]
            cell["retained_return_frac"] = {k: (p[k]["return"] - p["FIX"]["return"]) / gap for k in p if k != "FIX"}
            cell["gate"] = dict(
                ORA_minus_FIX_task_score=head, ORA_minus_FIX_return=p["ORA"]["return"] - p["FIX"]["return"],
                AGN_minus_FIX_task_score=p["AGN"]["task_score"] - p["FIX"]["task_score"],
                CAU_minus_FIX_task_score=p["CAU"]["task_score"] - p["FIX"]["task_score"],
                CAU_minus_FIX_return=p["CAU"]["return"] - p["FIX"]["return"],
                AGN_minus_FIX_return=p["AGN"]["return"] - p["FIX"]["return"],
                CAU_capture=(p["CAU"]["task_score"] - p["FIX"]["task_score"]) / head if head > 0 else float("nan"),
                GS1_cell=bool(head >= 0.02 and p["ORA"]["return"] > p["FIX"]["return"]))
            out["cells"].append(cell)
    out["GS1_regimes"] = [r for r in regimes if all(c["gate"]["GS1_cell"] for c in out["cells"] if c["regime"] == r)]
    Path(args.out).write_text(json.dumps(out, indent=1))
    for c in out["cells"]:
        p = c["policies"]
        print(c["fold"], c["regime"], " ".join(
            f"{k}:{p[k]['task_score']:.3f}/R{p[k]['return']:+.1f}/T{p[k]['share_TOKEN']:.2f},C{p[k]['share_CROP']:.2f},F{p[k]['share_FRAME']:.2f}"
            for k in ("FIX", "AGN", "CAU", "ORA")), "| fixed", {m: round(v["task_score"], 3) for m, v in c["fixed_modes"].items()},
            "| GS1", c["gate"]["GS1_cell"], flush=True)
        print("    retained ORA-FIX return:", {k: round(v, 2) for k, v in c["retained_return_frac"].items()}, flush=True)
    print("GS1 regimes (both folds):", out["GS1_regimes"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("precomp"); b.add_argument("out"); b.add_argument("--zip", required=True)
    r = sub.add_parser("run"); r.add_argument("npz"); r.add_argument("out")
    r.add_argument("--regimes", default="ref,K5,phy_mcs_blind"); r.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    (build if a.cmd == "build" else run)(a)


if __name__ == "__main__":
    main()
