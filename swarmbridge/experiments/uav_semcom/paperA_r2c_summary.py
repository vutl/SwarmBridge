"""Paper A robustness: (a) decision latency of every MPC arm in E1-E3 (+E2b); (b) test-dev video-level uncertainty:
two-way (environment seed x video) bootstrap CIs of Qmin-compliant recall contrasts. The A5 return is an episode sum
and cannot be split per video, so the video-level analysis uses detector recall.
Usage: python paperA_r2c_summary.py -> runs/uav_semcom/paperA_r2/summary_c.json"""
import collections
import gzip
import json

import numpy as np

import a5_common as C
import a5_eval as AE
import at1_task as AT
import paperA_r2 as R
from atc_summary import two_way


def load(name):
    p = R.OUT / f"{name}.json"
    if not p.exists():
        return None, None, None
    te = json.loads(p.read_text())
    dl = json.load(gzip.open(R.OUT / f"{name}.deliv.json.gz", "rt"))
    by = collections.defaultdict(list)
    for r in te["results"]:
        by[(r["phy"], r["combo"])].append(r)
    return te, dl, by


def main():
    AE.init(True)
    T = AT.tables(AE.G["table"], AE.G["pools"])
    rows_of = {int(mi): r for r, mi in enumerate(AE.G["man_idx"]) if mi >= 0}
    man = C.manifest()["images"]
    out = dict(latency={}, video_ci={})
    print("== decision latency of MPC arms: mean p99 / max p99 (ms), share of slots over 50 ms")
    for name in ("e12", "e3", "e1b", "e2b_test", "e2b_testdev"):
        te, dl, by = load(name)
        if te is None:
            continue
        for (phy, combo), rs in sorted(by.items()):
            if "+mpc" not in combo:
                continue
            p99 = np.array([x["ms_int_max_p99"] for r in rs for x in r["rows"]])
            o50 = np.array([x.get("int_over50", np.nan) for r in rs for x in r["rows"]])
            key = f"{name}|{phy}|{combo.split('|')[1]}"
            out["latency"][key] = dict(p99_mean=float(p99.mean()), p99_max=float(p99.max()), over50=float(np.nanmean(o50)))
            print(f"   {key:60s} {p99.mean():6.1f} / {p99.max():6.1f}  over50 {100 * np.nanmean(o50):.2f}%")
    te, dl, by = load("e3")
    S, H = te["sched"], te["H"]
    seqs = sorted({man[mi]["seq"] for k, v in dl.items() for ep in v for mi, *_ in ep})
    sidx = {s: i for i, s in enumerate(seqs)}

    def cells(phy, combo):
        acc_tp, acc_gt = 0, 0
        rs = by[(phy, combo)]
        for r in rs:
            eps = dl[f"{phy}|{combo}|{r['ckpt']}"]
            tp, gt = np.zeros((len(eps), len(seqs))), np.zeros((len(eps), len(seqs)))
            for e, ep in enumerate(eps):
                for mi, mode, ok, ot in ep:
                    rr, j = rows_of[mi], sidx[man[mi]["seq"]]
                    gt[e, j] += T["ngt"][rr]
                    if mode >= 0 and ok:
                        tp[e, j] += T["gt"][rr, mode]
            acc_tp, acc_gt = acc_tp + tp / len(rs), acc_gt + gt / len(rs)
        return acc_tp, acc_gt
    print(f"== test-dev, Qmin-compliant recall contrasts, two-way (seed x {len(seqs)} videos) bootstrap 95% CI (points)")
    for phy in ("S", "L"):
        nat = H["S"] if phy == "S" else H["L"]
        pairs = [("PPO-S - T:best (block A)", "tbest|ppo", "tbest|heur"),
                 ("H* + cal. MPC - H* + RobustMPC-style", f"{S}|h={nat}+mpc", f"{S}|h={nat}+mpcrh")]
        if phy == "L":
            pairs.append(("PPO-LA + cal. MPC - H* + cal. MPC", f"{S}|ppola+mpc", f"{S}|h={nat}+mpc"))
        for lab, a, b in pairs:
            d = [100 * x for x in two_way(*cells(phy, a), other=cells(phy, b), boot=5000)]
            out["video_ci"][f"{phy}|{lab}"] = d
            print(f"   {phy} {lab:42s} {d[0]:+6.2f} [{d[1]:+6.2f}, {d[2]:+6.2f}]")
    # (c) E2b ablation: SNR-binned vs pooled (no SNR bins) calibration, and pooled vs RobustMPC-style
    from ax2_summary import ci, fmt
    out["e2b"] = {}
    print("== E2b: calibrated MPC with vs without SNR bins (paired over seeds)")
    for base, abl in (("e12", "e2b_test"), ("e3", "e2b_testdev")):
        tb, _, bb = load(base)
        ta, _, ba = load(abl)
        if ta is None:
            continue
        assert tb["seeds"] == ta["seeds"]
        Sb = tb["sched"]

        def ret(by_, phy, c):
            return np.mean([[x["return"] for x in r["rows"]] for r in by_[(phy, c)]], 0)
        for (phy, c) in sorted({(p, cc) for p, cc in ba}):
            inc = c.split("|")[1].replace("+mpcpl", "")
            pooled = ret(ba, phy, c)
            binned = ret(bb, phy, f"{Sb}|{inc}+mpc")
            rh = ret(bb, phy, f"{Sb}|{inc}+mpcrh")
            nom = ret(bb, phy, f"{Sb}|{inc}+mpcnc") if (phy, f"{Sb}|{inc}+mpcnc") in bb else None
            row = {"binned - pooled": [float(x) for x in ci(binned - pooled)],
                   "pooled - RobustMPC-style": [float(x) for x in ci(pooled - rh)]}
            if nom is not None:
                row["pooled - nominal"] = [float(x) for x in ci(pooled - nom)]
            out["e2b"][f"{base}|{phy}|{inc}"] = row
            print(f"   {base:4s} {phy:8s} {inc:28s} " + "  ".join(f"{k}: {fmt(v)}" for k, v in row.items()))
    json.dump(out, open(R.OUT / "summary_c.json", "w"), indent=1)


if __name__ == "__main__":
    main()
