"""A8 mismatch ladder summary (reports/archive/a7-a8-b1/swarmbridge-a8-b1-preregistration.md): delta = PPO mode-only (frozen
phy_mcs_blind models, mean of 3) - H*(level), with the three A5.5 uncertainty levels and the pre-registered GO rule.
Usage: python a8_summarize.py RUNS_A8_DIR CODE_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

LADDER = [("phy_mcs", 0.0), ("phy_mcs_l75", 0.25), ("phy_mcs_l50", 0.5), ("phy_mcs_l25", 0.75),
          ("phy_mcs_blind", 1.0)]
TOL = 0.5


def main():
    O = Path(sys.argv[1])
    sys.path.insert(0, sys.argv[2])
    import a55_summarize as S
    out = dict(ladder=LADDER, pools={})
    for pool in ("test", "testdev"):
        ev = json.loads((O / f"eval_{pool}.json").read_text())
        rows = []
        for r, mis in LADDER:
            res = ev["results"][r]
            ppo = sorted(s for s in res if s.startswith("P:"))
            h = S.arr(res["T:HSTAR"])
            d = np.mean([S.arr(res[s]) for s in ppo], axis=0) - h
            m, lo, hi = S.ci1(d)
            row = dict(regime=r, mismatch=mis, hstar=ev["resolved"][r]["T:HSTAR"], hstar_return=float(h.mean()),
                       ppo_return=float(h.mean() + m), delta=float(m), ci1=[float(lo), float(hi)],
                       per_model=[float((S.arr(res[s]) - h).mean()) for s in ppo],
                       best_a5=float(S.arr(res["T5:best"]).mean()), sc=float(S.arr(res["T:SC"]).mean()),
                       ppo_ok=float(np.mean([S.arr(res[s], "ok_rate").mean() for s in ppo])),
                       hstar_ok=float(S.arr(res["T:HSTAR"], "ok_rate").mean()))
            bp = O / f"boot_{pool}_{r}.json"
            if bp.exists():
                boot = json.loads(bp.read_text())
                for lev, rm in (("ci2", False), ("ci3", True)):
                    dd = S.boot_deltas(boot, r, ppo, "T:HSTAR", rm)
                    row[lev] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
                row["boot_mean"] = float(S.boot_deltas(boot, r, ppo, "T:HSTAR", False).mean())
            rows.append(row)
        ds = [x["delta"] for x in rows]
        mono = all(b >= a - TOL for a, b in zip(ds, ds[1:]))
        n_pos = sum(1 for x in rows if "ci3" in x and x["ci3"][0] > 0)
        # pre-registered rule: GO = non-decreasing (tolerance 0.5) AND level-3 CI > 0 at >= 2 levels;
        # NO-GO = no trend, or a positive advantage at one level only; otherwise INCONCLUSIVE (trend, CIs too wide)
        if mono and n_pos >= 2:
            verdict = "GO"
        elif not mono or sum(d > 0 for d in ds) <= 1:
            verdict = "NO-GO"
        else:
            verdict = "INCONCLUSIVE"
        # manipulation check (added AFTER the run, conservative: it can only withdraw a GO): if the per-episode
        # returns of PPO and H* do not change along the ladder, the five levels are one experiment repeated and the
        # rule above is vacuous. The mechanical output is kept as rule_output.
        base = {s: S.arr(ev["results"][LADDER[0][0]][s]) for s in ppo + ["T:HSTAR"]}
        max_change = max(float(np.abs(S.arr(ev["results"][r][s]) - base[s]).max()) for r, _ in LADDER for s in base)
        same_h = len({x["hstar"] for x in rows}) == 1
        effective = not (max_change < 1e-9 and same_h)
        out["pools"][pool] = dict(rows=rows, monotone_within_tol=mono, n_levels_ci3_positive=n_pos,
                                  rule_output=verdict, max_return_change_along_ladder=max_change,
                                  manipulation_effective=effective,
                                  verdict=verdict if effective else "INVALID (manipulation has no effect)")
        verdict = out["pools"][pool]["verdict"]
        print(f"== {pool}: monotone {mono}, levels with level-3 CI > 0: {n_pos} -> rule {out['pools'][pool]['rule_output']}"
              f"; max |return change| along the ladder {max_change:.2g} -> {verdict}")
        for x in rows:
            print("  %-14s mis %.2f  H* %-26s %7.2f  PPO %7.2f  d %+6.2f [%+.2f,%+.2f]  L2 %s  L3 %s  models %s  ok %.3f/%.3f" % (
                x["regime"], x["mismatch"], x["hstar"][:26], x["hstar_return"], x["ppo_return"], x["delta"], *x["ci1"],
                "[%+.2f,%+.2f]" % tuple(x["ci2"]) if "ci2" in x else "-",
                "[%+.2f,%+.2f]" % tuple(x["ci3"]) if "ci3" in x else "-",
                " ".join("%+.1f" % v for v in x["per_model"]), x["ppo_ok"], x["hstar_ok"]))
    out["primary_pool"] = "test"
    out["verdict"] = out["pools"]["test"]["verdict"]
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
