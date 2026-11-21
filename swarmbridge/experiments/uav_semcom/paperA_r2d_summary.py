"""Paper A E2c summary: one-factor ladder from the RobustMPC-style predictor (rh) to the pooled calibrator (pl),
incumbent H*, paired over environment seeds (95% t-CI). Each step changes one element of the rate predictor:
  rh -> ab1  drop the pessimistic factor 1 / (1 + max recent relative error)
  ab1 -> ab2 arithmetic instead of harmonic mean (last 5 slot rates of the UAV)
  ab2 -> ab3 sample the last 5 rates instead of using their mean
  ab3 -> ab4 whole-episode memory instead of the last 5
  ab4 -> ab5 all UAVs pooled instead of one history per UAV
  ab5 -> pl  wait for 20 s of observations before use (rho-scaled Shannon fallback until then)
Side contrasts: pl - ab6 (sampling vs mean at the pooled end), ab0 - rh (zero-rate slots left out), cal - pl (SNR
bins, = column F). Output: runs/uav_semcom/paperA_r2/summary_e2c.json and printed table."""
import json

import numpy as np

import paperA_r2 as R
from ax2_summary import ci

CONF = [("S", "S", "e12", "e2b_test", "e2c_test"), ("L", "L", "e12", "e2b_test", "e2c_test"),
        ("L_fade", "L_fade", "e12", "e2b_test", "e2c_test"), ("L_bursty", "L_bursty", "e12", "e2b_test", "e2c_test"),
        ("L_badfb", "L_badfb", "e12", "e2b_test", "e2c_test"), ("TD-S", "S", "e3", "e2b_testdev", "e2c_testdev"),
        ("TD-L", "L", "e3", "e2b_testdev", "e2c_testdev")]
STEPS = [("no pessimism", "ab1", "rh"), ("mean vs harmonic", "ab2", "ab1"), ("sample vs mean (last 5)", "ab3", "ab2"),
         ("episode memory", "ab4", "ab3"), ("pool UAVs", "ab5", "ab4"), ("20 s warm-up", "pl", "ab5"),
         ("TOTAL pl - rh", "pl", "rh"), ("sample vs mean (pooled)", "pl", "ab6"), ("zero slots out", "ab0", "rh"),
         ("SNR bins (cal - pl)", "cal", "pl")]
cache = {}


def returns(name):
    if name not in cache:
        te = json.loads((R.OUT / f"{name}.json").read_text())
        cache[name] = {(r["phy"], r["combo"]): (r["seeds"], np.array([x["return"] for x in r["rows"]])) for r in te["results"]}
    return cache[name]


def main():
    out = {}
    for lab, phy, base, e2b, e2c in CONF:
        te = json.loads((R.OUT / f"{base}.json").read_text())
        S, h = te["sched"], te["H"]["S" if phy == "S" else "L"]
        src = {"cal": (base, "+mpc"), "rh": (base, "+mpcrh"), "pl": (e2b, "+mpcpl")}
        src.update({a: (e2c, f"+mpc{a}") for a in ("ab0", "ab1", "ab2", "ab3", "ab4", "ab5", "ab6", "ab7")})

        def get(a):
            name, suf = src[a]
            seeds, v = returns(name)[(phy, f"{S}|h={h}{suf}")]
            return seeds, v

        s0, pl = get("pl")
        s7, ab7 = get("ab7")
        assert s0 == s7
        # ab7 re-implements PoolRates; identical except rare episodes (floating-point differences in the sampling
        # probabilities can flip a draw and the episode then diverges) -> reported, not asserted
        row = {"level": {a: float(get(a)[1].mean()) for a in src},
               "ab7_check": dict(n_diff=int((np.abs(pl - ab7) > 1e-6).sum()), mean_diff=float((ab7 - pl).mean()))}
        print(f"   [check] ab7 vs pl: {row['ab7_check']['n_diff']} / {len(pl)} episodes differ, "
              f"mean diff {row['ab7_check']['mean_diff']:+.3f}")
        for txt, a, b in STEPS:
            sa, va = get(a)
            sb, vb = get(b)
            assert sa == sb
            row[txt] = [float(x) for x in ci(va - vb)]
        out[lab] = row
        print(f"== {lab}  levels: " + "  ".join(f"{a} {row['level'][a]:.1f}" for a in
                                               ("rh", "ab1", "ab2", "ab3", "ab4", "ab5", "pl", "ab6", "cal")))
        for txt, _, _ in STEPS:
            m, lo, hi = row[txt]
            print(f"   {txt:26s} {m:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]{' *' if lo > 0 or hi < 0 else ''}")
    json.dump(out, open(R.OUT / "summary_e2c.json", "w"), indent=1)


if __name__ == "__main__":
    main()
