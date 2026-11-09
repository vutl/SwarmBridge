"""A5.5 summary tables (markdown) from runs/uav_semcom/a55/, following the pre-registered rules of
reports/uav-semcom-a5.5.md section 0:
  * 0.3 confirmatory deltas (env seeds 8000..8099): PPO mode-only - H*, PPO joint - H*, PPO joint K5 5M - H*,
    T:SC - A5 T:best; test and test-dev, regimes ref / K5 / phy_mcs_blind;
  * 0.4 three uncertainty levels: (1) paired t over env seeds of the 3-model mean; (2) video cluster bootstrap with
    re-simulation (percentile of the B replicate deltas); (3) as (2) plus resampling the 3 training seeds in each
    replicate;
  * 0.5 conclusion rule; per-training-seed spread; T:SC vs A5 T:best over the 21 regimes; exploratory seeds 7000..7099.
Usage: python a55_summarize.py [sections...]   (main seeds sc expl tune)
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import t as _t

    def tq(n):
        return float(_t.ppf(0.975, n - 1))
except ImportError:   # pragma: no cover
    def tq(n):
        return 1.984 if n >= 100 else 2.01

REPO = Path(__file__).resolve().parents[3]
O = REPO / "runs" / "uav_semcom" / "a55"
SEL = ["ref", "K5", "phy_mcs_blind"]
POOLS = ["test", "testdev"]
L3_SEED = 555


def load(p):
    return json.load(open(p)) if p.exists() else None


def hstar_spec(regime):
    d = json.loads((O / "tune" / f"{regime}.json").read_text())
    return "T:SC" if d["hstar"] == d["tuned"]["sc"] and d["hstar"] != d["a5_best"] else "T5:best"


def ppo_groups(results):
    """{group: [spec of seed 0, 1, 2]} for the PPO specs of a results dict (label 'mode_best_s0' -> 'mode_best')."""
    g = {}
    for sp in results:
        if sp.startswith("P:"):
            lab = sp[2:].split("=", 1)[0]
            g.setdefault(lab.rsplit("_s", 1)[0], []).append(sp)
    return {k: sorted(v, key=lambda s: s[2:].split("=", 1)[0]) for k, v in g.items()}


def arr(eps, k="return"):
    return np.array([e[k] for e in eps], float)


def ci1(d):
    n = len(d)
    h = tq(n) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h


def fmt(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def boot_deltas(boot, regime, a_specs, b_spec, resample_models):
    """Per-replicate delta of mean(A over its models) - B; a_specs = list of specs (models) or one heuristic spec."""
    res = boot["results"][regime]
    out = []
    for b in sorted(res, key=int):
        r = res[b]
        if resample_models and len(a_specs) > 1:
            rng = np.random.default_rng([L3_SEED, int(b)])
            pick = [a_specs[j] for j in rng.integers(0, len(a_specs), len(a_specs))]
        else:
            pick = a_specs
        a = np.mean([np.array(r[s], float) for s in pick], axis=0)
        out.append(float(np.mean(a - np.array(r[b_spec], float))))
    return np.array(out)


def verdict(m, c1, c3):
    if c3 is not None and (c3[0] > 0 or c3[1] < 0):
        return "**bền**" + (" (PPO hơn)" if m > 0 else " (PPO kém)")
    if c1[0] > 0 or c1[1] < 0:
        return "chưa chắc" + (" (+)" if m > 0 else " (−)")
    return "không có khác biệt"


def main_table():
    print("### Δ xác nhận (seed 8000..8099), ba mức độ bất định\n")
    print("| regime | pool | H* | so sánh | Δ return [CI mức 1: seed env] | CI mức 2: + video | CI mức 3: + seed train"
          " | kết luận (0.5) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in SEL:
        hs = hstar_spec(r)
        for pool in POOLS:
            heur = load(O / "final" / f"{pool}_heur.json")
            pp = load(O / "final" / f"{pool}_ppo_{r}.json")
            boot = load(O / "boot" / f"{pool}_{r}.json")
            if heur is None or pp is None or r not in heur["results"]:
                continue
            assert heur["seeds"] == pp["seeds"]
            H = heur["results"][r]
            hret = arr(H[hs])
            groups = ppo_groups(pp["results"][r])
            rows = [(f"PPO {g} − H*", specs) for g, specs in groups.items()]
            rows.append(("T:SC − A5 T:best", None))
            for lab, specs in rows:
                if specs is None:
                    d = arr(H["T:SC"]) - arr(H["T5:best"])
                    a_specs, b_spec, three = ["T:SC"], "T5:best", False
                else:
                    d = np.mean([arr(pp["results"][r][s]) for s in specs], axis=0) - hret
                    a_specs, b_spec, three = specs, hs, True
                c1 = ci1(d)
                c2 = c3 = None
                if boot is not None and r in boot["results"] and all(s in next(iter(boot["results"][r].values()))
                                                                       for s in a_specs + [b_spec]):
                    d2 = boot_deltas(boot, r, a_specs, b_spec, False)
                    c2 = (np.percentile(d2, 2.5), np.percentile(d2, 97.5))
                    d3 = boot_deltas(boot, r, a_specs, b_spec, True) if three else d2
                    c3 = (np.percentile(d3, 2.5), np.percentile(d3, 97.5))
                s2 = f"[{c2[0]:+.2f}, {c2[1]:+.2f}]" if c2 else "—"
                s3 = f"[{c3[0]:+.2f}, {c3[1]:+.2f}]" if c3 else "—"
                v = verdict(c1[0], c1[1:], c3) if c3 else "chờ bootstrap"
                print(f"| {r} | {pool} | {hs} | {lab} | {fmt(*c1)} | {s2} | {s3} | {v} |")
    print()


def seeds_table():
    print("### Từng model PPO (seed huấn luyện) − H*, seed 8000..8099\n")
    print("| regime | pool | nhóm | s0 | s1 | s2 |")
    print("|---|---|---|---|---|---|")
    for r in SEL:
        hs = hstar_spec(r)
        for pool in POOLS:
            heur = load(O / "final" / f"{pool}_heur.json")
            pp = load(O / "final" / f"{pool}_ppo_{r}.json")
            if heur is None or pp is None:
                continue
            hret = arr(heur["results"][r][hs])
            for g, specs in ppo_groups(pp["results"][r]).items():
                cells = [fmt(*ci1(arr(pp["results"][r][s]) - hret)) for s in specs]
                print(f"| {r} | {pool} | {g} | " + " | ".join(cells) + " |")
    print()


def sc_table(name="test_heur.json", title="T:SC − A5 T:best trên pool test, seed 8000..8099 (21 regime)"):
    d = load(O / "final" / name)
    if d is None:
        return
    print(f"### {title}\n")
    print("| regime | A5 T:best | T:SC | H* | return A5 T:best | Δ T:SC − T:best [CI] | Δ ok_rate | Δ PSNR |")
    print("|---|---|---|---|---|---|---|---|")
    for r, H in d["results"].items():
        nm = d["resolved"][r]
        dd = arr(H["T:SC"]) - arr(H["T5:best"])
        dok = (arr(H["T:SC"], "ok_rate") - arr(H["T5:best"], "ok_rate")).mean()
        dps = (arr(H["T:SC"], "mean_psnr") - arr(H["T5:best"], "mean_psnr")).mean()
        print(f"| {r} | {nm['T5:best']} | {nm['T:SC']} | {'SC' if hstar_spec(r) == 'T:SC' else 'A5'} | "
              f"{arr(H['T5:best']).mean():+.2f} | {fmt(*ci1(dd))} | {dok:+.3f} | {dps:+.2f} |")
    print()


def tune_table():
    print("### Tinh chỉnh trên train (seed 0..29)\n")
    print("| regime | T:SC | return | A5 T:best | return | H* |")
    print("|---|---|---|---|---|---|")
    for p in sorted((O / "tune").glob("*.json")):
        d = json.loads(p.read_text())
        sc = d["tuned"]["sc"]
        print(f"| {d['regime']} | {sc} | {d['rows'][sc]['return']:+.2f} | {d['a5_best']} | "
              f"{d['a5_best_train_return']:+.2f} | {'SC' if d['hstar'] == sc and sc != d['a5_best'] else 'A5'} |")
    print()


def expl():
    sc_table("expl_test_heur_7000.json", "Khám phá: T:SC − A5 T:best, pool test, seed A5 7000..7099")
    sc_table("expl_testdev_heur_7000.json", "Khám phá: T:SC − A5 T:best, pool test-dev, seed A5 7000..7099")


def main():
    secs = sys.argv[1:] or ["tune", "main", "seeds", "sc", "expl"]
    for s in secs:
        {"tune": tune_table, "main": main_table, "seeds": seeds_table, "sc": sc_table, "expl": expl}[s]()


if __name__ == "__main__":
    main()
