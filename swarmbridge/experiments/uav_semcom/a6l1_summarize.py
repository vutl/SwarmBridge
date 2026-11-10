"""A6.0-L1 summary tables (markdown) from runs/uav_semcom/a6l1/ (reports/archive/uav-semcom-a/uav-semcom-a6-l1.md section 0.5).
Run in the WSL env (policy-level AP50 needs the cached per-image matching of a6l1_ap).

  tune    tuned members under the task reward (train) vs the A5 PSNR-tuned ones
  train   PPO learning curves (val task return), best / last checkpoints
  main    per policy: task + PSNR returns, recall, small-object recall, FP, precision, pooled policy-level AP50, bits,
          deadline success, energy, mission time (PPO = mean over the 3 training seeds)
  cmp     pre-registered paired comparisons (env-seed CI; + video cluster bootstrap levels 2 and 3 where available):
          PPO task - PPO PSNR, T6:best - T5:best, PPO task - T6:best
  pareto  recall vs bits / deadline / energy; non-dominated marks
  modes   mode shares overall and by tercile of the decision state
  beta    task score re-scored with beta = 0 / 0.5 / 1 (exploratory)
Usage: python a6l1_summarize.py [sections...]
"""
import gzip
import json
import os
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
O = Path(os.environ.get("A6L1_OUT", REPO / "runs" / "uav_semcom" / "a6l1"))
SEL = ["ref", "K5", "phy_mcs_blind"]
POOLS = ["test", "testdev"]
L3_SEED = 556
LABEL = {"T6:best": "T:best_task", "T6:tv": "tv (task-tuned)", "T6:reuse": "reuse (task-tuned)",
         "T6:budget": "budget (task-tuned)", "T6:fixed": "fixed (task-tuned)", "T5:best": "A5 T:best (PSNR-tuned)",
         "T5:fixed": "A5 fixed (PSNR-tuned)", "H:greedy+budget1+reuse": "A4 greedy+budget1+reuse",
         "task_best": "**PPO task**", "psnr_best": "PPO PSNR (A5)", "task_last": "PPO task, ckpt 2M"}
_AP, _MODES = {}, None


def modes():
    global _MODES
    if _MODES is None:
        import a5_common as C
        _MODES = list(C.MODES)
    return _MODES


def load(p):
    return json.load(open(p)) if p.exists() else None


def loadgz(p):
    return json.load(gzip.open(p, "rt")) if p.exists() else None


def groups(results):
    """{group: [specs]}: heuristics are their own group; PPO specs P:label_sN=... grouped by label."""
    g = {}
    for sp in results:
        if sp.startswith("P:"):
            lab = sp[2:].split("=", 1)[0]
            g.setdefault(lab.rsplit("_s", 1)[0], []).append(sp)
        else:
            g[sp] = [sp]
    return g


def gval(res, specs, k):
    """Per-env-seed value of a group (mean over its models)."""
    return np.mean([[e[k] for e in res[s]] for s in specs], axis=0)


def ci1(d):
    n = len(d)
    h = tq(n) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h


def fmt(m, lo, hi, p=2):
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


# ---------------------------------------------------------------- policy-level AP50
def built(pool):
    import a6l1_ap as AP
    key = "val" if pool == "test" else "testdev"
    if key not in _AP:
        b = AP.build(key)
        _AP[key] = (b, {n: i for i, n in enumerate(b["images"])})
    return key, _AP[key]


def ap_instances(deliv_ep, names, pool):
    key, (b, idx) = built(pool)
    M = modes()
    return [(key, idx[names[i]], (M[m] if m >= 0 else None)) for i, m in deliv_ep]


def ap_of(eps, names, pool):
    import a6l1_ap as AP
    key, (b, _) = built(pool)
    inst = [x for ep in eps for x in ap_instances(ep, names, pool)]
    return AP.ap50(inst, {key: b})


def ap_table(r, pool):
    """{spec: (pooled AP50, per-episode AP50 array)} from the deliveries file."""
    dl = loadgz(O / "final" / f"{pool}_{r}.deliv.json.gz")
    if dl is None:
        return {}
    names = dl["images"]
    out = {}
    for sp, eps in dl["results"][r].items():
        out[sp] = (ap_of(eps, names, pool), np.array([ap_of([ep], names, pool) for ep in eps]))
    return out


# ---------------------------------------------------------------- sections
def tune():
    print("### Tinh chỉnh dưới reward tác vụ (train, seed 0..29)\n")
    print("| regime | T:best_task | R_task | R_PSNR | recall | tv tốt nhất | R_task | A5 T:best (PSNR) | R_task | R_PSNR |"
          " recall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    import a5_eval as AE
    for r in SEL:
        p = O / "tune" / f"{r}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rw, bt, tv, a5 = d["rows"], d["best_task"], d["tuned"]["tv"], AE.tuned(r, "best")
        f = lambda n: f"{rw[n]['return']:+.2f} | {rw[n]['return_psnr']:+.2f} | {rw[n]['recall']:.3f}"
        print(f"| {r} | {bt} | {f(bt)} | {tv} | {rw[tv]['return']:+.2f} | {a5} | {f(a5)} |")
    print()


def train():
    print("### Đường học PPO task (val, seed 3000..3019; return tác vụ)\n")
    print("| run | best@ | val best | val @2M | val @1M | còn tăng ở 2M? |")
    print("|---|---|---|---|---|---|")
    for p in sorted((O / "train").glob("*/train.json")):
        d = json.loads(p.read_text())
        log = {e["steps"]: e["val_return"] for e in d["log"]}
        last = max(log)
        tail = [log[s] for s in sorted(log)[-3:]]
        rising = "có" if tail[-1] >= max(log.values()) - 1e-9 and tail[-1] > tail[0] else "không"
        print(f"| {p.parent.name} | {d['best_steps']} | {log[d['best_steps']]:+.2f} | {log[last]:+.2f} | "
              f"{log.get(1000000, float('nan')):+.2f} | {rising} |")
    print()


KEYS = [("return", "R_task", 2, 1), ("return_psnr", "R_PSNR", 2, 1), ("recall", "recall", 3, 1),
        ("recall_small", "recall nhỏ", 3, 1), ("fp_per_deliv", "FP/khung giao", 1, 1), ("precision", "precision", 3, 1),
        ("bits_sent", "Mbit gửi", 1, 1e-6), ("ok_rate", "đúng hạn", 3, 1), ("energy_kj", "năng lượng kJ", 1, 1),
        ("mission_s", "thời gian s", 0, 1)]


def main_table():
    for r in SEL:
        for pool in POOLS:
            d = load(O / "final" / f"{pool}_{r}.json")
            if d is None:
                continue
            res = d["results"][r]
            aps = ap_table(r, pool)
            print(f"### {r}, pool {pool} (seed {d['seeds'][0]}..{d['seeds'][-1]})\n")
            print("| policy | " + " | ".join(k[1] for k in KEYS) + " | AP50 (gộp) |")
            print("|---" * (len(KEYS) + 2) + "|")
            for g, specs in groups(res).items():
                cells = [f"{gval(res, specs, k).mean() * sc:.{p}f}" for k, _, p, sc in KEYS]
                ap = np.mean([aps[s][0] for s in specs]) if aps else float("nan")
                nm = LABEL.get(g, g) + (f" ({d['resolved'][r][g]})" if not g.startswith("task") and
                                        not g.startswith("psnr") and g in d["resolved"][r] else "")
                print(f"| {nm} | " + " | ".join(cells) + f" | {ap:.4f} |")
            print()


CMP = [("task_best", "psnr_best", "PPO task − PPO PSNR"), ("T6:best", "T5:best", "T:best_task − A5 T:best"),
       ("task_best", "T6:best", "PPO task − T:best_task")]
CKEYS = [("return", "R_task", 2, 1), ("return_psnr", "R_PSNR", 2, 1), ("recall", "recall", 4, 1),
         ("recall_small", "recall nhỏ", 4, 1), ("bits_sent", "Mbit", 1, 1e-6), ("ok_rate", "đúng hạn", 4, 1)]


def boot_ci(boot, r, ga, gb, k, resample):
    res = boot["results"][r]
    v = []
    for b in sorted(res, key=int):
        rr = res[b]
        ga_s = groups(rr)[ga] if ga in groups(rr) else None
        gb_s = groups(rr)[gb] if gb in groups(rr) else None
        if ga_s is None or gb_s is None:
            return None

        def mean_of(specs, side):
            if resample and len(specs) > 1:
                rng = np.random.default_rng([L3_SEED, int(b), side])
                specs = [specs[j] for j in rng.integers(0, len(specs), len(specs))]
            return np.mean([[e[k] for e in rr[s]] for s in specs], axis=0)
        v.append(float(np.mean(mean_of(ga_s, 0) - mean_of(gb_s, 1))))
    v = np.array(v)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def cmp():
    print("### So sánh định trước (theo cặp seed; CI mức 1 = seed môi trường; mức 2 = + video; mức 3 = + seed train)\n")
    print("| regime | pool | so sánh | chỉ số | Δ [CI mức 1] | CI mức 2 | CI mức 3 |")
    print("|---|---|---|---|---|---|---|")
    for r in SEL:
        for pool in POOLS:
            d = load(O / "final" / f"{pool}_{r}.json")
            boot = load(O / "boot" / f"{pool}_{r}.json")
            if d is None:
                continue
            res = d["results"][r]
            G = groups(res)
            aps = ap_table(r, pool)
            for a, b, lab in CMP:
                if a not in G or b not in G:
                    continue
                for k, kl, p, sc in CKEYS:
                    dd = (gval(res, G[a], k) - gval(res, G[b], k)) * sc
                    c2 = c3 = None
                    if boot is not None and r in boot["results"] and k in next(iter(next(iter(
                            boot["results"][r].values())).values()))[0]:
                        c2 = boot_ci(boot, r, a, b, k, False)
                        c3 = boot_ci(boot, r, a, b, k, True)
                    s2 = f"[{c2[0] * sc:+.{p}f}, {c2[1] * sc:+.{p}f}]" if c2 else "—"
                    s3 = f"[{c3[0] * sc:+.{p}f}, {c3[1] * sc:+.{p}f}]" if c3 else "—"
                    print(f"| {r} | {pool} | {lab} | {kl} | {fmt(*ci1(dd), p=p)} | {s2} | {s3} |")
                if aps:
                    da = np.mean([aps[s][1] for s in G[a]], axis=0) - np.mean([aps[s][1] for s in G[b]], axis=0)
                    pa = np.mean([aps[s][0] for s in G[a]]) - np.mean([aps[s][0] for s in G[b]])
                    print(f"| {r} | {pool} | {lab} | AP50 mỗi episode (gộp: {pa:+.4f}) | {fmt(*ci1(da), p=4)} | — | — |")
    print()


def pareto():
    print("### Pareto: recall vs Mbit gửi (★ = không bị policy nào khác trội hơn ở cả recall↑ và Mbit↓)\n")
    print("| regime | pool | policy | recall | Mbit | đúng hạn | năng lượng kJ | thời gian s | ★ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in SEL:
        for pool in POOLS:
            d = load(O / "final" / f"{pool}_{r}.json")
            if d is None:
                continue
            res = d["results"][r]
            rows = []
            for g, specs in groups(res).items():
                rows.append((LABEL.get(g, g), *(gval(res, specs, k).mean() for k in
                                                ("recall", "bits_sent", "ok_rate", "energy_kj", "mission_s"))))
            for row in sorted(rows, key=lambda x: -x[1]):
                dom = any(o[1] >= row[1] and o[2] <= row[2] and (o[1] > row[1] or o[2] < row[2]) for o in rows)
                print(f"| {r} | {pool} | {row[0]} | {row[1]:.4f} | {row[2] / 1e6:.1f} | {row[3]:.3f} | {row[4]:.1f} |"
                      f" {row[5]:.0f} | {'' if dom else '★'} |")
    print()


STATE = ["log10_probe_bits", "vhat_max", "backlog_s", "snr_db", "slack_s"]
MODE_POL = ["task_best", "psnr_best", "T6:best", "T5:best"]


def modes_sec(pools=("test",)):
    M = modes()
    for r in SEL:
        for pool in pools:
            dc = loadgz(O / "final" / f"{pool}_{r}.dec.json.gz")
            if dc is None:
                continue
            res = dc["results"][r]
            G = groups(res)
            pols = [g for g in MODE_POL if g in G]
            rows = {g: np.array([x for s in G[g] for ep in res[s] for x in ep], float) for g in pols}
            allr = np.concatenate([rows[g] for g in pols])
            print(f"### Chọn mode theo trạng thái — {r}, pool {pool}\n")
            top = [m for m, _ in sorted(((m, np.mean(allr[:, -1] == m)) for m in range(len(M))), key=lambda x: -x[1])
                   if np.mean(allr[:, -1] == m) > 0.02]
            share = lambda R: " · ".join(f"{M[m]} {np.mean(R[:, -1] == m):.0%}" for m in top
                                         if len(R) and np.mean(R[:, -1] == m) >= 0.05)
            print("| policy | toàn bộ |\n|---|---|")
            for g in pols:
                print(f"| {LABEL.get(g, g)} | {share(rows[g])} |")
            print()
            for j, f in enumerate(STATE):
                q = np.nanpercentile(allr[:, j], [100 / 3, 200 / 3])
                print(f"| {f} | tam phân vị thấp (≤{q[0]:.2f}) | giữa | cao (>{q[1]:.2f}) |\n|---|---|---|---|")
                for g in pols:
                    R = rows[g]
                    t = np.digitize(R[:, j], q)
                    print(f"| {LABEL.get(g, g)} | " + " | ".join(share(R[t == i]) for i in range(3)) + " |")
                print()


def beta():
    print("### Độ nhạy β (khám phá; điểm tác vụ trung bình mỗi episode: Σ(TP − β·FP)/V̄ của khung giao đúng hạn)\n")
    print("| regime | pool | so sánh | β = 0 | β = 0,5 | β = 1 |")
    print("|---|---|---|---|---|---|")
    for r in SEL:
        for pool in POOLS:
            d = load(O / "final" / f"{pool}_{r}.json")
            if d is None:
                continue
            res = d["results"][r]
            G = groups(res)
            for a, b, lab in CMP:
                if a in G and b in G:
                    cells = [fmt(*ci1(gval(res, G[a], k) - gval(res, G[b], k)))
                             for k in ("task_score_b0", "task_score", "task_score_b1")]
                    print(f"| {r} | {pool} | {lab} | " + " | ".join(cells) + " |")
    print()


def main():
    secs = sys.argv[1:] or ["tune", "train", "main", "cmp", "pareto", "modes", "beta"]
    for s in secs:
        {"tune": tune, "train": train, "main": main_table, "cmp": cmp, "pareto": pareto, "modes": modes_sec,
         "beta": beta}[s]()


if __name__ == "__main__":
    main()
