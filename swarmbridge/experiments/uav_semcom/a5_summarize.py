"""A5 summary tables (markdown) from runs/uav_semcom/a5/: sweep (val), final test heuristics, PPO vs heuristics
(paired CIs), action replacement, A4 zero-shot, experience / learning curves, task-fidelity join.
Every comparison is paired (same seeds); CI = mean difference +- t_{0.975, n-1} * sd / sqrt(n). Pooling over training
seeds: the per-test-seed mean over the 3 PPO models is paired with the heuristic on that test seed.
Usage: python a5_summarize.py [sections...]   (sweep final ppo replace zeroshot fidelity)
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

import a5_common as C

try:
    from scipy.stats import t as _t

    def tq(n):
        return float(_t.ppf(0.975, n - 1))
except ImportError:   # pragma: no cover
    def tq(n):
        return 1.984 if n >= 100 else 2.01

POL_LABEL = {"T:fixed": "fixed (tuned)", "H:greedy": "greedy", "T:base": "base (tuned)", "T:budget": "budget (tuned)",
             "T:reuse": "reuse (tuned)", "T:best": "**best tuned**", "H:pacedB1.5+budget1.5": "A4 pacedB1.5+budget1.5",
             "H:greedy+budget1+reuse": "A4 greedy+budget1+reuse", "T:reuse@oracle": "reuse (tuned) + oracle info"}
F = C.A5


def load(p):
    return json.load(open(p)) if p.exists() else None


def arr(eps, k="return"):
    return np.array([e[k] for e in eps], float)


def ci(d):
    n = len(d)
    h = tq(n) * d.std(ddof=1) / np.sqrt(n)
    return f"{d.mean():+.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]"


def sweep():
    d = load(F / "sweep_val.json")
    if not d:
        return "(sweep chưa có)\n"
    sel = load(F / "selected.json")
    cols = [p for p in POL_LABEL if p in next(iter(d["results"].values()))]
    out = ["| regime | " + " | ".join(POL_LABEL[p] for p in cols) + " | reuse: ok / PSNR / nhiệm vụ |",
           "|---" * (len(cols) + 2) + "|"]
    for r, v in d["results"].items():
        m = {p: arr(v[p]).mean() for p in POL_LABEL if p in v}
        best = max(m[p] for p in ("T:fixed", "T:base", "T:budget", "T:reuse"))
        cells = [(f"**{m[p]:+.1f}**" if abs(m[p] - best) < 1e-9 else f"{m[p]:+.1f}") for p in POL_LABEL if p in m]
        ru = v["T:reuse"]
        out.append(f"| {r} | " + " | ".join(cells) +
                   f" | {arr(ru, 'ok_rate').mean():.3f} / {np.nanmean(arr(ru, 'mean_psnr')):.1f} / "
                   f"{arr(ru, 'mission_s').mean():.0f} s |")
    tuned = ["", "Heuristic được chọn trên train (seed 0..29) cho từng regime:", "",
             "| regime | fixed | base | budget | reuse |", "|---|---|---|---|---|"]
    for r in d["results"]:
        t = load(F / "tune" / f"{r}.json")["tuned"]
        tuned.append(f"| {r} | {t['fixed']} | {t['base']} | {t['budget']} | {t['reuse']} |")
    s = ""
    if sel:
        s = "\n\nÁp quy tắc 0.2 (`selected.json`):\n\n| regime | reuse − max(base, budget) | reuse − fixed | điều khiển (best − greedy) | đủ điều kiện |\n|---|---|---|---|---|\n"
        s += "\n".join(f"| {x['regime']} | {x['reuse_gain']:+.2f} | {x['adapt']:+.2f} | {x['control']:+.2f} | "
                       f"{'có' if x['eligible'] else 'không'} |" for x in sel["candidates"])
        s += f"\n\n**Chọn: R1 = {sel['R1']}, R2 = {sel['R2']}, R3 = {sel['R3']}.**\n"
    return "\n".join(out + tuned) + s + "\n"


def final_heur(name="test_heur.json"):
    d = load(F / "final" / name)
    if not d:
        return f"({name} chưa có)\n"
    cols = [p for p in POL_LABEL if p in next(iter(d["results"].values()))]
    out = [f"Pool `{d['pool']}`, n = {len(d['seeds'])} seed. Return trung bình; Δ = policy − best tuned (T:best), CI 95% theo cặp.", "",
           "| regime | " + " | ".join(POL_LABEL[p] for p in cols) + " | reuse − best | base − best | budget − best | T:best = |",
           "|---" * (len(cols) + 5) + "|"]
    for r, v in d["results"].items():
        ref = arr(v["T:best"])
        cells = [f"{arr(v[p]).mean():+.1f}" for p in cols]
        out.append(f"| {r} | " + " | ".join(cells) + f" | {ci(arr(v['T:reuse']) - ref)} | {ci(arr(v['T:base']) - ref)} | "
                   f"{ci(arr(v['T:budget']) - ref)} | {d['resolved'][r]['T:best']} |")
    return "\n".join(out) + "\n"


def ppo(regime):
    th = load(F / "final" / "test_heur.json")
    tp = load(F / "final" / f"test_ppo_{regime}.json")
    if not (th and tp):
        return f"({regime}: PPO chưa đánh giá)\n"
    H = th["results"][regime]
    P = tp["results"][regime]
    ref = arr(H["T:best"])
    groups = defaultdict(list)
    for spec, eps in P.items():
        label = spec[2:].split("=")[0]
        kind = label.rsplit("_s", 1)[0]
        groups[kind].append(eps)
    out = [f"**{regime}** (pool test, n = {len(ref)}; heuristic mạnh T:best = {th['resolved'][regime]['T:best']}, "
           f"return {ref.mean():+.2f})", "",
           "| policy | return | Δ vs T:best (gộp 3 seed) | Δ từng seed | ok | PSNR | nhiệm vụ (s) | năng lượng (kJ) |",
           "|---|---|---|---|---|---|---|---|"]
    for kind in ("joint_best", "joint_last", "mode_best", "mode_last"):
        if kind not in groups:
            continue
        runs = groups[kind]
        R = np.mean([arr(e) for e in runs], 0)
        per = ", ".join(f"{(arr(e) - ref).mean():+.2f}" for e in runs)
        ok = np.mean([arr(e, "ok_rate").mean() for e in runs])
        ps = np.mean([np.nanmean(arr(e, "mean_psnr")) for e in runs])
        mi = np.mean([arr(e, "mission_s").mean() for e in runs])
        en = np.mean([arr(e, "energy_kj").mean() for e in runs])
        out.append(f"| PPO {kind} | {R.mean():+.2f} | {ci(R - ref)} | {per} | {ok:.3f} | {ps:.1f} | {mi:.0f} | {en:.1f} |")
    for p in ("T:base", "T:budget", "T:reuse", "T:best", "H:greedy+budget1+reuse"):
        e = H[p]
        out.append(f"| {POL_LABEL[p]} | {arr(e).mean():+.2f} | {ci(arr(e) - ref) if p != 'T:best' else '—'} | | "
                   f"{arr(e, 'ok_rate').mean():.3f} | {np.nanmean(arr(e, 'mean_psnr')):.1f} | "
                   f"{arr(e, 'mission_s').mean():.0f} | {arr(e, 'energy_kj').mean():.1f} |")
    if "joint_best" in groups and "mode_best" in groups:
        J = np.mean([arr(e) for e in groups["joint_best"]], 0)
        M = np.mean([arr(e) for e in groups["mode_best"]], 0)
        out += ["", f"Joint − mode-only (best-val, gộp): {ci(J - M)}"]
    return "\n".join(out) + "\n"


def experience(regime):
    out = ["| run | best-val step | val return (best / last) | episodes | env-giây | RL slot / mode decisions |",
           "|---|---|---|---|---|---|"]
    for part in ("joint", "mode"):
        for s in (0, 1, 2):
            d = load(F / "train" / f"{regime}_{part}_s{s}" / "train.json")
            if not d:
                continue
            log = d["log"]
            b = [x for x in log if x["steps"] == d["best_steps"]][0]
            ex = log[-1]["experience"]
            out.append(f"| {part} s{s} | {d['best_steps']:,} | {b['val_return']:+.2f} / {log[-1]['val_return']:+.2f} | "
                       f"{ex.get('episodes', 0):,} | {ex.get('env_s', 0):,.0f} | {ex.get('rl_slot', 0):,} / {ex.get('rl_mode', 0):,} |")
    curves = []
    for part in ("joint", "mode"):
        for s in (0, 1, 2):
            d = load(F / "train" / f"{regime}_{part}_s{s}" / "train.json")
            if d:
                curves.append(f"{part} s{s}: " + " ".join(f"{x['val_return']:+.1f}" for x in d["log"]))
    return "\n".join(out) + "\n\nĐường học (return val mỗi 2·10⁵ bước):\n\n" + "\n".join(f"- {c}" for c in curves) + "\n"


def replace(regime):
    d = load(F / "final" / f"test_replace_{regime}.json")
    tp = load(F / "final" / f"test_ppo_{regime}.json")
    if not (d and tp):
        return f"({regime}: action replacement chưa có)\n"
    base = {s: arr(tp["results"][regime][k]) for k in tp["results"][regime] for s in (0, 1, 2)
            if k.startswith(f"P:joint_best_s{s}=")}
    rows = defaultdict(list)
    for spec, eps in d["results"][regime].items():
        label = spec[2:].split("=")[0]
        s, combo = int(label.split("_s")[1][0]), label.rsplit("_", 1)[1]
        rows[combo].append(arr(eps) - base[s])
    out = ["Thay phần hành động của PPO joint (best-val) bằng phần tương ứng của heuristic mạnh T:best "
           "(thứ tự: move, channel, power, mode; H = heuristic). Δ so với PPPP, gộp 3 seed:", "",
           "| combo | Δ return |", "|---|---|"]
    for combo in sorted(rows, key=lambda c: (c.count("H"), c)):
        out.append(f"| {combo} | {ci(np.mean(rows[combo], 0))} |")
    return "\n".join(out) + "\n"


def zeroshot():
    d = load(F / "final" / "test_a4_zeroshot.json")
    th = load(F / "final" / "test_heur.json")
    if not (d and th):
        return "(zero-shot chưa có)\n"
    out = ["| regime | A4 joint (3 model, gộp) | A4 mode-only | best tuned | joint − best | mode − best |",
           "|---|---|---|---|---|---|"]
    for r, v in d["results"].items():
        ref = arr(th["results"][r]["T:best"])
        J = np.mean([arr(e) for k, e in v.items() if "a4_joint" in k], 0)
        M = np.mean([arr(e) for k, e in v.items() if "a4_mode" in k], 0)
        out.append(f"| {r} | {J.mean():+.2f} | {M.mean():+.2f} | {ref.mean():+.2f} | {ci(J - ref)} | {ci(M - ref)} |")
    return "\n".join(out) + "\n"


def fidelity_tables():
    out = []
    for pool in ("val", "testdev"):
        d = load(F / "fidelity" / f"coco_{pool}.json")
        if not d:
            continue
        per = [json.loads(l) for l in open(F / "fidelity" / f"per_image_{pool}.jsonl")]
        agg = defaultdict(lambda: defaultdict(list))
        for r in per:
            for k in ("bytes", "psnr", "agree_f1"):
                agg[r["mode"]][k].append(r[k])
            agg[r["mode"]]["tp"].append(r["tp"]); agg[r["mode"]]["n_gt"].append(r["n_gt"])
            agg[r["mode"]]["tp_s"].append(r["tp_small"]); agg[r["mode"]]["n_s"].append(r["n_small"])
        for sub, res in d["results"].items():
            out += [f"Pool `{pool}`, tập con `{sub}` ({next(iter(res.values()))['n_images']} ảnh):", "",
                    "| mode | kB | PSNR | AP | AP50 | AP_small | AR500 | recall@0.25 | recall nhỏ | F1 so với ảnh gốc |",
                    "|---|---|---|---|---|---|---|---|---|---|"]
            for m, s in sorted(res.items(), key=lambda kv: np.mean(agg[kv[0]]["bytes"]) if agg[kv[0]]["bytes"] else 1e12):
                a = agg[m]
                kb = np.mean(a["bytes"]) / 1e3 if m != "original" else float("nan")
                ps = np.mean([x for x in a["psnr"] if np.isfinite(x)]) if m != "original" else float("inf")
                out.append(f"| {m} | {kb:.1f} | {ps:.1f} | {s['AP']:.3f} | {s['AP50']:.3f} | {s['AP_small']:.3f} | "
                           f"{s['AR500']:.3f} | {sum(a['tp']) / max(sum(a['n_gt']), 1):.3f} | "
                           f"{sum(a['tp_s']) / max(sum(a['n_s']), 1):.3f} | {np.mean(a['agree_f1']):.3f} |")
            out.append("")
    return "\n".join(out) if out else "(fidelity chưa có)\n"


def fidelity_join(evals):
    """Task fidelity of what each policy DELIVERS: object recall of delivered-on-time frames
    (sum of TPs of the delivered (image, mode) pairs / sum of labelled objects of all captured frames; a dropped,
    late or quality-violating frame counts 0), and relative to sending every frame as the original."""
    per = {}
    for pool in ("val", "testdev"):
        p = F / "fidelity" / f"per_image_{pool}.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                per[(r["image"], r["mode"])] = r
    if not per:
        return "(fidelity chưa có)\n"
    man = C.manifest()["images"]
    out = ["| file | regime | policy | object recall giao đúng hạn | so với gửi ảnh gốc | recall vật nhỏ |",
           "|---|---|---|---|---|---|"]
    for name in evals:
        p = F / "final" / name.replace(".json", ".deliv.json.gz")
        if not p.exists():
            continue
        d = json.load(gzip.open(p, "rt"))["results"]
        for r, v in d.items():
            for spec, eps in v.items():
                tp = gt = tps = ns = orig = 0
                for ep in eps:
                    for mi, mode, ok in ep:
                        im = man[mi]["name"]
                        row = per.get((im, C.MODES[mode])) if mode >= 0 else None
                        o = per.get((im, "original"))
                        if o is None:
                            continue
                        gt += o["n_gt"]; ns += o["n_small"]; orig += o["tp"]
                        if ok and row is not None:
                            tp += row["tp"]; tps += row["tp_small"]
                if gt:
                    out.append(f"| {name} | {r} | {spec[:48]} | {tp / gt:.3f} | {tp / max(orig, 1):.3f} | {tps / max(ns, 1):.3f} |")
    return "\n".join(out) + "\n"


def main():
    secs = sys.argv[1:] or ["sweep", "final", "ppo", "replace", "zeroshot", "fidelity"]
    sel = load(F / "selected.json")
    regs = ["ref", "K5", sel["R3"]] if sel else []
    for s in secs:
        print(f"\n<!-- {s} -->\n")
        if s == "sweep":
            print(sweep())
        elif s == "final":
            print(final_heur("test_heur.json"))
            print(final_heur("testdev_heur.json"))
        elif s == "ppo":
            for r in regs:
                print(ppo(r))
                print(experience(r))
        elif s == "replace":
            for r in regs:
                print(replace(r))
        elif s == "zeroshot":
            print(zeroshot())
        elif s == "fidelity":
            print(fidelity_tables())
            print(fidelity_join(["test_heur.json"] + [f"test_ppo_{r}.json" for r in regs]))


if __name__ == "__main__":
    main()
