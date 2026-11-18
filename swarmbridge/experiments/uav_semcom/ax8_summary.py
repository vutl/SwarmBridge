"""Review #43 (A-X8) summary. Usage: python ax8_summary.py runs/uav_semcom/ax8

Per movement (matched: same movement, channel scheduler JA, PHY, seeds), paired over the 100 held-out seeds:
  C1 calibration value in heuristics         hc=HC*   - h=H*
  C2 DECISION GATE                           ppo+mpc  - hc=HC*      (per PPO checkpoint too)
  C3 MPC+calibration on top of PPO           ppo+mpc  - ppo
  C4 selective MPC on the heuristic          hc=HC*+mpc - hc=HC*
  C5 value of the learned incumbent          ppo+mpc  - hc=HC*+mpc
  C6 zero-shot PPO vs calibrated heuristic   ppo      - hc=HC*
  I1 scheduler x mode vs calibrated heur.    (JA - LS | ppo+mpc) - (JA - LS | hc=HC*)
  I2 scheduler x calibration                 (JA - LS | hc=HC*)  - (JA - LS | h=H*)
Best separately tuned stacks: every stack at the movement it chose on train seeds.
Metrics: return, ok (deadline success), late, drop, goodput, energy, Jain, worst UAV, detector recall on the delivered
images, service-rate ratio, latency (slot p50/p99, mode p50/p99, integrated slot + mode p99, share over 50 ms).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import a5_common as C
from ax2_summary import FID, ci, fmt, recall

MOVES = {"paced": dict(ja="ja", ls="ls"), "mission": dict(ja="jae", ls="tbest_env")}
KEYS = ("return", "ok_rate", "late_rate", "drop_rate", "goodput_kbps", "energy_kj", "jain", "min_uav_ok",
        "ms_slot_p50", "ms_slot_p99", "ms_mode_p50", "ms_mode_p99", "ms_int_max_p99", "ms_int_sum_p99", "int_over50",
        "mode_over50")


def main(d):
    d = Path(d)
    tune = json.load(open(d / "tune.json"))
    sel, smv = tune["selected"], tune["stack_movement"]
    print("== tuning (TRAIN pool, seeds 0..29)")
    for mv, s in sel.items():
        top = sorted(s["H_all"].items(), key=lambda x: -x[1])[:5]
        topc = sorted(s["HC_all"].items(), key=lambda x: -x[1])[:3]
        print(f"  {mv:8s} H* {s['H']} ({s['H_ret']:.2f}); top5 {[(n, round(v, 1)) for n, v in top]}")
        print(f"  {'':8s} HC* {s['HC']} ({s['HC_ret']:.2f}); calibrated top3 {[(n, round(v, 1)) for n, v in topc]}")
        print(f"  {'':8s} PPO {s['PPO_ret']:.2f}  PPO+MPC {s['PPOMPC_ret']:.2f}  HC*+MPC {s['HCMPC_ret']:.2f}")
    print(f"  stack movement chosen on train: {smv}")

    test = json.load(open(d / "test.json"))
    dl = json.load(gzip.open(d / "test.deliv.json.gz", "rt"))
    per = {}
    for line in open(FID):
        r = json.loads(line)
        per[(r["image"], r["mode"])] = r
    man = C.manifest()["images"]
    rows = {}
    for r in test["results"]:
        rr = []
        for x, ep in zip(r["rows"], dl[f"{r['combo']}|{r['ckpt']}"]):
            x = dict(x)
            x["recall"] = recall([ep], per, man)
            x["svc"] = x["service"].get("ratio", np.nan)
            rr.append(x)
        rows.setdefault(r["combo"], {})[r["ckpt"]] = dict(seeds=r["seeds"], rows=rr)
    seeds = test["seeds"]

    def per_ck(c, k="return"):
        return [np.array([dict(zip(v["seeds"], v["rows"]))[s][k] for s in seeds], float) for _, v in sorted(rows[c].items())]

    def arr(c, k="return"):
        return np.mean(per_ck(c, k), 0)

    out = dict(selected=sel, stack_movement=smv, combos={}, contrasts={})
    print("== test (held-out seeds 8900..8999)")
    for c in sorted(rows):
        t = {k: float(np.nanmean(arr(c, k))) for k in KEYS + ("recall", "svc")}
        t["per_ckpt"] = [float(x.mean()) for x in per_ck(c)]
        out["combos"][c] = t
        print(f"  {c:40s} ret {t['return']:8.2f} {np.round(t['per_ckpt'], 1)} ok {t['ok_rate']:.3f} late {t['late_rate']:.3f}"
              f" goodput {t['goodput_kbps']:6.1f} E {t['energy_kj']:.1f} jain {t['jain']:.3f} minUAV {t['min_uav_ok']:.3f}"
              f" recall {t['recall']:.3f} svc {t['svc']:.2f} | ms slot {t['ms_slot_p50']:.1f}/{t['ms_slot_p99']:.1f}"
              f" mode {t['ms_mode_p50']:.1f}/{t['ms_mode_p99']:.1f} int p99 {t['ms_int_max_p99']:.1f}"
              f" (sum {t['ms_int_sum_p99']:.1f}) >50ms {t['int_over50']:.4f}")

    def con(name, a, b, with_ck=False):
        x = dict(ret=ci(arr(a) - arr(b)), recall=ci(arr(a, "recall") - arr(b, "recall")),
                 ok=ci(arr(a, "ok_rate") - arr(b, "ok_rate")))
        if with_ck and len(rows[a]) > 1:
            x["per_ckpt"] = [float((pa - arr(b)).mean()) for pa in per_ck(a)]
        out["contrasts"][name] = x
        print(f"  {name:64s} {fmt(x['ret'])}  recall {x['recall'][0]:+.4f}  ok {x['ok'][0]:+.4f}"
              + (f"  per ckpt {np.round(x['per_ckpt'], 2)}" if "per_ckpt" in x else ""))

    for mv, sc in MOVES.items():
        J, L, H, HC = sc["ja"], sc["ls"], sel[mv]["H"], sel[mv]["HC"]
        print(f"== contrasts, movement {mv} (scheduler JA = {J}, least-slack = {L}; H* {H}, HC* {HC})")
        con(f"{mv} C1 calibration in heuristics (HC* - H*)", f"{J}|{HC}", f"{J}|{H}")
        con(f"{mv} C2 GATE ppo+mpc - HC*", f"{J}|ppo+mpc", f"{J}|{HC}", True)
        con(f"{mv} C3 ppo+mpc - ppo", f"{J}|ppo+mpc", f"{J}|ppo", True)
        con(f"{mv} C4 HC*+mpc - HC*", f"{J}|{HC}+mpc", f"{J}|{HC}")
        con(f"{mv} C5 learned incumbent: ppo+mpc - HC*+mpc", f"{J}|ppo+mpc", f"{J}|{HC}+mpc", True)
        con(f"{mv} C6 ppo - HC*", f"{J}|ppo", f"{J}|{HC}", True)
        if f"{J}|{H}+mpc" in rows:
            con(f"{mv} C7 H*+mpc - HC*+mpc (incumbent calibration inside MPC)", f"{J}|{H}+mpc", f"{J}|{HC}+mpc")
        for m in (H, HC, "ppo", "ppo+mpc"):
            con(f"{mv} scheduler JA - LS under {m}", f"{J}|{m}", f"{L}|{m}")
        i1 = ci((arr(f"{J}|ppo+mpc") - arr(f"{L}|ppo+mpc")) - (arr(f"{J}|{HC}") - arr(f"{L}|{HC}")))
        i2 = ci((arr(f"{J}|{HC}") - arr(f"{L}|{HC}")) - (arr(f"{J}|{H}") - arr(f"{L}|{H}")))
        out["contrasts"][f"{mv} I1 interaction vs calibrated heuristic"] = dict(ret=i1)
        out["contrasts"][f"{mv} I2 scheduler x calibration"] = dict(ret=i2)
        print(f"  {mv + ' I1 (JA-LS | ppo+mpc) - (JA-LS | HC*)':64s} {fmt(i1)}")
        print(f"  {mv + ' I2 (JA-LS | HC*) - (JA-LS | H*)':64s} {fmt(i2)}")

    print("== best separately tuned stacks (each at its train-selected movement)")
    name = {"H": lambda mv: sel[mv]["H"], "HC": lambda mv: sel[mv]["HC"], "PPO": lambda mv: "ppo",
            "PPOMPC": lambda mv: "ppo+mpc", "HCMPC": lambda mv: sel[mv]["HC"] + "+mpc"}
    stack = {k: f"{MOVES[smv[k]]['ja']}|{name[k](smv[k])}" for k in name}
    out["stacks"] = stack
    for k, c in stack.items():
        print(f"  {k:7s} {c:40s} {out['combos'][c]['return']:8.2f}")
    heur_best = max(("H", "HC", "HCMPC"), key=lambda k: sel[smv[k]][k + "_ret"])
    con(f"STACK ppo+mpc ({stack['PPOMPC']}) - best non-learned stack by train ({heur_best}: {stack[heur_best]})",
        stack["PPOMPC"], stack[heur_best], True)
    con(f"STACK ppo+mpc - calibrated heuristic stack ({stack['HC']})", stack["PPOMPC"], stack["HC"], True)
    json.dump(out, open(d / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
