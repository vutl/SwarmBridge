"""Review #44 (A-X9) summary and the PRE-REGISTERED decision gate (reports/phases/2026-11-17_paperA-scheduling/swarmbridge-ax9-protocol.md).
Usage: python ax9_summary.py runs/uav_semcom/ax9
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import a5_common as C
from ax2_summary import FID, ci, fmt, recall

KEYS = ("return", "ok_rate", "late_rate", "drop_rate", "goodput_kbps", "energy_kj", "jain", "min_uav_ok",
        "ms_slot_p99", "ms_mode_p99", "ms_int_max_p99", "int_over50", "mode_over50")


def main(d):
    d = Path(d)
    sc = json.load(open(d / "sched.json"))
    f = sc["frozen"]
    print(f"== stage 1 (train): frozen {f['sched']} @ {f['move']}  score {f['score']:.2f}; A-X8 setting "
          f"{sc['reference_A_X8']['score']:.2f}; least-slack same movement {sc['least_slack_same_move']:.2f}")
    for mv in ("mission", "pacedB1.5", "pacedB2", "pacedB3"):
        cand = {k: v for k, v in sc["score"].items() if k.endswith("@" + mv)}
        b = max((k for k in cand if not k.startswith("ls")), key=cand.get)
        print(f"   {mv:10s} best {b:40s} {cand[b]:8.2f}   least-slack {cand['ls@' + mv]:8.2f}")
    he = json.load(open(d / "heur.json"))
    print(f"== stage 2 (train): best heuristic {he['heuristic_best']} ({he['heur_returns'][he['heuristic_best']]:.2f});"
          f" MPC incumbent I* {he['mpc_incumbent']}")
    for m, v in sorted(he["mpc_incumbent_returns"].items(), key=lambda x: -x[1]):
        print(f"   {m:32s} alone {he['heur_returns'][m]:8.2f}  +MPC {v:8.2f}")
    print("== stage 3 (PPO on LA): validation curves")
    for s in range(3):
        p = d / "train" / f"ppo_la_s{s}.json"
        if p.exists():
            tj = json.load(open(p))
            print(f"   seed {s}: best@{tj['best_steps']} val " +
                  " ".join(f"{r['val_return']:.1f}" for r in tj["log"]))

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
            rr.append(x)
        rows.setdefault(r["combo"].split("|")[1], {})[r["ckpt"]] = dict(seeds=r["seeds"], rows=rr)
    seeds = test["seeds"]

    def per_ck(c, k="return"):
        return [np.array([dict(zip(v["seeds"], v["rows"]))[s][k] for s in seeds], float) for _, v in sorted(rows[c].items())]

    def arr(c, k="return"):
        return np.mean(per_ck(c, k), 0)

    H, I = test["heuristic_best"], test["mpc_incumbent"]
    Inc = I.replace("hc=", "h=")
    names = {"H*": H, "I*": I, "I*+MPC": I + "+mpc", "I*+MPC no-cal": Inc + "+mpcnc", "PPO-LA": "ppola",
             "PPO-LA+MPC": "ppola+mpc", "PPO-Shannon": "ppo", "PPO-Shannon+MPC": "ppo+mpc"}
    out = dict(frozen=f, heur=dict(H=H, I=I), arms={}, contrasts={})
    print(f"== stage 4 test (held-out seeds {seeds[0]}..{seeds[-1]}), scheduler {test['sched']}")
    for lab, c in names.items():
        if c not in rows:
            continue
        t = {k: float(np.nanmean(arr(c, k))) for k in KEYS + ("recall",)}
        t["per_ckpt"] = [float(x.mean()) for x in per_ck(c)]
        out["arms"][lab] = t
        print(f"   {lab:16s} {c:34s} ret {t['return']:8.2f} {np.round(t['per_ckpt'], 1)} ok {t['ok_rate']:.3f} "
              f"drop {t['drop_rate']:.3f} goodput {t['goodput_kbps']:6.1f} E {t['energy_kj']:.1f} jain {t['jain']:.3f} "
              f"minUAV {t['min_uav_ok']:.3f} recall {t['recall']:.3f} | ms slot p99 {t['ms_slot_p99']:.1f} mode p99 "
              f"{t['ms_mode_p99']:.1f} int p99 {t['ms_int_max_p99']:.1f} >50 {t['int_over50']:.4f}")

    def con(lab, a, b):
        ca, cb = names[a], names[b]
        x = dict(ret=ci(arr(ca) - arr(cb)), recall=ci(arr(ca, "recall") - arr(cb, "recall")),
                 ok=ci(arr(ca, "ok_rate") - arr(cb, "ok_rate")))
        if len(rows[ca]) > 1:
            pc = [float((pa - arr(cb)).mean()) for pa in per_ck(ca)]
            m, s = np.mean(pc), np.std(pc, ddof=1)
            h = stats.t.ppf(0.975, len(pc) - 1) * s / np.sqrt(len(pc))
            x.update(per_ckpt=pc, train_seed_ci=[float(m), float(m - h), float(m + h)])
        out["contrasts"][lab] = x
        print(f"   {lab:50s} {fmt(x['ret'])} recall {fmt(x['recall'])}"
              + (f"  per seed {np.round(x['per_ckpt'], 2)}  train-seed CI {fmt(x['train_seed_ci'])}" if "per_ckpt" in x
                 else ""))
        return x

    print("== PRIMARY (gate): (PPO-LA + MPC) - (I* + MPC)")
    g = con("PRIMARY PPO-LA+MPC - I*+MPC", "PPO-LA+MPC", "I*+MPC")
    lat = out["arms"]["PPO-LA+MPC"]["ms_int_max_p99"]
    conds = {"mean >= +2.0": g["ret"][0] >= 2.0, "env-seed CI lower bound > 0": g["ret"][1] > 0,
             "all 3 training seeds > 0": all(v > 0 for v in g["per_ckpt"]),
             "recall not significantly worse (CI upper >= 0)": g["recall"][2] >= 0,
             "integrated latency p99 <= 50 ms": lat <= 50.0}
    for k, v in conds.items():
        print(f"      {'PASS' if v else 'FAIL'}  {k}")
    out["gate"] = dict(conditions=conds, decision="DRL GO" if all(conds.values()) else "STOP the DRL line")
    print(f"   DECISION: {out['gate']['decision']}")
    print("== secondary")
    con("calibration: I*+MPC - I*+MPC no-cal", "I*+MPC", "I*+MPC no-cal")
    con("MPC on incumbent: I*+MPC - I*", "I*+MPC", "I*")
    con("PPO-LA - H*", "PPO-LA", "H*")
    con("PPO-LA - PPO-Shannon", "PPO-LA", "PPO-Shannon")
    con("distribution shift: PPO-LA+MPC - PPO-Shannon+MPC", "PPO-LA+MPC", "PPO-Shannon+MPC")
    con("I*+MPC - H*", "I*+MPC", "H*")
    for lab in ("I*+MPC", "PPO-LA", "PPO-LA+MPC", "PPO-Shannon+MPC", "I*+MPC no-cal"):
        if names[lab] in rows:
            con(f"recall/return vs H*: {lab}", lab, "H*")
    json.dump(out, open(d / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
