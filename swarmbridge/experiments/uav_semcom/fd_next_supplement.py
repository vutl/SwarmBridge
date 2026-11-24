"""Post-chain completion checks and REGISTERED supporting outputs not produced by fd_next_summary.py (review of
c4a5e299). Uses fd_next_summary's classes unchanged; runs only on stored results (no simulation).
  coverage : every phase file parses; expected pools, seed lists, arms, configurations and checkpoint ids; episodes per
             file; 52 K deliveries per episode. Expected new evaluation episodes: ablation 2,400; Phase 4 4,800;
             RL zero-shot 3,600; retrained 14,400 (= 25,200).
  ckpts    : phase3/ckpts.json paths == selection.json paths per (family, training seed); sha256 recomputed; the nine
             training runs (selected step, best validation return, nominal / actual steps, updates, wall time).
  extras   : R5 recall contrasts (X - H*, X+matched - H*+matched; seed level + video-cluster; Phase 2 H* rows, same
             seeds); band / bias quantiles and absolute recall levels per arm; abort / truncation diagnostics (only
             the per-episode abort COUNT is logged; abort categories were not recorded -> reported as unavailable).
Usage: python fd_next_supplement.py coverage|ckpts|extras"""
import argparse
import hashlib
import json

import numpy as np

import fd_common as FD
from fd_next import AB_ARMS, AB_TEST, CFG, GRID_ARMS, GRID_TEST, RL_ARMS, RL_TEST
from fd_next_summary import Pool, ci

EXPECT = {
    "ablation": [("test", p, s, [("N5x2", a, [-1]) for a in AB_ARMS]) for p, s in AB_TEST.items()],
    "phase4": [("test", p, s, [(c, a, [-1]) for c in CFG if c != "N5x2" for a in GRID_ARMS]) for p, s in GRID_TEST.items()],
    "phase3": [("zeroshot", "TD", RL_TEST["TD"], [("N5x2", x + a, [0, 1, 2]) for x in ("ppola", "qrla", "dsla") for a in RL_ARMS])]
              + [("retrained", p, s, [("N5x2", x + a, [0, 1, 2]) for x in ("ppofd", "qrfd", "dsfd") for a in RL_ARMS])
                 for p, s in RL_TEST.items()],
}


def ctx():
    t, pools, rec, man_idx = FD.load()
    man = json.loads(FD.MANIFEST.read_text())["images"]
    return (rec, {int(mi): r for r, mi in enumerate(man_idx)}, {k: e["seq"] for k, e in enumerate(man)}, pools)


def coverage():
    C = ctx()
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    tune4 = json.loads((FD.OUT / "phase4" / "tune.json").read_text())["H"]
    total, ok = 0, True
    for phase, items in EXPECT.items():
        for name, pool, seeds, arms in items:
            p = FD.OUT / phase / f"{name}_{pool}.json"
            if not p.exists():
                print(f"MISSING {phase}/{name}_{pool}.json")
                ok = False
                continue
            P = Pool(p, FD.OUT / phase / f"{name}_{pool}.deliv.json.gz", pool, *C)
            n_ep, problems = 0, []
            if P.seeds != list(seeds):
                problems.append("seed list differs")
            for cfg, arm, cks in arms:
                inc = arm if arm.split("+")[0] in ("ppola", "qrla", "dsla", "ppofd", "qrfd", "dsfd") else \
                    f"h={(H if cfg == 'N5x2' else tune4[cfg])}{arm}"
                key = (cfg, inc)
                if key not in P.arms:
                    problems.append(f"missing arm {key}")
                    continue
                if sorted(P.arms[key]) != sorted(cks):
                    problems.append(f"checkpoints {sorted(P.arms[key])} != {cks} for {key}")
                for ck, (rows, eps, sd) in P.arms[key].items():
                    n_ep += len(rows)
                    if len(rows) != len(seeds) or len(eps) != len(seeds):
                        problems.append(f"{key} ck {ck}: {len(rows)} rows / {len(eps)} delivery lists")
                    if any(len(ep) != 52 * CFG[cfg]["K"] for ep in eps):
                        problems.append(f"{key} ck {ck}: delivery count != 52 K")
            extra = set(P.arms) - {(c, a if a.split("+")[0] in ("ppola", "qrla", "dsla", "ppofd", "qrfd", "dsfd")
                                    else f"h={(H if c == 'N5x2' else tune4[c])}{a}") for c, a, _ in arms}
            if extra:
                problems.append(f"unexpected arms {sorted(extra)}")
            total += n_ep
            ok &= not problems
            print(f"{phase}/{name}_{pool}: {n_ep} episodes, {'OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
    print(f"TOTAL new evaluation episodes {total} (expected 25,200); {'ALL OK' if ok and total == 25200 else 'NOT COMPLETE'}")


def ckpts():
    ck = json.loads((FD.OUT / "phase3" / "ckpts.json").read_text())
    sel = json.loads((FD.OUT / "phase3" / "selection.json").read_text())
    print("versions", sel.get("versions"))
    for key, paths in ck.items():
        for s, path in enumerate(paths):
            v = sel["selection"][key][str(s)]
            same = v["ckpt"] == path
            sha_ok = hashlib.sha256(open(path, "rb").read()).hexdigest() == v["ckpt_sha256"]
            print(f"{key} seed {s}: ckpts.json == selection.json {same}; sha256 ok {sha_ok}; selected step "
                  f"{v['best_steps']} best val {v['best_val']:.2f}; nominal {v['last_steps']} actual "
                  f"{v.get('actual_timesteps')} updates {v.get('n_updates')} wall {v.get('train_wall_s', 0) / 3600:.2f} h")
            assert same and sha_ok


def extras():
    C = ctx()
    H = json.loads((FD.OUT / "phase2" / "tune.json").read_text())["H"]
    out = {}
    for pool in RL_TEST:
        p = FD.OUT / "phase3" / f"retrained_{pool}.json"
        p2 = FD.OUT / "phase2" / f"test_{pool}.json"
        if not (p.exists() and p2.exists()):
            continue
        P = Pool(p, FD.OUT / "phase3" / f"retrained_{pool}.deliv.json.gz", pool, *C)
        P2 = Pool(p2, FD.OUT / "phase2" / f"test_{pool}.deliv.json.gz", pool, *C)
        assert P.seeds == P2.seeds
        for x in ("ppofd", "qrfd", "dsfd"):
            for tag, a, b in (("alone", x, f"h={H}"), ("matched", x + "+mpcng", f"h={H}+mpcng")):
                A, Bb = P.tpgt("N5x2", a), P2.tpgt("N5x2", b)
                per = lambda T: np.array([100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for d in T])
                out[f"R5_recall_{pool}_{x}_{tag}"] = dict(seed=ci(per(A) - per(Bb)), levels=[float(per(A).mean()),
                                                                                           float(per(Bb).mean())])
                print(f"R5 recall {pool} {x} {tag}: {out[f'R5_recall_{pool}_{x}_{tag}']['seed']}")
    for phase, items in EXPECT.items():
        for name, pool, seeds, arms in items:
            p = FD.OUT / phase / f"{name}_{pool}.json"
            if not p.exists():
                continue
            P = Pool(p, FD.OUT / phase / f"{name}_{pool}.deliv.json.gz", pool, *C)
            for key, cks in P.arms.items():
                rows = [x for v in cks.values() for x in v[0]]
                q = lambda k: [float(np.mean([x[k][i] for x in rows if x.get(k)])) for i in range(3)] \
                    if any(x.get(k) for x in rows) else None
                rec = None
                if pool in ("TD", "F", "V", "TD_all"):
                    T = P.tpgt(*key)
                    rec = float(np.mean([100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for d in T]))
                out[f"{phase}|{name}|{pool}|{key[0]}|{key[1]}"] = dict(
                    band_lo_q=q("band_lo_q"), band_hi_q=q("band_hi_q"), bias_q=q("bias_q"), recall=rec,
                    aborts_per_episode=float(np.mean([x.get("aborts", 0) for x in rows])) if rows[0].get("aborts") is not None else None,
                    mode_over50=float(np.mean([x.get("mode_over50", 0) for x in rows])),
                    abort_categories="unavailable (only the per-episode abort count was logged)")
    json.dump(out, open(FD.OUT / "supplement_extras.json", "w"), indent=1)
    print(f"extras written: {len(out)} entries -> supplement_extras.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("coverage", "ckpts", "extras"))
    dict(coverage=coverage, ckpts=ckpts, extras=extras)[ap.parse_args().cmd]()
