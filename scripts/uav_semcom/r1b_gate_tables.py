"""Descriptive tables for the R1b-0a H1 report from runs/uav_semcom/r1b/gate_v1{,_rows}.json (no re-scoring)."""
import json, sys
from collections import defaultdict
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "runs/uav_semcom/r1b"
rows = json.load(open(f"{D}/gate_v1_rows.json")); G = json.load(open(f"{D}/gate_v1.json"))
lock = json.load(open(f"{D}/lock_v1.json"))
LV = {round(v): k for k, v in lock["R_ul_levels"].items()}
gate, aux = set(G["gate"]), set(G["aux"])
CONDS = ["A1", "B", "Bp", "C1", "D", "A0", "C0"]
out = {}


def micro(rs, sfx=""):
    tp = sum(r["tp" + sfx] for r in rs); fp = sum(r["fp" + sfx] for r in rs); fn = sum(r["fn" + sfx] for r in rs)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return 100 * 2 * P * R / max(P + R, 1e-12), 100 * P, 100 * R


def sel(var="main", clips=gate, cond=None, lv=None):
    return [r for r in rows if r["variant"] == var and r["clip"] in clips and (cond is None or r["cond"] == cond)
            and (lv is None or LV[round(r["level_R"])] == lv)]


levels = [k for k in lock["R_ul_levels"]]
for lv in levels:
    t = {}
    for c in CONDS:
        rs = sel(cond=c, lv=lv)
        f1, p, r = micro(rs); f1a, _, _ = micro(rs, "_ag")
        n_fr = sum(x["n_frame"] for x in rs); n_tk = sum(x["n_tok"] for x in rs); n_sk = sum(x["n_skip"] for x in rs)
        t[c] = dict(F1=f1, P=p, R=r, F1_ag=f1a, IDF1=float(np.mean([x["IDF1"] for x in rs])),
                    ul_kbps=float(np.mean([x["ul_kbps"] for x in rs])), dl_kbps=float(np.mean([x["dl_kbps"] for x in rs])),
                    frac_frame=n_fr / max(n_fr + n_tk + n_sk, 1), frac_tok=n_tk / max(n_fr + n_tk + n_sk, 1),
                    frac_skip=n_sk / max(n_fr + n_tk + n_sk, 1),
                    frame_ok=1 - sum(x["frame_lost"] + x["frame_late"] for x in rs) / max(n_fr, 1),
                    tok_ok=1 - sum(x["tok_lost"] + x["tok_late"] for x in rs) / max(n_tk, 1) if n_tk else None,
                    dropped_compute=sum(x["dropped_compute"] for x in rs) / max(sum(x["ticks"] for x in rs), 1),
                    belief_f1=float(np.mean([x["belief_f1"] for x in rs if x["belief_f1"] is not None])),
                    ghost_s_per_min=float(np.mean([x["ghost_s_per_min"] for x in rs])),
                    ghost_uav_absent=float(np.mean([x["ghost_uav_absent_frac"] for x in rs])),
                    ghost_known_absent=float(np.mean([x["ghost_known_absent_frac"] for x in rs])),
                    desync_ticks=sum(x["desync_ticks"] for x in rs), snapshots=sum(x["snapshots"] for x in rs))
        a = sel(clips=aux, cond=c, lv=lv)
        t[c]["F1_aux"] = micro(a)[0]
    out[lv] = dict(main=t)
    sens = {}
    for v in ["shared_A1_params", "pipelined", "DL16", "p_ul0", "p_ul5"]:
        vs = {c: micro(sel(var=v, cond=c, lv=lv))[0] for c in ["A1", "B", "Bp", "C1", "D"] if sel(var=v, cond=c, lv=lv)}
        if vs:
            sens[v] = dict(F1=vs, d_C1_A1=vs["C1"] - vs["A1"], d_D_A1=vs["D"] - vs["A1"],
                           frame_ok_C1=1 - sum(x["frame_lost"] + x["frame_late"] for x in sel(var=v, cond="C1", lv=lv))
                           / max(sum(x["n_frame"] for x in sel(var=v, cond="C1", lv=lv)), 1))
    out[lv]["sens"] = sens
    out[lv]["per_source"] = {s: {c: float(np.mean([x["F1"] for x in sel(clips={s}, cond=c, lv=lv)])) for c in CONDS}
                             for s in sorted(gate)}
json.dump(out, open(f"{D}/gate_v1_tables.json", "w"), indent=1)
for lv in levels:
    print("==", lv)
    for c, v in out[lv]["main"].items():
        print(f" {c:3s} F1 {v['F1']:.2f} P {v['P']:.1f} R {v['R']:.1f} F1ag {v['F1_ag']:.2f} IDF1 {v['IDF1']:.1f} aux {v['F1_aux']:.2f} "
              f"UL {v['ul_kbps']:.0f} DL {v['dl_kbps']:.1f} fr/tok/skip {v['frac_frame']:.2f}/{v['frac_tok']:.2f}/{v['frac_skip']:.2f} "
              f"frame_ok {v['frame_ok']:.2f} tok_ok {v['tok_ok'] if v['tok_ok'] is None else round(v['tok_ok'],3)} drop {v['dropped_compute']:.2f} "
              f"belief {v['belief_f1']:.0f} ghost {v['ghost_s_per_min']:.1f} uav_abs {v['ghost_uav_absent']:.2f} known_abs {v['ghost_known_absent']:.2f} desync {v['desync_ticks']} snap {v['snapshots']}")
    for k, s in out[lv]["sens"].items():
        print(f"  sens {k}: C1-A1 {s['d_C1_A1']:.2f} D-A1 {s['d_D_A1']:.2f} frame_ok(C1) {s['frame_ok_C1']:.2f} F1 " + " ".join(f"{c}={x:.2f}" for c, x in s["F1"].items()))
