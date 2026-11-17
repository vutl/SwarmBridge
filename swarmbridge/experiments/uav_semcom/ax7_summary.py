"""Review #42 (A-X7) summary. Usage: python ax7_summary.py runs/uav_semcom/ax7

E1 information decomposition (A-X6 PHY, i.i.d. LoS per slot): selection gain (JA-NR-c - NR); LoS-only oracle gain
   (JA-NR-l - JA-NR-c) and full oracle gain (JA-NR-o - JA-NR-c), the share of the full gap that is large-scale (LoS)
   information; whether rare LoS events dominate (per-episode oracle gain vs number of realised LoS candidate-slots).
E2 Monte Carlo count (4 / 8 / 32 / 128): return vs the default 32, and scheduling latency p50 / p99.
E3 spatially correlated LoS: filter gain (JA-NR-f - JA-NR-c), share of the LoS-only oracle gap it recovers, prediction
   quality (Brier, prior vs filter), probe (SRS) effect, reuse on top of the filter.
E4 A-X2 re-check: JA vs NR and vs the stored A-X2 C-only RL on K5 / ref (paired by seed).
D  mode controller under LA: scheduler {tbest_env = LA-tuned no-reuse rule, ja} x mode {heur_env, heur, ppo, ppo+mpc}.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from ax2_summary import ci, fmt

REPO = Path(__file__).resolve().parents[3]


def load(p):
    d = json.load(gzip.open(p, "rt"))
    by = {}
    for r in d["results"]:
        by.setdefault(r["variant"], {})[r["policy"]] = r
    return by


def col(P, p, k="return"):
    return np.array([x.get(k, np.nan) for x in P[p]["rows"]], float)


def mean(P, p, k):
    return float(np.nanmean(col(P, p, k)))


def diff(P, a, b):
    assert P[a]["seeds"] == P[b]["seeds"]
    return ci(col(P, a) - col(P, b))


def table(P, pols):
    t = {}
    for p in pols:
        if p in P:
            t[p] = {k: mean(P, p, k) for k in ("return", "ok_rate", "drop_rate", "cochannel_air_frac", "energy_kj",
                                              "ja_ms_p50", "ja_ms_p99", "los_frac", "prior_mean", "surprise_frac",
                                              "ambiguous_frac", "brier_prior", "brier_used", "ja_reuse_frac", "srs")}
            t[p]["member"] = P[p]["member"]
            x = t[p]
            print(f"    {p:12s} {x['member']:26s} ret {x['return']:8.2f} ok {x['ok_rate']:.3f} drop {x['drop_rate']:.3f} "
                  f"cochan {x['cochannel_air_frac']:.3f} ms50 {x['ja_ms_p50']:.2f} ms99 {x['ja_ms_p99']:.2f} "
                  f"brier {x['brier_prior']:.4f}->{x['brier_used']:.4f}")
    return t


def e1(path):
    by, out = load(path), {}
    for v, P in sorted(by.items()):
        print(f"== E1 {v}")
        o = dict(policies=table(P, ["NR", "JA-NR-c", "JA-NR-l", "JA-NR-o", "JA-R-c", "JA-R-l", "JA-R-o"]))
        con = {}
        for r in ("NR", "R"):
            c, l, oo = f"JA-{r}-c", f"JA-{r}-l", f"JA-{r}-o"
            con[f"{r}: c - NR"] = diff(P, c, "NR")
            con[f"{r}: LoS-only oracle gain (l - c)"] = diff(P, l, c)
            con[f"{r}: full oracle gain (o - c)"] = diff(P, oo, c)
            con[f"{r}: fast-fading part (o - l)"] = diff(P, oo, l)
        for k, x in con.items():
            print(f"    {k:40s} {fmt(x)}")
        gl = con["NR: LoS-only oracle gain (l - c)"][0]
        go = con["NR: full oracle gain (o - c)"][0]
        o["los_share_of_full_gap_NR"] = float(gl / go) if abs(go) > 1e-9 else float("nan")
        # rare events: per-episode gains vs number of realised LoS candidate-slots (same for every policy: CRN)
        nlos = col(P, "JA-NR-c", "los_frac") * col(P, "JA-NR-c", "diag_n")
        gl_e = col(P, "JA-NR-l") - col(P, "JA-NR-c")
        go_e = col(P, "JA-NR-o") - col(P, "JA-NR-c")
        rho_l = stats.spearmanr(nlos, gl_e)
        rho_o = stats.spearmanr(nlos, go_e)
        q = np.quantile(nlos, [1 / 3, 2 / 3])
        terc = [np.where(nlos <= q[0])[0], np.where((nlos > q[0]) & (nlos <= q[1]))[0], np.where(nlos > q[1])[0]]
        o["rare"] = dict(los_slots_per_episode=float(nlos.mean()), episodes_without_los=float((nlos == 0).mean()),
                         spearman_los_gain=[float(rho_l[0]), float(rho_l[1])],
                         spearman_full_gain=[float(rho_o[0]), float(rho_o[1])],
                         los_gain_by_tercile=[float(gl_e[i].mean()) for i in terc],
                         full_gain_by_tercile=[float(go_e[i].mean()) for i in terc])
        print(f"    LoS share of the full gap (NR): {o['los_share_of_full_gap_NR']:.2f}; LoS candidate-slots / episode "
              f"{nlos.mean():.1f} (episodes without any {o['rare']['episodes_without_los']:.2f}); Spearman(nLoS, l-c) "
              f"{rho_l[0]:+.2f} p={rho_l[1]:.3f}; (nLoS, o-c) {rho_o[0]:+.2f}; l-c by nLoS tercile "
              f"{np.round(o['rare']['los_gain_by_tercile'], 2)}")
        o["contrasts"] = con
        out[v] = o
    return out


def e2(path):
    by, out = load(path), {}
    for v, P in sorted(by.items()):
        print(f"== E2 {v}")
        o = {}
        for r in ("NR", "R"):
            for n in ("n4", "n8", "", "n128"):
                p = f"JA-{r}-c" + (f"-{n}" if n else "")
                ref = f"JA-{r}-c"
                o[p] = dict(ret=mean(P, p, "return"), ms_p50=mean(P, p, "ja_ms_p50"), ms_p99=mean(P, p, "ja_ms_p99"),
                            vs_n32=diff(P, p, ref) if p != ref else None)
                print(f"    {p:14s} ret {o[p]['ret']:8.2f} ms p50 {o[p]['ms_p50']:.2f} p99 {o[p]['ms_p99']:.2f}"
                      + (f"  vs n32 {fmt(o[p]['vs_n32'])}" if o[p]["vs_n32"] else ""))
        out[v] = o
    return out


def e3(path, e1_path):
    by, out = load(path), {}
    if e1_path.exists():
        by0 = load(e1_path)
        if "nom_sched_persist" in by0:
            by["c0_sched_persist (=E1 nom)"] = by0["nom_sched_persist"]
    for v, P in sorted(by.items()):
        print(f"== E3 {v}")
        o = dict(policies=table(P, ["NR", "JA-NR-c", "JA-NR-f", "JA-NR-l", "JA-NR-o", "JA-R-c", "JA-R-f"]))
        con = {}
        pairs = [("c - NR", "JA-NR-c", "NR"), ("filter gain (f - c)", "JA-NR-f", "JA-NR-c"),
                 ("LoS-only oracle gain (l - c)", "JA-NR-l", "JA-NR-c"), ("full oracle gain (o - c)", "JA-NR-o", "JA-NR-c"),
                 ("reuse on filter (R-f - NR-f)", "JA-R-f", "JA-NR-f"), ("f - NR", "JA-NR-f", "NR")]
        for name, a, b in pairs:
            if a in P and b in P:
                con[name] = diff(P, a, b)
                print(f"    {name:34s} {fmt(con[name])}")
        if "filter gain (f - c)" in con:
            g_f, g_l = con["filter gain (f - c)"][0], con["LoS-only oracle gain (l - c)"][0]
            o["recovered_share_of_los_gap"] = float(g_f / g_l) if abs(g_l) > 1e-9 else float("nan")
            print(f"    filter recovers {o['recovered_share_of_los_gap']:.2f} of the LoS-only oracle gap")
        o["contrasts"] = con
        out[v] = o
    print("== E3 probe (SRS every slot) effect, same policy, paired by seed")
    srs = {}
    for d in (10, 30, 100):
        a, b = by.get(f"c{d}_srs_sched_persist"), by.get(f"c{d}_sched_persist")
        if a and b:
            for p in ("NR", "JA-NR-c", "JA-NR-f"):
                if a[p]["seeds"] == b[p]["seeds"]:
                    srs[f"c{d} {p}"] = ci(col(a, p) - col(b, p))
                    print(f"    c{d} {p:10s} srs - no srs {fmt(srs[f'c{d} {p}'])}")
    out["srs_effect"] = srs
    return out


def e4(path):
    by, out = load(path), {}
    for v, P in sorted(by.items()):
        print(f"== E4 {v}")
        o = dict(policies=table(P, ["NR", "JA-NR-c", "JA-R-c", "JA-NR-o", "JA-R-o"]))
        con = {f"{p} - NR": diff(P, p, "NR") for p in ("JA-NR-c", "JA-R-c", "JA-NR-o", "JA-R-o")}
        st_path = REPO / "runs" / "uav_semcom" / "ax2" / f"eval_{v}.json"
        if st_path.exists():
            st = json.load(open(st_path))
            seeds = P["NR"]["seeds"]

            def stored(combo):
                rs = [x for x in st["results"] if x["combo"] == combo]
                return np.mean([[dict(zip(x["seeds"], x["rows"]))[s]["return"] for s in seeds] for x in rs], 0)
            tb, cc = stored("tbest|heur"), stored("c|heur")
            con["NR - stored T:best (sanity)"] = ci(col(P, "NR") - tb)
            con["stored A-X2 C-only RL - NR"] = ci(cc - col(P, "NR"))
            for p in ("JA-NR-c", "JA-R-c"):
                con[f"{p} - stored RL"] = ci(col(P, p) - cc)
        for k, x in con.items():
            print(f"    {k:32s} {fmt(x)}")
        o["contrasts"] = con
        out[v] = o
    return out


def dmode(path, extra_paths=()):
    d = json.load(open(path))
    rows = {}
    for r in d["results"] + [x for q in extra_paths for x in json.load(open(q))["results"]]:
        rows.setdefault(r["combo"], {})[r["ckpt"]] = r
    seeds = next(iter(next(iter(rows.values())).values()))["seeds"]

    def arr(cmb, k="return"):
        ck = rows[cmb]
        return np.mean([[dict(zip(ck[i]["seeds"], ck[i]["rows"]))[s][k] for s in seeds] for i in sorted(ck)], 0)

    def svc(cmb):
        return float(np.nanmean([x["service"].get("ratio", np.nan) for i in rows[cmb] for x in rows[cmb][i]["rows"]]))
    out = dict(env_extra=d.get("env_extra"), ja=d.get("ja"), combos={})
    print(f"== D {Path(path).name}  env {d.get('env_extra')}  ja {d.get('ja')}")
    for cmb in sorted(rows):
        a = arr(cmb)
        out["combos"][cmb] = dict(ret=float(a.mean()), ok=float(arr(cmb, "ok_rate").mean()), service_ratio=svc(cmb),
                                  mode_p99=float(np.mean([x["ms_mode_p99"] for i in rows[cmb] for x in rows[cmb][i]["rows"]])),
                                  per_ckpt=[float(np.mean([x["return"] for x in rows[cmb][i]["rows"]])) for i in sorted(rows[cmb])])
        x = out["combos"][cmb]
        print(f"    {cmb:22s} ret {x['ret']:8.2f} {np.round(x['per_ckpt'], 1)} ok {x['ok']:.3f} svc {x['service_ratio']:.2f} "
              f"mode p99 {x['mode_p99']:.1f}")
    con = {}
    for s in ("tbest_env", "ja"):
        for a, b in (("ppo", "heur_env"), ("ppo+mpc", "ppo"), ("ppo+mpc", "heur_env"), ("heur", "heur_env")):
            if f"{s}|{a}" in rows and f"{s}|{b}" in rows:
                con[f"{s}: {a} - {b}"] = ci(arr(f"{s}|{a}") - arr(f"{s}|{b}"))
    for m in ("heur_env", "heur", "ppo", "ppo+mpc"):
        if f"ja|{m}" in rows and f"tbest_env|{m}" in rows:
            con[f"{m}: ja - tbest_env"] = ci(arr(f"ja|{m}") - arr(f"tbest_env|{m}"))
    if all(f"{s}|{m}" in rows for s in ("ja", "tbest_env") for m in ("ppo+mpc", "heur_env")):
        con["CONFOUNDED interaction (ja gain under ppo+mpc - under heur_env)"] = ci(
            (arr("ja|ppo+mpc") - arr("tbest_env|ppo+mpc")) - (arr("ja|heur_env") - arr("tbest_env|heur_env")))
    for s in ("ls", "jae"):
        for a, b in (("ppo", "heur_env"), ("ppo+mpc", "ppo"), ("ppo+mpc", "heur_env"), ("heur", "heur_env")):
            if f"{s}|{a}" in rows and f"{s}|{b}" in rows:
                con[f"{s}: {a} - {b}"] = ci(arr(f"{s}|{a}") - arr(f"{s}|{b}"))
    for m in ("heur_env", "heur", "ppo", "ppo+mpc"):
        for a, b, tag in (("ja", "ls", "scheduler, paced movement"), ("jae", "tbest_env", "scheduler, mission movement"),
                          ("ls", "tbest_env", "movement, least-slack channels"), ("ja", "jae", "movement, JA channels")):
            if f"{a}|{m}" in rows and f"{b}|{m}" in rows:
                con[f"{m}: {a} - {b} ({tag})"] = ci(arr(f"{a}|{m}") - arr(f"{b}|{m}"))
    for a, b, tag in (("ja", "ls", "paced"), ("jae", "tbest_env", "mission")):
        if all(f"{x}|{m}" in rows for x in (a, b) for m in ("ppo+mpc", "heur_env")):
            con[f"interaction {tag}: ({a}-{b}) under ppo+mpc - under heur_env"] = ci(
                (arr(f"{a}|ppo+mpc") - arr(f"{b}|ppo+mpc")) - (arr(f"{a}|heur_env") - arr(f"{b}|heur_env")))
    for k, x in con.items():
        print(f"    {k:52s} {fmt(x)}")
    out["contrasts"] = con
    return out


def main(d):
    d = Path(d)
    res = {}
    if (d / "e1_decomp.json.gz").exists():
        res["e1"] = e1(d / "e1_decomp.json.gz")
    if (d / "e2_mc.json.gz").exists():
        res["e2"] = e2(d / "e2_mc.json.gz")
    if (d / "e3_corr.json.gz").exists():
        res["e3"] = e3(d / "e3_corr.json.gz", d / "e1_decomp.json.gz")
    if (d / "e4_ax2.json.gz").exists():
        res["e4"] = e4(d / "e4_ax2.json.gz")
    for f in ("d_mode_nom.json", "d_mode_c30.json"):
        if (d / f).exists():
            ex = [d / "d_mode_nom_matched.json"] if f == "d_mode_nom.json" and (d / "d_mode_nom_matched.json").exists() else []
            res[f[:-5]] = dmode(d / f, ex)
    json.dump(res, open(d / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
