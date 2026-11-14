"""G-L2 step 2 (reports/archive/r1-visdrone/uav-semcom-g-l2.md 0.4): oracle headroom over the strongest content-agnostic baseline, on
train U val only (test / test-dev untouched).

  H(B)   = O(B) - BL(B): optimistic oracle (true detector outcome per image and mode), full 14-mode menu;
  H_x(B) = O_x(B) - BL_x(B): cross-checked oracle, codec menu (10 codecs + thumb64_up): choose with one measurement
           (original / flipped-image), score with the other, both directions averaged; BL_x scored with the mean of the
           two measurements;
  breakdowns per video, per tercile of object count and of small-object share; video cluster bootstrap (exploratory).
Writes runs/uav_semcom/g_l2/headroom.json and prints the markdown tables.
Usage: python g2_headroom.py [--boot 500]
"""
import argparse
import json

import numpy as np

import g2_common as G

PT = 100.0     # recall points


def oracle(d, I, menu, tp_choose, tp_score, fp=None, tps=None):
    return G.Curve(G.lagrange_choices(tp_choose, d.bits, I, menu), I, d.bits, tp_score, d.ngt, fp, tps, d.nsmall)


def video_oracle(d, I, menu, tp, fp=None, tps=None):
    """Exploratory (added after pre-registration, prior-art audit): best FIXED mode per video, chosen with the true
    outcomes of that video (argmax_m sum TP - lam sum bits per video) -> how much of O - BL lies BETWEEN videos."""
    I = np.asarray(I)
    seqs = d.seq[I]
    us = np.unique(seqs)
    menu = np.asarray(menu)
    T = np.stack([tp[I[seqs == s]][:, menu].sum(0) for s in us])
    Bt = np.stack([d.bits[I[seqs == s]][:, menu].sum(0) for s in us])
    pos = np.searchsorted(us, seqs)
    ch = []
    for lam in G.LAMS:
        j = np.argmin(Bt, 1) if np.isinf(lam) else np.argmax(T - lam * Bt, 1)
        ch.append(menu[j][pos])
    return G.Curve(ch, I, d.bits, tp, d.ngt, fp, tps, d.nsmall)


def headroom(d, I, train, menu, B, tp, fp=None, tps=None):
    o = oracle(d, I, menu, tp, tp, fp, tps)
    base = G.baselines(d, I, train, menu, tp, fp, tps)
    ro, mo = o.at(B)
    rb, name, mb, allb = G.bl_at(base, B)
    rv = video_oracle(d, I, menu, tp).at(B)[0]
    return dict(O=ro * PT, BL=rb * PT, H=(ro - rb) * PT, BL_name=name, BL_all={k: v * PT for k, v in allb.items()},
                O_video=rv * PT, H_video=(rv - rb) * PT, mix_O=mo, mix_BL=mb)


def cross(d, I, train, menu, B, tpA, tpB):
    oab = oracle(d, I, menu, tpA, tpB).at(B)[0]
    oba = oracle(d, I, menu, tpB, tpA).at(B)[0]
    M = (tpA + tpB) / 2
    rb, name, _, _ = G.bl_at(G.baselines(d, I, train, menu, M), B)
    oa = oracle(d, I, menu, tpA, tpA).at(B)[0]
    ba = G.bl_at(G.baselines(d, I, train, menu, tpA), B)[0]
    return dict(Ox=(oab + oba) / 2 * PT, BLx=rb * PT, Hx=((oab + oba) / 2 - rb) * PT, BLx_name=name,
                O_codec=oa * PT, BL_codec=ba * PT, H_codec=(oa - ba) * PT)


def groups(d, I, res):
    """Decompose the pool-level gain O - BL (expected TP per image) by video and by terciles."""
    mo, mb = res["mix_O"], res["mix_BL"]
    N = d.ngt[I].sum()
    dtp, dbits = mo["tp"] - mb["tp"], mo["bits"] - mb["bits"]
    out = {}
    seqs = d.seq[I]
    per = {s: dict(n=int((seqs == s).sum()), dTP_pts=float(dtp[seqs == s].sum() / N * PT),
                   dMbit_per_frame=float(dbits[seqs == s].mean() / 1e6)) for s in np.unique(seqs)}
    out["video"] = per
    tot = sum(v["dTP_pts"] for v in per.values())
    out["video_max_share"] = max(v["dTP_pts"] for v in per.values()) / tot if tot > 0 else np.nan
    for key, x in (("n_obj", d.ngt[I]), ("small_share", d.nsmall[I] / np.maximum(d.ngt[I], 1))):
        q = np.quantile(x, [1 / 3, 2 / 3])
        t = np.digitize(x, q)
        out[key] = [dict(lo=float(x[t == k].min()), hi=float(x[t == k].max()), n=int((t == k).sum()),
                         dTP_pts=float(dtp[t == k].sum() / N * PT),
                         gain_rel_pts=float(dtp[t == k].sum() / d.ngt[I][t == k].sum() * PT),
                         dMbit_per_frame=float(dbits[t == k].mean() / 1e6)) for k in range(3)]
    return out


def boot_rows(d, I, train, menuA, menuC, Bs, tpA, tpB, nb, seed=558):
    seqs = d.seq[I]
    us = np.unique(seqs)
    by = {s: I[seqs == s] for s in us}
    rng = np.random.default_rng(seed)
    out = {m: dict(H=[], Hx=[]) for m in Bs}
    for b in range(nb):
        J = np.concatenate([by[s] for s in rng.choice(us, len(us), replace=True)])
        for m, B in Bs.items():
            out[m]["H"].append(headroom(d, J, train, menuA, B, d.tp)["H"])
            if tpB is not None:
                out[m]["Hx"].append(cross(d, J, train, menuC, B, tpA, tpB)["Hx"])
    return {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
                for k, v in r.items()} for m, r in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--no-flip", action="store_true")
    a = ap.parse_args()
    d = G.Data(False)
    train, val = d.pools["train"], d.pools["val"]
    tv = np.sort(np.concatenate([train, val]))
    Bs = G.budgets(d)
    menuA = list(range(len(d.modes)))
    tpB = None
    if not a.no_flip:
        tpB, fpB, tpsB = d.flip()
        menuC = [j for j in menuA if not np.isnan(tpB[tv, j]).any()]
    else:
        menuC = []
    res = dict(budgets_bits=Bs, modes=d.modes, menu_codec=[d.modes[j] for j in menuC], pools={})
    print("## Bước 2: dư địa oracle (train ∪ val; điểm recall)\n")
    print("Mức bit:", ", ".join(f"{m} = {B / 1e6:.3f} Mbit/khung" for m, B in Bs.items()), "\n")
    print("| tập | mức bit | O | BL (loại) | F | MX | RV | CB | **H** | O_x | BL_x | **H_x** | H (menu codec) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pname, I in (("train∪val", tv), ("train", train), ("val", val)):
        res["pools"][pname] = {}
        for m, B in Bs.items():
            r = headroom(d, I, train, menuA, B, d.tp, d.fp, d.tps)
            x = cross(d, I, train, menuC, B, d.tp, tpB) if tpB is not None else {}
            row = {k: v for k, v in r.items() if not k.startswith("mix")}
            row.update(x)
            # FP and small-object recall of O / BL at B
            for lab, mix in (("O", r["mix_O"]), ("BL", r["mix_BL"])):
                if mix is not None and mix.get("fp") is not None:
                    row[f"{lab}_fp_per_frame"] = float(mix["fp"].mean())
                    row[f"{lab}_recall_small"] = float(mix["tps"].sum() / d.nsmall[I].sum() * PT)
            if pname == "train∪val":
                row["groups"] = groups(d, I, r)
            res["pools"][pname][m] = row
            ba = r["BL_all"]
            print(f"| {pname} | {m} | {r['O']:.2f} | {r['BL']:.2f} ({r['BL_name']}) | {ba['F']:.2f} | {ba['MX']:.2f} | "
                  f"{ba['RV']:.2f} | {ba['CB']:.2f} | **{r['H']:+.2f}** | {x.get('Ox', np.nan):.2f} | "
                  f"{x.get('BLx', np.nan):.2f} | **{x.get('Hx', np.nan):+.2f}** | {x.get('H_codec', np.nan):+.2f} |")
    print()
    print("Khám phá (thêm sau tiền đăng ký): oracle chọn một mode cố định cho mỗi video (biết đáp án của video đó)\n")
    print("| tập | mức bit | O (theo khung) | O_video (theo video) | BL | phần dư địa nằm giữa các video |")
    print("|---|---|---|---|---|---|")
    for pname in res["pools"]:
        for m in Bs:
            r = res["pools"][pname][m]
            share = r["H_video"] / r["H"] if r["H"] > 0 else np.nan
            print(f"| {pname} | {m} | {r['O']:.2f} | {r['O_video']:.2f} | {r['BL']:.2f} | {share * 100:.0f}% |")
    print()
    tvr = res["pools"]["train∪val"]
    print("FP/khung và recall vật nhỏ ở cùng mức bit (train ∪ val):\n")
    print("| mức bit | FP/khung O | FP/khung BL | recall nhỏ O | recall nhỏ BL |")
    print("|---|---|---|---|---|")
    for m in Bs:
        r = tvr[m]
        print(f"| {m} | {r.get('O_fp_per_frame', np.nan):.1f} | {r.get('BL_fp_per_frame', np.nan):.1f} | "
              f"{r.get('O_recall_small', np.nan):.2f} | {r.get('BL_recall_small', np.nan):.2f} |")
    print()
    print("Phân rã (train ∪ val): đóng góp vào O − BL theo nhóm (điểm recall của cả pool; 'tương đối' = trong nhóm)\n")
    for m in Bs:
        g = tvr[m]["groups"]
        v = sorted(g["video"].items(), key=lambda kv: -kv[1]["dTP_pts"])
        print(f"- {m}: video đóng góp lớn nhất {v[0][0]} ({v[0][1]['dTP_pts']:+.2f} điểm, "
              f"{g['video_max_share'] * 100:.0f}% tổng); số video có đóng góp > 0: "
              f"{sum(x['dTP_pts'] > 0 for _, x in v)}/{len(v)}")
        for key in ("n_obj", "small_share"):
            print(f"  - {key}: " + "; ".join(f"[{t['lo']:.2g}, {t['hi']:.2g}] {t['dTP_pts']:+.2f} điểm "
                                             f"(tương đối {t['gain_rel_pts']:+.2f}, ΔMbit/khung {t['dMbit_per_frame']:+.3f})"
                                             for t in g[key]))
    if tpB is not None:
        tvi = tv
        A, Bm = d.tp[tvi][:, menuC], tpB[tvi][:, menuC]
        rel = dict(corr_level=float(np.corrcoef(A.ravel(), Bm.ravel())[0, 1]))
        q3, a50, a80 = (d.modes.index(x) for x in G.BUDGET_MODES)
        for lab, (j1, j2) in (("avif_50−mbt_q3", (q3, a50)), ("avif_80−avif_50", (a50, a80)),
                              ("mbt_q3−avif_20", (d.modes.index("avif_20"), q3))):
            da, db = d.tp[tvi, j2] - d.tp[tvi, j1], tpB[tvi, j2] - tpB[tvi, j1]
            rel[lab] = dict(corr=float(np.corrcoef(da, db)[0, 1]), sd_A=float(da.std()), sd_B=float(db.std()),
                            mean_A=float(da.mean()), mean_B=float(db.mean()))
        rel["mean_tp_A_minus_B"] = float(np.nanmean(A - Bm))
        res["reliability"] = rel
        print("\nĐộ lặp lại giữa hai phép đo (train ∪ val, menu codec):\n")
        print(f"- tương quan TP (mức): {rel['corr_level']:.3f}; TP trung bình đo gốc − đo lật: {rel['mean_tp_A_minus_B']:+.2f}")
        for lab in ("mbt_q3−avif_20", "avif_50−mbt_q3", "avif_80−avif_50"):
            r = rel[lab]
            print(f"- hiệu TP {lab}: tương quan giữa hai phép đo {r['corr']:.3f}; trung bình {r['mean_A']:+.2f} / "
                  f"{r['mean_B']:+.2f}; độ lệch chuẩn {r['sd_A']:.2f} / {r['sd_B']:.2f}")
    if a.boot:
        ci = boot_rows(d, tv, train, menuA, menuC, Bs, d.tp, tpB, a.boot)
        res["boot_ci_video"] = ci
        print(f"\nCI 95% theo video (cluster bootstrap B = {a.boot}, khám phá):\n")
        for m in Bs:
            print(f"- {m}: H {ci[m]['H']}" + (f"; H_x {ci[m]['Hx']}" if ci[m]["Hx"] else ""))
    ok = [m for m in Bs if tvr[m]["H"] >= 1.0 and tvr[m].get("Hx", -np.inf) >= 0.5]
    res["decision"] = dict(pass_budgets=ok, proceed=bool(ok))
    print(f"\n**Quy tắc 0.4:** mức bit đạt H ≥ 1,0 và H_x ≥ 0,5: {ok or 'không có'} → "
          f"{'ĐI TIẾP sang bước 3' if ok else 'DỪNG (FAIL: không có dư địa thực chất)'}")
    G.G2.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("mix")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (G.G2 / "headroom.json").write_text(json.dumps(clean(res), indent=1))


if __name__ == "__main__":
    main()
