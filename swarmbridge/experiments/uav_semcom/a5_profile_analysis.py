"""A5.1 analysis of the A5 mode table (no simulation):
  * reproducibility vs A4: for the 100 A4 frames, payload of every non-generative mode compared byte for byte with
    the A4 profile (same codec settings, native resolution);
  * per-mode statistics per split (train / val / test / test-dev): payload (kB), PSNR, LPIPS -- mean, coefficient
    of variation, p10 / p90; mean encode / decode latency (val pool; learned codecs on CPU);
  * content dependence: per image, the cheapest mode reaching PSNR >= 30 / 35 / 40 dB (non-generative modes) -- how
    often the 'best mode' differs between images; correlation of payload with object count and with resolution;
  * the controller's predictor (fitted on the train split, A4 rule: log-linear in the avif_5 probe payload):
    R^2 of log payload and MAE of PSNR per split (does it generalise to unseen sequences / test-dev?);
  * A4 table (100 frames) vs A5 val-pool table: mean payload / PSNR per mode.
Usage: python a5_profile_analysis.py OUT_JSON
"""
import json
import sys
from collections import Counter

import numpy as np

import a5_common as C
import env as E

A4 = C.REPO / "runs" / "uav_semcom" / "a4_native"
GENERATIVE = ("cap_t2i", "thumb64_up", "cap_thumb64_i2i_t0.6", "cap_thumb128_i2i_t0.3")


def rows(files):
    out = {}
    for f in files:
        if f.exists():
            for l in open(f):
                r = json.loads(l)
                out[(r["image"], r["mode"])] = r
    return out


def main():
    man = C.manifest()["images"]
    info = {e["name"]: e for e in man}
    a5 = rows(C.PROFILES_VAL + C.PROFILES_TD)
    a4 = rows([A4 / "a11" / "profile.jsonl", A4 / "a12" / "profile.jsonl"])
    res = {}
    # reproducibility
    same, tot, diffs = 0, 0, []
    for (im, m), r in a4.items():
        if m in GENERATIVE or m not in C.MODES or (im, m) not in a5:
            continue
        tot += 1
        same += r["bytes"] == a5[(im, m)]["bytes"]
        diffs.append(abs(r["bytes"] - a5[(im, m)]["bytes"]) / r["bytes"])
        res.setdefault("repro_psnr_absdiff", []).append(abs(r["psnr"] - a5[(im, m)]["psnr"]))
    res["repro"] = dict(pairs=tot, identical_bytes=same, max_rel_bytes_diff=float(max(diffs) if diffs else 0),
                        max_psnr_absdiff=float(max(res.pop("repro_psnr_absdiff", [0]))))
    # per-split statistics
    tab, pools = C.load_table(testdev=(C.PROFILES_TD[0].exists() and C.PROFILES_TD[1].exists()))
    stats = {}
    for sp, idx in pools.items():
        if sp in ("none", "test_a4_excluded"):
            continue
        s = {}
        for j, m in enumerate(tab.modes):
            kb, q, lp = tab.bits[idx, j] / 8e3, tab.psnr[idx, j], tab.lpips[idx, j]
            s[m] = dict(kB=float(kb.mean()), kB_cv=float(kb.std() / kb.mean()), kB_p10=float(np.percentile(kb, 10)),
                        kB_p90=float(np.percentile(kb, 90)), psnr=float(q.mean()), psnr_sd=float(q.std()),
                        psnr_p10=float(np.percentile(q, 10)), psnr_p90=float(np.percentile(q, 90)),
                        lpips=float(lp.mean()))
        stats[sp] = dict(n=len(idx), modes=s)
    res["per_split"] = stats
    res["latency_s"] = {m: dict(enc=float(tab.enc_s[j]), dec=float(tab.dec_s[j])) for j, m in enumerate(tab.modes)}
    # content dependence: cheapest mode reaching a PSNR target
    ng = [j for j, m in enumerate(tab.modes) if m not in GENERATIVE]
    cd = {}
    for thr in (30, 35, 40):
        pick = []
        for i in range(len(tab.images)):
            ok = [j for j in ng if tab.psnr[i, j] >= thr]
            pick.append(tab.modes[min(ok, key=lambda j: tab.bits[i, j])] if ok else "none")
        cnt = Counter(pick)
        cd[f"psnr>={thr}"] = dict(counts=dict(cnt.most_common()), top_share=cnt.most_common(1)[0][1] / len(pick))
    res["best_mode_by_content"] = cd
    nobj = np.array([info[im]["n_obj"] for im in tab.images], float)
    pix = np.array([info[im]["w"] * info[im]["h"] for im in tab.images], float)
    corr = {}
    for m in ("avif_50", "avif_80", "mbt2018-mean_q5"):
        j = tab.modes.index(m)
        bpp = tab.bits[:, j] / pix
        corr[m] = dict(bits_vs_pixels=float(np.corrcoef(tab.bits[:, j], pix)[0, 1]),
                       bpp_vs_objects=float(np.corrcoef(bpp, nobj)[0, 1]),
                       psnr_vs_objects=float(np.corrcoef(tab.psnr[:, j], nobj)[0, 1]))
    res["correlations"] = corr
    # predictor generalisation
    pred = {}
    for sp, idx in pools.items():
        if sp in ("none", "test_a4_excluded"):
            continue
        lb, lq = [], []
        for i in idx:
            b, q = tab.predict(i)
            lb.append(np.log(b[ng]) - np.log(tab.bits[i, ng]))
            lq.append(q[ng] - tab.psnr[i, ng])
        lb, lq = np.array(lb), np.array(lq)
        tot_var = np.var(np.log(tab.bits[idx][:, ng]), axis=0)
        pred[sp] = dict(log_bits_rmse=float(np.sqrt((lb ** 2).mean())), psnr_mae=float(np.abs(lq).mean()),
                        log_bits_r2_mean=float(np.mean(1 - (lb ** 2).mean(0) / tot_var)))
    res["predictor"] = pred
    # A4 vs A5 means
    a4tab = E.ModeTable([str(A4 / "a11" / "profile.jsonl"), str(A4 / "a12" / "profile.jsonl")], C.MODES,
                        str(A4 / "a11" / "cpu_learned.jsonl"))
    vi = np.concatenate([pools[s] for s in ("train", "val", "test", "test_a4_excluded") if s in pools])
    res["a4_vs_a5"] = {m: dict(a4_kB=float(a4tab.bits[:, j].mean() / 8e3), a5_kB=float(tab.bits[vi, j].mean() / 8e3),
                               a4_psnr=float(a4tab.psnr[:, j].mean()), a5_psnr=float(tab.psnr[vi, j].mean()),
                               a4_enc=float(a4tab.enc_s[j]), a5_enc=float(tab.enc_s[j]))
                       for j, m in enumerate(C.MODES)}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(dict(repro=res["repro"], predictor=pred, best_mode=cd), indent=1))


if __name__ == "__main__":
    main()
