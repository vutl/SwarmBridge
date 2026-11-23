"""Paper A full-data Phase 1 summary (no GPU): coverage, failures, timing, probe-predictor error per pool, artifact
hashes. Reads runs/uav_semcom/paperA_fd/profile/profile.jsonl + manifest_fd_v1.json; writes
runs/uav_semcom/paperA_fd/phase1_summary.json and .log.

Probe predictor = env.ModeTable.fit_info ('predicted' mode info): per-mode least squares of log(bits) and PSNR on
log(bits of the avif_5 probe of the same image). Fitted on rl_train rows of fd_v1 ('new'), evaluated on every pool;
compared with the A5 predictor fitted on the 304 A5 train frames ('old', DET-val), evaluated on the same fd_v1 rows.
Errors: median / mean absolute error of log(bits) (reported as a multiplicative factor exp(MAE)) and of PSNR (dB),
over all (image, non-probe mode) pairs; also sequence-weighted (each sequence counts once).
NOTE: the old predictor was fitted on 304 DET-val frames, so its error on V is NOT out-of-sample; compare on F / TD /
TD_extra / rl_val (images unseen by both fits)."""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

import a5_common as C
from fd_manifest import OUT

POOLS = ["rl_train", "rl_val", "F", "TD", "TD_extra", "V"]


def sha_text(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fit(bits, psnr, j):
    x = np.log(bits[:, j])
    X = np.stack([np.ones_like(x), x], 1)
    return np.linalg.lstsq(X, np.log(bits), rcond=None)[0], np.linalg.lstsq(X, psnr, rcond=None)[0]


def main():
    man = json.loads((OUT / "manifest_fd_v1.json").read_text())["images"]
    split = {e["name"]: e["split"] for e in man}
    seq = {e["name"]: e["seq"] for e in man}
    rows = [json.loads(l) for l in open(OUT / "profile" / "profile.jsonl")]
    cfgs = Counter(r["config"] for r in rows)
    lines = [f"rows {len(rows)}, configs {dict(cfgs)}"]
    assert len(cfgs) == 1, "more than one config hash in the main profile"
    # record integrity (review of 33aae4a9): one accepted record per (image, mode, config); rows of an image
    # interrupted mid-write and redone on resume would appear twice -> keep the LAST complete write of that image
    # (deterministic; the raw file is kept unchanged); sha256 and split must match the manifest
    ent = {e["name"]: e for e in man}
    keys = Counter((r["image"], r["mode"], r["config"]) for r in rows)
    dup = {k: v for k, v in keys.items() if v > 1}
    bad_sha = sum(r["sha256"] != ent[r["image"]]["sha256"] for r in rows)
    bad_split = sum(r["split"] != ent[r["image"]]["split"] for r in rows)
    bad_seed = sum(r.get("gen_seed") not in (None, ent[r["image"]]["gen_seed"]) for r in rows)
    lines.append(f"integrity: duplicate (image, mode, config) keys {len(dup)}, sha256 mismatches {bad_sha}, "
                 f"split mismatches {bad_split}, gen_seed mismatches {bad_seed}")
    assert bad_sha == bad_split == bad_seed == 0
    by = defaultdict(dict)
    for r in rows:                                     # later rows overwrite earlier ones (resume = last write)
        by[r["image"]][r["mode"]] = r
    # coverage
    cov = {}
    for p in POOLS:
        names = [e["name"] for e in man if e["split"] == p]
        need = list(C.MODES) + (["original"] if p in ("F", "TD", "TD_extra", "V") else [])
        miss = [(n, m) for n in names for m in need if m not in by.get(n, {})]
        nonfinite = [(n, m) for n in names for m in C.MODES if m in by.get(n, {}) and
                     not np.isfinite(by[n][m]["psnr"])]
        cov[p] = dict(images=len(names), missing_rows=len(miss), nonfinite_psnr=len(nonfinite),
                      missing_examples=miss[:5])
        lines.append(f"coverage {p:9s} {cov[p]}")
    # timing (wall per image, enc / dec means per mode on rl_train)
    wall = defaultdict(list)
    for n, d in by.items():
        r = next(iter(d.values()))
        wall[split[n]].append(r["wall_s"])
    tim = {p: dict(mean_s=float(np.mean(v)), total_h=float(np.sum(v) / 3600)) for p, v in wall.items()}
    lines.append("wall per image: " + ", ".join(f"{p} {t['mean_s']:.2f}s ({t['total_h']:.2f} h)" for p, t in tim.items()))
    tr = [n for n in by if split[n] == "rl_train" and all(m in by[n] for m in C.MODES)]
    # latency (review of 33aae4a9): the main profile has GPU timings for the learned codecs; env.ModeTable replaces
    # their ENCODE time by the UAV-CPU timing (cpu_learned.jsonl, A5 protocol: os.cpu_count() threads). Both reported.
    lat = {m: dict(device=by[tr[0]][m]["device"], enc_s_profile=float(np.mean([by[n][m]["enc_s"] for n in tr])),
                   dec_s_profile=float(np.mean([by[n][m]["dec_s"] for n in tr]))) for m in C.MODES}
    for fname, key in (("cpu_learned.jsonl", "enc_s_cpu_uav"), ("cpu_learned_default_threads.jsonl",
                                                                "enc_s_cpu_default_threads_unused")):
        f = OUT / "profile" / fname
        if f.exists():
            cr = defaultdict(list)
            for l in open(f):
                r = json.loads(l)
                cr[r["mode"]].append(r["enc_s"])
            for m, v in cr.items():
                lat[m][key] = float(np.mean(v))
    for m, d in lat.items():
        d["enc_s_table"] = d.get("enc_s_cpu_uav", d["enc_s_profile"])     # what the fd_v1 ModeTable will use
    old_t = C.load_table(False)[0]
    for i, m in enumerate(C.MODES):
        lat[m]["enc_s_table_A5"], lat[m]["dec_s_table_A5"] = float(old_t.enc_s[i]), float(old_t.dec_s[i])
    for m, d in lat.items():
        lines.append(f"latency {m:24s} {d['device']:4s} enc profile {d['enc_s_profile']:.3f}s  "
                     f"cpu(UAV) {d.get('enc_s_cpu_uav', float('nan')):.3f}s  table {d['enc_s_table']:.3f}s "
                     f"(A5 {d['enc_s_table_A5']:.3f}s)  dec {d['dec_s_profile']:.3f}s (A5 {d['dec_s_table_A5']:.3f}s)")
    # predictor
    modes = list(C.MODES)
    j = modes.index("avif_5")

    def mats(names):
        return (np.array([[8 * by[n][m]["bytes"] for m in modes] for n in names], float),
                np.array([[by[n][m]["psnr"] for m in modes] for n in names], float))

    bt, pt = mats(tr)
    cb_new, cp_new = fit(bt, pt, j)
    t_old, pools_old = C.load_table(False)
    cb_old, cp_old = t_old.coef_bits, t_old.coef_psnr
    assert list(t_old.modes) == modes
    pred = {}
    for p in POOLS:
        names = [n for n in by if split[n] == p and all(m in by[n] for m in modes)]
        if len(names) < 2:
            continue
        b, q = mats(names)
        X = np.stack([np.ones(len(names)), np.log(b[:, j])], 1)
        mask = np.ones(len(modes), bool)
        mask[j] = False
        res = {}
        for tag, (cb, cp) in (("new", (cb_new, cp_new)), ("old", (cb_old, cp_old))):
            eb = np.abs(X @ cb - np.log(b))[:, mask]
            eq = np.abs(X @ cp - q)[:, mask]
            s = np.array([seq[n] for n in names])
            sw = np.array([np.mean(eb[s == u]) for u in np.unique(s)])
            sq = np.array([np.mean(eq[s == u]) for u in np.unique(s)])
            res[tag] = dict(bits_factor_mae=float(np.exp(eb.mean())), bits_factor_median=float(np.exp(np.median(eb))),
                            psnr_mae_db=float(eq.mean()), psnr_median_db=float(np.median(eq)),
                            bits_factor_mae_seqw=float(np.exp(sw.mean())), psnr_mae_db_seqw=float(sq.mean()))
        pred[p] = res
        lines.append(f"predictor {p:9s} new: x{res['new']['bits_factor_mae']:.3f} bits, {res['new']['psnr_mae_db']:.2f} dB "
                     f"(seq-w x{res['new']['bits_factor_mae_seqw']:.3f}, {res['new']['psnr_mae_db_seqw']:.2f} dB) | "
                     f"old: x{res['old']['bits_factor_mae']:.3f}, {res['old']['psnr_mae_db']:.2f} dB")
    # pool statistics of the profiled quantities (for the Phase 2 pre-registration)
    stats = {}
    for p in POOLS:
        names = [n for n in by if split[n] == p and all(m in by[n] for m in modes)]
        if not names:
            continue
        b, q = mats(names)
        stats[p] =dict(mean_kbit={m: float(b[:, i].mean() / 1e3) for i, m in enumerate(modes)},
                        mean_psnr={m: float(q[:, i].mean()) for i, m in enumerate(modes)})
    hashes = {f: sha_text(OUT / f) for f in ("manifest_fd_v1.json", "profile/profile.jsonl")}
    if (OUT / "profile" / "cpu_learned.jsonl").exists():
        hashes["profile/cpu_learned.jsonl"] = sha_text(OUT / "profile" / "cpu_learned.jsonl")
    lines.append(f"hashes {hashes}")
    print("\n".join(lines))
    json.dump(dict(coverage=cov, wall=tim, latency_rl_train=lat, predictor=pred, pool_stats=stats, hashes=hashes,
                   config=list(cfgs)[0]), open(OUT / "phase1_summary.json", "w"), indent=1)
    (OUT / "phase1_summary.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
