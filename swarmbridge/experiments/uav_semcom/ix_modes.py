"""Observational 14-mode utilisation audit of the STORED fd_v1 results (no simulation, no training, no tuning).

Delivery records (fd_phase2.py / fd_next.py, ax2_eval.DELIV_EXT): [manifest idx, mode or -1, ok, ontime] for every
finished frame. Outcome classes per record:
  * no_mode    : mode = -1 (dropped before any mode decision);
  * compliant  : ok = 1 (on time and quality floor met);
  * below_floor: ok = 0, ontime = 1 (delivered on time, PSNR below the floor);
  * late_drop  : ok = 0, ontime = 0 (dropped at deadline expiry or delivered late).
"Selected" = the mode attached to the frame when it finished (the executed decision). RL training action counts are
NOT recorded in the stored training logs (phase3/train/*.train.json hold validation returns only); they are reported
as unavailable, never inferred from test outcomes.
Per-mode profile statistics come from profile/profile.jsonl: payload, PSNR, encode / decode time over each pool; YOLO
recall (sum tp / sum n_gt) over the recall pools F / TD / V only (YOLO was trained on the rl_* scenes).
Outputs: runs/uav_semcom/paperA_ix/modes_fd_v1.csv, modes_profile_fd_v1.csv, modes_fd_v1.log.
Usage: python ix_modes.py"""
import csv
import gzip
import json
from collections import defaultdict

import numpy as np

import a5_common as C
from fd_manifest import REPO

FD = REPO / "runs" / "uav_semcom" / "paperA_fd"
OUT = REPO / "runs" / "uav_semcom" / "paperA_ix"
FILES = [("phase2", f"phase2/test_{p}.deliv.json.gz") for p in ("TD", "F", "V", "TD_all")] + \
        [("ablation", f"ablation/test_{p}.deliv.json.gz") for p in ("TD", "F", "V")] + \
        [("phase3_retrained", f"phase3/retrained_{p}.deliv.json.gz") for p in ("TD", "F", "V", "TD_all")] + \
        [("phase3_zeroshot", "phase3/zeroshot_TD.deliv.json.gz")] + \
        [("phase4", f"phase4/{f}") for f in ("test_TD.deliv.json.gz", "test_F.deliv.json.gz",
                                             "test_V.deliv.json.gz")]
M = len(C.MODES)
CLS = ("compliant", "below_floor", "late_drop")


def parse(block, key):
    p = key.split("|")
    if block == "phase2":
        return dict(cfg="ref", pool=p[0], combo="|".join(p[1:]), ckpt="")
    return dict(cfg=p[0], pool=p[1], combo="|".join(p[2:-1]), ckpt=p[-1])


def arm(combo):
    return combo.split("|")[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, agg = [], {}
    for block, f in FILES:
        path = FD / f
        if not path.exists():
            print("missing (skipped):", f)
            continue
        for key, eps in json.load(gzip.open(path, "rt")).items():
            k = parse(block, key)
            cnt = np.zeros((M, 3), int)
            nomode = 0
            for ep in eps:
                for rec in ep:
                    m, ok, ontime = rec[1], rec[2], rec[3] if len(rec) > 3 else None
                    if m < 0:
                        nomode += 1
                        continue
                    c = 0 if ok else (1 if ontime else 2)
                    cnt[m, c] += 1
            tot = cnt.sum()
            # learned incumbents: aggregate the 3 checkpoints of one algorithm
            ck = k["ckpt"].rsplit("_s", 1)[0] if k["ckpt"] not in ("", "-1") else k["ckpt"]
            g = (block, k["cfg"], k["pool"], arm(k["combo"]), ck)
            a = agg.setdefault(g, dict(cnt=np.zeros((M, 3), int), nomode=0, eps=0))
            a["cnt"] += cnt
            a["nomode"] += nomode
            a["eps"] += len(eps)
    for (block, cfg, pool, armname, ck), a in sorted(agg.items()):
        cnt, tot = a["cnt"], a["cnt"].sum()
        for j, m in enumerate(C.MODES):
            n = int(cnt[j].sum())
            rows.append(dict(block=block, cfg=cfg, pool=pool, arm=armname, incumbent_ckpt=ck, episodes=a["eps"], mode=m,
                             selected=n, share=round(n / tot, 5) if tot else 0.0,
                             **{c: int(cnt[j, i]) for i, c in enumerate(CLS)},
                             compliant_rate=round(cnt[j, 0] / n, 4) if n else "",
                             no_mode_frames_arm=a["nomode"]))
    with open(OUT / "modes_fd_v1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per-mode profile statistics
    man = {e["name"]: e["split"] for e in json.loads((FD / "manifest_fd_v1.json").read_text())["images"]}
    st = defaultdict(lambda: defaultdict(list))
    tp, ngt = defaultdict(float), defaultdict(float)
    by_img = defaultdict(dict)
    for line in open(FD / "profile" / "profile.jsonl"):
        r = json.loads(line)
        by_img[r["image"]][r["mode"]] = r
    for im, d in by_img.items():
        sp = man.get(im)
        for m in C.MODES:
            r = d.get(m)
            if r is None:
                continue
            for pool in (sp, "all"):
                st[(pool, m)]["kbit"].append(r["bytes"] * 8 / 1000)
                st[(pool, m)]["psnr"].append(r["psnr"])
                st[(pool, m)]["enc_s"].append(r["enc_s"])
                st[(pool, m)]["dec_s"].append(r["dec_s"])
            if sp in ("F", "TD", "V") and "tp" in r:
                tp[m] += r["tp"]
                ngt[m] += d["original"]["n_gt"]
        if man.get(im) in ("F", "TD", "V") and "original" in d and "tp" in d["original"]:
            tp["original"] += d["original"]["tp"]
            ngt["original"] += d["original"]["n_gt"]
    prow = []
    for m in C.MODES:
        s = st[("all", m)]
        prow.append(dict(mode=m, images=len(s["kbit"]), kbit_median=round(float(np.median(s["kbit"])), 1),
                         kbit_mean=round(float(np.mean(s["kbit"])), 1), psnr_mean=round(float(np.mean(s["psnr"])), 2),
                         enc_s_mean_profile=round(float(np.mean(s["enc_s"])), 3),
                         dec_s_mean_profile=round(float(np.mean(s["dec_s"])), 3),
                         recall_FTDV=round(tp[m] / ngt[m], 4) if ngt[m] else ""))
    if ngt["original"]:
        prow.append(dict(mode="original", recall_FTDV=round(tp["original"] / ngt["original"], 4)))
    with open(OUT / "modes_profile_fd_v1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)

    # log: overall selection per block / arm on TD (reference cfg), and never-selected modes per arm
    lines = ["RL training action counts: NOT RECORDED in phase3/train/*.train.json (validation returns only).", ""]
    for (block, cfg, pool, armname, ck), a in sorted(agg.items()):
        if pool != "TD" or cfg not in ("ref", "K5_la", "N5x2", "base") and block not in ("phase2", "ablation"):
            continue
        cnt, tot = a["cnt"], a["cnt"].sum()
        sh = cnt.sum(1) / max(tot, 1)
        top = ", ".join(f"{C.MODES[j]} {100 * sh[j]:.1f}%" for j in np.argsort(-sh)[:5] if sh[j] > 0)
        never = [C.MODES[j] for j in range(M) if cnt[j].sum() == 0]
        lines.append(f"{block} | {cfg} | {armname} | {ck or '-'} | n={tot} no_mode={a['nomode']} | {top} | "
                     f"never: {', '.join(never) or '-'}")
    gen = C.MODES[:4]
    tot_all = sum(a["cnt"].sum() for a in agg.values())
    gen_all = sum(a["cnt"][:4].sum() for a in agg.values())
    gen_ok = sum(a["cnt"][:4, 0].sum() for a in agg.values())
    lines += ["", f"generative / thumbnail modes {gen}: selected {gen_all} of {tot_all} finished frames "
                  f"({100 * gen_all / max(tot_all, 1):.3f}%), compliant {gen_ok}"]
    for j, m in enumerate(C.MODES):
        n = sum(a["cnt"][j].sum() for a in agg.values())
        ok = sum(a["cnt"][j, 0].sum() for a in agg.values())
        lines.append(f"  {m:24s} selected {n:9d} ({100 * n / max(tot_all, 1):6.3f}%)  compliant {ok:9d}"
                     f"  ({100 * ok / n:5.1f}% of selected)" if n else f"  {m:24s} selected 0")
    (OUT / "modes_fd_v1.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
