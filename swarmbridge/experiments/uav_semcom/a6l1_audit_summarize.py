"""Summary tables (markdown) of the A6.0-L1 method audit (reports/uav-semcom-a6-l1-audit.md section 0), from
runs/uav_semcom/a6l1/audit/. Run in the WSL env (policy-level AP50 uses the cached matching of a6l1_ap).
Sections: cross (0.2), qmin (0.3 a), val (0.3 b / 0.4), final (0.4). Usage: python a6l1_audit_summarize.py [sections]
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

import a5_common as C
import a6l1_audit as AU
import a6l1_ap as AP
from a6l1_summarize import ci1, fmt

O = Path(os.environ.get("A6L1_AUDIT_OUT", AU.OUT))
POOLS = ["test", "testdev"]
_B = {}


def built(testdev):
    key = "testdev" if testdev else "val"
    if key not in _B:
        b = AP.build(key)
        tab, _ = C.load_table(testdev)
        idx = {n: i for i, n in enumerate(b["images"])}
        _B[key] = (b, [idx.get(n, -1) for n in tab.images])
    return key, _B[key]


def ap_eps(eps, testdev, col):
    """Per-episode AP50 and pooled AP50 of a list of deliveries (col 1 = delivered ok, col 2 = on time)."""
    key, (b, tmap) = built(testdev)
    inst = lambda ep: [(key, tmap[r[0]], C.MODES[r[col]] if r[col] >= 0 else None) for r in ep]
    per = np.array([AP.ap50(inst(ep), {key: b}) for ep in eps])
    pooled = AP.ap50([x for ep in eps for x in inst(ep)], {key: b})
    return pooled, per


def group(results, prefix):
    """PPO specs whose label minus the '_sN' seed tag equals prefix ('task_best', 'task_best@T5best', ...)."""
    import re
    return sorted(s for s in results if s.startswith("P:") and "=" in s
                  and re.sub(r"_s\d+", "", s[2:].split("=", 1)[0]) == prefix)


def gval(res, specs, k):
    return np.mean([[e[k] for e in res[s]] for s in specs], axis=0)


def modeshare(modes, specs, top=3):
    c = np.sum([np.sum(modes[s], 0) for s in specs], 0)
    c = c / max(c.sum(), 1)
    o = np.argsort(-c)[:top]
    return ", ".join(f"{C.MODES[j].replace('mbt2018-mean_', 'mbt ')} {c[j] * 100:.0f}%" for j in o if c[j] > 0)


def cross():
    print("## 0.2 Đánh giá chéo với cùng phần slot (seed 9000..9099; PPO = trung bình 3 model)\n")
    verdict = {}
    for pool in POOLS:
        p = O / f"cross_{pool}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        dl = json.load(gzip.open(str(p)[:-5] + ".deliv.json.gz", "rt"))["results"]
        for r in d["results"]:
            R, M, D = d["results"][r], d["modes"][r], dl[r]
            cells = {
                "PPO task @ S_task": group(R, "task_best"), "PPO task @ S_psnr": group(R, "task_best@T5best"),
                "PPO PSNR @ S_psnr": group(R, "psnr_best"), "PPO PSNR @ S_task": group(R, "psnr_best@T6best"),
                "heur mode task @ S_task (T:best_task)": ["T6:best"], "heur mode task @ S_psnr": ["X:T5:best|T6:best"],
                "heur mode PSNR @ S_psnr (A5 T:best)": ["T5:best"], "heur mode PSNR @ S_task": ["X:T6:best|T5:best"]}
            ap = {}
            for lab, specs in cells.items():
                per = [ap_eps(D[s], pool == "testdev", 1) for s in specs]
                ap[lab] = (float(np.mean([x[0] for x in per])), np.mean([x[1] for x in per], 0))
            print(f"### {r}, {pool}\n")
            print("| ô | Mbit | recall | recall nhỏ | AP50 | đúng hạn | R_task | mode chính |")
            print("|---|---|---|---|---|---|---|---|")
            for lab, specs in cells.items():
                g = lambda k: gval(R, specs, k).mean()
                print(f"| {lab} | {g('bits_sent') / 1e6:.1f} | {g('recall'):.4f} | {g('recall_small'):.4f} | "
                      f"{ap[lab][0]:.4f} | {g('ok_rate'):.3f} | {g('return'):+.2f} | {modeshare(M, specs)} |")
            print()
            rows = [("Δ_mode @ S_psnr (PPO)", "PPO task @ S_psnr", "PPO PSNR @ S_psnr"),
                    ("Δ_mode @ S_task (PPO)", "PPO task @ S_task", "PPO PSNR @ S_task"),
                    ("Δ_slot, PPO task", "PPO task @ S_task", "PPO task @ S_psnr"),
                    ("Δ_slot, PPO PSNR", "PPO PSNR @ S_task", "PPO PSNR @ S_psnr"),
                    ("Δ_mode @ S_psnr (heur)", "heur mode task @ S_psnr", "heur mode PSNR @ S_psnr (A5 T:best)"),
                    ("Δ_mode @ S_task (heur)", "heur mode task @ S_task (T:best_task)", "heur mode PSNR @ S_task")]
            print("| hiệu | ΔMbit [CI] | Δrecall điểm [CI] | ΔAP50 điểm [CI] | Δđúng hạn [CI] |")
            print("|---|---|---|---|---|")
            for lab, a, b in rows:
                db = (gval(R, cells[a], "bits_sent") - gval(R, cells[b], "bits_sent")) / 1e6
                drc = (gval(R, cells[a], "recall") - gval(R, cells[b], "recall")) * 100
                dap = (ap[a][1] - ap[b][1]) * 100
                dok = (gval(R, cells[a], "ok_rate") - gval(R, cells[b], "ok_rate")) * 100
                c_b, c_r = ci1(db), ci1(drc)
                print(f"| {lab} | {fmt(*c_b, 1)} | {fmt(*c_r)} | {fmt(*ci1(dap))} | {fmt(*ci1(dok))} |")
                if lab.startswith("Δ_mode") and "(PPO)" in lab:
                    verdict.setdefault((r, pool), []).append(c_b[2] < 0 and c_r[2] < 0)
            print()
    print("**Quy tắc 0.2 (C1):**\n")
    for (r, pool), v in verdict.items():
        print(f"- {r} {pool}: " + ("không phụ thuộc slot" if all(v) else "PHỤ THUỘC SLOT") +
              f" (bit < 0 và recall < 0, CI loại 0, ở S_psnr / S_task: {v})")
    print()


def qmin():
    print("## 0.3 (a) Qmin ~ U(22, 30) dB: mode nào bị loại (ảnh train)\n")
    import a6l1_task as TK
    tab, pools = C.load_table(False)
    tt = TK.task_table(tab)
    tr = pools["train"]
    print("| mode | P(vi phạm Qmin) | PSNR trung vị | v trung bình | recall | payload trung vị (Mbit) |")
    print("|---|---|---|---|---|---|")
    for j, m in enumerate(C.MODES):
        ps = tab.psnr[tr, j]
        pv = float(np.mean(np.clip((30 - ps) / 8, 0, 1)))
        rc = tt.tp[tr, j].sum() / tt.ngt[tr].sum()
        print(f"| {m} | {pv:.2f} | {np.median(ps):.1f} | {np.mean(tt.v[tr, j]):+.2f} | {rc:.3f} | "
              f"{np.median(tab.bits[tr, j]) / 1e6:.3f} |")
    print()


def val():
    p = O / "val.json"
    if not p.exists():
        return
    d = json.load(open(p))
    print("## 0.3 (b) / 0.4 Trọng số ξ_q và Qmin: heuristic tinh chỉnh trên train, đánh giá trên val (seed 3000..3049)\n")
    for r in d["results"]:
        R = d["results"][r]
        print(f"### {r}\n")
        print("| cấu hình | policy | recall_ontime | recall | Mbit | đúng hạn |")
        print("|---|---|---|---|---|---|")
        for c in AU.CONFIGS:
            k = f"{c}|TC:{c}:best"
            g = lambda x: np.mean([e[x] for e in R[k]])
            print(f"| {c} | {d['resolved'][r][k]} | {g('recall_ontime'):.4f} | {g('recall'):.4f} | "
                  f"{g('bits_sent') / 1e6:.1f} | {g('ok_rate'):.3f} |")
        for lab, specs in (("A5 T:best (PSNR)", ["T5:best"]), ("T:best_task (L1)", ["T6:best"]),
                           ("PPO PSNR", group(R, "psnr_best")), ("PPO task", group(R, "task_best"))):
            g = lambda x: gval(R, specs, x).mean()
            print(f"| L1 (ξ_q 0,5, Qmin) | {lab} | {g('recall_ontime'):.4f} | {g('recall'):.4f} | "
                  f"{g('bits_sent') / 1e6:.1f} | {g('ok_rate'):.3f} |")
        c, feas, cap, _ = AU.choose(r, d)
        print(f"\nChọn theo quy tắc 0.4: **{c}** ({'thoả' if feas else 'KHÔNG cấu hình nào thoả'} điều kiện bit ≤ PPO PSNR "
              f"= {cap / 1e6:.1f} Mbit)\n")


def final():
    print("## 0.4 Cấu hình chọn trên val, đánh giá một lần (seed 9000..9099)\n")
    for pool in POOLS:
        p = O / f"final_{pool}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        dl = json.load(gzip.open(str(p)[:-5] + ".deliv.json.gz", "rt"))["results"]
        for r in d["results"]:
            R, D = d["results"][r], dl[r]
            c = d["picks"][r]["config"]
            ck = f"{c}|TC:{c}:best"
            cells = {f"heur {c} ({d['resolved'][r]})": [ck], "PPO PSNR": group(R, "psnr_best"),
                     "T:best_task (L1)": ["T6:best"]}
            print(f"### {r}, {pool}\n")
            print("| policy | recall_ontime | recall | recall nhỏ (ontime) | AP50 (ontime) | Mbit | đúng hạn |")
            print("|---|---|---|---|---|---|---|")
            ap = {}
            for lab, specs in cells.items():
                per = [ap_eps(D[s], pool == "testdev", 2) for s in specs]
                ap[lab] = (float(np.mean([x[0] for x in per])), np.mean([x[1] for x in per], 0))
                g = lambda x: gval(R, specs, x).mean()
                print(f"| {lab} | {g('recall_ontime'):.4f} | {g('recall'):.4f} | {g('recall_small_ontime'):.4f} | "
                      f"{ap[lab][0]:.4f} | {g('bits_sent') / 1e6:.1f} | {g('ok_rate'):.3f} |")
            print()
            a = list(cells)[0]
            print("| hiệu | Δrecall_ontime điểm [CI] | ΔAP50 điểm [CI] | ΔMbit [CI] |")
            print("|---|---|---|---|")
            for b in list(cells)[1:]:
                drc = (gval(R, cells[a], "recall_ontime") - gval(R, cells[b], "recall_ontime")) * 100
                db = (gval(R, cells[a], "bits_sent") - gval(R, cells[b], "bits_sent")) / 1e6
                print(f"| {c} − {b} | {fmt(*ci1(drc))} | {fmt(*ci1((ap[a][1] - ap[b][1]) * 100))} | {fmt(*ci1(db), 1)} |")
            print()


def main():
    secs = sys.argv[1:] or ["cross", "qmin", "val", "final"]
    for s in secs:
        dict(cross=cross, qmin=qmin, val=val, final=final)[s]()


if __name__ == "__main__":
    main()
