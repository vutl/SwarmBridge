"""Join protocol outputs (mAP rows) with tracking metrics (AMOTA) and print markdown tables.

Usage: python -I summarize.py RUNS_DIR   (RUNS_DIR = runs/uav3d_proto)
Every joined row checks that tracking/<name>/input.json records the same sha256 as the prediction row; rows without
tracking are shown with AMOTA '-'. Seeds of the same scenario are averaged (n and min-max spread reported).
Writes RUNS_DIR/summary.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def track(d, name, sha):
    p = d / "tracking" / name
    if not (p / "metrics_summary.json").exists():
        return None
    inp = json.loads((p / "input.json").read_text())
    assert inp["det_sha256"] == sha, f"sha mismatch for {p}"
    m = json.loads((p / "metrics_summary.json").read_text())
    return {"amota": m["amota"], "mota": m["mota"], "ids": m["ids"], "recall": m["recall"]}


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.max(vals) - np.min(vals)) / 2 if len(vals) > 1 else 0.0)


def fmt(m, s=None):
    if m is None:
        return "-"
    return f"{m:.3f}" + (f" ±{s:.3f}" if s else "")


def amota_cell(x):
    """mean AMOTA over seeds; '*' marks a cell where only some seeds have been tracked yet."""
    v = [r["trk"]["amota"] if r["trk"] else None for r in x]
    am, sp = agg(v)
    n = sum(a is not None for a in v)
    return fmt(am, sp) + ("*" if 0 < n < len(v) else "")


def main():
    runs = Path(sys.argv[1])
    out = {}
    # ---- box_proto: methods x scenarios ----
    d = runs / "box"
    rows = json.loads((d / "box_proto.json").read_text())["rows"]
    for r in rows:
        r["trk"] = track(d, f"{r['method']}__{r['ledger']}", r["sha256"])
    out["box"] = rows
    ego = [r for r in rows if r["method"] == "ego"][0]
    print(f"## box_proto\n\nego: mAP {ego['mAP']:.3f}, AMOTA {fmt(ego['trk'] and ego['trk']['amota'])}\n")
    g = defaultdict(list)
    for r in rows:
        if r["method"] != "ego":
            g[(r["method"], r["ledger"].rsplit("_s", 1)[0])].append(r)
    scen = ["sync", "fix1", "fix2", "rand", "loss0.2", "loss0.5", "rand_loss0.2"]
    meths = ["box_max", "box_nor", "kf_max", "kf_nor", "kf0_nor", "kfv_nor", "kfp_nor", "kfw_nor"]
    print("| scenario | " + " | ".join(meths) + " |\n|---|" + "---|" * len(meths))
    for sc in scen:
        cells = []
        for m in meths:
            x = g.get((m, sc))
            if not x:
                cells.append("")
                continue
            mp = agg([r["mAP"] for r in x])[0]
            cells.append(f"{mp:.3f} / {amota_cell(x)}")
        print(f"| {sc} | " + " | ".join(cells) + " |")
    print("\n(cell = mAP / AMOTA, seeds averaged, ± = half min-max spread)\n")

    # ---- budget_proto ----
    d = runs / "budget"
    if (d / "budget_proto.json").exists():
        B = json.loads((d / "budget_proto.json").read_text())
        rows = B["rows"]
        for r in rows:
            r["trk"] = track(d, f"{r['policy']}_B{r['budget_kbit']}__{r['ledger']}", r["sha256"])
        out["budget"] = rows
        print(f"## budget_proto (full message {B['full_kbit']} kbit/frame)\n")
        pols = list(dict.fromkeys(r["policy"] for r in rows))
        for chan in ["sync", "rand"]:
            print(f"### channel {chan}\n\n| policy | " + " | ".join(f"B={b}" for b in B["budgets"]) + " |\n|---|"
                  + "---|" * len(B["budgets"]))
            for p in pols:
                cells = []
                for b in B["budgets"]:
                    x = [r for r in rows if r["policy"] == p and r["budget_kbit"] == b
                         and r["ledger"].rsplit("_s", 1)[0] == chan]
                    mp = agg([r["mAP"] for r in x])[0]
                    cells.append(f"{mp:.3f} / {amota_cell(x)}")
                print(f"| {p} | " + " | ".join(cells) + " |")
            print()

    # ---- queue_proto ----
    d = runs / "queue"
    if (d / "queue_proto.json").exists():
        Q = json.loads((d / "queue_proto.json").read_text())
        rows = Q["rows"]
        for r in rows:
            r["trk"] = track(d, f"{r['policy']}_R{r['rate_kbit']}__{r['channel']}", r["sha256"])
        out["queue"] = rows
        print("## queue_proto\n")
        pols = list(dict.fromkeys(r["policy"] for r in rows))
        for R in Q["rates"]:
            print(f"### link rate {R} kbit/frame per helper\n\n| policy | const mAP / AMOTA | fade mAP / AMOTA | "
                  "fade mean age | fade on-air kbit/frame |\n|---|---|---|---|---|")
            for p in pols:
                c = [r for r in rows if r["policy"] == p and r["rate_kbit"] == R and r["channel"].startswith("const")]
                f = [r for r in rows if r["policy"] == p and r["rate_kbit"] == R and r["channel"].startswith("fade")]
                cell = []
                for x in (c, f):
                    mp = agg([r["mAP"] for r in x])[0]
                    cell.append(f"{mp:.3f} / {amota_cell(x)}")
                age = agg([r["mean_age"] for r in f])[0]
                air = agg([r["kbit_onair_mean"] for r in f])[0]
                print(f"| {p} | {cell[0]} | {cell[1]} | {fmt(age)} | {air:.1f} |")
            print()
    (runs / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
