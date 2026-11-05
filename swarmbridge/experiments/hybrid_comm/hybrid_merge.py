"""Hybrid receiver (gate 1, no wireless, age 0): box branch outside the received tiles, feature branch inside.

For every dev frame: U = union over helpers of the tiles the receiver decoded (sent_tiles.json from m0_receiver.py;
first K tiles of each helper's rank order). Output = feature-decoder detections whose centre lies in U  +
box-branch detections (M1, box_proto.py, synchronous ledger) whose centre lies outside U. A box is never taken from
both branches, so the same evidence is not counted twice. Diagnostic 'densein': the feature branch inside U is
replaced by the decoder fed with ALL helper features (what a better sparse receiver could reach in U).

Usage: python -I hybrid_merge.py DATA_ROOT M0_DIR BOX_DETS_DIR OUT_JSON
  BOX_DETS_DIR/<method>/sync_s0/results_nusc.json, methods kf_nor (KF-PV) and box_nor (noisy-OR boxes)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uav3d_comm"))
from evaldet import MiniDB, evaluate, load_nusc_results  # noqa: E402

HELD = ["scene_11", "scene_13", "scene_15"]


def tile_of(b, ego, half, cell, tile, axis):
    ix = int((b["translation"][0] - ego[0] + half) // cell)
    iy = int((b["translation"][1] - ego[1] + half) // cell)
    g = int(2 * half / cell + 0.5)
    if not (0 <= ix < g and 0 <= iy < g):
        return -1
    if axis == "yx":
        ix, iy = iy, ix
    return (ix // tile) * (g // tile) + iy // tile


def main():
    db = MiniDB(sys.argv[1])
    m0, boxdir, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    st = json.loads((m0 / "sent_tiles.json").read_text())
    held = set(db.tokens(HELD))
    load = lambda p: json.loads(Path(p).read_text())["results"]
    rows = {}

    def score(name, results):
        p = out.parent / "hybrid" / name / "results_nusc.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"meta": {"use_camera": True}, "results": results}))
        r = evaluate(db, load_nusc_results(p, held), db.tokens(HELD))
        r["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows[name] = r
        print(f"{name:32s} mAP {r['mAP']:.3f}  ({r['ap_0.5']:.3f}/{r['ap_1.0']:.3f}/{r['ap_2.0']:.3f}/{r['ap_4.0']:.3f})"
              f"", flush=True)
        return r

    dense = load(m0 / "feat_dense" / "results_nusc.json")
    for meth in ("kf_nor", "box_nor"):
        box = load(boxdir / meth / "sync_s0" / "results_nusc.json")
        score(f"m1_{meth}", {t: box.get(t, []) for t in st["sent"]})
        for rank in ("obj", "norm"):
            for k in (16, 64, 256):
                feat = load(m0 / f"tiles_{rank}_{k}" / "results_nusc.json")
                for tag, src in (("", feat), ("_densein", dense)):
                    res, frac = {}, []
                    for t, s in st["sent"].items():
                        U = set()
                        for j in ("1", "2", "3", "4"):
                            U.update(s[rank][j][:k])
                        f = lambda b: tile_of(b, s["ego_xy"], st["half"], st["cell"], st["tile"], st["axis"]) in U
                        res[t] = [b for b in src.get(t, []) if f(b)] + [b for b in box.get(t, []) if not f(b)]
                        frac.append(len(U) / 1024)
                    r = score(f"{meth}+tiles_{rank}_{k}{tag}", res)
                    r["tile_area_frac"] = sum(frac) / len(frac)
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
