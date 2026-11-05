"""Profile the existing receiver input builder, without offline cache disk I/O.

Uses validation contexts and already serialized packets. No detector, training,
prediction or labels. Copies are synchronized and therefore cannot be treated
as an optimized deployment latency or added to old timing percentiles.
Run with the common GPU lock through run_matched.sh profile.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import core as C
from receiver import ReceiverData


def stats(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out")
    args = ap.parse_args()
    root = Path(args.out)
    split = json.loads((root / "prepared/split.json").read_text())
    data = ReceiverData(root / "prepared")
    torch.set_num_threads(4)
    tokens = split["tokens"]["val"][:3]
    # 3 ego maps fit the cache. Packet set fits its 64-entry limit.
    for tok in tokens:
        data.ego(tok)
        for j in range(1, 5):
            for kind in ("box", "tile"):
                data.packet(tok, str(j), "obj", kind)
    # Assert there can be no silent filesystem access during measurement.
    def resident_ego(tok):
        return data.cache[tok]
    def resident_packet(tok, sender, rank, kind):
        folder = "boxes" if kind == "box" else "tiles"
        name = tok + "_%s" % sender + ("" if kind == "box" else "_" + rank) + ".bin"
        return data.packet_cache[folder + "/" + name]
    data.ego, data.packet = resident_ego, resident_packet
    rows = {}
    start = time.monotonic()
    for mode, count, helpers in [("none", 0, False), ("m1", 0, True), ("m2", 16, True), ("m3", 64, True)]:
        elapsed, copies, residual = [], [], []
        for n in range(15):
            tok = tokens[n % len(tokens)]
            ctx = dict(token=tok, helpers={str(j):dict(token=tok, age=0) for j in range(1,5)} if helpers else {}, k=count, rank="obj")
            timing = {}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x, _, _ = data.build(ctx, profile=timing)
            torch.cuda.synchronize()
            total = 1000 * (time.perf_counter() - t0)
            if n == 0:
                reference, _, _ = data.build(ctx)
                assert torch.equal(x, reference), "profiling changed input"
                del reference
            if n >= 3:
                elapsed.append(total); copies.append(timing["transfer_wall_ms"])
                residual.append(total - timing["transfer_wall_ms"])
            del x
        rows[mode] = dict(samples=len(elapsed), resident_build_wall_ms=stats(elapsed),
                          cpu_to_gpu_copy_and_ego_cast_wall_ms=stats(copies),
                          other_build_work_wall_ms=stats(residual))
    C.save_json(root / "input_profile.json", dict(
        validation_tokens=tokens, modes=rows, warmup_per_mode=3,
        gpu_job_seconds=time.monotonic()-start,
        limitations=["Resident bytes and ego BEV: disk I/O excluded; no encoder, sender encoding, radio or receiver decoder.",
                     "Synchronized copies include ego fp16-to-fp32 conversion; other work includes decode, scatter, allocation, warp and assembly.",
                     "Twelve measurements over three validation frames per mode; implementation microbenchmark, not an end-to-end deadline test."],
        source_hashes={p:C.sha(Path(__file__).parent/p) for p in ["receiver.py", "profile_inputs.py"]}))
    print("INPUT_PROFILE", json.dumps(rows), flush=True)


if __name__ == "__main__": main()
