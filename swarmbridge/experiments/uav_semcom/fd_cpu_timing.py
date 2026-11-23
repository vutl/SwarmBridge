"""fd_v1: CPU timing of the learned codecs (UAV on-board encoder), with EXACTLY the A5 protocol of profile_a11.py:
torch.set_num_threads(os.cpu_count()), the first image encoded/decoded once and discarded (warm-up), then the first
K rl_train images (manifest order). env.ModeTable replaces the learned codecs' ENCODE time by the mean of these
measurements (decode stays the BS-side GPU time of the main profile).

Why a separate script (review of 33aae4a9): `fd_profile.py cpu`, the last step of run_phase1.sh, leaves torch at its
default thread count (3 in this WSL, versus os.cpu_count() = 6 in the A5 protocol). Its output is kept as
cpu_learned_default_threads.jsonl and NOT used. This script must run AFTER the GPU profiling job, so that the CPU
measurement is not disturbed by it. Output: runs/uav_semcom/paperA_fd/profile/cpu_learned.jsonl (+ .meta.json).
Usage (WSL): python fd_cpu_timing.py [--k 10]"""
import argparse
import json
import os
import time

import torch

import imcodecs as IC
from fd_manifest import OUT, SRC
from fd_profile import CODEC_MODES, MANIFEST
from profile_a11 import hardware, load, timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    out = OUT / "profile"
    old = out / "cpu_learned.jsonl"
    if old.exists() and not (out / "cpu_learned.meta.json").exists():
        old.rename(out / "cpu_learned_default_threads.jsonl")      # written by `fd_profile.py cpu` (3 threads)
    torch.set_num_threads(os.cpu_count())
    man = [e for e in json.loads(MANIFEST.read_text())["images"] if e["split"] == "rl_train"][:a.k]
    cms = [IC.CompressAICodec(m.arch, m.quality, "cpu") for m in IC.default_modes("cpu")
           if isinstance(m, IC.CompressAICodec) and m.name in CODEC_MODES]
    t0 = time.time()
    with open(out / "cpu_learned.jsonl", "w") as fo:
        for ii, e in enumerate([man[0]] + man):
            img = load(SRC[e["source"]] / "images" / e["name"], 0)
            for m in cms:
                data, te = timed(m.encode, img, False)
                rec, td = timed(m.decode, data, False)
                if ii:
                    fo.write(json.dumps(dict(image=e["name"], mode=m.name, bytes=len(data), enc_s=te, dec_s=td)) + "\n")
    (out / "cpu_learned.meta.json").write_text(json.dumps(dict(
        protocol="A5 profile_a11.py CPU timing: set_num_threads(os.cpu_count()), warm-up image discarded",
        threads=torch.get_num_threads(), k=a.k, images=[e["name"] for e in man], hardware=hardware(),
        wall_s=time.time() - t0), indent=1))
    print(f"CPU_TIMING_DONE threads {torch.get_num_threads()} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
