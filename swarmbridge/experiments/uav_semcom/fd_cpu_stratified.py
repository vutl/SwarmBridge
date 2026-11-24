"""fd_v1: resolution-stratified CPU timing of the learned codecs (MBT2018 q1/q3/q5/q8), approved option (b) of the
Phase 1 review (c30690a8). Replaces, for fd_v1, the first-10-images rule (kept as cpu_learned.jsonl, not used).

Pre-registered rules (fixed BEFORE any timing is seen; the sample is committed first with `select`):
  * population: rl_train of manifest_fd_v1.json (1,523 frames);
  * strata by resolution: 1400x1050 (14 frames), 1360x765 (12), 1920x1080 (6), 960x540 (6), 1400x788 (5),
    all other resolutions (5): 48 frames, without replacement;
  * RNG numpy default_rng(20261124); within a stratum, sequences are shuffled and one random frame is taken per
    sequence in that order (distinct sequences first); further rounds only if a stratum has fewer sequences than
    its quota;
  * timing: torch.set_num_threads(os.cpu_count()) (= 6, as A5 and Phase 1), same MBT2018 models / settings as Phase 1,
    one warm-up image (the first sampled frame) encoded / decoded once and discarded, then every sampled frame once;
  * primary estimate per mode: sum_h (N_h / 1523) * mean_h, N_h = stratum size in rl_train (NOT the unweighted mean of
    the stratum means); per-mode SD over frames and per-stratum means are reported, and the first-10-images values
    for comparison;
  * the fd_v1 ModeTable uses this weighted estimate as the UAV encode time of MBT2018 (never GPU timings);
  * label: measured CPU timing on the stated hardware (i5-14400F, 6 threads), not latency validated on UAV hardware.
Outputs (runs/uav_semcom/paperA_fd/profile/): cpu_stratified_sample.json (select), cpu_learned_stratified.jsonl +
cpu_learned_stratified.meta.json (measure).
Usage (WSL): python fd_cpu_stratified.py select ; python fd_cpu_stratified.py measure"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from fd_manifest import OUT, SRC

STRATA = [("1400x1050", 14), ("1360x765", 12), ("1920x1080", 6), ("960x540", 6), ("1400x788", 5), ("other", 5)]
SEED = 20261124
SAMPLE = OUT / "profile" / "cpu_stratified_sample.json"
MODES = ["mbt2018-mean_q1", "mbt2018-mean_q3", "mbt2018-mean_q5", "mbt2018-mean_q8"]


def stratum(e):
    r = f"{e['w']}x{e['h']}"
    return r if r in dict(STRATA) else "other"


def select():
    assert not SAMPLE.exists(), "sample already fixed"
    man = json.loads((OUT / "manifest_fd_v1.json").read_text())["images"]
    tr = [e for e in man if e["split"] == "rl_train"]
    rng = np.random.default_rng(SEED)
    sizes = Counter(stratum(e) for e in tr)
    picked = []
    for h, n in STRATA:
        by_seq = defaultdict(list)
        for e in sorted((e for e in tr if stratum(e) == h), key=lambda e: e["name"]):
            by_seq[e["seq"]].append(e)
        seqs = sorted(by_seq)
        seqs = [seqs[i] for i in rng.permutation(len(seqs))]
        left = {s: list(by_seq[s]) for s in seqs}
        got = []
        while len(got) < n:
            for s in seqs:
                if len(got) == n:
                    break
                if left[s]:
                    got.append(left[s].pop(int(rng.integers(len(left[s])))))
        picked += [dict(name=e["name"], seq=e["seq"], sha256=e["sha256"], w=e["w"], h=e["h"], stratum=h) for e in got]
    meta = dict(rule=__doc__.split("Pre-registered rules")[1].split("Outputs")[0].strip(), seed=SEED,
                strata={h: dict(quota=n, N_h=sizes[h], weight=sizes[h] / len(tr),
                                sequences_in_sample=len({p["seq"] for p in picked if p["stratum"] == h}))
                        for h, n in STRATA}, n_rl_train=len(tr), sample=picked)
    SAMPLE.write_text(json.dumps(meta, indent=1))
    for h, s in meta["strata"].items():
        print(h, s)
    print("sample sha256", hashlib.sha256(SAMPLE.read_bytes().replace(b"\r\n", b"\n")).hexdigest())


def measure():
    import torch
    import imcodecs as IC
    from profile_a11 import hardware, load, timed
    meta = json.loads(SAMPLE.read_text())
    torch.set_num_threads(os.cpu_count())
    cms = [IC.CompressAICodec(m.arch, m.quality, "cpu") for m in IC.default_modes("cpu")
           if isinstance(m, IC.CompressAICodec) and m.name in MODES]
    assert sorted(m.name for m in cms) == sorted(MODES)
    src = {e["name"]: e["source"] for e in json.loads((OUT / "manifest_fd_v1.json").read_text())["images"]}
    sample = meta["sample"]
    t0 = time.time()
    rows = []
    with open(OUT / "profile" / "cpu_learned_stratified.jsonl", "w") as fo:
        for ii, p in enumerate([sample[0]] + sample):
            img = load(SRC[src[p["name"]]] / "images" / p["name"], 0)
            for m in cms:
                data, te = timed(m.encode, img, False)
                rec, td = timed(m.decode, data, False)
                if ii:
                    r = dict(image=p["name"], stratum=p["stratum"], mode=m.name, bytes=len(data), enc_s=te, dec_s=td)
                    rows.append(r)
                    fo.write(json.dumps(r) + "\n")
    est = {}
    old = defaultdict(list)
    for l in open(OUT / "profile" / "cpu_learned.jsonl"):
        r = json.loads(l)
        old[r["mode"]].append(r["enc_s"])
    for m in MODES:
        v = np.array([r["enc_s"] for r in rows if r["mode"] == m])
        sm = {h: float(np.mean([r["enc_s"] for r in rows if r["mode"] == m and r["stratum"] == h])) for h, _ in STRATA}
        est[m] = dict(weighted_mean=float(sum(meta["strata"][h]["weight"] * sm[h] for h in sm)),
                      unweighted_frame_mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()),
                      max=float(v.max()), stratum_means=sm, first10_mean=float(np.mean(old[m])))
        print(f"{m}: weighted {est[m]['weighted_mean']:.3f}s (frame mean {v.mean():.3f}, SD {v.std(ddof=1):.3f}, "
              f"range {v.min():.3f}-{v.max():.3f}) vs first-10 {np.mean(old[m]):.3f}s; strata "
              + ", ".join(f"{h} {x:.3f}" for h, x in sm.items()), flush=True)
    (OUT / "profile" / "cpu_learned_stratified.meta.json").write_text(json.dumps(dict(
        protocol="resolution-stratified CPU timing (fd_cpu_stratified.py), 6 threads, warm-up discarded",
        threads=torch.get_num_threads(), n=len(sample), estimates=est, hardware=hardware(),
        sample_sha256=hashlib.sha256(SAMPLE.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        wall_s=time.time() - t0), indent=1))
    print(f"STRATIFIED_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "measure"))
    dict(select=select, measure=measure)[ap.parse_args().cmd]()
