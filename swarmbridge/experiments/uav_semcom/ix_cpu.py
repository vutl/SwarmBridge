"""paperA_ix stage A: resolution-stratified CPU timing of the learned codecs (MBT2018 q1/q3/q5/q8) on U_train, the
fd_v1 procedure (fd_cpu_stratified.py) applied to the UAVDT training pool. Rules fixed before any timing:
  * population: U_train of manifest_ix_v1.json (780 frames: 1024x540 750, 960x540 30);
  * sample size 48 (as fd_v1); quotas proportional to stratum size, at least 2: 1024x540 46, 960x540 2;
  * RNG numpy default_rng(20261127); within a stratum, sequences are shuffled and one random frame is taken per
    sequence in that order, further rounds only when a stratum has fewer sequences than its quota (fd_v1 rule);
  * timing: torch.set_num_threads(os.cpu_count()), one warm-up image discarded, then every sampled frame once;
  * PRIMARY estimate per mode: frame-weighted (fd_v1 `estimate`, review of 2cd2ec77): within stratum h,
    mu_h = sum_s n_sh t_s / sum_s n_sh, then sum_h (N_h / N) mu_h; the sequence-balanced estimate is also written.
Outputs (runs/uav_semcom/paperA_ix/cpu/): cpu_stratified_sample.json, cpu_learned_stratified.jsonl,
cpu_learned_stratified_estimators.json (read by ix_common.fit_retrained).
Usage (WSL): python ix_cpu.py select ; python ix_cpu.py measure ; python ix_cpu.py estimate"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

import ix_common as IX
from ix_audit import UAVDT

D = IX.OUT / "cpu"
SAMPLE = D / "cpu_stratified_sample.json"
ROWS = D / "cpu_learned_stratified.jsonl"
N, SEED = 48, 20261127
MODES = ["mbt2018-mean_q1", "mbt2018-mean_q3", "mbt2018-mean_q5", "mbt2018-mean_q8"]


def stratum(e):
    return f"{e['w']}x{e['h']}"


def quotas(sizes):
    tot = sum(sizes.values())
    q = {h: max(2, int(round(N * n / tot))) for h, n in sizes.items()}
    big = max(q, key=lambda h: sizes[h])
    q[big] += N - sum(q.values())
    return q


def select():
    assert not SAMPLE.exists(), "sample already fixed"
    D.mkdir(parents=True, exist_ok=True)
    tr = [e for e in IX.read_manifest() if e["split"] == "U_train"]
    rng = np.random.default_rng(SEED)
    sizes = Counter(stratum(e) for e in tr)
    q = quotas(sizes)
    picked = []
    for h in sorted(q, key=lambda h: -sizes[h]):
        by_seq = defaultdict(list)
        for e in sorted((e for e in tr if stratum(e) == h), key=lambda e: e["name"]):
            by_seq[e["seq"]].append(e)
        seqs = sorted(by_seq)
        seqs = [seqs[i] for i in rng.permutation(len(seqs))]
        left = {s: list(by_seq[s]) for s in seqs}
        got = []
        while len(got) < q[h]:
            for s in seqs:
                if len(got) == q[h]:
                    break
                if left[s]:
                    got.append(left[s].pop(int(rng.integers(len(left[s])))))
        picked += [dict(name=e["name"], seq=e["seq"], sha256=e["sha256"], w=e["w"], h=e["h"], stratum=h) for e in got]
    meta = dict(rule=__doc__.split("Rules fixed before any timing:")[1].split("Outputs")[0].strip(), seed=SEED,
                strata={h: dict(quota=q[h], N_h=sizes[h], weight=sizes[h] / len(tr)) for h in q}, n_U_train=len(tr),
                sample=picked)
    SAMPLE.write_text(json.dumps(meta, indent=1))
    print({h: (q[h], sizes[h]) for h in q}, "sample sha256", IX.sha_text(SAMPLE))


def measure():
    import torch
    import imcodecs as IC
    from profile_a11 import hardware, load, timed
    meta = json.loads(SAMPLE.read_text())
    torch.set_num_threads(os.cpu_count())
    cms = [IC.CompressAICodec(m.arch, m.quality, "cpu") for m in IC.default_modes("cpu")
           if isinstance(m, IC.CompressAICodec) and m.name in MODES]
    assert sorted(m.name for m in cms) == sorted(MODES)
    sample, t0 = meta["sample"], time.time()
    with open(ROWS, "w") as fo:
        for ii, p in enumerate([sample[0]] + sample):
            path = UAVDT / p["name"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == p["sha256"], p["name"]
            img = load(path, 0)
            for m in cms:
                data, te = timed(m.encode, img, False)
                rec, td = timed(m.decode, data, False)
                if ii:
                    fo.write(json.dumps(dict(image=p["name"], stratum=p["stratum"], mode=m.name, bytes=len(data),
                                             enc_s=te, dec_s=td)) + "\n")
    (D / "cpu_measure.meta.json").write_text(json.dumps(dict(threads=torch.get_num_threads(), n=len(sample),
                                                             hardware=hardware(), wall_s=time.time() - t0), indent=1))
    print(f"MEASURE_DONE {time.time() - t0:.0f}s", flush=True)


def estimate():
    meta = json.loads(SAMPLE.read_text())
    tr = [e for e in IX.read_manifest() if e["split"] == "U_train"]
    n_sh = Counter((stratum(e), e["seq"]) for e in tr)
    rows = [json.loads(l) for l in open(ROWS)]
    seq = {p["name"]: p["seq"] for p in meta["sample"]}
    est = {}
    for m in MODES:
        sb = fw = 0.0
        per = {}
        for h, s in meta["strata"].items():
            rs = [r for r in rows if r["mode"] == m and r["stratum"] == h]
            w = np.array([n_sh[(h, seq[r["image"]])] for r in rs], float)
            t = np.array([r["enc_s"] for r in rs])
            per[h] = dict(seq_balanced=float(t.mean()), frame_weighted=float((w * t).sum() / w.sum()), N_h=s["N_h"])
            sb += s["weight"] * per[h]["seq_balanced"]
            fw += s["weight"] * per[h]["frame_weighted"]
        est[m] = dict(primary="frame_weighted", frame_weighted=fw, seq_balanced=sb, strata=per)
    (D / "cpu_learned_stratified_estimators.json").write_text(json.dumps(dict(
        rule="frame-weighted primary (fd_v1 estimate rule)", sample_sha256=IX.sha_text(SAMPLE),
        rows_sha256=IX.sha_text(ROWS), estimates=est), indent=1))
    print("ESTIMATE_DONE", {m: round(v["frame_weighted"], 4) for m, v in est.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "measure", "estimate"))
    dict(select=select, measure=measure, estimate=estimate)[ap.parse_args().cmd]()
