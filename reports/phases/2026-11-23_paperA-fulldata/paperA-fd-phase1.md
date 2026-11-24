# Paper A extended evaluation: Phase 1 record (profiling)

**Authorization:** ChatGPT review of 50ce9022. Phase 1 may run once the integrity checks and smoke tests pass.

## 1. Before the full run (23/11)

**Manifest `fd_v1`:**
- built by `fd_manifest.py`;
- file `runs/uav_semcom/paperA_fd/manifest_fd_v1.json`, sha256 `f0760601515f9663f2b7747bd247eddabc3ab8c997755c73ae8bf2f2339d4d50`;
- 4,223 images to profile and 4,406 excluded and recorded with their reason.

**Integrity checks** (`integrity_fd_v1.log`, `neardup_fd_v1.log`):
- Images and annotation files match one-to-one in all three splits.
- The YOLO train list and hold-out list equal exactly the expected DET-train subsets.
- All pools are sequence-disjoint. TD and TD_extra are one pool, TD_all.
- **Exact duplicate files:** 3 groups, all within one split:
  - two DET-train frames of sequence 0000239;
  - two test-dev frames of 9999950;
  - two DET-val frames with *different* sequence prefixes (0000022_00000_d_0000004 = 0000023_00000_d_0000008). Both
    are in V.
- **Near-duplicates:** checked with perceptual hashes (dHash and aHash, both ≤ 20/256) over all 8,629 frames. Because
  a sequence prefix does not always identify a scene, this check runs across pools.
  - 16 pairs found: 10 within one pool, 6 across pools.
  - All 6 cross-pool pairs link a used frame to a frame excluded by the cap (x_cap, never used).
  - **No near-duplicate pair links two used pools.**

**Smoke tests** (`smoke.log`, `smoke_repro/repro.log`), run under the GPU lock:
- **Reproduction:** 5 DET-val images profiled with the A5 generator seeds, compared with the A5 rows. For all 14 modes,
  bytes are equal (5/5) and |ΔPSNR| = |ΔLPIPS| = 0. Detector true positives are equal (|ΔTP| = 0) for all 14 modes and
  for the original frame. The one-pass pipeline therefore reproduces A5 exactly.
- **Timing:**
  - 20 rl_train images (profile only): 3.8 s per image;
  - 5 F images (profile + detector): 5.4 s per image.
- **Resume:** rerunning the 20 images finds 20 of 20 cached and does nothing.
- **Phase 1 estimate:** 1,897 × 3.8 s + 2,326 × 5.4 s ≈ **5.5 h GPU**.

**Implementation changes relative to the A5 pipeline:**
- one pass per image, so the profiled reconstruction is the image the detector scores;
- generator seed = manifest `gen_seed`;
- no reuse of A5 rows;
- detector fidelity only in the recall pools.

**Full run:**
- `run_phase1.sh`, launched detached (WMI) through `fd_gpulock_win.py`, which holds the GPU lock;
- log `runs/uav_semcom/paperA_fd/phase1.log`, rows in `runs/uav_semcom/paperA_fd/profile/profile.jsonl`.

## 2. Results (run 23/11 23:18 – 24/11 06:15, EXIT 0; summary 06:17)

**Sources:**
- `runs/uav_semcom/paperA_fd/phase1.log`, `after_phase1.log`;
- `phase1_summary.log` / `.json` (from `fd_phase1_summary.py`).

### 2.1 Completeness and integrity

**Config:** each of the six pool runs and the final coverage pass logged config `f9d361f37bc386c8`. `profile.jsonl`
holds 61,448 rows, all with that config.

**Coverage and integrity:**
- **Accepted unique records: 61,448 of the expected 61,448** (4,223 × 14 + 2,326 originals).
- No image had more than one attempt.
- Every image has a complete attempt.
- Missing rows: 0 in every pool and mode.
- Non-finite PSNR: 0.
- Mismatches against the manifest: 0 for sha256, 0 for split, 0 for gen_seed (seeded modes equal the manifest seed;
  all other modes have none).
- No failure, no imputation.

**Artifact hashes** (sha256 after CRLF→LF normalisation):

| File | sha256 |
|---|---|
| manifest_fd_v1.json | `f0760601515f9663f2b7747bd247eddabc3ab8c997755c73ae8bf2f2339d4d50` |
| profile/profile.jsonl | `078e0e231dfeadaf9c9c84f8db2cb2a0d2c4166e4c3470cd35907d10f898781c` |
| profile/cpu_learned.jsonl | `d40ed56853b98c06160b027726df3c0ff07c00955d49a704933f8aa18b3b95ba` |

### 2.2 Runtime

| Pool | Images | Mean wall time per image | Total |
|---|---|---|---|
| rl_train | 1,523 | 6.40 s | 2.71 h |
| rl_val | 374 | 6.72 s | 0.70 h |
| F | 168 | 5.58 s | 0.26 h |
| TD | 582 | 5.69 s | 0.92 h |
| TD_extra | 1,028 | 5.57 s | 1.59 h |
| V | 548 | 4.59 s | 0.70 h |

- The recall pools include the detector.
- The total is 6.9 h, against the 7.7 h estimate.

### 2.3 Latency used by the simulator (means over rl_train)

| Mode | Encode, profile device | Encode, UAV CPU (6 threads) | Table encode | A5 table encode | Decode (BS) | A5 decode |
|---|---|---|---|---|---|---|
| mbt2018 q1 | 0.106 s (GPU) | **0.374 s** | 0.374 s | 0.463 s | 0.119 s | 0.068 s |
| mbt2018 q3 | 0.097 s (GPU) | **0.386 s** | 0.386 s | 0.476 s | 0.116 s | 0.068 s |
| mbt2018 q5 | 0.369 s (GPU) | **0.735 s** | 0.735 s | 0.912 s | 0.419 s | 0.110 s |
| mbt2018 q8 | 0.364 s (GPU) | **0.718 s** | 0.718 s | 0.894 s | 0.544 s | 0.113 s |
| avif 5 / 20 / 50 / 80 | 0.091 / 0.118 / 0.177 / 0.176 s (CPU) | — | same | 0.072 / 0.091 / 0.134 / 0.129 s | 0.006–0.009 s | 0.004–0.007 s |
| jpeg2000 100 / 25 | 0.276 / 0.275 s (CPU) | — | same | 0.198 / 0.199 s | 0.028 / 0.044 s | 0.020 / 0.031 s |
| cap_t2i, cap_thumb i2i | 0.24–0.27 s (GPU caption) | — | same | 0.21–0.24 s | 0.45–0.73 s | 0.41–0.50 s |
| thumb64_up | 0.017 s | — | same | 0.016 s | 0.008 s | 0.006 s |

**Diagnostic (not used):** the 3-thread CPU file gives 0.53 / 0.53 / 1.04 / 1.02 s for MBT2018 q1 / q3 / q5 / q8,
about 40% slower than the 6-thread A5 protocol. Separating the two measurements was therefore needed.

**Issue for the Phase 2 pre-registration: representativeness of the CPU sample.**
- The frozen A5 protocol times the first 10 rl_train images in manifest order. Here these are 3 frames at 960×540 and
  7 at 1360×765, a mean of 0.88 MP.
- The rl_train mean is 1.36 MP, and 43% of rl_train frames are 1400×1050 or 1920×1080.
- Encode time grows with pixel count, so the table's MBT2018 encode time likely **underestimates** the rl_train mean.
- This contrasts with every other codec's latency, which is a mean over all rl_train frames.
- Options for the reviewer:
  - (a) keep the frozen A5 rule and state the limitation;
  - (b) a pre-registered resolution-stratified CPU sample, for example 2 frames from each of the 5 main resolutions
    in proportion, run before any Phase 2 test.
- In both A5 and fd_v1, the simulator uses one encode time per mode, not per image. That simplification is unchanged.

### 2.4 Probe-predictor error

Mean absolute error over (image, non-probe mode) pairs. "bits" is the multiplicative factor exp(MAE of log bits).
"seq-w" weights each sequence equally.

| Pool | New fit (rl_train): bits / PSNR | seq-w | Old A5 fit (304 DET-val frames): bits / PSNR | Out-of-sample for |
|---|---|---|---|---|
| rl_train | ×1.190 / 1.69 dB | ×1.202 / 1.86 dB | ×1.205 / 1.80 dB | old fit only |
| rl_val | ×1.193 / 1.71 dB | ×1.218 / 2.17 dB | ×1.235 / 2.10 dB | both |
| F | ×1.158 / 1.44 dB | ×1.184 / 1.75 dB | ×1.162 / 1.44 dB | both |
| TD | ×1.166 / 1.47 dB | ×1.183 / 1.77 dB | ×1.181 / 1.71 dB | both |
| TD_extra | ×1.175 / 1.47 dB | ×1.149 / 1.21 dB | ×1.183 / 1.67 dB | both |
| V | ×1.177 / 1.73 dB | ×1.181 / 1.77 dB | ×1.146 / 1.54 dB | new fit only (the old fit was trained on V frames) |

- On the pools that are out-of-sample for both fits (rl_val, F, TD, TD_extra), the new fit is equal or better:
  - payload error ×1.16–1.19 against ×1.16–1.24;
  - PSNR error 1.44–1.71 dB against 1.44–2.10 dB.
- The old fit's advantage on V is in-sample and is not interpreted.

### 2.5 Status

- **Phase 1 is complete.** All mechanical checks pass.
- **Open decision for the review:** the CPU-sample rule (section 2.3).
- **Next deliverable:** the Phase 2 pre-registration. Phase 2–4 have not been started.

## 1b. Checks after the review of 33aae4a9 (23/11, 23:30; the run continues)

1. **Config hash.**
   - The running job uses config `f9d361f37bc386c8`.
   - Recomputing `fd_profile.config()` from the working tree gives the same hash.
   - The last change to `fd_profile.py` (`e5b9a5cd`) was committed at 23:18:06; the run was launched at 23:18:19.
   - None of the hashed files (`fd_profile.py`, `imcodecs.py`, `metrics.py`, `a5_fidelity.py`, `profile_a11.py`)
     differs from git, and none will be edited until the run ends.
   - At 23:29, `profile.jsonl` held one config and no duplicate (image, mode, config) key.
2. **Learned-codec UAV latency.**
   - `env.ModeTable` replaces the learned codecs' encode time by a CPU measurement. The A5 file was measured with
     `torch.set_num_threads(os.cpu_count())` = 6 threads.
   - The `fd_profile.py cpu` step of `run_phase1.sh` would measure with torch's default thread count, which is 3 in this
     WSL. This would be a silent protocol change. Its output is therefore kept as `cpu_learned_default_threads.jsonl`
     and not used.
   - `fd_cpu_timing.py` repeats the A5 protocol: 6 threads, warm-up image discarded, first 10 `rl_train` images. It
     runs after the GPU job (`run_after_phase1.sh`, a detached waiter), so the CPU timing is not disturbed.
   - The fd_v1 ModeTable will use these CPU encode times for MBT2018. The GPU timings of the main profile (MBT2018
     encode 0.07–0.28 s) are reported separately and never used as the UAV encode time.
3. **Record integrity** (`fd_phase1_summary.py`):
   - exactly one accepted record per (image, mode, config);
   - if a resumed image was written twice, the last complete write is kept, deterministically, and the raw file is not
     changed;
   - sha256, split and gen_seed must match the manifest;
   - coverage of the 14 modes, plus the original frame in the recall pools;
   - non-finite PSNR is counted.
4. **Near-duplicates.** The wording is "no near-duplicate pair between used pools *at the tested threshold* (dHash and
   aHash both ≤ 20/256)". This is evidence, not proof of scene independence.
5. **Predictor comparison.**
   - The old predictor was fitted on 304 DET-val frames, so its error on V is not out-of-sample.
   - Comparisons are made on rl_val, F, TD and TD_extra.
6. **The A5 generative discrepancy** (test-dev PSNR and recall from differently seeded samples) is documented. The old
   metrics are not treated as paired measurements of one output. E1–E7 are unchanged.

## 1c. Aggregation corrections after the review of ab2860e0 (23/11, 23:45; the run continues)

**Changes to `fd_phase1_summary.py`** (post-processing only; the profiler is untouched):
- **Seed check.** `gen_seed` must equal the manifest seed for cap_t2i, cap_thumb64_i2i_t0.6 and cap_thumb128_i2i_t0.3,
  and must be `None` for every other mode, including `original`.
- **Reconciliation by image-level attempt.** The rows of an image are written as one contiguous block, so each
  contiguous run of an image's rows is one attempt.
  - The last *complete* attempt (every required mode exactly once) is accepted as a whole. Modes are never mixed from
    different attempts.
  - Images with more than one attempt are listed, together with a bytes-consistency check across complete attempts.
- **Expected count** of accepted unique records: 4,223 × 14 + 2,326 originals = **61,448**.
- **CPU timing sample.** The 10-image CPU timing is reported as a per-mode estimate from 10 rl_train images, with their
  resolutions listed next to the rl_train mix. It does not characterise every resolution.

**Runtime.** The time per image grows with resolution:
- 960×540: about 3 s; 1360×765: about 3.9 s; 1920×1080: about 11 s (GPU memory close to the 8 GB limit at that size);
- the recall pools add about 1.5 s for the detector;
- revised estimate: **about 7.7 h** in total, finishing around 07:00 on 24/11.
