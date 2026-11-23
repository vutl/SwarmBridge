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

## 2. Results

To be written when the run finishes:
- coverage per pool and mode;
- failures;
- timing;
- probe-predictor error per pool;
- artifact hashes.
