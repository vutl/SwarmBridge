# Paper A independent evaluation (namespace `paperA_ix`): revised pre-registration, DRAFT v1 (26/11)

**Status:** draft for review. Data acquisition and the integrity audit are done where possible (§1). No profiling,
detector run on new images, tuning, training or simulation has been done. Nothing in §2–§7 is executed until this
draft is approved and committed as final (`paperA-ix-prereg.md`, with its commit hash quoted in every later report).

**Reviews answered:** ChatGPT stage 1 review (26/11): focused scope of 10,200 episodes, data terms, independence audit,
freeze and order, detection, statistics correction, mode utilisation.

**Corrections to the stage 1 proposal (e3ae8998):**
- The fd_v1 ablation **confirmatory** family is {A1, A2, A4, A4b} (Holm over 4, amended). nc-H, A7 and A8 are
  supporting contrasts.
- A2 is **band only − unguarded (nominal)**, not guarded − nominal.

The stage 1 file is kept unchanged as history.

## 1. Data acquisition and integrity audit (done 26/11)

### 1.1 X1: VisDrone2019-DET test-challenge

- **Source:** the official GitHub README link (Google Drive id `1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT`), downloaded to WSL
  `~/data/visdrone_tc/`.
  - zip sha256 `635e899e…a2da69`.
  - Contents: 1,580 JPEG files, images only (no annotation folder).
- **Terms:**
  - The README states: "Researchers can use test-dev to publish papers. testset-challenge is used for VisDrone2020
    Challenge and the annotations is unavailable."
  - It contains no licence text and no restriction on research use of the test-challenge images. The toolkit says
    "research purpose only".
  - The official guideline recommends test-dev for detection benchmarking. Our use is not detection benchmarking:
    - no detector score is computed on test-challenge;
    - the images feed communication simulation only (payload and PSNR against the image itself);
    - nothing is redistributed.
  - **Open:** the authors (user / professor) confirm this use is acceptable. Images and per-image artifacts stay out of
    git, as for all VisDrone data.
- **Audit** (`ix_audit.py x1`, output `runs/uav_semcom/paperA_ix/audit_x1.{json,log}`; names, ids, hashes only):

  | Item | Result |
  |---|---|
  | Frames / sequences | 1,580 / 53 |
  | Frames per sequence | 1 to 452; nine sequences with more than 30 frames |
  | Resolutions | 1400×788 712, 1400×1050 491, 1360×765 150, 1916×1078 127, 960×540 64, 1920×1080 36 |
  | Reference set (every VisDrone image on disk) | DET-train 6,471 + DET-val 548 + DET-test-dev 1,610 + VID-train/val 27,047 frames |
  | Sequence prefix shared with any DET split | 0 |
  | Identical sha256 | 0 |
  | Near-duplicate (dHash and aHash ≤ 20/256, the fd_v1 threshold) | 1 frame: `0000332_00600_d_0000028` ≈ DET-train `0000331_00601_d_0000844` (4 / 2 bits) |
  | Nearest-reference distance, max(dHash, aHash) percentiles 0 / 1 / 5 / 50 % | 4 / 56 / 66 / 85 bits |
  | Near-duplicate pairs within test-challenge | 0 |

- **Note on sequence ids.** VisDrone numbers sequences globally, and the splits interleave: 52 of 53 test-challenge
  sequences have a DET-train/val/test-dev sequence within ±2 ids. The pair 0000331 / 0000332 shows that one recording can
  span two ids and two splits. Id adjacency is therefore **not** used as an exclusion rule (it would exclude almost
  everything and is not evidence of shared content). Content hashes are the rule.

### 1.2 X2: UAVDT (UAVDT-Benchmark-M, DET/MOT part)

- **Sources:** the authors' project page, sites.google.com/view/daweidu/projects/uavdt:
  - README.txt: Drive id `1FV7zNa_…`, sha256 `72d63451…`;
  - DET/MOT toolkit `UAV-benchmark-MOTD_v1.0.zip`: Drive id `19498uJd…`, sha256 `1e565da8…`. This archive holds all
    ground truth: `GT/<seq>_gt_whole.txt` (DET), `_gt.txt` (MOT) and `_gt_ignore.txt`.
- **Images** (`UAV-benchmark-M.zip`, Drive id `1m8KA6oP…`): **not downloaded yet**. gdown hit the Drive download quota,
  as it did for VisDrone-VID on 10/11. The user downloads it in a browser from the official link (see the reply).
  The image-level hash audit (§1.3) runs before anything else touches the images.
- **Terms:**
  - The toolkit README says: "This code library is for research purpose only … We distribute our library under the
    GNU-GPL license. If you use this library or the dataset, please cite our paper [ECCV 2018]."
  - No separate dataset licence was found in the README.txt or the toolkit. Use: research, cite ECCV 2018, no
    redistribution of images or per-image artifacts. **Open:** authors confirm.
  - No toolkit code is executed (MATLAB / C sources and binaries are present; we only read the GT text files).
- **Ground truth facts** (read from the archive, `ix_audit` scratch reader):
  - 50 sequences; 40,421 annotated frames (maximum frame index per sequence, 265–2,035 per sequence).
  - DET format: `frame, id, left, top, w, h, out-of-view (1–3), occlusion (1–4), category`.
  - **Category: car (1), truck (2), bus (3). No van definition is given anywhere in the documentation.** Box counts:
    car 755,688, truck 25,086, bus 18,021.
  - Ignore regions are in `_gt_ignore.txt`, in the MOT format.
- **Official test split:** the 20 sequences with released baseline detections (`RES_DET/det_*`): M0203, M0205, M0208,
  M0209, M0403, M0601, M0602, M0606, M0701, M0801, M0802, M1001, M1004, M1007, M1009, M1101, M1301, M1302, M1303,
  M1401. This matches the paper (30 train / 20 DET-MOT test).
  - The paper (§2.1) says training videos "are taken at different locations from the testing videos, but share
    similar scenes and attributes".
  - The id groups (Mxx..) mix train and test sequences (for example M0201–M0210 contain 4 test sequences). Location
    independence therefore rests on the authors' statement and is reported as such.
- **Prior project contact:** none with UAVDT images. The Huang step 0 audit (04–06/11) counted UAVBench **question
  JSON files** that name UAVDT; no image or annotation was used.

### 1.3 Rules fixed now, applied mechanically after download

- **Exclusion rule (X1 and X2):** a sequence is excluded **entirely** if any of its frames has
  - an identical sha256 with, or
  - a near-duplicate (both hashes ≤ 20/256) of,
  any frame in the reference set (§1.1 table, plus every image of the other new dataset). For X2 test, the reference
  set also includes UAVDT train frames.
- **No other exclusion,** in particular none based on content, detector output, payload or any controller outcome.
- **X1 result (rule applied):** exclude 0000332. 52 sequences, 1,574 frames remain.
- **Minimum size:** X1 is reported as a confirmatory block if at least 15 sequences remain after exclusion (this is the
  number fixed in the stage 1 proposal; it was stated before the audit). Otherwise it is supplementary. **Met: 52.**
  X2 test: if more than 5 of the 20 test sequences are excluded, X2 is supplementary.

## 2. Pools and sampling (fixed now)

**Cap rule (fd_v1):** at most 30 frames per sequence, at positions `round(linspace(0, n−1, 30))` of the name-sorted
(X1) or frame-index-sorted (UAVDT) frame list. All frames of a sequence with ≤ 30 frames are used.

| Pool | Source | Sequences | Frames (cap 30) | Use |
|---|---|---|---|---|
| `X1` | test-challenge after exclusion | 52 | 631 | test, return endpoints only |
| `U_train` | UAVDT train sequences (30 − 4 val) | 26 | 780 | X2-retrained: table statistics, predictor, H\* tuning, RL training |
| `U_val` | UAVDT train sequences: sorted, permuted with `numpy.default_rng(0)`, first `round(0.15·30) = 4` (Python round) | 4 | 120 | X2-retrained checkpoint selection only |
| `U_test` | the 20 official test sequences | 20 | 600 | test (X2-frozen and X2-retrained), return + recall |

- UAVDT frames are numbered `img000001.jpg …` inside each sequence folder. The manifest stores
  `<seq>/<file>`, the frame index, the sha256 of image and annotation lines, resolution and object counts.
- Resolution is not rescaled (fd_v1 `--size 0`), and the network settings are not rescaled for the smaller UAVDT
  frames. The difference in payload and load is reported as a property of the dataset.
- **Generator seed** of the generative modes = position of the image in the `paperA_ix` manifest (pools in the order
  U_train, U_val, U_test, X1; names sorted).

## 3. What is frozen and what is refitted

### 3.1 X1 and X2-frozen (frozen VisDrone artifacts)

All controller-side, data-dependent quantities are taken from the fd_v1 table **as computed from fd_v1 rl_train**,
never recomputed from new rows:
- per-mode `enc_s` and `dec_s` (rl_train means + the CPU stratified MBT2018 estimates);
- `mean_bits` and `mean_psnr` (used by the MPC for future arrivals);
- probe predictor `coef_bits` and `coef_psnr` (avif_5 probe);
- H\* names (reference setting);
- RL checkpoints: not used in these two blocks;
- every MPC, gate, band, calibration, LA, scheduler and reward parameter.

**Implementation (to be reviewed before any run):** `ix_common.load(dataset, frozen=True)`.
- It builds the image table from the new profile rows (true outcomes: payload, PSNR, LPIPS, times, tp per image),
  then **copies** the controller-side arrays from `fd_common.load()`.
- It never calls `fit_info` and never averages over new rows.
- Unit tests:
  - (a) the copied arrays are bit-identical to fd_v1's, checked by hash;
  - (b) `fit_info` is not called on the new table (monkeypatched to raise);
  - (c) changing any new-row value leaves `predict`'s coefficients unchanged, while `predict(img)` uses the new
    probe payload of that image (the probe is executed on the UAV, as in fd_v1);
  - (d) an fd_v1 reproduction: loading fd_v1 through `ix_common` with fd_v1 rows gives byte-identical episode returns
    on 3 seeds.

### 3.2 X2-retrained (procedure re-applied on UAVDT train)

Refitted on **U_train only**, with the fd_v1 code paths and budgets:
- per-mode latency means;
- the MBT2018 CPU estimate (fd_cpu_stratified procedure on a U_train sample of the same size);
- `mean_bits` and `mean_psnr`; the probe predictor;
- H\* among the same 22 rules on training seeds, for the reference setting and each grid setting;
- RL: PPO, QR-DQN and Discrete SAC × seeds 0–2 × 2M decisions, fd_v1 hyperparameters;
- checkpoint selection on U_val with the fd_v1 rule (20 validation seeds, best validation return).

### 3.3 Detector

- Frozen VisDrone YOLO11s (`a5_det`, sha256 recorded in fd_v1) for every recall in X2. No fine-tuning on UAVDT.
- Fidelity is computed on **U_test only**, together with profiling. Detector outputs on U_test are not looked at
  before the lock in §4; only the integrity counts (rows, NaN) are checked.

## 4. Execution order (lock points)

1. Approve this draft, commit the final pre-registration (L0).
2. User downloads `UAV-benchmark-M.zip`.
3. Image hash audit for X2 and the manifest `paperA_ix` (pools §2). Commit the manifest hash (L1).
4. Implement and test `ix_common` (§3.1 tests). Review. Commit (L2).
5. Profiling (14 modes) of U_train, U_val, U_test and X1; fidelity on U_test. Integrity checks only (L3: profile
   hashes committed).
6. X2-retrained fitting: table statistics, CPU estimate, H\* tuning (training seeds), RL training, checkpoint
   selection on U_val.
7. **Lock L4:** commit sha256 of every X2-retrained checkpoint, the H\* names and the fitted predictor and statistics.
   This happens **before any test episode of any block** (X1, X2-frozen or X2-retrained).
8. Run all test blocks. Fail-fast verify (seed coverage, one result per seed / arm / checkpoint, hashes).
9. One analysis pass with the registered script. Report everything. No re-tuning, arm change or data change after
   step 8 begins.

## 5. Evaluation blocks (10,200 episodes)

Arms, network settings and learned incumbents are exactly those of fd_v1. Seeds are fresh.

| Block | Content | Episodes | Seeds |
|---|---|---|---|
| X1-ref | 6 Phase 2 arms (H\*, nominal, guarded, calibrated, RobustMPC-style, RobustMPC no band), reference setting | 600 | 30000–30099 |
| X1-abl | 8 ablation arms | 800 | 30100–30199 |
| X2F-ref | as X1-ref, on U_test, frozen | 600 | 31000–31099 |
| X2F-abl | as X1-abl, on U_test, frozen | 800 | 31100–31199 |
| X2R-ref | as X1-ref, on U_test, retrained | 600 | 32000–32099 |
| X2R-abl | as X1-abl, on U_test, retrained | 800 | 32100–32199 |
| X2R-grid | 6 settings × 4 arms (H\*, nominal, guarded, calibrated) | 2,400 | 32200–32299 |
| X2R-RL | 3 algorithms × 3 checkpoints × 4 arms (X, X+nominal, X+guarded, X+calibrated) | 3,600 | 32000–32099 (pairs with X2R-ref H\*) |

X2-frozen tests transfer of the heuristic-incumbent controller stack only. **No claim about zero-shot RL transfer**
is made.

## 6. Statistics

**Primary endpoint:** episode return J, paired over seeds; two-sided paired t-tests; 95% CIs.

**Supporting endpoints:**
- recall (X2 only): seed CI and sequence-cluster bootstrap, as in fd_v1;
- on-time, drop and quality-violation rates; mean PSNR and payload;
- mode utilisation (§7);
- decision time.

**X1 recall is "unavailable"** (no public labels), never zero and never imputed.

**Families.** Holm is applied within each block. Each X block is its own evidence line:

| Family | Blocks | Tests | Status |
|---|---|---|---|
| Reference replication | X1-ref, X2F-ref, X2R-ref (each separately) | H1 cal − H\*, H3b cal − matched nominal design, i.e. guarded MPC (Holm over 2) | replication of fd_v1 Phase 2 |
| Ablation replication | X1-abl, X2F-abl, X2R-abl (each separately) | A1, A2, A4, A4b (Holm over 4) | replication of fd_v1 ablation |
| **New: paper findings** | X1, X2F, X2R (each separately; contrasts from the ablation block) | N1 = nominal − H\* (nc-H), N2 = guarded − H\* (A8), N3 = calibrated − guarded (A7) (Holm over 3) | **newly registered confirmatory** (these are the contrasts the manuscript's findings rest on; supporting in fd_v1) |
| Grid | X2R-grid | G1_c, G2_c × 6 settings (Holm over 12) | replication of fd_v1 Phase 4 |
| Learned incumbents | X2R-RL | R1_X, R2_X × 3 algorithms (Holm over 6) | replication of fd_v1 Phase 3 |

- Supporting contrasts (A3, A5, A6, A9, H2, H2b, H3, R5, per-checkpoint contrasts) get unadjusted CIs and no
  confirmatory statement.
- **The two X2 conditions share U_test.** They are two analyses of one test set, not two replications. They are never
  pooled, and their agreement is not counted as independent evidence.
- **No cross-block multiplicity correction.** Each block answers its own question (§0 of the stage 1 proposal). Where
  the paper summarises several blocks, it says how many blocks support a finding and quotes each block's own Holm
  result.

**Reporting rules:**
- Every registered contrast in every block is reported, including reversals and non-rejections.
- A non-rejection is "no detected difference", never "equivalent".
- The manuscript keeps fd_v1 as "development / internal evaluation" and adds the X blocks as "independent
  evaluation".

## 7. Mode utilisation (observational)

For every block, arm and checkpoint, report:
- per-mode selection counts and shares;
- outcomes by mode: compliant / delivered on time below the floor / late or dropped / dropped before a mode decision;
- per-mode profile statistics: payload, PSNR, encode and decode time, recall on recall pools;
- modes never or rarely selected (≤ 0.5%), with the generative modes named.

RL training action counts are not logged by the fd_v1 trainer. For X2-retrained they are added as **logging only**
(action histogram per validation evaluation); this changes no training computation. For fd_v1 they are "not
recorded".

**fd_v1 result** (`ix_modes.py`, stored logs only; `runs/uav_semcom/paperA_ix/modes_fd_v1.{csv,log}`,
`modes_profile_fd_v1.csv`):
- Share of finished frames over all fd_v1 test blocks (7.18 M frames):
  - cap_t2i 0.22% (compliant 0 times: PSNR 9.6 dB is below every quality floor);
  - thumb64_up 1.11% (8.7% of them compliant);
  - cap_thumb64_i2i 0.10% (1.0% compliant);
  - cap_thumb128_i2i 0.23% (8.5% compliant).
- Reference TD, Phase 2:
  - guarded MPC: avif_20 77.9%, avif_5 9.9%, mbt q3 5.3%; all three generative modes and thumb64_up never selected;
  - nominal MPC: avif_20 30.2%, mbt q3 19.5%, avif_5 16.6%, avif_50 16.5%;
  - calibrated: avif_20 72.4%, avif_5 14.6%.
- The RobustMPC no-band variant is the only arm that uses a thumbnail mode heavily: thumb64_up 24.3%.
- Zero-shot PPO-LA checkpoints use 4–5 of 14 modes.
- Per-mode YOLO recall on F/TD/V (sum tp / sum n_gt):
  - original 0.582;
  - avif_80 0.579, mbt q8 0.585;
  - avif_20 0.476, avif_5 0.363;
  - cap_thumb128_i2i 0.137, thumb64_up 0.028, cap_thumb64_i2i 0.007, cap_t2i 0.003.
- **Reading (observational only):** the three generative modes deliver almost no detectable objects and rarely meet
  the PSNR floor. They are dominated in this mode set, and the controllers almost never choose them. No mode is
  removed or changed.

## 8. Cost (measured fd_v1 times)

| Work | Estimate |
|---|---|
| Profiling: 631 + 780 + 120 + 600 = 2,131 frames × 6.4 s | ≈ 3.8 h GPU |
| Fidelity on U_test: 600 × 4.5 s | ≈ 0.75 h GPU |
| CPU MBT2018 estimate (U_train sample) | ≈ 0.5 h CPU |
| H\* tuning (reference + 6 settings, training seeds) | ≈ 1–2 h CPU |
| RL: 9 runs, 2 in parallel | ≈ 13 h CPU |
| 10,200 test episodes | ≈ 3–4 h CPU |
| **Total** | **≈ 1.5–2 days** of unattended detached jobs, plus review |

## 9. Open items for the authors

1. UAVDT images: browser download of `UAV-benchmark-M.zip` from the official Drive link into
   `\\wsl$\Ubuntu\home\uservutl\data\uavdt\` (or any folder; tell Claude the path).
2. Confirm the data terms: test-challenge images used for communication simulation without labels; UAVDT for
   research with citation; nothing redistributed.
3. **Recall class mapping (decision needed; UAVDT documents no van):**
   - (a) **proposed primary:** class-agnostic vehicle recall. A UAVDT box (car / truck / bus) counts as detected when a
     YOLO detection of any VisDrone vehicle class {car, van, truck, bus} matches it at IoU ≥ 0.5 (greedy, by
     confidence, fd_v1 operating threshold). Non-vehicle detections are discarded. Detections with ≥ 50% of their area
     inside an ignore region are removed.
   - (b) **secondary:** class-aware recall with car ← {car, van}, truck ← truck, bus ← bus.
   - (c) **secondary:** class-aware recall with car ← car only.
   - (b) and (c) bracket the undocumented van question.
   - IoU 0.5 is the paper's system-level metric, not the UAVDT AP protocol (which uses IoU 0.7). Original-image recall
     under (a)–(c) is reported first.
