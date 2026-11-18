# Dataset feasibility audit: synchronized multi-UAV views with cross-view identity (2026-11-11)

Problem under test: several UAVs observe overlapping ground areas at the same time. Each UAV decides what perception information to send to a base station (BS) over a limited uplink, exploiting redundancy (the same object seen by several UAVs) and complementarity.

The hard rule: independent single-view videos (VisDrone etc.) are NOT treated as synchronized views. Ground truth for "redundant vs unique object" needs synchronized views plus the same physical object carrying one identity across views.

This audit resolves the open cells of `reports/archive/r1-visdrone/r0-notes/collaborative-swarm.md` section 2 and adds 2024-2026 datasets. Method:
- Landing pages, GitHub (via the REST API), HuggingFace and Figshare/Dryad APIs, arXiv HTML.
- No dataset bulk was downloaded.
- One small label sample was fetched: the MDMT test ground truth, 28 txt files, 16 MB, publicly in the GitHub repo. Saved to `C:\Users\USER\AppData\Local\Temp\claude\434d0fc3082c421a\211e39df-8890-414c-a6ef-90f1625f6d0a\scratchpad\datasets_meta\mdmt_test_gt\`, analysed with `...\scratchpad\mdmt_overlap.py`.
- `swarmbridge/experiments/uav3d_comm/` and `runs/active_recovery/` were NOT opened.

"nf" = not found in the sources checked.

## 0. New measured fact: MDMT cross-drone redundancy (from the public test GT)

What the files are:
- The MDMT repo ships the test-split ground truth in `demo/eval/test/<seq>-1.txt` / `<seq>-2.txt` (drone 1 / drone 2): 14 pairs, MOT format `frame,id,x,y,w,h,0,class,vis`.
- The official MDA evaluation script (`demo/eval/mango_eval_MDA_perImage.py`, lines 148-152) builds the ground-truth cross-drone pairs as `gtA.ID == gtB.ID`. So **an identical ID in both drones' files = the same physical object**. Cross-view identity is given, not derived.

Statistics computed by us on all 14 test pairs:

| Quantity | Value |
|---|---|
| Synchronized frame pairs | 5,883 (both drones have identical frame counts per sequence, 150-700 frames) |
| Boxes | 600,923 (about 51 per frame per drone) |
| **Redundant fraction**, per frame: objects visible in both drones / union of objects | **0.456 pooled**; per sequence 0.23 (seq 26) to 0.83 (seq 31) |
| IDs seen by both drones / all IDs | from 45/80 to 151/158 per sequence |
| Class field | 3 values: 418K / 155K / 28K boxes. The repo config lists ("person", "car", "bicycle"); the exact id-to-name mapping is not documented |
| Visibility field | 1 (443K) or 0.5 (158K = occluded) |
| Box area (px^2 at 1920x1080) | p10 = 209, p50 = 1,078, p90 = 5,049 |

So MDMT spans the whole useful range: mostly-redundant pairs (0.6-0.8) and mostly-complementary pairs (0.2-0.4). This is exactly the regime the problem needs.

This also resolves the r0 box-count conflict: about 51 boxes per frame x 39,678 frames is consistent with the repo's 2.2M boxes. The "220K" figure elsewhere is M3OT's count.

## 1. Feasibility matrix

| Dataset | Real/sim | #views (aerial) | Sync | Poses / intrinsics | 2D boxes per view | Classes | Track IDs | Cross-view ID | Overlap | Size | Access | License | UAV3D-related? | Verdict for our problem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MDMT** (Liu et al., TMM 2023), github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking | Real (2 drones) | 2 | Yes, paired sequences with equal frame counts (verified on test GT); fps nf | **No.** Issue #27 (2025-10-11) asks for extrinsics/intrinsics, unanswered | Yes (MOT txt; XML in the full release per issue #4; COCO json per view in branch `tba`) | 3 (person, car, bicycle). CARNet (arXiv 2610.26312) says the original labels lump all vehicles into one class | Yes | **Given** (same ID across drones; MDA script uses `gtA.ID == gtB.ID`) | **Measured:** 0.23-0.83 of per-frame objects shared, pooled 0.46 (test split) | 88 sequences (44 pairs), 39,678 frames, 11,454 IDs, 2.2M boxes, 25.8 GB | Images: Baidu Netdisk only (pwd 9un6), uncompressed small files. Issues #8/#20/#22 (2023-2026) report registration and download pain and ask for GDrive, unanswered. **Test GT (14 pairs, 16 MB) is directly on GitHub** | **None** (no LICENSE file; GitHub API `license: null`) | No (real) | **Usable-with-work.** Only real dataset with verified cross-drone IDs and a quantified redundancy mix. Risks: Baidu download, no license, N = 2 only, label noise (CARNet re-annotated as MDMT-CD; the "release in future" note is dated 2026-09-10 and has not happened), no poses, `tba` json filenames mismatch images (issue #30) |
| **VisionTrack** (GMT, arXiv 2407.01007; CVPR 2026), github.com/FoxCanned/GMT | Real (2 moving drones) | 2 | Manual time alignment; 30 fps | Not mentioned | Yes (annotations json in the repo's data links) | Pedestrians (the paper frames the task as pedestrian tracking; class list nf) | Yes | **Given:** "targets visible in both views ... shared IDs"; one ID per target across both cameras | "Overlapping fields of view", not quantified | 15 scenes, ~116K frames, ~1.18M boxes, 1920x1080; GB nf | **OneDrive + Baidu** (code GMTT). Links added in a 2026-07-05 commit "Release VisionTrack"; the README still says "will be released soon"; OneDrive folder contents not verifiable without a browser | Dataset: nf (code repo Apache-2.0; the paper notes subject consent "for scientific research") | No | **Usable-with-work** (second real option). Not Baidu-only. But: pedestrians only, N = 2, no poses, the repo self-declares its code "unusable" (2026-11-05 notice), and it has not been inspected by us |
| **MDMOT** (FusionTrack, arXiv 2505.18727) | Real (DJI Mini 3 Pro fleet) | 3-5 | Start/end alignment using drone timestamps; fps nf | nf | Yes | nf (vehicles implied) | Yes | **Given** ("manually associate ... to guarantee identity consistency") | Mixed: partially overlapping and fully non-overlapping clips | 20 sequences, ~122K frames, 2.48M or 24.81M boxes (internal 10x inconsistency), 30-100 m, 1920x1080 | **No link.** arXiv v1 only, no code or data URL; no GitHub repo found (API search) | nf | No | **Unusable now** (only N > 2 real option; would need an e-mail request to the authors) |
| **M3OT** (Sci. Data 2025), figshare 10.6084/m9.figshare.28308887 | Real (2x DJI Mavic 3T, RGB+IR) | 2 | "Manually aligned", same start/end per pair; fps nf | Partial: camera spec table (focal length, sensor) only; no poses/GPS | Yes, per drone and per modality (MOT gt.txt + COCO json) | Vehicles (all < 1024 px^2, 640x512) | Yes | **Not stated.** "Different ID for each target ... no duplicate ID labels"; cross-drone linkage undocumented. Derivable via a ground-plane homography per frame pair (100-120 m, near-planar) | nf | 21,580 frames, ~220K boxes; **single 10.74 GB zip** | Figshare direct download (published 2025-10-23) | **CC BY 4.0** (figshare API) | No | **Usable-with-work.** Cleanest access and license, but cross-view ID must be built and validated by hand. Tiny objects plus the RGB/IR 640x512 sensor make matching error-prone |
| MDOT (Zhu et al., TCSVT 2021) | Real | 2 / 3 | Yes | nf | Only the single target | 1 target per clip | n/a | Target only | Yes | Two-MDOT 92 groups, Three-MDOT 63 groups | Baidu (pwd 1895) | nf | No | **Unusable** (single-object; no redundant/unique object population) |
| CoPerception-UAVs (Where2comm, NeurIPS 2022), siheng-chen.github.io/dataset/coperception-uav | Sim (AirSim + CARLA, 3 towns) | 5 UAVs x 5 cams (BEV + 4 at -45 deg, 800x450, 90 deg FoV) | Simulator-synchronous; sample every ~4 s (r0, paper p.22-24) | Yes (global 3D boxes; camera params in sim) | Yes (projected from 3D) + semantic cameras | Vehicles | nf (no tracking stated) | **Derivable** from global 3D boxes (per-instance token not verified) | High (formation flight) | 131.9K images; GB nf | Full set Baidu only; sample on Google Drive | nf (site says "Copyright 2022 SJTU MAGIC"); repo no license | **Adjacent:** UAV3D uses the identical rig (5x5 cams, BEV + 4 at -45 deg, 800x450) | Usable-with-work (sim). Baidu-only; reviewers will see it as the UAV3D family |
| CoPerception-UAVs+ (CoCa3D, CVPR 2023) | Sim | 10 | Sim | Sim | nf | nf | nf | Derivable from 3D | Yes | nf | Not checked beyond repo (MIT badge for code) | nf | Adjacent | Not prioritized |
| **Air-Co-Pred** (DHD, NeurIPS 2024), github.com/WangzcBruce/DHD | Sim (CARLA Town10) | 4 drones at 50 m, ~100x100 m | 2 Hz, synchronous | Yes (nuScenes calibrated_sensor/ego_pose) | Via projection (r0: 2D/3D labels) | Vehicles, cycles, pedestrians | Yes (nuScenes instances over 200 scenes) | **Derivable exactly** (nuScenes `instance_token` + projection/visibility per drone camera) | High (all drones cover the same area) | ~32K images 1600x900, ~430K 3D boxes; **HF repo 39.5 GB** | **HF datasets/BRUCE-WZC/AIR, direct, not gated** (updated 2025-10-30); Baidu mirror | nf (HF repo has no license tag; GitHub no license) | **Adjacent:** CARLA Town10, same simulator family as UAV3D (UAV3D: Towns 3/10/6/7) | **Usable-with-work** (best-accessible sim option for N = 4) |
| AeroCollab3D (UCDNet, arXiv 2406.04648) | Sim (CARLA, 8 maps) | 6 (one cam at -45 deg, 50 m) | Sim | Sim | Via projection | 4 | No | Derivable | Yes | 3.2K samples, 19.2K images, 218K 3D boxes | nf (AirCopBench reuses it, so some copy exists) | nf | Adjacent (CARLA; same CAS-AIR group as Air-Co-Pred) | Unusable now (no access) |
| AirCo-MultiTasks (RS-DFM, arXiv 2406.07032) | Sim (CARLA) | 5 | Synchronous | Sim | 2D/3D | nf | nf | Derivable | nf | 38K images 1600x900, 200 scenes | nf (not verified) | nf | Adjacent | Unusable now (no access found) |
| U2UData (ACM MM 2024) / U2UData+ (AAAI 2026), github.com/fengtt42/U2UData | Sim (UE5 "U2USim") | 3 / 15 | Sim | Per-UAV coordinates | Not stated (3D-centric) | 3 animal classes | Tracking | Derivable from 3D | Formation-dependent | Claimed 3.62 TB (U2UData+). **HF fengtt42/U2UData-2 holds only ~62 GB** ("Scene_Sunny_Rain_drone_1..15.rar") | HF direct; Baidu by e-mail | HF tag Apache-2.0 | No (UE5, wildlife) | Usable-with-work but off-domain (wildlife, LiDAR-centric, partial release) |
| AirV2X-Perception (NeurIPS 2025 D&B), github.com/taco-group/AirV2X-Perception | Sim (CARLA + AirSim, Towns 1-4/6/7/12) | 3-5 drones (1 down-facing cam 1280x720 + LiDAR) + vehicles + RSUs | 5 Hz | Sim (metadata.pkl) | Not mentioned | Vehicles | 3D tracking | nf, likely derivable | nf | 6.73 h; 1.96M objects | **HF only has `mini/` (0.31 GB, last modified 2025-05-15)**; full set not found | HF tag CC BY 4.0 | Adjacent (CARLA + AirSim) | Unusable now (full data not public) |
| Griffin (AAAI 2026), HF wjh-svm/Griffin | Sim (CARLA + AirSim; Town03/10HD/06/07) | **1 drone** (5 down-oriented cams) + 1 vehicle | CARLA synchronous mode | Yes | No (3D only) | 6 | Persistent IDs | Drone-vehicle, derivable | n/a | 37.7K frames, 339K images; HF 973 GB | HF direct + Baidu | MIT (HF tag) | **Adjacent:** same 4 towns as UAV3D | Unusable (single UAV) |
| AGC-Drive (NeurIPS 2025 D&B), github.com/PercepX/AGC-Drive | Real | **1 UAV** + 2 cars | Hardware sync | Yes | Not mentioned | 13 | nf | nf | n/a | 350 scenes, ~80K LiDAR frames, 360K images | Baidu (lidar_only); **PNG images "coming soon"** | nf | No | Unusable (single UAV; images unreleased) |
| V2U4Real (CVPR 2026, arXiv 2603.25275) | Real | 1 UAV + 1 vehicle | Yes | Yes | nf | 4 | Yes | nf | n/a | 56K LiDAR frames, 700K 3D boxes | Not checked | nf | No | Unusable (single UAV) |
| CoPeD (RA-L 2024), github.com/arplaboratory/CoPeD | Real | 2 aerial + 3 ground robots | Yes (ROS) | GPS + SLAM poses | Auto-generated 2D labels from baseline models | nf | nf | No | Low / unclear | nf | GitHub | nf | No | Unusable (no human GT, no cross-view ID) |
| **MATRIX** (arXiv 2511.08615), github.com/KostaDakic/MATRIX | Sim (UE 5.2 + AirSim/Colosseum) | 8 drones (moving) | 2 fps synchronized frames, 1920x1080 | **Yes, per-timestep intrinsics + extrinsics** | Yes | Pedestrians only (40) | Yes (world positions) | **Given** (2D box matchings + POM ground-plane positions + per-camera LoS) | 30x30 m area, high | 1,000 timesteps, 8,000 frames | Google Drive (direct) + **generator code public** | nf (no license on repo) | No (UE5, pedestrians) | Usable-with-work as a **generator**; the released set is tiny and toy-scale |
| MCOP sets (ICCV 2025) | Sim + real GauUScene-Occ | 4-5 | Yes | Yes | No (occupancy) | n/a | No | No | Yes | nf | nf | nf | **UAV3D-Occ is derived from UAV3D** | Unusable |
| AirSim-MAP (When2com) | Sim | 5-6 | Yes | Yes | No (segmentation only) | n/a | No | No | Partial | nf | Repo | nf | No | Unusable |
| AirCopBench (arXiv 2511.11025) | Mixed: reuses CoPerception-UAV, AeroCollab3D, EmbodiedCity, MDMT | 2-6 | "Simultaneous" | nf | Yes | Varies | n/a | Object-matching QA only | n/a | 2,920 multi-view images, 14.6K VQA | embodiedcity.github.io/AirCopBench | nf | No | Unusable (VQA benchmark) |
| MAVREC (CVPR 2024) | Real | 1 drone + 1 ground camera | Time-synchronized | nf | Yes | Several | nf | nf | Partial | 0.5M frames, 1.1M boxes | mavrec.github.io | nf | No | Unusable (one aerial view) |
| DIVOTrack (IJCV 2023) | Real | 3 moving cameras; in v1 one was a drone ("Drone, View1, View2") | Manual sync | nf | Yes | Pedestrians | Yes | **Given** (953 cross-view tracks) | High | 15 scenes | HF syhao777/DIVOTrack; zip password after e-mailing a signed license | Custom research license | No | Analog only (1 aerial view, ground-dominant) |
| WILDTRACK / MultiviewX / CityFlow | Real / Unity sim / real | 7 static / 6 / 40 ground cams | Yes | Yes (calibrated) | Yes | Pedestrians / pedestrians / vehicles | Yes | Given (ground-plane position IDs) | High / high / mostly non-overlapping | 400 annotated frames / 400 / large | Public / GitHub / agreement | Research licenses | No | Analog only (not aerial; acceptable as a sanity benchmark, not as the main claim) |
| MMOT (NeurIPS 2025 D&B) | Real | 1 | n/a | n/a | OBB | 8 | Yes | n/a | n/a | 125 sequences | HF | CC BY-NC-ND 4.0 | No | Unusable (single view) |
| SkyScenes (ECCV 2024) | Sim (CARLA) | Single camera, 12 height/pitch settings | n/a | Yes | No (segmentation/depth) | 28 | No | No | n/a | 33.6K images | HF | MIT | Adjacent (CARLA) | Unusable |
| Cross-View Urban Traffic (arXiv 2606.07708) | Real | 1 drone + bicycle camera | Yes | nf | Yes | Traffic | Yes | Street-to-drone matches | n/a | nf | nf | nf | No | Unusable (one aerial view) |
| Not found | | | | | | | | | | | | | | "MUSt", "AirCo" (other than AirCo-MultiTasks), and a TUMTraf variant with UAV views were not found. The TUMTraf family is roadside/vehicle |

### UAV3D overlap flag

- None of the real options (MDMT, VisionTrack, M3OT, MDMOT) is UAV3D-related.
- Every CARLA multi-UAV option is UAV3D-adjacent:
  - CoPerception-UAVs has the identical camera rig.
  - Griffin uses the identical town set.
  - Air-Co-Pred uses Town10.
  - MCOP's UAV3D-Occ is UAV3D-derived.
- If our main claim rests on CARLA multi-UAV data, it lands in the same evaluation family as the other team's UAV3D work.

## 2. Communication-side data

| Source | Type | Link | Content | Access / license | Fit for driving a link simulator |
|---|---|---|---|---|---|
| DeepSense 6G Scenario 23 (local PDF, Table I; deepsense6g.net) | **Measured** | Drone to BS (A2G), 60 GHz, BS with 16-element ULA, 64-beam codebook over 90 deg | Per sample: 64-entry received-power vector, BS-side RGB, drone GPS, height, speed, orientation | Public website. The scenario page did not render for our fetcher. License not verified | Beam/power traces for **one** drone; usable as an SNR proxy for a mmWave uplink. No rate/BLER, no multi-UAV contention |
| AERPAW Ericsson 5G NSA KPI (Dryad 10.5061/dryad.wh70rxx06) | **Measured trace** | A2G, LTE band 66 + NR n77; UAV zigzag at 30 m (2023-09-15) | Time-stamped RSRP, RSRQ, SINR, CQI, MCS, RI, cell ID, **throughput (Mbps)**, UAV lat/lon/alt | 1.3 MB, direct; Dryad site terms (Dryad normally publishes CC0; not stated on the page) | **Best trace for a rate-vs-time uplink/downlink emulator.** Single UAV, single altitude, rural |
| AERPAW LTE I/Q (arXiv 2303.01635; IEEE DataPort) | Measured raw I/Q | A2G LTE, 5 altitudes 30-110 m | I/Q (SigMF) + GPS; RSRP after post-processing | IEEE DataPort (account needed); license nf | Altitude-dependent RSRP; needs DSP to get SNR |
| AERPAW AFAR challenge data | Measured + digital twin | A2G, 3.51 GHz SDR eNB, 30-110 m | RSS/RSQ, GPS, velocity, orientation | Per the AERPAW catalog (arXiv 2510.08752); license nf | Signal quality vs position |
| AERPAW A2A channel sounding (Dryad 10.5061/dryad.z34tmpgws, 2026-08-27) | **Measured A2A** | UAV to UAV, 3.4 GHz, 56 MHz | 8,883 Zadoff-Chu I/Q captures, received power, RMS delay spread, `uav2uav_dist`, altitudes | 2.03 GB, Dryad "licensed for reuse" | Only public measured sub-6 GHz U2U set found; relevant only if U2U relaying is in scope |
| 60 GHz UAV-to-UAV (Polese et al., mmNets 2020), github.com/wineslab/uav-to-uav-60-ghz-channel-model | Measured | A2A, 60 GHz Terragraph sounders | dataset.csv path loss + fitted model | GitHub, GPL-3.0 | Path-loss model for mmWave U2U |
| AERPAW catalog (arXiv 2510.08752) | Catalog | A2G only (no A2A or multi-UAV listed in the first 100K characters read) | Spectrum, LoRa, channel sounding at 30/60/90 m, TDOA, emulated RSRP/throughput CSVs | Dryad / AERPAW site; licenses "not stated" | Index for further traces |
| 3GPP TR 36.777 (Rel-15 aerial UE) | **Model** | A2G cellular: UMa-AV, UMi-AV, RMa-AV, UAV height up to 300 m | LoS probability, path loss, shadowing, fast fading (Annex B) | 3GPP, free | Standard for "UAV to cellular BS"; reviewers accept it |
| 3GPP TR 38.901 (Rel-19 ISAC extension) | Model | UMa-AV reused for UAV sensing targets (25-300 m) | Background + target channel | 3GPP | Sensing, not uplink; context only |

**No measured dataset has several UAVs sharing one uplink simultaneously.** Multi-UAV contention and scheduling must be modeled: TR 36.777 path loss plus a MAC/scheduler model, optionally modulated by the AERPAW throughput trace. That is defensible, but it is a model, and the paper should say so.

## 3. Recommendation

### Option A (primary, real): MDMT, 2 drones, cross-view ID given

Why:
- Verified synchronized pairs.
- Verified identical-ID-means-same-object semantics.
- A measured redundancy spread from 0.23 to 0.83 gives natural "mostly redundant" vs "mostly complementary" strata.
- The 14-pair test GT is already in hand from GitHub. Redundancy statistics and an oracle dedup upper bound can be computed today, without Baidu.

Work needed:
1. Download the 25.8 GB from Baidu. This needs a Baidu account; a slow but known path. Fallback: ask the authors (issues #8/#20 show no GDrive reply since 2023).
2. Re-split by sequence pair, with no pair crossing splits.
3. Run a label-noise audit: spot-check about 200 cross-drone ID pairs per class. CARNet reports errors, and MDMT-CD is unreleased.
4. Treat the class mapping as {1, 2, 3} to be confirmed visually.
5. Confirm the license with the authors before any redistribution. Use only, no redistribution.

Risks:
- N = 2 only, so "several UAVs" degenerates to pairs.
- No poses, so no geometry-based dedup baseline unless image homography is used.
- Label errors. The dataset is a Chinese-campus scene set with only 3 classes.

### Option B (real, second source / replication): VisionTrack (2 drones, pedestrians) or M3OT (2 drones, vehicles, CC BY 4.0)

- **VisionTrack** has given cross-view IDs and a OneDrive link. It is worth a 1-day inspection: check that the OneDrive folder holds images plus json and that the IDs are shared across the two cameras.
- **M3OT** has the cleanest license and a direct Figshare download, but needs a built cross-view ID. Procedure:
  - Per frame pair, estimate a ground-plane homography with feature matching (e.g. SIFT/LoFTR + RANSAC on RGB, not IR).
  - Project box centers and match by Hungarian assignment with a distance gate.
  - Smooth over tracks: one cross-view ID per track pair, by majority vote.
  - Validate against a manually labelled subset of at least 300 pairs, and report precision/recall of the built GT.
  - Risks: objects under 1024 px^2 at 640x512, parallax near buildings, night/IR frames.
- Using two real datasets (MDMT plus one of these) answers the "single dataset" criticism.

### Option C (simulation, for N >= 3): Air-Co-Pred (4 drones, HF direct 39.5 GB, nuScenes)

- Cross-view ID is exact via `instance_token`. Project 3D boxes into each drone camera and use visibility/occlusion to label each object as redundant (visible in 2 or more drones) or unique.
- Cheapest credible sim route: no generation needed, direct download, standard devkit.
- If custom formations or altitudes are needed, the MATRIX generator (UE5 + AirSim, public code, per-timestep calibration, 8 drones) is the only multi-drone generator with published code we found. It is pedestrian-only and 30x30 m as released. CoPerception-UAVs, Griffin, AirV2X and UAV3D publish data or converters but no generation scripts (repo trees checked).

Sim-only reviewer risk:
1. CARLA Town10 multi-UAV vehicle detection is the UAV3D/CoPerception-UAVs evaluation family, so there is overlap with the other team's UAV3D line and with Where2comm-style prior art.
2. Sim-to-real gap: perfect synchronization, no GPS/pose noise, clean rendering. Redundancy statistics are set by the chosen formation, not by real operations.
3. The comm side would also be modeled, so an all-synthetic pipeline is easy to dismiss.

Use sim only for the N-scaling study. Anchor the headline claims on MDMT (Option A).

### Blockers summary

- MDMOT (the only real 3-5 drone set with cross-view IDs) has no public link.
- MDMT's images are Baidu-only and it has no license.
- AeroCollab3D and AirCo-MultiTasks are not accessible.
- AirV2X has only its mini split public.
- AGC-Drive and V2U4Real have one UAV.
- No multi-UAV measured comm trace exists.

**Statement:** A real, synchronized multi-UAV dataset with given cross-view identity is accessible: MDMT (Baidu images, GitHub test GT) and, pending inspection, VisionTrack (OneDrive). Both have only **2 drones**. **No accessible real dataset with 3 or more synchronized UAVs and cross-view identity was found.**
