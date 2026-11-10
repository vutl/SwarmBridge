# R0 notes: multi-UAV cooperation x what-to-send-to-BS (prior art, datasets, gaps)

Date 2026-11-10. Read-only extraction. UAV3D and Active Semantic Recovery (ASR) were NOT opened; both are treated as known overlap risks.

Conventions:
- Depth tags: F = full read, P = partial (method/experiments/dataset sections, located with grep on pdftotext output), A = abstract, landing page or snippet only.
- "p.N" = PDF page number from the pdftotext form-feed count, not the printed page.
- [audit] = taken from reports/novelty-audit-a5-l1-2026-11-10.md (sec. 2.2) and not re-read.
- [web] = found by a web-research sub-agent (WebSearch/WebFetch). No web paper was read in full.
- "not found" = grep/search found nothing. It does not mean the paper lacks it.

KEY SURPRISES (neither appears in either novelty audit):
1. CCSRA (Zheng et al., EURASIP JWCN 2026, in pdf/) already does cross-UAV redundancy-aware uplink to an edge server: an embedding memory bank, a redundancy score in the reward, and a "redundancy ratio vs overlap ratio" plot. It runs on VisDrone, which is single-view; the overlap is simulated in some way the paper does not explain.
2. GOSC (Rezazadeh, Chergui, Liu, Debbah, arXiv 2609.39477, Sep 2026, in pdf/ as "Resource-Efficient Semantic Communication for Heterogeneous Agentic Teams"):
   - multi-UAV/agent search-and-rescue reporting to a BS edge coordinator;
   - message value is computed against the "common knowledge" the BS has broadcast, so redundancy is suppressed;
   - shared-uplink variant, mission metrics (completion time, survivors saved), tuned periodic baselines, 95% bootstrap CIs.
   - Perception is a grid with Pd/Pfa: no images and no detector.
   - It is the closest match to scenario (ii) and partly to (iii).
3. CodeFilling (Hu et al., CVPR 2024, in pdf/): selection across several senders with a saturating per-receiver "information demand", so messages are explicitly non-redundant. It is V2X, the receiver is an agent rather than a BS, and the byte budget is ideal.

## 1. Source extraction

Field order in each entry: scenario | agents -> receiver | what is sent | redundancy/overlap handling | comm model | mission metric | method | dataset | depth, pages.

### 1A. Collaborative perception (V2X / drone, feature-level). These are the overlap-risk family.

- **Where2comm** (Hu et al., NeurIPS 2022)
  - 3D detection | N cars or drones -> each other (ego), no BS | sparse BEV features chosen by a spatial confidence map.
  - Redundancy: the receiver sends a request map R = 1 - C (p.5); senders pick cells that are both confident and requested. This is pairwise, not a coordinated choice among senders.
  - Comm: ideal byte count (log2 volume); no channel or contention [audit p.7] | metric AP@0.5/0.7.
  - Method: supervised end-to-end; RL explicitly not used [audit p.3].
  - Data: OPV2V, V2X-Sim, DAIR-V2X, CoPerception-UAVs (sec 2).
  - Depth P: p.5, p.7-8, p.22-24 (dataset appendix).
- **CoSDH** (arXiv 2503.03430, 2025)
  - V2V/V2I 3D detection | agent -> ego | features plus boxes (intermediate-late hybrid).
  - Redundancy: supply (sender confidence) x demand (receiver weakness) mask, with hand-set thresholds (p.1-2, p.4).
  - Comm: 27 Mbps split into per-agent caps; latency robustness at 100/200 ms only [audit p.6, p.12-13].
  - Metric AP 3D. Data: OPV2V, V2X-Sim, DAIR-V2X. Depth P [audit].
- **How2comm** (NeurIPS 2023)
  - V2X 3D detection | agent -> ego | spatial-channel filtered features.
  - Redundancy: a mutual-information-aware mechanism names "communication redundancy" as one of three challenges; the ego also requests complementary info (p.1-2).
  - Comm: fixed delay tau = 100 ms, compensated with flow; no deadline [audit p.24].
  - Data: DAIR-V2X, V2XSet, OPV2V. Depth P.
- **Who2com** (ICRA 2020) [web, A]
  - Aerial agents in sim, segmentation | a degraded agent sends a compressed request; others reply with match scores; the agent picks ONE peer.
  - Redundancy: implicit, through peer selection.
  - Comm: ideal; ~1/4 of the bandwidth of a centralized scheme.
  - Data: "AirSim-CP" (abstract). Whether this is the same data as AirSim-MAP is unconfirmed.
- **When2com** (CVPR 2020) [web, P]
  - 5-6 AirSim drones in "CityEnviron" whose views "periodically and partially overlap" | query broadcast; supporters send feature maps.
  - No explicit dedup: learned attention, links below 1/N pruned.
  - Comm: ideal MB per frame plus link count; no contention or latency.
  - Results: mIoU 57.28 at 0.385 MBpf vs 2.5 MBpf for fully connected.
  - Code is MIT: github.com/GT-RIPL/MultiAgentPerception.
- **CodeFilling: Information Filling with Codebook** (Hu et al., CVPR 2024)
  - V2X 3D detection | each supporter i -> receiver j | sparse BEV regions coded as codebook indices (integers instead of floats).
  - Redundancy: joint selection maximizes sum over receivers of min(C_j + sum_i M_ij*C_i, u) under a total-regions budget b, eq.(3) p.4. Selection stops once the receiver's demand u is met. This is the explicit multi-sender anti-redundancy rule; it is solved on the supporter side.
  - Comm: ideal budget (number of regions/bits); robustness tests for pose error and latency up to 100 ms (p.7) | AP.
  - Method: supervised plus a hand-designed selection rule | data DAIR-V2X, OPV2VH+.
  - Depth P: p.1-2, p.4, p.7.
- **CERF** (2025)
  - Multi-UAV 3D detection and tracking | drone -> ego | compact detection results converted into a virtual modality "Poture" fused with local BEV; history-based prediction compensates delay (p.1-2).
  - Retraining-free and heterogeneity-tolerant; no explicit cross-sender dedup found.
  - Comm: bandwidth reported in Mbps (Table 1, p.3); ideal.
  - Data: UAV3D (OVERLAP RISK) and Air-Co-Pred. Depth P.
- **LIF** (Is Intermediate Fusion All You Need for UAV CP?)
  - Late-intermediate fusion | drone -> ego | detection results plus instance-level cues (VPE, box-based virtual augmented features).
  - Redundancy: uncertainty-driven selection of "high-quality and reliable shared areas" (p.1-2).
  - Comm: ideal | data: UAV3D (OVERLAP RISK). Depth P: p.1-3.
- **HyComm**
  - V2X 3D detection | top-b boxes plus raw points by a fixed rule [audit p.3-4].
  - Redundancy: the receiver removes redundant detections at fusion (p.5).
  - Comm: ideal byte budget | data DAIR-V2X, OPV2V. Depth P.
- **DiffCP** (2025)
  - V2X | the co-agent sends ultra-low-bit info; the ego's diffusion model, using the ego's own geometric prior, reconstructs the co-agent's BEV (p.1-2).
  - Redundancy: implicit (exploits cross-view correlation).
  - Comm: about 150 kbps vs C-V2X ~10 Mbps (p.1) | data OPV2V. Depth A/P.
- **CoDS / SComCP / HMS-SCP** [audit]
  - V2V feature transmission through VQ+LDPC+QAM (CoDS), JSCC (SComCP) or multi-scale top-k JSCC (HMS-SCP).
  - Real PHY (AWGN/Rayleigh), point-to-point. Importance masks; no multi-sender dedup.
  - Data OPV2V / DAIR-V2X. Depth P.
- **SchedCP** (DRL user scheduling for CP, 2025) [audit]
  - Ego plus N collaborators on SHARED spectrum: one link per 5 ms slot, 200 ms frame.
  - Double DQN picks which collaborator transmits. Reward is rate plus detection-loss reduction (label-dependent), or a label-free version from the confidence map (p.6-7).
  - Baselines nearest / round-robin / max-rate | data V2X-Sim. No explicit cross-sender dedup found.
  - This is the closest precedent for "who sends under contention", but the receiver is a vehicle, not a BS.
- **Drones Help Drones (DHD)** (NeurIPS 2024)
  - 4 drones, trajectory prediction | drone -> ego | sparse BEV, top-K cells by complementarity, 25% transmission ratio (p.7).
  - Comm: ideal; the paper itself calls the setting "idealized" and defers latency and communication frequency (p.10).
  - Data: Air-Co-Pred. Depth P: p.2, p.6-7, p.10, p.14.
- **MCOP** (arXiv 2510.12679)
  - Multi-UAV 3D semantic occupancy | the ego broadcasts a request mask (low-confidence regions); collaborators send features where their support mask overlaps it (p.4-6).
  - Comm: ideal; 0.23 MB vs 17.5/19.1 MB (p.2).
  - Data: Air-to-Pred-Occ, UAV3D-Occ (OVERLAP RISK), GauUScene-Occ (real). Depth P.
- **CoCa3D: Collaboration Helps Camera Overtake LiDAR** (CVPR 2023)
  - Camera-only collaborative 3D detection; introduces CoPerception-UAVs+ (10 drones). Depth A (p.1-2).

### 1B. Multi-UAV -> BS/edge semantic uplink (receiver is infrastructure)

- **CCSRA** (Zheng et al., EURASIP JWCN 2026:112). Closest match to (i).
  - U = 4 (also 6, 8) UAVs -> edge server for "global situational awareness" | object-level semantic tokens from YOLOv8-n, 2/4/6/8-bit quantization.
  - Redundancy (sec 3.5, p.11-12):
    - the edge keeps a semantic memory bank of size M = 64;
    - redundancy score = max cosine similarity between the token and the bank, eq.(24);
    - the authors explicitly REJECT cross-view 2D IoU because boxes from different UAVs are not comparable without registration (p.11-12);
    - the redundancy penalty enters the reward (sec 4.3, p.14).
  - Comm: OFDM, 16 x 180 kHz, Rician, 23 dBm; single shot, no retransmission (p.20-21).
  - Metrics: mAP@50, CCSE, "redundancy ratio".
    - Redundancy ratio drops from 0.28 to 0.19 as lambda3 rises (Table 1, p.22).
    - At overlap 0.5 it is -23..35% vs baselines (Fig. 4, p.23-24).
  - Method: hierarchical hybrid SAC.
  - Data: VisDrone2019-DET. The "inter-UAV overlap ratio is controlled by adjusting relative deployment positions, default 0.35" (p.20). How overlapping images come from a single-view dataset is NOT described (likely a crop/synthetic split).
  - No cross-view GT identity, so the redundancy metric is defined by embedding similarity rather than true object identity.
  - Depth P: p.1-2, 6, 8-12, 14, 20-24.
- **GOSC: Resource-Efficient SemCom for Heterogeneous Agentic Teams** (Rezazadeh et al., arXiv 2609.39477). Closest match to (ii).
  - Scenario:
    - 32x32 grid of 10 m cells, 1 s slots;
    - 6 agents (also 10), 8 survivors (uniform or clustered);
    - agents launch next to a BS with an edge coordinator (p.3, p.7).
  - Perception: each agent senses a (2r+1)^2 footprint with per-cell detection/false-alarm probabilities. The edge keeps a log-odds belief map with additive Bayesian fusion, and declares a survivor when the posterior passes a threshold (p.3). There is NO imagery and no detector.
  - Sent: CONFIRM (survivor cell, 18 bits), EVIDENCE (aggregated detection records), INTENT (rescue claims) (Table II, p.4); natural-language and periodic-semantic variants as baselines (p.8).
  - Redundancy: message value is computed against the common knowledge the BS broadcasts, i.e. what teammates already know. Delivered messages are never re-sent; claims prevent duplicated travel (p.2-5).
  - Comm: Rayleigh block fading, rate choice, HARQ, per-packet overhead; orthogonal per-agent channel uses by default, plus a SHARED uplink with price coordination (sec IV-F, V-F2) (p.3, p.6).
  - Mission metrics: search = completion time until all survivors are declared; rescue = fraction saved under deadlines U[60,180] slots; plus false declarations and confirmation latency (p.3, p.8).
  - Method: value-threshold rule with a closed-form scheduler (theorems), not learned. Baselines: 38 tuned periodic configurations, AoI, throughput and belief-divergence valuations. Fresh-seed validation with 95% bootstrap CIs (p.7-8). Planner or LLM core.
  - Result: 1.2-8.5x fewer uplink channel uses at matched mission targets.
  - Depth P: p.1-8.
- **Semantic-Driven Multi-UAV Cooperative Comm** (Wen et al., WCL 2026) [audit]: 3-UAV CoMP DOWNLINK to users, text QoE. "Cooperation" here is joint transmission, not joint sensing. Not relevant to (i)-(iii) except as a naming clash.
- **JOSC, PASC** [audit]: K UAVs send video frames or patches over shared resources (send/skip plus compression ratio). There is NO cross-UAV content coupling: each UAV's value is independent.

### 1C. Mission/search environments and semantic metrics

- **DSSE** (Drone Swarm Search Env, PettingZoo; capstone-insper)
  - Maritime SAR grid with a drifting target and a probability matrix | actions are 4 moves plus "search"; finding means "descend and verify" (p.2).
  - Reward: +cell probability on search, a time-scaled find bonus, -1000 for not finding, -2000 for collision, -100000 for leaving the grid (p.4).
  - No communication model and no imagery. Depth P: p.1-4. Useful only as a search-dynamics scaffold.
- **GOST** (Goal-Oriented Semantic Twinning tracking, satellite-UAV, TMC 2026)
  - 2 clusters x 6 UAVs -> satellite | point targets; UAVs send position/state estimates.
  - MADDPG; AoI, positioning error, target loss; latency and packet-loss model (p.4-5, p.10).
  - No images or detector; no explicit cross-UAV dedup found. Depth P.
- **AoII** (Maatouk et al.)
  - Single source -> monitor over an unreliable channel with a rate constraint; penalizes the TIME the receiver's estimate is WRONG (threshold and dissatisfaction error functions, p.3-5); optimal policy is a randomized threshold.
  - Single sender, so no multi-sender redundancy. It is the natural metric family for "BS belief about object set is wrong" (missed or unverified objects). Depth P: p.1-5.

### 1D. Cross-camera redundancy in video analytics (static ground cameras) [web]

- **CrossRoI** (Guo et al., ACM MMSys 2021, not MobiSys) [P]
  - 5 static cameras at one intersection -> server.
  - Offline ReID profiling finds where the same objects appear across cameras; tile optimization gives each camera a minimal RoI mask; cameras stream only RoI tiles.
  - Emulated 30 Mbps WiFi; vehicle query; AI City data.
  - Results: -42..65% network, -25..34% delay, >99% query accuracy.
  - Heuristic plus optimization, not RL. The mask is STATIC, from offline profiling; it cannot follow moving cameras.
- **Spatula** (Jain et al., SEC 2020) [A]: cross-camera spatio-temporal correlation prunes which cameras/frames are searched for one identity; compute -8.3x. Related roadmap: HotMobile 2019.
- **Polly** (Li et al., INFOCOM 2023) [P]: overlapping co-located cameras share detections. ReID finds overlap; a regression tree maps boxes between cameras; low-confidence regions are re-inferred. Saves COMPUTE (-71% latency), not uplink. AI City data, Jetson Nano.
- **Caesar** (SenSys 2019) [A]: cross-camera complex activity with NON-overlapping cameras; claims bandwidth savings. Not a dedup method.
- **Argus** (arXiv 2401.14132) [A]: object-wise association across overlapping smart cameras skips redundant identification (7.13x fewer).
- **STAC** (arXiv 2401.15288) [A]: association plus frame filtering and RoI masking on AI City.
- **CONVINCE** (PerCom-W 2020) [A]: ~25% of frames sent at ~91% identification accuracy.

### 1E. Multi-UAV reporting, dedup, verification with imagery [web]

- **Semantic-Aware Multiple Access via Spatial Redundancy Exploitation** (Mazandarani et al., arXiv 2606.31715) [A]. Structurally closest to (i), but with VEHICLES.
  - Users exchange patches over V2V to estimate overlap redundancy, then send only important, non-redundant patches over a shared uplink to a BS.
  - Joint which/when/channel decision; metric is the fraction of users reaching high-fidelity reconstruction (pixel-oriented, not mission-oriented).
  - Dataset and numbers not found. MUST READ.
- **Vision-based multi-view cooperative perception for UAV swarms (GNSS-denied hub recon)** (Liu et al., Drones 10(7):546, 2026) [A]
  - Onboard detection; cross-view association by projected-polygon IoU; 50% of overlapping redundancy eliminated, 0 duplicates on SUES-200.
  - Infrastructure footprints, not small objects; no comm model found.
- **CARNet** (arXiv 2610.26312, Pattern Recognition) [P]: multi-drone collaborative 2D detection on MDMT-CD (cross-view pair: class or "no-match"). mAP50 25.23 vs 23.45 single-drone. No comm or GS model.
- **UCDNet** (arXiv 2406.04648) [A]: multi-UAV 3D detection, AeroCollab3D; comm delay left to future work.
- **Cooperative-search fusion of detection / false-alarm probabilities** (Khan, Yanmaz, Rinner, ICRA 2014 / TCNS 2015) [A]: mission time -27..70% for 2-5 UAVs. Probabilistic; not about imagery or uplink cost.
- No paper was found that combines "an uncertain detection triggers a second UAV view vs a higher-quality re-send" with an uplink cost. Several publisher pages (MDPI, Springer, DOAJ) were blocked, so this gap is weakly established.

## 2. DATASETS: multi-UAV overlapping views with labels (for a 2D-detection reporting pilot)

VisDrone-DET/VID is single-view; CCSRA's "overlap" on VisDrone is synthetic and undocumented.

| Dataset | Real/Sim | Drones/views | Size | Labels | Per-view 2D boxes | Cross-view ID | Overlap | Access/license | Depth |
|---|---|---|---|---|---|---|---|---|---|
| MDMT (Liu et al., IEEE TMM 2023) | Real | 2 | 88 seq, 39,678 frames, 11,454 IDs; box count conflicts (2.2M in repo vs 220K elsewhere) | person, bicycle, car; MOT | yes | **yes** (MDA metric) | overlapping pairs | Baidu only, 25.8 GB (pwd 9un6); license not found; CARNet reports label errors, and a corrected MDMT-CD (7 cls) is "will be available" | web A/P |
| MDOT (Zhu et al., TCSVT 2021) | Real | 2 / 3 | Two-MDOT 92 groups, 113,918 fr; Three-MDOT 63 groups, 145,875 fr | single-object tracking (1 target per clip) | only the target | target only | yes | Baidu (pwd 1895); license not found | web A |
| MDMOT (FusionTrack, arXiv 2505.18727) | Real (5 DJI Mini 3 Pro) | 3-5 | 20 seq, ~122K frames, 2.48M (or 24.8M) boxes | per-view boxes, no class field | yes | **yes** (manual) | mixed overlap / non-overlap | download link not found | web P |
| VisionTrack (cited in GMT, arXiv 2407.01007) | Real | 2 | 15 scenes, 116K fr, 1.18M boxes | MOT | yes | yes (per citing paper) | yes | original paper/access not found | web A |
| M3OT (Sci. Data 2025) | Real (DJI Mavic 3T, 100-120 m, RGB+IR) | 2 | 21,580 fr (10,790 RGB-IR pairs), ~220K boxes; 17,260/2,400/1,920 split | vehicles (small objects, all < 1024 px^2); MOT gt.txt plus COCO JSON (p.5) | yes, per drone and per modality (Table 3, p.5) | NOT stated: each target has a unique ID within its annotation; cross-drone linkage not documented | "image pairs for multi-view MOT" (p.3); degree not quantified | Figshare figshare.com/s/01fa8d1163f4e9a5a13a; article CC BY 4.0 (p.10); dataset file license unverified | P (pdf) |
| CoPerception-UAVs (Where2comm) | Sim (AirSim+CARLA towns 4-6) | 5 UAVs x 5 cams (BEV + 4 at -45 deg) | 131.9K images (123.8K discipline + 8.1K dynamic), samples every ~4 s; 3 altitudes (40 m covers ~200x200 m) | vehicles: 3.6M 2D boxes (projected from 3D), 1.94M 3D boxes, segmentation, BEV map; nuScenes format | yes | derivable from global 3D boxes (nuScenes instance tokens likely; not verified) | high (discipline formation) | Baidu full / GDrive sample; license not found (simulators MIT) | P (pdf p.22-24) |
| CoPerception-UAVs+ (CoCa3D, CVPR 2023) | Sim | 10 | not checked | 3D | not checked | likely via global 3D | yes | not checked | A |
| Air-Co-Pred (DHD) | Sim (CARLA Town10) | 4 drones at 50 m, ~100x100 m | ~32K images at 1600x900, 2 Hz, 200 scenes; ~430K 3D boxes | vehicles, cycles, pedestrians; nuScenes | via projection | derivable from 3D | high (cover the same area) | HF datasets/BRUCE-WZC/AIR or Baidu; license not found | P (pdf p.6, p.14) |
| AeroCollab3D (UCDNet) | Sim (CARLA) | 6 | 3.2K samples, 19.2K images, 218K 3D boxes | 4 classes, 3D, no track IDs | via projection | derivable | yes | access not found | web A |
| U2UData (ACM MM 2024) | Sim (UE5 "U2USim", Yunnan map) | 3 | 945K RGB/depth, 315K LiDAR frames, 2.41M 3D boxes, 3 classes (animals) | 3D det, tracking, segmentation | not stated (3D-centric) | 3D, per-UAV coordinates | formation-dependent | github fengtt42/U2UData; paper CC BY 4.0 | P (pdf p.1-3) |
| U2UData+ (AAAI 2026) | Sim | 15 | 12.96M RGB, 4.32M LiDAR, 720 traces x 600 s, 3.62 TB | 3D boxes per UAV global frame (p.6), wildlife ELH tasks incl. "collaborative communication" | not stated | 3D | yes | fengtt42.github.io/U2UData-2 | P (pdf p.1-6) |
| AirSim-MAP (When2com) | Sim | 5-6 | not recorded | segmentation only, NO boxes | no | no | partial, periodic | repo README | web P |
| MCOP sets | Sim plus real GauUScene-Occ | multi | not recorded | 3D occupancy | no | no | yes | no link found; UAV3D-Occ = OVERLAP RISK | P |
| DIVOTrack (IJCV 2023) | Real, 3 moving cams (one was a drone in v1) | 3 | not recorded | pedestrians, 953 cross-view tracks | yes | **yes** | yes | password after signed license | web A |
| Ground analogs: WILDTRACK (7 static cams, 400 fr), MultiviewX (synthetic, 6 cams), CityFlow (40 cams) | Real/Sim | 6-40 | small / large | pedestrians / vehicles | yes | **yes** | high | research licenses | web A |

Dataset takeaways:
1. **Real, overlapping, with 2D boxes AND cross-view IDs:**
   - MDMT has 2 drones but a Baidu-only download and noisy labels.
   - MDMOT has 3-5 drones but no download link was found.
   - Either would allow a ground-truth duplicate/unique-object metric.
2. **M3OT has the cleanest access** (Figshare, CC BY article), but has 2 drones, vehicles only, and no documented cross-drone identity. Dedup ground truth would have to be built with a homography or manual linking; feasible for vehicles on near-planar ground at 100-120 m.
3. **Simulated with 4-5 views and high overlap:** CoPerception-UAVs (2D boxes provided) and Air-Co-Pred. Cross-view identity can be derived from global 3D boxes, which gives exact dedup ground truth.
   - Risk: CARLA/AirSim imagery is close to the UAV3D style (UAV3D is also CARLA Town10, like Air-Co-Pred).
   - Reviewers may see "multi-UAV CARLA plus vehicles" as UAV3D-adjacent.
4. **None of these has a communication trace.** Every comm model would be ours.

## 3. GAPS (candid; "not found in sources checked", never "first")

### Scenario (i): cross-UAV redundancy-aware reporting to a BS over a shared uplink

Already covered:
- CCSRA: multi-UAV -> edge uplink, redundancy score vs a memory bank in the RL reward, OFDM, VisDrone + YOLO, redundancy ratio vs overlap. It is essentially scenario (i) with tokens and an RL scheduler.
- CodeFilling: multi-sender saturating demand (V2X ego receiver, ideal budget).
- Where2comm, CoSDH, MCOP: request/demand masks, pairwise.
- SchedCP: who transmits on shared spectrum (V2X ego).
- CrossRoI: static cameras, offline ReID masks, 30 Mbps.
- Mazandarani 2606.31715: V2V-estimated redundancy -> BS uplink for vehicles; must read.

Not found in the sources checked:
- (a) Redundancy evaluated with GROUND-TRUTH cross-view identity on REAL multi-drone imagery. CCSRA's overlap on VisDrone is synthetic and undocumented, and its redundancy is defined by embedding similarity, not object identity.
- (b) A mission metric of "unique objects correctly reported to the BS before the deadline" (on-time unique recall / duplicate airtime) under contention or queuing. CCSRA is single-shot, so every payload arrives.
- (c) Moving-camera dedup without offline profiling (CrossRoI's masks are static).
- (d) A comparison of BS-side dedup (send all, NMS at BS) vs sender-side coordination vs BS-broadcast "already known" lists, with the cost of the coordination traffic counted.

Risk: reviewers will cite CCSRA directly. Any claim must be framed against it (real overlap plus GT identity plus a deadline/contention metric plus tuned heuristics).

### Scenario (ii): cooperative search-and-report (where to look x what to send)

Already covered:
- GOSC: grid SAR to a BS, value against broadcast common knowledge, shared uplink, completion time and survivors saved, tuned periodic baselines, bootstrap CIs. Almost the whole communication and evaluation side of (ii), with ABSTRACT Pd/Pfa sensing.
- DSSE: search MDP, no comm. GOST: tracking with MADDPG, point targets.
- Khan/Yanmaz/Rinner: probabilistic cooperative search.

Not found:
- (a) The same loop driven by REAL imagery and a real detector, where Pd/Pfa depend on altitude, resolution and the codec/bitrate chosen. GOSC's sensing is per-cell Bernoulli.
- (b) Coupling between the codec/bitrate choice and the BS posterior (declaration threshold). GOSC sends fixed-size symbolic records.

Data obstacle: no real dataset supports closed-loop "where to look". Real multi-drone sets are fixed trajectories. A closed loop needs a sim (CARLA/AirSim, close to UAV3D) or a replay of a large orthomosaic. This makes (ii) costly and easy to dismiss as "GOSC with images".

### Scenario (iii): cooperative verification of uncertain detections (second UAV view vs higher-quality re-send)

Already covered:
- ASR (other team): recovery requests; known overlap for the "re-send" arm.
- GOSC: EVIDENCE accumulation until the BS posterior passes a threshold (multi-look via fusion, no image re-send).
- HARQ cooperative perception [audit]: retransmit by SimCRC.
- Polly: re-infer low-confidence regions locally.
- MCOP, Where2comm: request masks for low-confidence regions.
- DSSE: "descend and verify" as an action.

Not found:
- An explicit decision, per uncertain object, among (1) a re-send of a higher-quality crop from the same UAV, (2) a fresh view from a second UAV that already sees it (overlap) or could, (3) accepting or dropping it. The decision would be judged by verified-correct declarations per airtime under a deadline.
- The independence argument (a second viewpoint de-correlates detector errors; a re-send only removes compression error) was not found tested on real overlapping imagery.

Risks:
- Arm (1) overlaps ASR.
- The decision needs both views at the same time: MDMT/M3OT give 2 synchronized views, so arm (2) is only possible where the objects lie in the overlap.

## 4. Implications for choosing ONE candidate (input for the design report, not a decision)

1. **Scenario (i) on MDMT, or M3OT plus homography IDs, looks most feasible with real data.** It must differ from CCSRA on four points:
   - true cross-view identity ground truth;
   - a deadline/contention metric (on-time unique recall, duplicate airtime);
   - comparison of BS-broadcast "known set" (GOSC-style) vs sender-side vs BS-side NMS;
   - tuned heuristics vs learned.

   It reuses the existing SB uplink simulator, codec menu and detector.
2. **Scenario (iii) can be added inside (i)** as a sub-decision for objects in the overlap: a second view vs a re-send. It stays distinct from ASR only if the "second view" arm is central.
3. **Scenario (ii) needs closed-loop motion,** so it needs sim data (UAV3D-adjacent risk), and it is GOSC-covered on the comm side. Lowest priority.
4. **Must-read before committing:**
   - arXiv 2606.31715 (Mazandarani et al., V2V-estimated redundancy -> BS);
   - GOSC sec IV-F/V-F2 in full (shared-uplink pricing);
   - CCSRA sec 3.5 / 6.1 again: how the VisDrone overlap was built; ask the authors or check code if any;
   - CARNet / MDMT-CD release status.
