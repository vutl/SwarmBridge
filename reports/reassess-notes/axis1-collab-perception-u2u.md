# Axis 1 reassessment notes: multi-UAV / multi-agent collaborative perception, communication efficiency, U2U/U2B cooperative reporting

Date 2026-11-11. Scope: prior art for the candidate problem below. It adds to `reports/r0-notes/collaborative-swarm.md` (R0) and does not repeat it. Entries already in R0 appear only where this pass found something new about them.

Candidate problem (not validated):
- K UAVs with overlapping views report to a BS over a shared, limited uplink, with U2U links possibly available.
- Each UAV decides WHAT to contribute and WHETHER to send directly (U2B) or through another UAV (U2U relay or cooperation).
- One UAV's choice changes the value of what the others should send.

Conventions:
- Depth: F = full method and evaluation read; P = partial (abstract, system model and evaluation located by grep on pdftotext output); A = abstract or search snippet only.
- "p.N" is the PDF page number counted from pdftotext form feeds.
- Fields per entry: venue/year | scenario | senders -> receiver | what is sent | redundancy/overlap handling | comm model (and whether coordination overhead is counted) | U2U/relay | metric | method | dataset | code | depth.
- PDFs downloaded in this pass are in the session scratchpad (`papers_axis1/`), not in the repo.
- "not found" means not found in the sources checked. It is never a claim that the work does not exist.

---

## 1. MUST-READ: Mazandarani et al., arXiv 2606.31715 (read in full, depth F)

ID verified, but the title has changed:
- v1 (30 Jun 2026): "Semantic-Aware Multiple Access via Spatial Redundancy Exploitation for Uplink-Dominant 6G Use Cases".
- v3 (23 Sep 2026), current title: "Exploiting Overlapping Fields of View for Redundancy-Aware Uplink Transmission in Vehicular 6G".
- Authors: H. Mazandarani, M. Shokrnezhad, T. Taleb (Ruhr Univ. Bochum), O. Günlü.
- Venue: accepted at **IEEE MECOM 2026**, a 6-page conference paper (p.1 header). The old title is now the name of Algorithm 1.

**System model (sec II, p.2-3)**
- V vehicular UEs, each with a front camera, upload frames to a BS.
- Time frame = camera frame (10 Hz), split into |T| slots (100 slots per frame in the evaluation).
- Each frame is cut into K = 100 patches (10x10 grid).
- Channels are split into C_V2V (low-power, high-bandwidth, directed V2V) and C_V2I (uplink). Both are drawn from one pool C, so V2V spectrum is taken from the same budget.
- Collision cliques per channel group. A UE uses at most one channel per slot (eq.5); at most one UE per clique per channel per slot (eq.6); a patch is sent at most once per frame per channel group (eq.3).
- Decision variable m_{i,c,p}^tau: UE i sends patch p on channel c in slot tau (eq.1).
- Local information Omega_i = patches received over V2V before tau (eq.7). This is causal, so a UE only knows what it has been shown.
- BS reconstruction (eq.8): concatenate the received patches. Patches from UE j are "transformed into UE i's viewpoint (GenAI multi-view synthesis)".
- Objective (P): number of UEs whose reconstruction quality Q (SSIM and/or LPIPS) passes a threshold. The metric is **pixel fidelity**, not task accuracy.

**How redundancy is estimated (sec III-A, p.4)**
- Two phases per frame.
- (1) Observation Sharing (OS):
  - Only "preceding" vehicles send selected patches over V2V to "following" vehicles. Pairs are defined by distance < 50 m and heading < 90 deg; each vehicle has ONE role per frame.
  - Which patches to share is ordered by a geometric FoV prior: vertical image bars are mapped onto planar FoV segments, and the overlap area gives a prior similarity (FoV-Priority).
  - The OS duration D_OS is fixed, and the BS must estimate it without seeing the data.
- (2) PatchSelection at the follower:
  - All-pairs patch similarity (SSIM, better after a GenAI view transform; or JEPA-embedding cosine).
  - One-to-one matching by Hungarian / linear-sum assignment.
  - The UE keeps only patches whose matched similarity is below a threshold.

**Is the V2V exchange cost counted?**
- YES in time: the OS slots are taken from the frame, so fewer slots remain for V2I (Alg.1; Fig.4 bottom).
- V2V uses its own channel subset with B_V2V = 10 patches/slot.
- Not counted: signaling for the BS scheduler. UEs report Z_init (their selected patch sets) to the BS, and that cost is not modelled.

**Receiver / scheduler (sec III-B, p.4-5)**
- The BS schedules centrally per frame.
- Patch importance (eq.9) = target-object coverage from CLIPSeg (vehicles, pedestrians, bikes, lights, signs, lanes, crosswalks) x a depth term x (1 + beta x r), where r = 1 if the patch will be used to reconstruct another UE's patch.
- That last factor is the only explicit cross-user coupling in the scheduler: a patch that serves two UEs gets a bonus.
- Reconstruction: received patches plus matched patches from neighbours. The paper discusses but does NOT implement inpainting, RGB-D/LoFTR warping or LMM reconstruction (sec III-B, p.5); these are listed as future work (sec V, p.6).

**Evaluation (sec IV, p.5-6)**
- CARLA, 40 vehicles in an urban zone, 10 s (10K slots), RGB 1280x720 at 10 fps. One scenario. No seeds or number of runs reported; error bars are "variability across time frames and UEs".
- Overlap is REAL GEOMETRY from synchronized simulated cameras, so it is not synthetic like CCSRA. It is limited to front cameras on following vehicles, and the data are simulated.
- Baselines:
  - RND;
  - "SOTA" = importance-aware digital semcom of Liu, Guo, Yang, Ni, Quek, TCOM 2024 (ref [3]), which treats users independently;
  - Ours-LB = the method with the preceding vehicle's unmodified patches, i.e. no view transform;
  - Ours-UB = the method with PERFECT view transformation. The paper states this is "equivalent to directly accessing the following user's observation", so it is an oracle.
- Metrics: satisfied-user ratio for SSIM > 0.8 and for LPIPS < 0.2.

**Results** (Fig.4, p.6; values read by eye from the embedded figure images)
- Satisfied-user ratio vs number of channels:

  | Channels | Metric | UB | SOTA | LB | RND |
  |---|---|---|---|---|---|
  | 30 | SSIM | ~0.46 | ~0.27 | ~0.20 | ~0.08 |
  | 30 | LPIPS | ~0.28 | ~0.19 | ~0.15 | ~0 |
  | 35 | SSIM | ~0.97 | ~0.82 | ~0.71 | ~0.85 |
  | 35 | LPIPS | ~0.79 | ~0.46 | ~0.35 | ~0.18 |
  | 40 | (all) | all reach ~0.9-1.0 | | | |

- **The implementable variant (LB) is BELOW the independent-user SOTA at every plotted point. Only the oracle UB beats SOTA.** The text claims only that UB "consistently outperforms" SOTA; it does not mention that LB loses.
- Gain region: "moderate-resource regime" (~25-35 channels for 40 UEs).
- Sharing duration 2/4/6/8 slots (bottom of Fig.4; channel count not stated, values match the 30-channel point):
  - UB is unimodal: SSIM ~0.28 / 0.38 / 0.46 / 0.39.
  - LB DECREASES monotonically: ~0.26 / 0.23 / 0.20 / 0.20.
  - SOTA ~0.31 to 0.27. It is unclear why a scheme that does not share varies with sharing duration.

**What it does NOT do**
- No detector or task metric at the BS.
- No multi-way (more than 2 UEs) redundancy resolution.
- No relaying: V2V is used only to DISCOVER redundancy, never to forward payload.
- No packet loss (collision-free by construction).
- No real data. Code not found.

**Relevance**
- Structurally the closest precedent to "a sender decides what to send, given what a neighbour sees, using a side link".
- Its own Fig.4 shows that sender-side redundancy discovery over a side link, done without an oracle view transform, did not beat a redundancy-oblivious importance scheduler. That is direct evidence that coordination overhead plus estimation error can cancel the redundancy gain.

**Companion paper, same group: PRISM** (Mazandarani, Shokrnezhad, Taleb, arXiv 2609.03559, Sep 2026; accepted IEEE TNSM) [P: p.1-2, 4-7, 10-14]
- Scenario: semantic-aware distributed MAC for uplink-dominant services. The motivating examples include multiple UAVs with overlapping views of a parking lot (Fig.2, p.2; Table II "Multi-UAV Area Coverage": "two UAVs detect the same object -> same segment", p.6).
- Senders -> receiver: N UEs -> SBS, contending for C slotted channels; variable packet length (macro-actions).
- Abstraction: each UE is associated with "segments" through a binary association matrix A. A segment received several times in a slot counts once.
- New metrics: self-throughput and assisted-throughput (one UE's transmission credits others that share the segment) (p.5, p.11-12).
- Redundancy is resolved AT THE SBS after reception. A UE "neither observes nor assumes a neighbour's current payload" (p.11).
- Feedback: one short downlink control message per macro-action, assumed error-free, delay and quantization not modelled (p.12). Overhead of estimating the association matrix: "assumed negligible ... does not explicitly account for descriptor-signaling overhead or matrix-estimation errors" (p.7).
- Comm: collisions plus 10% channel failure per frame; a collided packet is not retransmitted.
- Method: model-free MADRL (CTDE, double-dueling DQN, MacDec-POMDP). Baselines: CB (centralized Gurobi optimum), PRISM-lite (same MADRL without semantic observations), RND, ORTH (idealized semantic orthogonality).
- Data: **synthetic association matrices A_{N,q}** (N = 8, C = 4, Rmax = 5, T = 30000). The paper states they are "synthetic stress-test instances ... not a substitute for application-level datasets" (p.13). No images; no U2U.
- Results: up to 90% of CB; up to 2x on alpha-fairness and energy efficiency vs semantic-oblivious schemes; the gain grows with overlap q (p.14).
- Scope limit stated by the authors: "no semantic-redundancy gain for independent payloads or when encryption prevents segment matching" (p.6). Code not found.
- Relevance: this is the MAC-layer, distributed, learned version of "one UAV's transmission changes another's value". It explicitly uses multi-UAV overlap as a use case but evaluates nothing UAV-specific.

---

## 2. New entries, not in R0

### 2A. Receiver is infrastructure (RSU / edge / BS), several overlapping senders

- **C-MASS** (Jia, Sun, Mao, Nan, Zhou, Niu; arXiv 2407.00412, 2024; journal version listed 2026-04 per TRID, venue not verified) [P: p.1-5, 7-11]
  - Senders -> receiver: CoVs -> RSU edge server ("edge-assisted CP"), or CoVs -> ego ("distributed CP"). The edge pulls BEV features.
  - Overlap: an explicit "perception topology" (which subsets of CoVs jointly detect each object), approximated to 2nd order (singles + pairs) by replaying detections. Objects jointly detectable only by a PAIR make the utility NON-submodular (Examples 1-2, p.7).
  - This is the clearest formal statement of "one sender's selection changes another's value", including complementarity, not just redundancy.
  - Comm: AWGN rate; minimum bandwidth per CoV for lossless delivery within the frame time; total bandwidth W_t; pull-based; CAM beacons give channel gains (p.4-5). Coordination is cheap because the topology is learned from replays at the receiver.
  - Method: hybrid greedy for budgeted max coverage with a worst-case bound, plus exploration for mobility.
  - Data: statistical detection model fitted on OPV2V and **V2V4Real (real)**, then SUMO mobility simulation (p.3-4, p.10).
  - Results: recall +5.8% / weighted recall +4.2% over object-level CPM; gap to offline optimum cut by up to 75% / 71% vs distance-based and area-based greedy (p.1, p.11).
  - No U2U relaying.
- **RACooper: RSU-Assisted Resource Allocation for CP** (G. Liu, Liang, Guo, Ye, Jin; arXiv 2509.17691, Sep 2025) [P: p.1-4]
  - Senders -> receiver: CAVs -> RSU (V2I, C-V2X); the RSU fuses with its own features.
  - Method: hierarchical RL allocates resources using confidence maps plus CSI.
  - Overlap handling: the RSU broadcasts a request map R_t. The confidence map is updated recursively from features already fused at the RSU, so senders' values are coupled through the receiver state (p.3-4).
  - Broadcast overhead is "considered negligible" (p.4).
  - Related work notes EdgeCooper (JSAC 2023), which uses multihop to cut redundant raw data and extend the RSU's range (p.2).
- **SchedCP**, R0 entry corrected (Y. Liu, G. Liu, Liang, Ye, Guo, Jin; arXiv 2502.10456) [P: p.3-7, 9-10, local pdf]
  - R0 said "no explicit cross-sender dedup found". In fact the ego regenerates its confidence map after each 5 ms slot from the fused features, and transmitted grids are masked (eq.11, p.5).
  - So the value of collaborator j depends on what i already delivered: sequential coupling through the receiver.
  - The RSU is one of the senders; the receiver is still the ego. Shared spectrum: one link per slot. Bandwidth 300 kHz (train), 200-600 kHz (test).
  - Results: +12.9% / +16.3% AP@0.5 / 0.7 vs Nearest at 300 kHz (p.10).
- **LGCP: Local-to-Global CP** (H. Zhang et al., USTC/KAIST; arXiv 2601.12749, Jan 2026; venue not found) [P: p.1-10]
  - Hierarchical, the closest vehicular analogue to "U2U then U2B":
    1. The RSU partitions the road into non-overlapping areas and assigns a CAV group plus a leader per area.
    2. Members send area-specific features to the leader over 5G-V2X sidelink.
    3. The leader fuses and uploads the RESULT to the RSU.
    4. The RSU builds the global view and broadcasts it.
  - Comm: Z shared subchannels, one per link, half-duplex, packet-level transmission scheduling; latency includes transmission plus computation (p.4-6). Leader-selection and grouping messages are counted as latency terms t1..t4 (p.5).
  - Data: OPV2V, V2XSet (OpenCOOD) plus CARLA/OpenCDA/ns-3 co-simulation (p.7).
  - Results: 44x less data on average.
    - At 4 CAVs, latency is 19.2x lower than vehicle-based (V2V) and 4.9x lower than edge-assisted; at 7 CAVs, ~8x lower than edge-assisted (p.8-10).
    - CAUTION: the gain over edge-assisted mixes (a) per-area SUBSET selection (dedup) and (b) splitting fusion compute across leaders (edge fusion latency grows linearly). The paper does not show the side-link aggregation alone helping.
  - Also: vehicle-based V2V sharing has ~2x the latency of edge-assisted even at 2 CAVs, from half-duplex self-interference (p.8).
- **TAMP: Timeliness-Oriented Scheduling in Multi-Region CP** (Zhu, Sun, Jia, Chen, Ai, Zhou; arXiv 2601.04542, Jan 2026) [P: p.1-3]
  - Senders -> receiver: static roadside sensors in several regions -> BS + edge.
  - The BS selects regions and allocates communication volume. Penalty is an empirical function of AoI and volume; Lyapunov per-slot priority.
  - Data: **real RCooper roadside dataset**. Results: up to +27% AP vs best baseline.
  - Regions are disjoint, so no cross-sender redundancy. Relevant only as a BS-side, deadline-aware CP scheduler on real data.
- **BHU: Large Vision Model for Multi-UAV Co-perception in Low-Altitude Wireless Networks** (Xu, Wang, ..., Niyato, Kim; arXiv 2603.16927, Mar 2026) [P: p.1-6]
  - Senders -> receiver: **multi-UAV -> BS / ground server**; uplink MU-MIMO-OFDM with a 3D geometry-based Rician channel and MMSE equalizer (p.3-4).
  - Sent: Top-K pixel-sparsified RGB images. The server runs MaskDINO (Swin-L) for BEV, fuses, and does vehicle segmentation / motion (IoU, VPQ).
  - Decision: a diffusion-model DRL jointly selects WHICH UAVs cooperate, the sparsification ratio per UAV, and precoders. The objective includes max latency over UAVs, since all must arrive before fusion (p.5).
  - Redundancy: only implicit, through UAV selection and a set-level fused utility. "Redundan*" not found in the text.
  - Data: Air-Co-Pred (CARLA). Results: >5% perception gain, -85% overhead vs CNN BEV fusion baselines.
  - No U2U. This is the closest multi-UAV-to-BS perception paper in IEEE-style comms framing found in this pass; it already claims "who cooperates + how much + PHY" for multi-UAV -> BS.
- **EMP** (Zhang, Zhang, Sun, Zhu, Guo, Qian, Mao; ACM MobiCom 2021) [P: p.1-5, 7-8]
  - Senders -> receiver: CAVs -> edge, sharing raw point clouds.
  - Overlap: the edge partitions the overlapped area into DISJOINT upload regions, with a bandwidth-aware Voronoi (REAP). A CAV with a slow uplink "offloads" its upload region to neighbours; chunks are uploaded with priorities (p.2, p.4-5, p.7).
  - Control plane: vehicles send [location, bandwidth] to the edge; this is counted in the system.
  - Comm: trace-driven emulation over real cellular traces; -49..65% end-to-end latency vs V2V sharing.
  - Code: github.com/Shawnxm/EMP.
  - Relevance: this is exactly "who sends which part of an overlapped region to a central receiver given heterogeneous uplinks", for ground vehicles, done with geometry rather than learning.
- **Harbor: Boosting Collaborative Vehicular Perception on the Edge with V2V** (R. Zhu et al.; ACM SenSys 2024, DOI 10.1145/3666025.3699328) [P: p.1-6, 9]
  - The direct analogue of "send directly or relay via another agent". V2I to the edge is the base; vehicles with poor V2I (helpees) relay sensor data over V2V through helpers with good V2I.
  - Motivation: measured cellular uplink heterogeneity, with two carriers at the same place showing 8-15 s periods of 0-1 Mbps (Fig.1, p.2).
  - Helper assignment: weighted bipartite b-matching. Score uses distance and V2V interference; constraint is the helper's spare V2I capacity, with 4.8 Mbps per encoded point-cloud stream at 10 fps (p.5-6).
  - The edge enforces a merge deadline.
  - Evaluation: real-world tests at Mcity plus trace-driven emulation (CARLA, 2-20 CAVs; 20-100 vehicles at scale).
  - Results: +12% average (up to 36%) detection accuracy vs pure V2V or pure V2I; at scale, -43% latency and +14% accuracy (p.1, p.3).
  - No cross-sender redundancy beyond merging. Code not found.
- **MVSparse** (Nalaie, Zheng; arXiv 2312.15858, Dec 2023; venue not checked) [A/P: p.1-2]
  - Static synchronized cameras -> edge.
  - Per-camera RL agents pick informative blocks using history AND the edge's feedback of neighbours' detections (cross-view redundancy). An oracle quantifies the overlap.
  - Results: 1.88x / 1.60x faster inference with marginal tracking loss. Ground cameras, not UAV.
- **DeepStream** (arXiv 2306.15129) [A]: multi-camera -> edge; spatial-redundancy removal plus DP bandwidth allocation under fluctuating bandwidth.
- **ETSI Collective Perception redundancy mitigation** (ETSI TR 103 562; Thandavarayan, Sepulcre, Gozalvez et al., VTC/TITS 2019-2020, JNCA vol.216 2023) [A]
  - Decentralized SENDER-side rules on a SHARED 802.11p / C-V2X broadcast channel: frequency, dynamics, confidence, entropy, object self-announcement and distance-based omission of objects already reported by others.
  - Evaluated with channel load and DCC in network simulation.
  - Long-standing, standardized prior art for "omit what a neighbour already announced". The receiver is every neighbour, not a BS.

### 2B. Multi-UAV collaborative perception (CV side), new in 2025-2026

- **UCP: Uncertainty-Driven Efficient Communication for UAV-Based CP** (Hao, Wan, Sun, Zhang, Wu, Lv; arXiv 2611.06635, 4 Nov 2026; same group as LIF) [P: p.1-6]
  - Request/supply maps from observational uncertainty (LiDAR density / LSS depth confidence) plus discriminative uncertainty, and U-Fusion.
  - Receiver is the ego UAV. Ideal bandwidth axis only.
  - Data: UAV3D (OVERLAP RISK for SB), Air-Co-Pred, and **AGC-Drive (real)**. Code not found.
- **MC-DSA: Multi-UAV Cooperative Deep Semantic Autoencoders** (Han, Lin, Wu, Quek; arXiv 2611.07374, 5 Nov 2026) [P: p.1-8]
  - One T-UAV broadcasts a low-dimensional query; S-UAVs compute query-key similarity and upload features only above a threshold (who2com / when2com).
  - The receiver is the T-UAV (a UAV, not a BS). Channel: AWGN / fading, TDMA uplink, latency and energy model; segmentation on AirSim-SRMS.
  - The authors note cooperative mode "always requires multi-UAV feature transmission ... which may degrade the fused representation".
  - Relevant only as the newest who2com-style multi-UAV semcom.
- **U2UData** (ACM MM 2024; already in R0 dataset table): the asynchronous benchmark samples per-link delay U[0,150] ms (from search snippet; not re-verified in the pdf).
- **AirV2X-Perception** (arXiv 2506.19283, 2025) [A/P]: CARLA+AirSim, up to 5 vehicles + 5 RSUs + 5 drones; benchmarks When2com / Where2comm. No channel model found in the text checked.
- **Griffin** (arXiv 2503.06983, 2025) [A]: aerial-ground cooperative detection/tracking, CARLA-AirSim, 20-60 m altitude, "realistic noise" (not verified to include comm). Receiver is a vehicle.
- **Robust-comm CP (vehicular) found, not UAV**:
  - asynchronous fusion robust to 500 ms (WACV 2025) [A];
  - masked quantized point clouds for packet loss (arXiv 2602.21667) [A];
  - CoDynTrust (arXiv 2502.08169) [A];
  - V2X-INCOP (interruption-aware) [A].
  - All are ego-receiver, single-hop, and none handles multi-sender redundancy.
- **Perception-Aware Communication Middleware for UAV Swarms** (Kaur et al., Cal State LA; arXiv 2609.24964, Sep 2026) [P: p.1-4]
  - Hierarchical swarm (followers -> cluster managers -> swarm leader). UDP pub-sub with image fragmentation, concurrent transmission and priority scheduling; YOLOv8n; hardware testbed (RPi4 / NUC).
  - Uses U2U as transport to in-swarm inference nodes. No redundancy handling and no BS.
  - Shows that the "complete image by deadline" vs packet-QoS issue is recognized.

### 2C. U2U relaying / hierarchy for UAV data (non-perception or partly)

- **Hu, Zhu, Zhou, Wu, Hu, Zhu: "Resource Allocation for Multi-Modal Semantic Communication in UAV Collaborative Networks"** (IEEE TCOM 73(9), Sep 2025; local pdf) [P: p.1-4, 7-8, 13-15]
  - M non-overlapping regions, one UAV each, ground users requesting images (image-to-image or image-to-text ViT semcom) from their own or ADJACENT regions.
  - For adjacent-region requests the local UAV RELAYS (A2A + A2G). Hybrid-action multi-UAV DRL (MHDCD-MSC) over trajectory, bandwidth, power and number of symbols.
  - KEY RESULT for U2U (Fig.12, p.15): with inter-region distance 300 m, **non-relay is better in QoE and cost** ("relay introduces unnecessary resource and time overhead"). As distance grows, non-relay degrades; at 1200 m both use all resources, but relay keeps better QoE.
  - Fig.11: equal when all requests are local; relay helps as the share of remote requests grows.
  - No cross-UAV overlap (regions disjoint) and the link is a downlink to users.
- **A2-UAV** (Coletta et al., IEEE INFOCOM 2023, arXiv 2301.06363; journal version Computer Networks 2024 "resilient") [P: p.1-5]
  - UAV fleet in a "connected coverage formation" whose multi-hop tree is rooted at the edge server.
  - U2U is REQUIRED because targets lie beyond the edge's communication radius (Def. III.1, p.4). Per-class compression-vs-accuracy tables; maximizes correctly executed tasks within the deadline (0.1-0.5 s).
  - Real testbed: 4 DJI Mavic Air 2. Code: GitHub (stated in the paper). No redundancy across UAVs (each target is assigned to one UAV).
- **Aerial Semantic Relay-Enabled SAGIN** (Yin et al., arXiv 2511.18456, Nov 2025) [A]: UAVs relay satellite semcom downlink to users (forward semantic or convert to bits). Not perception reporting.
- **Relay-Aware Semantic Resource Allocation for Multi-Hop UAV Swarm Systems** (Physical Communication, 2026; ScienceDirect S1874490726001114) [A, snippet only; publisher page 403]
  - Multi-hop semantic image forwarding in a swarm. Relays decode / re-encode, so quality degrades hop by hop; destination reconstruction feeds cooperative perception fusion.
  - Possibly the closest "U2U relay of perception semantics" paper. **Must be obtained before any U2U claim.**
- **Control-certified resource allocation for DT-enabled multi-hop UAV swarms** (Luo, Wang, Cheng; arXiv 2605.17791) [A]: multi-hop route-slot TDMA scheduling for telemetry/command with delivery and delay certificates, ns-3. Control traffic, not perception.
- **Distributed DJSCC for correlated multi-view images** [A]:
  - RDJSCC (arXiv 2411.12228): two sensors encode independently, the BS decodes jointly with complementarity-consistency fusion; code public.
  - D-NTSCC (arXiv 2503.21249, 2506.07391): learned joint entropy model, each reconstruction is side information for the other.
  - Earlier: stereo DJSCC (arXiv 2201.10340).
  - This exploits overlap at the RECEIVER with zero sender coordination. It is the natural "no-U2U" baseline family for any overlap-exploiting uplink, and it is missing from all the CP/MAC papers above.

### 2D. Local-library check (only new items)

- SchedCP: correction in 2A (sequential confidence-map masking = receiver-mediated coupling).
- HARQ cooperative perception (Sheng, Liang, Ye, Jin, Li; IEEE TCCN 2025): one CAV -> ego, OFDM, SimCRC semantic error detection, Chase / IR HARQ (p.4, p.9-10, p.19-28). Point-to-point; no multi-sender aspect. Nothing new for this axis.
- LIF (arXiv 2504.21774v2) and MCOP (ICCV 2025): ideal comm, ego receiver; nothing beyond R0. UCP (2B) is LIF's direct follow-up.
- PASC (Yang et al., WCL 2026): CIFAR-100, orthogonal sub-channels; no cross-UAV coupling (p.3-4). Confirms R0.
- Semantic-Driven Multi-UAV Cooperative Comm (Wen et al., WCL 2026): CoMP downlink, N = K = 3 (p.1-4). Confirms R0.
- Hu TCOM 2025 (in pdf/, not in R0): relay evidence, see 2C.
- "Collaborative Computation and Migration in Multi-UAV Networks with Semantic Communication" (Li, Ou, Liu, Jamalipour; arXiv 2609.14476): MEC offloading and migration with a HAN-MATD3 UAV-UAV attention graph. Not perception, no overlap.

---

## 3. Mapping to the candidate problem (what is already claimed)

| Element of candidate | Closest prior (depth) | Status |
|---|---|---|
| Several overlapping senders -> BS/edge, choose what to send | EMP (P), C-MASS (P), CCSRA (R0 P), BHU (P), Mazandarani (F), RACooper (P) | Covered for vehicles; for UAVs covered by CCSRA (synthetic overlap) and BHU (no explicit redundancy) |
| Sender value depends on others' choices | C-MASS non-submodular pair topology (P), PRISM assisted-throughput (P), CodeFilling (R0), SchedCP / RACooper recursive receiver confidence (P), ETSI CPM rules (A) | Covered in several forms |
| Shared uplink with contention / collisions | PRISM (synthetic), Mazandarani (collision-free schedule), SchedCP (1 link/slot), LGCP (subchannels, ns-3), GOSC (R0) | Covered (abstract or vehicular) |
| Side link used to discover redundancy | Mazandarani (V2V patch sharing; practical variant loses to the oblivious scheduler) | Covered, with a negative practical result |
| Side link used to relay payload to BS | Harbor (V2V helper relays, real tests), A2-UAV (multi-hop, range), Hu TCOM'25 (A2A relay, distance threshold), PhyCom'26 multi-hop UAV semantic (A) | Covered for vehicles and for range-limited UAVs |
| Hierarchical in-network fusion before uplink | LGCP (vehicles, sidelink -> leader -> RSU), swarm middleware (A/P) | Vehicular only; UAV version with redundancy not found |
| Receiver-side exploitation of overlap without coordination | Distributed DJSCC (A) | Exists; not used as a baseline in any CP paper checked |
| Real multi-UAV overlapping imagery + BS + realistic comm + redundancy | not found | Gap (as in R0) |

---

## GAPS (candid)

**Covered**
- Overlap-aware "who sends what" to a central receiver: EMP (geometry, bandwidth-aware), C-MASS (learned topology, coverage), RACooper / SchedCP (receiver confidence recursion), CCSRA (multi-UAV, memory-bank redundancy), BHU (multi-UAV -> BS selection plus sparsification plus precoding), Mazandarani / PRISM (semantic MAC with overlap).
- A reviewer can cite each of these against a generic version of the candidate.
- Hybrid direct-vs-relay perception uplink: Harbor (V2V helpers for poor-V2I vehicles, SenSys'24, real tests). Multi-hop UAV-to-edge CV under deadline: A2-UAV.
- Sender-coupled redundancy under contention: PRISM (abstract segments), ETSI CPM rules (broadcast, not BS).

**Not found in the sources checked**
1. A multi-UAV -> BS perception-reporting study that jointly decides content AND route (direct U2B vs U2U relay or in-network fusion) under a shared uplink, with redundancy defined by true cross-view object identity. Harbor does route without redundancy; Mazandarani does redundancy via V2V discovery without relaying; LGCP does hierarchy but for vehicles.
2. Any comparison of the three ways to remove cross-sender redundancy with coordination cost counted on the same budget, on real overlapping imagery:
   - (a) sender-side via side-link exchange (Mazandarani);
   - (b) BS-mediated via downlink feedback or broadcast of "already known" (PRISM, RACooper, GOSC);
   - (c) receiver-side joint decoding with no coordination (DJSCC).
   Mazandarani compares only against a redundancy-oblivious scheduler, and its practical variant loses.
3. A task metric at the BS (unique objects correctly reported by a deadline) in any of the overlap-aware MAC papers. Mazandarani uses SSIM / LPIPS; PRISM uses throughput / fairness.
4. Packet loss and retransmission in any multi-sender overlap-aware scheme. PRISM drops collided packets; Mazandarani is collision-free by construction; CCSRA is single-shot.
5. Full text of the Physical Communication 2026 multi-hop UAV semantic relay paper (paywalled; snippet only). It is the main unread item for the U2U question.

**Weakly established**
- The web search was single-pass per query. MDPI, ScienceDirect and IEEE Xplore pages were mostly not readable.
- IEEE TWC / JSAC / TMC / IoTJ 2025-2026 multi-UAV semcom papers with cross-UAV overlap: none found besides CCSRA (EURASIP JWCN) and BHU (arXiv). This may be a search limit.

---

## Does U2U add anything?

Evidence collected, for perception or sensing data going to an edge/BS:

**FOR U2U (conditions where it helped)**
- **Range / coverage**: A2-UAV needs multi-hop because targets are beyond the edge's radius (p.4).
- **Distance**: Hu TCOM'25 relay beats direct only when inter-region distance is large (around 1200 m in their setting). The paper does not give an exact crossover distance.
- **Uplink heterogeneity / outage**: Harbor, with real cellular traces showing 8-15 s periods of 0-1 Mbps (p.2), gets +12% average / up to +36% accuracy by relaying helpees through helpers with spare V2I capacity. The constraint is helper spare capacity (p.6).
- **Capacity asymmetry (cheap side link vs scarce uplink)**: Mazandarani assumes V2V is low-power, high-bandwidth, on separate channels (B_V2V = 10 patches/slot). Gains appear only in the "moderate-resource regime" and only with an oracle view transform (Fig.4).
- **Compute distribution plus subset selection**: LGCP's leader-based hierarchy beats edge-assisted 4.9-8x in latency. The gain is largely from per-area dedup and parallel fusion, not from the side link per se.

**AGAINST or neutral**
- **Short range / good direct links**: Hu TCOM'25 at 300 m, non-relay is better: "relay introduces unnecessary resource and time overhead" (p.15).
- **Side-link discovery overhead**: in Mazandarani, the implementable redundancy discovery (LB) is below the redundancy-oblivious scheduler at every channel count, and gets worse with longer V2V sharing (Fig.4, p.6).
- **Pure peer-to-peer exchange scales poorly**: data grows quadratically; half-duplex gives ~2x latency even at 2 CAVs (LGCP p.8); EMP is -49..65% latency vs V2V sharing; Harbor sec 2 says the same.
- **Redundancy can be removed without U2U**: via BS-mediated downlink feedback (PRISM, RACooper request map, GOSC common knowledge), geometric partition decided at the edge from tiny control messages (EMP: location + bandwidth), or joint decoding at the receiver (DJSCC). In each case the coordination traffic goes on the downlink, which is usually less constrained than the multi-UAV uplink.
- **Multi-hop costs**: relays add hop latency and per-hop loss; semantic re-encoding degrades quality hop by hop (PhyCom'26 snippet; general surveys).

**Net reading for the candidate (inference, not a measured result)**
- For UAVs at typical altitude, U2B links are mostly LoS and available. On that evidence, U2U relaying of perception payload is unlikely to be NECESSARY unless the scenario builds in at least one of:
  - (i) UAVs beyond BS range or behind blockage (urban canyon, terrain, low altitude);
  - (ii) strong, time-varying U2B heterogeneity (a Harbor-like outage);
  - (iii) a U2U link much cheaper or wider than U2B (separate band or mmWave sidelink) together with a tight uplink;
  - (iv) in-network fusion that reduces uplink bytes MORE than the U2U traffic it costs (high overlap, and a receiver that needs only the fused product).
- "Who sends what among overlapping UAVs" can be solved, and has been in vehicles, with BS-side coordination (EMP / C-MASS / RACooper style) at small downlink cost.
- A credible "U2U is needed" claim would have to beat a BS-coordinated, no-U2U baseline with all control and side-link traffic counted on the same budget. None of the papers checked report such a baseline: Mazandarani compares only with an oblivious scheduler, and Harbor with V2V-only / V2I-only.
