# R0 design notes: what the closest papers actually model (representation choice, onboard compute, contention, mission objective, cooperation)

Date: 2026-11-10. Builds on `reports/archive/uav-semcom-a/novelty-audit-a5-l1-2026-11-10.md` (A5L1) and `reports/archive/r1-visdrone/novelty-audit-g-l2-2026-11-10.md` (GL2).
Only facts not already in those audits, or facts needed for the R0 questions, are added here.
"p.N" = PDF page counted by pdftotext page breaks (not the journal page). Eq. numbers are the paper's own.
Depth: F = method + evaluation read from full text; P = full text available, key sections grepped/read; A = abstract/metadata only.
"Not found" = not found in the text I read; it is not a claim that the paper or the field lacks it.

Sources: library PDFs in `pdf/`; web: A²-UAV arXiv:2301.06363v2 (https://arxiv.org/pdf/2301.06363), DAO (https://mason.gmu.edu/~zyan4/papers/dao_mm22.pdf), AccMPEG arXiv:2204.12534 (https://arxiv.org/abs/2204.12534). Downloads and text dumps: scratchpad `dl_web/`, `txt/pg_*.txt`.

---

## 1. Huang et al., JSAC 2025: hybrid bit + generative SemCom in SAGIN. Depth F (system, framework, MDP, sim); also in A5L1

- **Scenario and episode:** 5 LEO → 3 UAV relays → 8 ground users, downlink (p.9, sim params). Image tasks (3.5 MB each, G = 100 tasks) arrive at each LEO following FTP Model 3. Goal: deliver a specified image to a ground user within D_max ~ U[2, 11] s and with PSNR ≥ Q_min ~ U[10, 30] (p.9).
- **What is transmitted:**
  - Mode 1: JPEG2000 + LDPC bits.
  - Mode 2: BLIP-2 caption → CLIP text features; the user regenerates the image with diffusion.
  - Mode 3_1..3_3: caption plus VQ-VAE latent at 3 spatial resolutions, costing hw·log2 V bits (§III-A p.5–6).
  - The encoder runs **on the LEO satellite**; the decoder runs at the ground user; UAVs only relay (p.6). There is no UAV-side encoding or sensing.
- **Decisions:** UAV–user pairing, UAV–LEO pairing, ISL forwarding, UAV trajectory, and **mode ∈ {1, 2, 3_1, 3_2, 3_3} per task** ((15) p.7, p.8).
- **Constraints:**
  - D_g ≤ D_max (15h) and Q_g ≥ Q_min (15i).
  - **Computation delay is a fixed constant per mode: 0.1323 s for Mode 1, 1.33 s for the other modes** (p.9, sim params). It is not tied to device FLOPs or load.
  - Channel: Rician/Shannon, 10 MHz "for all transmissions". Tasks arrive over time, but no explicit queue or contention model was found.
- **Objective:** SCE = ω_q(Q − Q_min) + ω_d(D_max − D) (14); reward = ΣSCE − penalty (18). Q is PSNR, with MS-SSIM in Fig. 11. **There is no task metric and no detector.**
- **Do representation types compete?** **Yes:** bits vs. text vs. text + latent, in one action. All are reconstructions of an image, judged by pixel fidelity. There is no onboard detector, so the onboard-detector-miss question does not apply.
- **Cooperation:** UAVs act as relays and are paired. There is no sensing redundancy.
- **Baselines, data, results:**
  - Baselines: MAPPO, DSAC + hovering UAV, menu without Mode 3', single fixed modes (Fig. 7–11).
  - Dataset not named.
  - Result: Mode 3_3 is the "most balanced" in Fig. 7–8. Only 2 episodes are shown in Fig. 6 (p.10).
- **Limitation (mine):** the generative modes are scored by PSNR. The paper itself says computational delay "cannot be overlooked" (p.10), but models it as a constant.

## 2. Zhao et al., TNSE 2026 (UA-TSC): task-oriented SemCom with trajectory + resources. Depth F (system model, eval); also in A5L1

- **Scenario and episode:** 1 UAV flies (0,0,0.1 km) → 4 waypoints → (3,7,0.1 km) and sends images to 1 BS at (3,4,0) km. Uplink 1 MHz, 1-s slots, v_max = 30 m/s. Mission: 30-class scene classification on AID (p.13–14).
- **What is transmitted:** **learned features.** A DeiT-small encoder with 6, 9 or 12 layers runs onboard, followed by an FC channel encoder producing K ∈ {64, 128, 256} JSCC symbols ((1) p.7; p.8–9; p.13). The BS decodes and classifies.
- **Decisions:** model depth i, number of symbols K, power P (PPO); trajectory via acceleration (Diffuser) ((18)–(19) p.10).
- **Constraints:**
  - **Onboard compute is modelled explicitly:**
    - E_comp = k·D·f²·C_i/c (9);
    - τ_comp = D·C_i/(c·f) (11);
    - **total latency τ = τ_tran + τ_comp (12)**, with τ ≤ τ_max = 0.25 s and accuracy ≥ 60 % (p.9, p.13);
    - f = 1.5 GHz, c = 32 FLOPs/cycle.
  - BS compute is assumed negligible (p.9).
  - Channel: LoS path loss + AWGN, single link. No contention, no queue.
- **Objective:** ω1·accuracy − ω2·energy + trajectory reward (18). Accuracy = **fitted lookup f(i, K, SNR)** from offline runs (p.8, Fig. 3–4), not a per-image model run.
- **Do representation types compete?** **Partly, and only as separate schemes, not one decision.** The baselines are:
  - **LI** (local inference): onboard classifier, sends only the label, 5 bits with repetition 2–16;
  - **RIO** (raw image offloading);
  - SemCom (features).

  Each scheme is optimized by its own PPO (p.14–16). Findings:
  - LI ≈ SemCom in accuracy at moderate SNR but costs more compute energy;
  - SemCom beats LI by 15 % and RIO by 5× at −10 dB (p.15).

  LI misclassification is captured only through the accuracy curve. There is no notion of missed objects, because the task is classification.
- **Cooperation:** none (single UAV).
- **Baselines:** LI, RIO; SAC, TD3; IQL, DP for the trajectory.
- **Stated limitation:** PPO is tailored to one setting; future work is "intelligent connectivity in low-altitude UAV networks" (p.18).

## 3. CCSRA (Zheng et al., EURASIP JWCN 2026): confidence-calibrated semantic RA, multi-UAV. Depth F

- **Scenario and episode:**
  - U = 4 UAVs (also 6 and 8) monitor a common 1000×1000 m area at 80–120 m altitude and 8–16 m/s.
  - Mean inter-UAV overlap is 0.35.
  - Each slot, every UAV captures an image and uploads semantics to **one edge server, which outputs a global object set Ô_t** of boxes, classes and fused scores ((1)–(2) p.5–6; §6.1 p.20).
- **What is transmitted:** **object-guided feature tokens**, not boxes and not images.
  - An onboard **YOLOv8-n (fine-tuned on VisDrone2019-DET)** produces candidates (box, class, raw conf p, localisation quality q) (6) p.7.
  - A semantic encoder produces N ≤ 64 tokens of d = 128 dimensions, each tied to a receptive field (3) p.6; (15) p.9.
  - Tokens are prioritised by summing the "reliable semantic value" ρ = w(c)·p̃·q·(1 − redundancy) over overlapping detections ((12) p.8; (16) p.9).
  - Tokens are quantised to {2, 4, 6, 8} bits (17)–(18).
  - The server fuses tokens across UAVs (4) and runs a detection head (5).
  - Calibration uses temperature scaling with an uncertainty-head MLP 128→64→1, trained post hoc on a val split ((8)–(11) p.7–8).
- **Decisions:** subchannel assignment a, power P, token subset I, bit depth b ((26)–(28) p.12–13). HH-SAC: the upper level decides wireless, the lower level decides content under action masking (Alg. 1 p.19).
- **Constraints:**
  - Reuse-enabled OFDMA, 16 subchannels × 180 kHz, **co-channel interference in the SINR** (20) p.10.
  - Per-slot bit budget B = T·ΣW·log2(1 + SINR) (21); payload ≤ budget (22)/(36); P ≤ 23 dBm (35).
  - "Single-shot, no retransmission" (Alg. 1).
  - **No queue, no deadline beyond the slot, no onboard compute or energy cost** for YOLO or the encoder. Only policy inference time is reported: < 5 ms on an RTX 4090 (p.20).
- **Objective:** λ1·CCSE + λ2·U_t − λ3·R_t − λ4·F_t ((34) p.14). CCSE = delivered Σρ / (payload + power + token count) (29). **U_t = Utility(Ô_t, O_t) against ground truth**, i.e. mAP/recall from the real server detector (33). Reported metric: mAP@50.
- **Do representation types compete?** **No.** There is one representation (tokens) with variable subset and precision.
- **Onboard-detector misses:** **not modelled explicitly.** Because token priority is driven only by onboard detections (16), tokens over objects that YOLOv8-n misses get priority 0 from (16). The paper does not analyse this; it only motivates calibration by "overconfident or underconfident predictions" for small objects (p.7).
- **Cooperation:** **yes, redundancy-aware.**
  - The server keeps a semantic memory bank of 64 prototypes, built with K-means and updated by EMA (23)–(25) p.11–12.
  - Token redundancy = max cosine similarity to that bank (24). Embedding similarity is chosen over cross-view box IoU because no registration exists (p.11–12).
  - The fairness penalty is the variance of delivered value across UAVs (31).
  - Fig. 4: redundancy ratio vs. overlap 0.1–0.7.
- **Baselines:** HARL-SemCom, ARA-DRL, Lyapunov-SCA, all with the same detector and encoder (p.21). **No tuned heuristic** and no "send image" option.
- **Key numbers:**
  - mAP@50 = 44.1 % at 2.88 MHz, +11.1/14.2/21.2 % vs. baselines; 48.3 % at 5.76 MHz (Fig. 2, p.21–22).
  - Payload 112 vs. 126/133/141 kbit/slot (Fig. 5).
  - λ2 sensitivity: 42.8 → 44.9 (Table 1).
  - No seeds or CIs found.
- **Limitations (mine):** the formulation is generic ("Utility may be accuracy, mAP, recall", p.14). How the edge "global detection head" is trained on fused quantised tokens is not specified in what I read.

## 4. A²-UAV (Coletta et al., INFOCOM 2023). Depth P (arXiv v2 full text; system model, MILP, greedy, eval skimmed)

- **Scenario and episode:**
  - A fleet U of UAVs; testbed: **4 DJI Mavic Air 2 + Raspberry Pi, WiFi, Jetson Nano edge**.
  - Targets T sit in an area. A UAV covering a target periodically captures images. Tasks are offloaded over a **multi-hop UAV relay tree** to the edge.
  - The edge runs DenseNet, ResNet152/50, MobileNet-V2 (classification) or YOLOv4.
  - Mission: maximise **"accomplished tasks" = delivered within deadline τ AND correctly predicted at the edge** (Def. III.2, p.4).
- **What is transmitted:** full JPEG images at compression level l ∈ {1..100}. There is no onboard inference; the onboard unit only compresses.
- **Decisions:**
  - UAV positions (connected coverage formation), multi-hop routing tree, target assignment, and per-target JPEG level.
  - Binary ξ_{iuj}: UAV u monitors target i at level j; flow variables e^s (data) and e^a (accuracy) ((2)–(11) p.4–5).
- **Constraints:**
  - Energy-reachability (1); sensing and communication ranges (2), (4).
  - **Each target covered by at most one UAV and each UAV covers one target (3)**, so there is no redundant coverage.
  - Link capacity per edge (7); tree topology (8); flow conservation (9)–(11).
  - Contention enters through the shared bottleneck along paths: Alg. 2 lowers JPEG quality to fit the bottleneck bandwidth (p.6).
  - Deadline τ is in the task definition. There is no onboard compute cost.
- **Objective:** max α·Σ accuracy flow + β·covered targets − γ·distance (12). NP-hard (Thm III.1); GREEDY-A2-TPP is polynomial.
- **Task-metric source:** **lookup Q(s, l)** = (mean accuracy, mean size) per **application scenario** (class group) and JPEG level, profiled offline on ImageNet-class subsets (A²-TA, p.3–4, Fig. 4). It is not per image.
- **Do representation types compete?** No (JPEG levels only). Onboard misses: not applicable.
- **Cooperation:** relaying and coverage only. No overlapping views or fusion.
- **Baselines, results:**
  - Baselines: STBA variants (Steiner-tree coverage, network-only).
  - Results: about **38 % more accomplished tasks** than the state of the art and up to 400 % with many targets. In simulation with 10 targets, 63 % vs. 38 % (L-STBA).
  - The testbed also uses a deadline τ.
- **Relevance to R0:** this is the closest prior for a **mission-level "correct and on-time" objective with multi-UAV shared multi-hop links**. However, accuracy comes from a per-scenario lookup and the representation is a single type.

## 5. DAO (Murad, Nguyen, Yan, ACM MM 2022). Depth P (method, eval, overhead); also in GL2 (F there)

- **Scenario:** 1 end device (Jetson AGX Xavier) offloads video chunks to 1 server running an object detector (SSD-family / YOLO). Bandwidth is throttled with real traces. Data: ILSVRC videos (and YouTube-VOS per GL2).
- **What is transmitted:** an H.264 chunk at a chosen (bitrate, resolution).
- **Encoder side:** a light CNN accuracy estimator (AE) on the device predicts server mAP for every configuration.
- **Decisions:** per-chunk bitrate and resolution, maximising Σ predicted accuracy subject to bitrate ≤ predicted bandwidth ((3) §4.3).
- **Constraints:**
  - Bandwidth only. No deadline per chunk, no queue/buffer: the paper says a buffer is future work (p.9).
  - **AE runtime is measured (26 ms) but not charged against any deadline**; it is called "negligible" per chunk (§5.7, p.9).
- **Task metric:** a real detector, used both for labels and evaluation.
- **Do representation types compete?** No (one codec, two knobs). Cooperation: none, a single device.
- **Result:** 20 % bandwidth saving and +59 % category-wise mAP vs. DASH (abstract).
- **Stated limitation:** "supports a single analytics task at the server" (p.9).

## 6. AccMPEG (Du et al., MLSys 2022). Depth P (motivation, design, eval); also in GL2 (F)

- **Scenario:** 1 camera → 1 server DNN (detection, segmentation, keypoint). 5 streams share 2.5 Mbps in one experiment (GL2).
- **What is transmitted:** **ROI-quality H.264**: macroblocks with predicted high "AccGrad" use QP 30, the rest QP 40.
  - AccModel (MobileNet-SSD features + 3 conv) runs on the camera every 10 frames.
  - This is exactly an "ROI enhancement" representation within one codec.
- **Delay model:** end-to-end delay = camera-side encoding (including AccModel) + streaming + server inference. **Camera-side compute is measured and counted** (Xeon CPU and Jetson Nano; "same camera-side compute resources", abstract/§6).
- **Accuracy:** measured against the server DNN on the high-quality frame, used as pseudo-ground truth (footnote 3).
- **Onboard-detector misses, explicitly argued:** "local heuristics ... giving rise to false negatives — object-related pixels are treated as background and thus filtered out or sent in low quality, causing the DNN to miss objects of interest". Camera DNNs such as MobileNet-SSD at 300×300 are "prone to missing small objects" (§2.2).
  - In eval, the **Vigil** baseline (camera-side detector selects regions) has lower streaming delay but "low accuracy (many small objects are missed)" (§6).
  - AccModel itself is said to struggle with very small objects in drone video (GL2).
- **Do representation types compete?** Not as a decision. Full-frame uniform quality (AWStream), camera-detector regions (Vigil), server-driven regions (DDS/EAAR), frame filtering (Reducto) and an autoencoder appear only as **separate baselines**.
- **Cooperation:** none. **Result:** 10–43 % lower delay at the same accuracy.

## 7. GOSC: Resource-Efficient SemCom for Heterogeneous Agentic Teams (Rezazadeh et al., arXiv 2609.39477, Sep 2026). Depth F

- **Scenario and episode:**
  - **N = 6 agents (10 in one test)** on a 32×32 grid (10 m cells, 1-s slots) with 8 survivors, plus a BS with an edge coordinator (Table III p.7).
  - **Search task:** the metric is mission completion time, i.e. time until all survivors are declared.
  - **Rescue task:** each survivor has a deadline U[60, 180] slots and needs 8 slots of service. The metric is the fraction saved (§III-A p.3).
- **Onboard perception is abstract:** each agent senses a 5×5 footprint with **p_d = 0.85, p_fa = 0.05 per cell** (Table III) and keeps a log-odds belief map.
  - So onboard detection misses and false alarms **are modelled**, as Bernoulli rates, not with a real detector or real images.
- **What is transmitted:** typed lossless messages:
  - CONFIRM (survivor cell, 18 bits);
  - INTENT (target cell, 18 bits);
  - EVIDENCE (aggregated detection records, about 34 + … bits) (§IV-A p.4).
- **Compared representations (Table II, Table V p.7):**
  - **raw** (position + 8-bit score per footprint cell, 218 bits);
  - **natural language** (≈ 1,480 bits);
  - **semantic periodic** (33 + 5 bits);
  - **report-only**.
- **Decisions:** per agent per slot, which message to send (what/when) and at which rate in R (how reliably). This is a multiple-choice knapsack (5) solved by price/Lagrangian bisection (Alg. 1). It is training-free; there is no RL.
- **Constraints:**
  - Rayleigh block fading; outage (2) or finite blocklength.
  - Per-agent budget W = 100 channel uses/slot, **or a shared network-wide budget allocated by the BS each slot** (§IV-F, §V-F2).
  - Per-packet overhead 16–64 bits; lossy/delayed downlink broadcast; HARQ.
  - Queue of pending records with a relevance filter. Data lifetime 20 slots.
  - **No onboard compute cost.**
- **Objective:** min E[mission time (or survivors not saved) + μ·channel uses] (3). Message value = ToM value of information (4), belief divergence, throughput or AoI.
- **Do representation types compete in one decision?** **No.** The representation is fixed per scheme; the competing message *types* (confirm / intent / evidence) are all symbolic.
- **Cooperation:** **yes, central.**
  - The edge fuses records and broadcasts common knowledge, so each agent knows what the team knows (Prop. 1).
  - Message value is computed against teammates' knowledge; duplicate claims and travel are avoided.
- **Baselines (strong, tuned):**
  - A family of 38 tuned periodic schedules (rate-aware, filter, suppressing), frozen on calibration seeds 1000–1079 and evaluated on seeds 0–99 and fresh seeds 3000–3099;
  - paired bootstrap 95 % CIs (§V-A).
- **Key results:**
  - 1.2–8.5× fewer uplink channel uses at matched mission targets (Table IV). Rescue: +3.6 and +7.2 pts survivors saved at W = 100/50.
  - Search vs. raw: −60.9 [48.1, 74.2] slots completion time (Table V).
  - **Vs. semantic periodic: no significant time difference (5.8 [−2.4, 13.8]) but 3.4× fewer channel uses.**
  - With a shared uplink, the search-task advantage is significant only with the largest budget (§V-F2, Table XI).
  - Value precision barely matters: "coarse task relevance ... suffices"; ToM value has no consistent gain over throughput/AoI values (p.1, Table VI).
- **Stated limitations (§V-H p.11):** two grid tasks, one simulator, generic channel; gains are relative to their tuned periodic family; published systems not reimplemented; ablations do not isolate joint selection from an equally tuned sequential scheduler.
- **Relevance to R0:** the best template found for **(d) a mission objective, (e) cooperation through common knowledge, and strong tuned baselines with frozen operating points**. It is abstract on perception (Bernoulli detections, no images, no codecs).

## 8. CCM: Collaborative Computation and Migration in multi-UAV networks with SemCom (Li et al., arXiv 2609.14476). Depth P

- **Scenario:** **N = 3 UAVs** serve **M = 10 mobile terminals (MTs)** in a 1000×1000 m area. MTs offload computation tasks to UAVs (MEC). Tasks can migrate UAV→UAV over A2A links (Table III; p.3–4).
- **What is transmitted:** semantic feature vectors from a deep encoder **on the MT**, quantised to n_b bits (15) p.5. The task type is generic, not vision-specific.
- **Decisions:** UAV trajectory, binary offload I_{n,m}, migration J_{n,n',m}, bandwidth and CPU allocation (p.4, p.7). Solved with HAN-MATD3.
- **Constraints:**
  - **Compute is in the latency:** L_off = L_enc (MT cycles/bit, (24)) + L_tra (25) + L_dec at UAV (26) + L_comp (28); migration adds A2A latency (30).
  - Separate A2G and A2A bandwidth pools with interference.
  - Latency appears only as an objective term; no hard deadline or queue found.
- **Objective:** weighted latency + energy − semantic similarity. Similarity comes from a **fitted ABG function of SINR and n_b** (p.5). The paper itself calls this "a tractable system-level abstraction"; task-oriented encoders and online quality estimation are left as extensions (p.10).
- **Do representation types compete?** No: features only, offload vs. local. Cooperation: migration between UAVs (load balancing), graph attention over UAV–UAV metapaths. No sensing redundancy.
- **Baselines and results:**
  - Baselines: MADDPG, MASAC, MATD3, NSGA-II.
  - Results: +5.3 % reward vs. MATD3; −12.2..14.6 % latency; similarity 0.835 vs. 0.782 at 14 MTs (p.19).
- **Limitation (p.10):** centralized critic cost grows with the number of UAVs.

## 9. Multi-modal SemCom in UAV collaborative networks (Hu et al., IEEE TCOM 2025). Depth P

- **Scenario:** UAVs, one per region, sense images and serve ground users in their own and adjacent regions. **If a user wants another region's image, the local UAV relays it (A2A + A2G)** (p.4, p.8). The A5L1 audit says 2 UAVs × 5 users.
- **What is transmitted:** **image modality** (ViT image-to-image semantic symbols, K_I ∈ {192, 256, 320, 384}) **or text modality** (image-to-text caption symbols, K_T ∈ {36, 48, 60, 72}) (p.5; sim p.12). Data: COCO Karpathy splits.
- **Decisions:** trajectory, bandwidth, power, and **K ∈ K_T ∪ K_I**. The choice of K implicitly selects the modality, including at the relay UAV (p.8, (19)). Solved with hybrid DDPG + DQN (MHDCD).
- **Constraints:**
  - FDMA bandwidth budget; transmission time ≤ D_th (soft, through a sigmoid QoE).
  - **Relay processing time assumed negligible** (p.8). No onboard compute cost found.
- **Objective:** QoE(transmission time, received semantic entropy) − cost. Accuracy comes from **SSIM/BLEU fitted functions of SNR** (Fig. 5), with no task detector.
- **Do representation types compete in one decision?** **Yes, image vs. text, in one discrete action.** But fidelity is judged by SSIM/BLEU fits, not by a downstream task. Onboard misses: not applicable.
- **Cooperation:** relaying only (Fig. 11–14: relay improves QoE and cost). No overlapping-view fusion.
- **Baselines:** fixed position, image-only (MHDCD-ISC), fixed K = 256, JPEG/BPG traditional (MADDPG-TC), EA-MSC. Not tuned heuristics.

## 10. PASC (Yang et al., IEEE WCL 2026). Depth P; also in A5L1

- **Scenario:** N = 3 UAVs, 10 subchannels, 1.5 MHz, Jakes fading; CIFAR-100 classification; **hard deadline T_max = 150 ms** (p.4).
- **What is transmitted:** MobileViT patches. An attention map splits patches into important (w ≥ θ) and non-essential, coded with different symbols per patch: K = N_imp(θ)·k_low + N_non(θ)·k_high (8). This is a **soft ROI / importance-based allocation inside one learned representation**.
- **Decisions:** threshold θ, subchannel, bandwidth, power (PPO + GAE, with a GRU-predicted channel in the state).
- **Constraints:** T_total = semantic codec latency (**linear fit in symbol volume**) + transmission + decision time (GRU + DRL) (5); **T_trans ≤ T_max − T_comp** (p.3). Orthogonal subchannels avoid inter-UAV interference. No queue found.
- **Objective:** QoE = sigmoid utility × latency term; **utility = 0 on timeout**. The task metric is a modelled sigmoid, not a classifier run per image.
- **Do representation types compete?** No. Cooperation: none, beyond fair sharing.
- **Result:** +58.8 % QoE vs. non-predictive baselines at low SNR (abstract).

---

## Summary matrix (Y = yes, p = partial, – = not found in what I read)

| Paper | Depth | ≥ 2 representation kinds in one decision | Onboard detector misses | Onboard compute in deadline | Shared channel + contention | Queue | Mission objective (BS-side) | Real detector metric | Cooperation / redundancy |
|---|---|---|---|---|---|---|---|---|---|
| Huang JSAC'25 | F | **Y** (bits / text / text + latent) | – | p (constant per mode) | p (shared 10 MHz; contention –) | – | – (per-task SCE) | – (PSNR) | relay only |
| Zhao TNSE'26 | F | p (LI / RIO / features as separate schemes) | – | **Y** (τ_comp from FLOPs) | – (1 UAV) | – | – | – (fitted lookup; classification) | – |
| CCSRA JWCN'26 | F | – (tokens only) | p (implicit via det-driven priority) | – | **Y** (OFDMA reuse, interference) | – | – (per-slot mAP) | **Y** (YOLOv8-n, VisDrone) | **Y** (memory-bank redundancy, overlap) |
| A²-UAV INFOCOM'23 | P | – (JPEG levels) | – | – | **Y** (multi-hop bottleneck) | – | **Y** (tasks correct and on time) | p (lookup per class group) | relay / coverage, no overlap |
| DAO MM'22 | P | – | – | – (26 ms measured, not charged) | – | – | – | **Y** | – |
| AccMPEG MLSys'22 | P | – (ROI QP in one codec; others only as baselines) | **Y, argued + Vigil baseline** | **Y** (measured end-to-end) | p (5 streams on one link) | – | – | **Y** (pseudo-GT) | – |
| GOSC arXiv'26 | F | – (fixed per scheme; raw / NL / semantic compared) | **Y** (Bernoulli p_d, p_fa) | – | **Y** (shared budget variant) | **Y** (pending records) | **Y** (completion time, survivors saved) | – (grid abstraction) | **Y** (common-knowledge broadcast) |
| CCM arXiv'26 | P | – | – | **Y** (enc/dec/compute latency) | p (bandwidth pools, interference) | – | – | – (ABG fit) | p (migration) |
| Multi-modal UAV TCOM'25 | P | **Y** (image vs. text via K set) | – | – | p (FDMA) | – | – | – (SSIM/BLEU fit) | relay only |
| PASC WCL'26 | P | – (soft ROI threshold) | – | **Y** (T_trans ≤ T_max − T_comp) | p (orthogonal subchannels) | – | – | – (sigmoid model) | – |

## Gaps relevant to R0 (candid, limited to the 10 papers above plus the two audits)

**(a) Image / token / ROI / feature choice in one decision, with onboard-detector misses modelled: not found in the papers read.**
- Two papers have one decision over different representation kinds:
  - Huang: bits vs. caption vs. caption + latent;
  - Hu TCOM: image vs. text.

  Both score by PSNR / SSIM / BLEU. Neither has an onboard detector, so tokens can never "lose" an object.
- Zhao compares labels-only (LI), raw image and features, but only as separately optimised schemes, for classification.
- Detector-driven token selection exists (CCSRA), but tokens are the only option and the miss effect is not analysed.
- AccMPEG explicitly argues that camera-side detectors miss small objects (false negatives) and shows it with the Vigil baseline. Still, it is a codec, not a per-frame choice among representation kinds.
- GOSC models misses abstractly (p_d = 0.85) with no images.
- So "pick tokens vs. ROI-enhanced image vs. full image vs. features per frame, where tokens inherit the onboard detector's misses that a server detector on the image would recover" is the clearest unoccupied combination I found.
- Caveat: the split-computing and edge-video-analytics literature beyond these papers (e.g. Vigil, DDS, Glimpse, or collaborative-perception late vs. intermediate fusion such as HyComm and CoSDH in A5L1) touches pieces of it. CoSDH and HyComm mix boxes and features (V2X, ideal byte budget).

**(b) Onboard compute cost charged against a deadline: covered, but separately from the rest.**
- Zhao (τ = τ_tran + τ_comp from FLOPs), PASC (T_trans ≤ T_max − T_comp), CCM (encode/decode/compute latency), AccMPEG (measured end-to-end), Huang (constant per mode).
- Not found together with (a) or (c): no paper read charges a *different* compute cost per representation kind (e.g. onboard detector + token encode vs. AVIF encode vs. feature head) against a per-frame deadline under shared-channel contention. Huang is closest, with two constants.

**(c) Shared channel contention with queues: partly covered.**
- Contention via interference / shared budgets: CCSRA, GOSC (shared-budget variant), A²-UAV (multi-hop bottleneck); also SchedCP and JOSC in A5L1.
- Explicit transmit queues/backlog: only GOSC's pending-record queue among the papers read here. CCSRA is explicitly single-shot with no retransmission, and DAO lists a buffer as future work.
- Not found: queueing + deadlines + representation choice together.

**(d) BS-side mission objective rather than per-frame accuracy: rare.**
- GOSC (completion time, survivors saved before deadlines) and A²-UAV (correct-and-on-time tasks) are the only two.
- Neither uses a real detector on real aerial images: GOSC is a grid with Bernoulli detections; A²-UAV uses a per-class-group accuracy lookup.
- All the SemCom papers optimise per-slot or per-task fidelity, accuracy or mAP.
- Not found: "objects reported on time" or "first-detection latency of each ground-truth object" measured with a real detector on VisDrone-like data.

**(e) Cooperation / redundancy among UAVs:**
- Overlapping-view redundancy: CCSRA (embedding memory bank, overlap sweep) and GOSC (common-knowledge broadcast, value relative to teammates).
- Relay-only cooperation: Huang, Hu TCOM, A²-UAV, CCM.
- Not found: redundancy reasoning that changes *which representation* a UAV sends. For example, sending tokens because a teammate already sent an image of the overlap, or using cross-UAV confirmation to recover onboard misses.

**Design hints drawn from these papers (not gaps):**
- GOSC's protocol is the methodological bar for R0 baselines: tuned periodic family, operating points frozen on disjoint seeds, matched-outcome comparison, paired bootstrap. Its result that coarse value + communication-aware scheduling ≈ precise value is a warning that mirrors our G-L2 finding.
- AccMPEG's false-negative argument plus its Vigil baseline give a citable mechanism for why tokens-only can lose. R0 should measure it on VisDrone small objects rather than assume it.
- CCSRA is the closest competitor to cite for "detector-guided tokens + multi-UAV + VisDrone + mAP". R0 differs if it adds the image/ROI alternatives, compute-in-deadline, queues and a mission metric.
