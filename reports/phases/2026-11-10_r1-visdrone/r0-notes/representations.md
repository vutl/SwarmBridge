# R0 prior-art scan: choosing among representation types (image / tokens / ROI-hybrid / features) for UAV->BS detection

Date: 2026-11-10. Scope: web scan (WebSearch + a few WebFetch of arXiv abstract/HTML pages). No PDFs read in full.
Reading depth legend:
- F = full text read this session (none reached F)
- P = partial (arXiv HTML / substantial page read this session)
- A = abstract / search-snippet level this session
- K = from background knowledge, NOT re-verified this session (treat details as "to verify")

Question checked: does prior work let each camera/UAV choose per frame among
(1) compressed full image re-detected at BS, (2) onboard-detector tokens (class/box/conf),
(3) hybrid (low-q image + ROI crops, or tokens + thumbnail), (4) intermediate features,
while charging onboard compute (latency/energy vs deadline), modelling shared-channel contention,
deadlines, and the fact that tokens lose objects the BS detector would find?

## A. Split computing / co-inference (mode 4 vs 1 vs local)

1. Neurosurgeon (Kang et al., ASPLOS 2017). [K]
   - Modes: layer-level split point incl. all-local / all-cloud. Costs: per-layer latency + energy models, upload bits.
   - Comm: single link, measured bandwidth. No deadlines (min latency or energy). No accuracy change (lossless split),
     so no "onboard misses". Decision: profile-based argmin. Task: classification + some vision DNNs.

2. JALAD (Li et al., ICPADS 2018). [K]
   - Split point + quantization/entropy coding of features; accuracy-loss constraint. Single link. ILP-style decision.
   - First-generation "feature compression changes accuracy" model; classification.

3. Edgent (Li, Zhou, Chen; MECOMM'18 -> IEEE TWC 2020). [K]
   - Joint split point + early exit (BranchyNet) under a LATENCY REQUIREMENT, maximize accuracy. Single device-edge link,
     static vs dynamic bandwidth (configuration map). Classification (AlexNet). Closest classic "deadline + mode" design,
     but modes are exits/splits, not representation types.

4. BottleNet (Eshratifar et al., ISLPED 2019) / BottleNet++ (Shao & Zhang, ICC Workshops 2020). [K]
   - Learned bottleneck + (for ++) JSCC-style feature encoding over AWGN/fading. Single link, no deadline, classification.

5. Matsubara, Levorato, Restuccia, "Split Computing and Early Exiting for DNN Applications: Survey", ACM CSUR 2022. [K]
   - Reference survey; categorises SC without/with bottleneck, early exit. Notes most SC work is single-device, single link.

6. Entropic Student / Supervised compression (Matsubara et al., WACV 2022) + SC2 Benchmark (TMLR 2023, arXiv 2203.08875). [A]
   - https://arxiv.org/abs/2108.11898 , https://arxiv.org/abs/2203.08875
   - Rate-distortion of learned feature codecs vs input (JPEG/WebP/BPG/neural) compression, INCLUDING COCO detection
     (Faster R-CNN FPN). Result: supervised feature compression gives better R-D than input compression in their setups.
   - Charges encoder compute on device (reported), bits; no channel/queue/deadline; no mode switching per frame.
   - Relevance: the best evidence that mode (4) can dominate mode (1) in bits-vs-AP; static choice only.

7. Choi & Bajic, "Deep feature compression for collaborative object detection" (ICIP 2018) + collaborative-intelligence line;
   MPEG FCM (Feature Coding for Machines) / VCM standard work (2023-2025, e.g. arXiv 2512.10209, 2512.10230). [K / A]
   - Feature-domain codecs for detection/segmentation/tracking; reported 75-85% bitrate savings vs remote inference
     at equal accuracy (authors' claims, A). One analysis (2512.10230, A) says VCM (pixel) underperforms FCM at low-mid
     bitrates for detection. Purely codec-level; no device compute budget, contention, or per-frame mode switching.

8. Feature compression for rate-constrained detection on the edge (NYU, arXiv 2204.07314). [A]
   - https://arxiv.org/abs/2204.07314 . Learned compressor trained jointly with YOLO under rate constraint; later layers
     sparser/easier to compress. Single link, static.

9. CoTeD (Yu, Mendula, Levorato, Papatriantafilou, Chiasserini), MeditCom 2024 -> IEEE IoT-J vol.13 no.8, 2026. [A]
   - https://research.chalmers.se/publication/550507 ; code https://github.com/Rexyyj/CoTeD
   - UAV head (YOLOv3/-tiny) + edge tail; on-UAV tensor selection/compression with dynamic ratio, application- and
     system-aware optimization coordinating compute + transmission in real time. Explicit deadline (detection must be
     output within ~10s-100ms autopilot deadline); reports >=90% inference-request success, +20-80% vs direct split,
     static JPEG, quantization; -90% radio data; -50% UAV compute vs UAV-only. Private 5G testbed.
   - Modes compared are BASELINES (JPEG offload / split / quantized onboard), not a per-frame runtime choice among
     representation types. Single UAV (confirmed in conf. version, see Deep reads R3). Onboard misses: N/A.
   - CLOSEST UAV-specific split-computing + deadline work found.

10. Ye et al., "E2E delay minimization ... joint DNN partitioning and resource allocation for cooperative edge inference",
    IEEE VTC2023-Fall (arXiv 2310.12937, v1 only). [P; ar5iv full HTML skimmed]
    - N UEs, one ES, FDMA shared uplink (5 MHz, sum of shares <=1); M/D/1 local queues (Poisson arrivals); Lyapunov
      virtual energy/memory queues; PPO picks split point, closed-form/CVX for CPU/bandwidth.
    - Costs: local compute latency+energy (kappa f^2), memory, feature bits. NO accuracy model, no misses, no hard deadlines
      (long-term avg delay). Simulation with AlexNet/ResNet18 (detection only in motivation).
    - Representative of the multi-user co-inference resource-allocation literature: shared channel + queues, but
      modes = split points with equal accuracy.

11. ENACHI: Tang, Sun, Chen, Ding, Ai, Y. Shao, "Hierarchical online scheduling for energy-efficient split inference with
    progressive transmission", arXiv 2601.08135 (Jan 2026). [A]
    - Multi-user, bandwidth allocation + per-slot power; hard delay + energy constraints; maximize accuracy;
      uncertainty-aware progressive feature transmission; two-tier Lyapunov. ImageNet classification.
      +43% accuracy, -62% energy under stringent deadlines (authors). Closest on "shared channel + deadline + accuracy"
      but only split/feature mode, classification.

12. Liu, Lan, K. Huang, "Resource allocation for multiuser edge inference with batching and early exiting",
    IEEE JSAC 41(4) 2023 (arXiv 2204.05223). [A]
    - Joint scheduling + bandwidth with batching/early exit at server, computation-time constraints, max throughput;
      NP-complete -> best-shelf-packing / tree search. Server-side inference only (all users upload); no device modes.

13. Zhuang, Mao, Ho, Yu, "UAV-assisted cooperative edge inference for low-altitude economy via MoE hierarchical DRL",
    arXiv 2605.19290 (May 2026). [A]
    - Ground devices offload intermediate features to UAV; discrete offloading + continuous feature-compression ratio +
      UAV trajectory (mission = trajectory deviation); CPOMDP + HDRL-MoE. Raw-image mode explicitly excluded as
      infeasible. Details (accuracy model, contention, deadlines) not visible at abstract level.

## B. Shao / Zhang task-oriented communication line (feature-only, IB)

14. Shao, Mao, Zhang, "Learning task-oriented communication for edge inference: an IB approach", IEEE JSAC 40(1) 2022. [A/K]
    - VIB feature encoder + variable-length coding for dynamic channel; single device; classification. Rate-relevance.
15. Shao, Mao, Zhang, "Task-oriented communication for multi-device cooperative edge inference", IEEE TWC 2023
    (arXiv 2109.00172). [A]
    - Multiple devices view same object; distributed IB / deterministic DIB, selective retransmission to drop redundant
      features; orthogonal channels (no contention modelled per their setting, as far as abstract shows); classification
      (multi-view). No local-only / image modes; no deadlines beyond latency motivation.
16. Shao, Zhang, Zhang, "Task-oriented communication for edge video analytics", arXiv 2211.14049 (IEEE TWC 2024). [A]
    - Deterministic IB features + temporal entropy model across frames; multi-device with shared resources in later
      sections [K, verify]. Tasks: pedestrian/ multi-view detection-like (verify). Feature-only representation.

## C. Local detect vs offload (mode 2 vs mode 1), cascades / confidence

17. Glimpse (Chen et al., SenSys 2015). [A]
    - Device sends selected "trigger frames" to server for heavy recognition; tracks locally between them to hide
      latency (>1 frame RTT). 1.8-2.5x precision vs naive. Single device; mode = send frame vs local tracking.
18. Vigil (Zhang, Chowdhery, Bahl, Jamieson, Banerjee, MobiCom 2015). [A]
    - MULTI-CAMERA, shared wireless uplink (whitespace/Wi-Fi). Edge nodes run lightweight detection, upload only
      relevant frames/best views, frame prioritization + traffic scheduling across cameras; up to 25% more query-relevant
      objects vs equal-throughput Wi-Fi under fixed bandwidth. Onboard-detector errors are the filter's risk but not
      the optimized quantity (verify). Closest early multi-device "onboard analytics decides what to send" system.
19. DeepDecision (Ran, Chen, Zhu, Liu, Chen, INFOCOM 2018). [A]
    - https://www.cs.ucr.edu/~jiasi/pub/deepdecision_infocom18.pdf
    - Per-period choice: local (small model) vs offload with chosen resolution/bitrate/model; utility = accuracy & fps
      under battery, data usage, latency constraint; measured accuracy vs resolution/bitrate. Optimization (MILP-like).
      Single device, single link. Accuracy per mode is EMPIRICAL (so local-model weakness is captured in aggregate,
      not per-object misses).
20. FastVA (Tan & Cao, INFOCOM 2020, arXiv 2001.04049) + "DL on mobile devices through NPU and edge" (arXiv 2112.02439). [A/K]
    - Local NPU (quantized, less accurate) vs offload frames at chosen resolution; deadline per frame; maximize
      accuracy / #frames meeting deadline; heuristic + DP. Single device. Detection/classification video.
21. Qiu, Wang, Hu, Guerin, Lu, "Optimizing edge offloading decisions for object detection", arXiv 2410.18919 (2024;
    WUSTL; venue not stated on arXiv). [P]
    - Weak onboard detector (YOLOv5n / SSD-MobileNet) on Jetson; offload image to strong detector (YOLOv5m / FRCNN).
      Reward ORIC = estimated mAP gain of offloading THIS image within a context set; error taxonomy explicitly includes
      MISSED GT, background, localization, duplicates of the weak detector -> estimator from weak detector's top-25
      proposals (MLP). Threshold at a target offload ratio. Costs measured (latency) but not charged in decision;
      single device, 1 Gbps Ethernet, no deadlines, no contention (authors note multi-device congestion as future work).
      COCO / VOC. +52% mAP over no-offload at 20% offload (SSD/FRCNN VOC); >60% better than random at 0.4.
    - Cites: Adaptive Feeding (ICCV 2017), DCSB (ICDCS 2023), IDK cascades, Qiu TCAD 2022 + Chakrabarti SEC 2021
      (offloading under TOKEN-BUCKET rate limits, classification), Yoon IEEE Access 2021, Ku TVT 2021.
    - CLOSEST work on "tokens lose objects the server would find" (mode 2 vs 1), but single device, no deadlines.
22. Hierarchical Inference: Moothedath, Champati, Gross, IEEE TMLCN 2024 (arXiv 2304.00891); "Case for HI" arXiv 2304.11763;
    Fresa & Champati MSWiM 2022 (accuracy max under time budget, small local models + large server model). [A]
    - Offload only when local tiny model is likely wrong (max-softmax), online learning with regret bounds.
      Single device, classification. Captures "local result wrong" but not per-object misses in detection.
23. Online learning of image-recognition offloading policies over wireless links (AUEB, IEEE ICC 2023). [A]
    - https://mm.aueb.gr/leadingedge/pubs/Y3-ICC-2023-OFFLOADING.pdf . Options: offload image / local simpler model /
      lightweight detector then send image only if objects detected. EXP3 bandit; cost = transmit energy + recognition
      error. Single device. Notable: three modes incl. a gating "detect-then-send".

## D. Hybrid ROI + background (mode 3), briefly

24. DDS (Du et al., SIGCOMM 2020), EAAR (Liu et al., MobiCom 2019), AccMPEG (Du et al., MLSys 2022). [K, already known]
    - Low-q full frame + high-q ROIs (DDS: server-driven second pass; EAAR: ROI encoding + motion-vector tracking on
      device; AccMPEG: cheap per-macroblock quality model). Single stream / single link; bandwidth-aware; no explicit
      per-frame choice between tokens vs image vs features.
25. EdgeDuet (Wang et al., IEEE INFOCOM 2021; ext. IEEE/ACM ToN 2023). [A]
    - https://zhouzimu.github.io/paper/infocom21-wang.pdf . HYBRID: large objects detected ON DEVICE (local results),
      small-object candidate tiles sent to edge at high quality (ROI encoding), tile-level parallel offloading.
      VisDrone (drone imagery): +44.7% accuracy, -34.2% E2E latency vs offloading SOTA; small-object AP 0.096 local vs
      0.319 EdgeDuet. Single device. Implicitly acknowledges onboard misses (small objects) by routing them to edge.
      CLOSEST work to "tokens + ROI crops" hybrid on aerial data.
26. Elf (Zhang et al., MobiCom 2021). [A]
    - Content-aware partition of a frame into region pieces sent in parallel to multiple edge servers; 4.85x speedup,
      52.6% bandwidth saving. Single device, multi-server.

## E. Representation/scheme choice in multi-user semantic comm / video analytics

27. He, You, Chen, Sun, Liu, Quek, Gao, "Joint user association and resource allocation for adaptive semantic
    communication in 5G and beyond", arXiv 2312.01049 (v4 Oct 2026). [P - full HTML read, see Deep reads R1]
    - CORRECTION after full read: the scheme library is (DNN width x transmitted feature fraction), i.e. feature-mode
      variants only; raw-data and result-only appear as motivating extremes / the raw-data "TC" BASELINE, not as
      choices. Generic analytic accuracy A(c,d); no task/dataset; no misses; one-shot per-user T^max, no queues.
28. JCAB (Wang, Zhang, Chen, Qian, Wu, Xiao, IEEE INFOCOM 2020; ToN 2022). [A]
    - Multiple cameras share one edge uplink; choose resolution/frame rate (configuration) + bandwidth; Lyapunov online;
      accuracy-energy-latency. Representation = image at different configs only (no tokens/features).
29. Jellyfish (Nigade et al., Real-Time Systems journal 2024). [A]
    - Multi-client edge serving with soft latency SLOs; jointly chooses client input size + server DNN variant.
      Image-only representation; server-side scheduling.
30. Kodan (Denby et al., ASPLOS 2023) - orbital edge computing analogue. [A]
    - Satellite onboard compute is the bottleneck; selects specialized models per tile, downlinks high-value tiles;
      +89-97% data value density vs bent-pipe. Analogue of "onboard filter vs raw downlink" with compute charged;
      single satellite downlink.
31. Huang JSAC 2025 hybrid bit + generative SemCom [known to project; not re-read].
32. Knowledge-graph UAV cognitive semantic comm for detection (Song et al., arXiv 2401.13995). [A]
    - UAV sends semantic features; detector at receiver; gains at low SNR/compression. Feature-only.
33. Maritime SAR (arXiv 2403.14281). [A] - caps #detections sent to fit a reduced drone-ground link (token budget),
    ranking boxes; a rare explicit "send detections not images under bandwidth" evaluation. Single UAV.

## Summary table (modes / misses / costs / comm / deadline / method / task / depth)

| # | Work | Modes competing | Onboard misses modelled | Costs charged | Comm | Deadline | Method | Task | Depth |
|---|------|-----------------|------------------------|---------------|------|----------|--------|------|-------|
| 1 | Neurosurgeon ASPLOS17 | split pts (incl local/cloud) | no (lossless) | lat+energy, bits | single | no | profile argmin | cls | K |
| 3 | Edgent TWC20 | split x early-exit | exit acc only | latency | single, dyn BW | yes (latency req) | search | cls | K |
| 6 | Entropic Student/SC2 | feature codec vs input codec | n/a | enc compute, bits | none | no | static R-D | det+seg | A |
| 9 | CoTeD Medit24/IoT-J26 | fixed split + adaptive tensor compr (JPEG/CS/edge-only = baselines) | n/a | measured head/tail time, bits | 1 UAV, tc-emulated/5G link | per-frame T_D (30 ms) | closed-form rank rule | 1 video / St. Marc; YOLOv3(-tiny) | F (conf) / A (journal) |
| 10 | Ye VTC23 2310.12937 | split pts | no acc | lat, energy, mem, bits | FDMA shared | avg delay, queues | Lyapunov+PPO | cls (AlexNet/ResNet18) | P |
| 11 | ENACHI 2601.08135 | split + progressive tx | acc objective | energy, bits | multi-user BW | hard | 2-tier Lyapunov | cls | A |
| 12 | Liu-Lan-Huang JSAC23 | server batching/exit | exit acc | server compute | multi-user BW | comp-time | packing/tree | cls | A |
| 15 | Shao TWC23 multi-dev | features (DIB) | n/a | bits | orthogonal | no | learned | multi-view cls | A |
| 19 | DeepDecision INFOCOM18 | local model vs offload x res/bitrate | aggregate acc | battery, data, lat | single | yes | optimization | det/cls video | A |
| 20 | FastVA INFOCOM20 | local NPU vs offload x res | aggregate acc | lat | single | per-frame | heuristic/DP | video | A/K |
| 21 | Qiu 2410.18919 | weak local det vs offload image | YES (TIDE 6 types incl. missed GT) | none in decision (ratio budget only) | single, 1GbE | no | MLP reward + percentile threshold | COCO/VOC det | F (HTML) |
| 22 | HI TMLCN24 | local tiny vs offload | yes (local wrong) | offload cost | single | no | online learning | cls | A |
| 25 | EdgeDuet INFOCOM21 | local det (large) + RoI-encoded tiles (small) at fixed rates | per-class size threshold (recall<90%) | latency only | 1 phone, emulated LTE/WiFi | none | heuristic rules | VisDrone 6 videos, pseudo-GT | F |
| 27 | He/Quek 2312.01049 | width x feature-fraction (raw only as baseline) | no (analytic A(c,d)) | cycles, gamma f^2 energy, bits | multi-SBS RBs, static interf. | per-user T^max=10ms | NP-hard -> 3-stage greedy/DP | none (generic) | P |
| 28 | JCAB INFOCOM20 | image configs | aggregate acc | energy, latency | shared uplink | latency | Lyapunov | video det | A |
| 18 | Vigil MobiCom15 | onboard filter -> upload frames | partly (filter) | bits | multi-cam shared Wi-Fi | real-time | scheduling heuristics | surveillance | A |

## Deep reads (follow-up, 2026-11-10; full text unless noted)
R1. He, You, Chen, Sun, Liu, Quek, Gao, arXiv 2312.01049 v4 (arXiv HTML, all sections read).
- What a user sends (Sec. III-B, IV-A): an ASC scheme = (DNN width in {0.25..1.0}) x (fraction of ranked feature
  channels sent, 25..100%, quantized), tuple (cycles c, bits d, utility u). Raw-data upload and "local inference,
  send only the result" are named in Sec. III-B as illustrative extremes; NOT in the evaluated scheme library.
  Raw upload appears only as baseline TC (Sec. V-A): raw 0.4-0.8 Mbit, max DNN at SBS -> accuracy eta2.
- Task: generic. Sec. III-B example is classification; Sec. IV-B1 mentions recognition/detection as utility sources;
  Sec. V-A uses an ANALYTIC accuracy A(c,d) = A_c(c) * A_d(d) / beta3 (log in cycles, power-law in bits) fitted
  to curves from cited papers; NO dataset, NO detector, NO per-object misses, no local-vs-server accuracy gap.
- Compute: T = c/f, E = c * gamma * f^2 (DVFS, Eqs. 2-3). Channel (Sec. III-A/C): 10 SBSs, 100 WDs, RBs per SBS
  (0.2 MHz each), Shannon rate with static average interference, each WD associated to one SBS.
- Deadline (Sec. III-D, Eq. 8b): one-shot per-user T^c + T^t <= T^max (10 ms) and energy <= 2 mJ; no arrivals,
  no queues, no frames/time slots.
- Method (Sec. IV): strongly NP-hard (knapsack reduction); per-WD scheme scheduling (budgets tight, fzero),
  RB allocation greedy (concave) / DP (general), association by relaxation-refinement. Results (Sec. V, Figs. 2-8,
  Table III): Prop > TC/FSC/SC/ARB/NUA on utility; TC (raw) worst under scarce RBs; ~1 s MATLAB for N=100.
- Verdict: multi-user shared-spectrum "how much semantic feature" selection; NOT a raw/feature/result menu and NOT
  detection. Weaker overlap than the abstract suggested.
R2. EdgeDuet, Wang, Yang, Wu, Zhao, Zhou, IEEE INFOCOM 2021 (PDF, Sec. I-VII read).
- Device (iPhone 11, Sec. VI-B): local YOLOv3-FP16 640x640 (Core ML) at fixed ~10 fps for medium/large objects;
  HEVC (Kvazaar) RoI encoding + 5x3 tiles offloaded at fixed ~5 fps; KCF trackers fill in between (detect+track).
  Edge (Sec. VI-A): YOLOv3-spp on 960x960 overlap tiles, 2x RTX 2080Ti.
- Tile/RoI choice (Sec. III-A): a block "contains small objects" if previous-frame results show an object below a
  PER-CLASS size threshold, the size below which the local detector's recall < 90% but the remote one is accurate
  (Fig. 3: car 5150 px, pedestrian 1698 px). Non-RoI blocks get QP 44 (RoI QP 22) chosen so the remote detector
  still LOCATES new objects with low confidence; those are re-requested next frame. Tiles prioritized by count of
  small objects (Sec. III-B2). Merge (Sec. V-1): drop local boxes for small objects, remote boxes for large ones.
- So onboard misses are handled by an OFFLINE size rule, not modelled per object/per frame at runtime.
- Network (Sec. VII-A4): single device over WiFi 5 GHz with Apple Network Link Conditioner emulating LTE / WiFi
  2.4 / 5 GHz; no contention, no deadline, no energy; latency = averaged per object.
- Data (Sec. VII-A1): 6 VisDrone 2K street videos, upsampled to 60/120 fps; GT = YOLOv3-spp 2560 pseudo-labels
  (cars, pedestrians); metric = average IoU (not AP). Results (Sec. VII-B, Figs. 9-10): small-object accuracy
  +161-292% vs local-only (LaT); overall accuracy +41-52% and latency -32..-49% vs Glimpse/EAAR; under LTE,
  local-only ~= offloading schemes in accuracy. Decision method: fixed heuristic rules, no optimization.
R3. CoTeD. MeditCom 2024 conference version READ IN FULL (par.nsf.gov PDF); IoT-J 2026 journal PDF blocked (403),
    journal known only from abstract + GitHub README.
- Modes (Sec. III-IV): ONE runtime mode, fixed split of YOLOv3-tiny at layer 8 (tensor 128x26x26); runtime knobs are
  pruning target rank theta_r and CP-decomposition scaling delta_R (Alg. 1-2). Onboard-only, edge-only (raw frame),
  JPEG-100 and compressive sensing are BASELINES (Sec. V, Fig. 4), not switchable modes. Boxes are never sent uplink;
  results return to the UAV autopilot.
- Deadline (Sec. IV-D): per frame T_D; transmission budget T_tr = T_D - T_head - T_tail - T_CoTeD (pre-measured);
  target compression r'_c = tensor bits / (T_tr * L_t) from measured bandwidth L_t; best effort if infeasible
  (deadline missed for L_t <= 8 Mbps at T_D = 30 ms). Compute is measured, not optimized.
- Setup (Sec. V): ONE UAV emulated by a Ryzen mini-PC, 100 Mbps Ethernet + tc shaping (conf.); README: Jetson Orin
  Nano + private 5G, YOLOv3 (split L10) and YOLOv3-tiny, "St. Marc" video dataset. Conf. eval = one ~30 s single-car
  video with YOLOv8x pseudo-labels; metrics detection "sensitivity" and "accuracy". Results: -95/-75/-80% latency vs
  CS/JPEG-100/edge-only; journal: up to -90% radio data, >=90% request success. No multi-UAV contention.
R4. Qiu, Wang, Hu, Guerin, Lu, arXiv 2410.18919 (arXiv HTML, all sections).
- Decision (Sec. III, Eq. 3): per image, offload to strong detector iff estimated reward > T, T = (1-r) percentile for
  offload ratio r; "static resource constraints" (ratio budget) only; no rate limit, token bucket, latency or deadline.
- Reward: ORI (Sec. II, Eq. 1) = per-image mAP gap strong-weak; ORIC (Sec. IV-A, Eqs. 4-5) = (|E|+1) x change in
  mAP of {image + ~1000-image context set} when this image's weak output is replaced by the strong output.
  Error taxonomy (Sec. IV-B2, TIDE): classification, localization, both, duplicate, background, MISSED GT.
- Estimator (Sec. V): MLP on features of the weak detector's top-25 proposals; target MORIC (CDF transform),
  weighted MSE. Costs (Sec. VI-2, Table III): weak inference ~8.9 ms, estimate ~0.5 ms, tx ~19 ms, strong ~12 ms are
  MEASURED, none enters the decision. Single Jetson Xavier NX + RTX 3090 over 1 GbE.
- Results: Fig. 9 >60% over random at r=0.4; Fig. 11 +52.2% mAP gain at r=0.2 (VOC, SSD/FRCNN); >50% of strong
  benefit at 20% offload. Limits (Sec. V-C, VI-2, VIII): two detector pairs; multi-device congestion, bandwidth/latency
  frameworks (token bucket) and inter-frame correlation left to future work.
R5. Ye et al. 2310.12937: published at IEEE VTC2023-Fall (arXiv comment). Object detection is only the motivating
    example in the abstract; evaluated models are AlexNet / ResNet18 classifiers in simulation, no accuracy model.

## GAPS (updated after deep reads; candid; "not found in sources checked", never "first")

Covered, by whom, how closely:
- Misses-aware choice tokens vs re-detected image: Qiu 2410.18919 is the closest. It models the weak detector's
  object-level errors (incl. missed GT) via an mAP-in-context reward and learns a predictor from weak outputs.
  It is single-device, wired, offload-ratio budget only: no bits/compute/deadline in the decision, no channel.
- Tokens + high-quality ROI hybrid on drone imagery: EdgeDuet. Local boxes for large objects + RoI-encoded tiles for
  small ones, with an offline per-class size rule standing in for "where the onboard detector misses". Single
  device, emulated link, fixed rates, no deadline/energy/contention, heuristic (no optimization), IoU metric.
- UAV split computing under a per-frame deadline: CoTeD. One fixed split, compression adapted to measured
  bandwidth; image/onboard-only are baselines; single UAV; compute measured, not traded.
- Multi-user shared spectrum + latency/energy budgets + representation size: He/Quek 2312.01049 (feature width x
  fraction, analytic accuracy, one-shot T^max, no detection), Ye VTC 2023 / ENACHI (split points, queues or
  Lyapunov, classification), Liu-Lan-Huang JSAC 2023 (server batching). None model detection misses.
- Multi-camera shared uplink with onboard filtering: Vigil (2015), JCAB (image configs). Image-only menus.

Not found in the sources checked (shallow web scan + 4 deep reads; IEEE Xplore not searched directly):
- A per-frame, per-UAV decision over a MENU of representation families {compressed image, onboard tokens,
  tokens/low-q image + ROI crops, intermediate features}. Every work above has one runtime family (+ knobs);
  the other families appear only as baselines (CoTeD, He/Quek TC) or as fixed pipeline stages (EdgeDuet).
- An object-level onboard-miss model used INSIDE a resource-constrained decision. Qiu has the miss-aware reward
  but no resources; the resource-allocation papers have resources but no detection/miss model.
- The combination of (i) miss-aware value, (ii) onboard inference latency/energy charged against a per-frame
  deadline, (iii) multi-UAV shared-channel queueing, (iv) BS mission objective (on-time recall/AP).
- Closest pairings to cite as building blocks: Qiu (value of re-detection) + EdgeDuet (ROI hybrid, VisDrone) +
  CoTeD (UAV split, deadline) + He/Quek and Ye/ENACHI (multi-user shared-resource scheduling).

Remaining risks: CoTeD JOURNAL full text not read (may add runtime mode switching/multi-UAV); ENACHI, Shao TWC 2024
video analytics, Zhuang 2605.19290, Chukhno INFOCOM-W 2025, Qiu TCAD 2022 / Chakrabarti SEC 2021 (token bucket)
only at abstract level; Chinese-venue UAV semantic-comm papers not checked.
Deep-read sources: arxiv.org/html/2312.01049v4 ; zhouzimu.github.io/paper/infocom21-wang.pdf ;
par.nsf.gov/servlets/purl/10544084 (CoTeD conf.) ; github.com/Rexyyj/CoTeD ; arxiv.org/html/2410.18919v1 ; arxiv.org/abs/2310.12937
