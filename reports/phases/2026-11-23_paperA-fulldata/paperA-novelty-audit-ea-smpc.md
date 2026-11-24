# Paper A: novelty audit of error-aware selective MPC (EA-SMPC)

**Date:** 24/11/2026.

**Scope:** a focused literature check of the method in the Phase 2 full-data evaluation
([paperA-fd-phase2-result.md](paperA-fd-phase2-result.md)). The working title under audit is "Error-aware selective model
predictive control for deadline-constrained UAV image communications under uncertain wireless service rates".

**Method:**
- References were checked by web search and fetch against publisher pages, DOI resolvers, DBLP/researchr,
  arXiv or the authors' PDFs.
- A DOI is given only when at least one bibliographic source showed it.
- "DOI not confirmed" means the paper, authors and venue were confirmed but the DOI was not.
- Most entries were checked at abstract or record level. Full text was read only where marked (RobustMPC's error
  rule via the Pensieve text; the OSAP fallback rule via the arXiv HTML).
- No reference in this file comes from memory alone.

## 0. The six method elements audited

| Tag | Element (as implemented) |
|---|---|
| **E1** | **Incumbent-agnostic override layer.** Any incumbent (tuned heuristic H\*, PPO, QR-DQN, DSAC) proposes a mode. The layer keeps it unless an override is justified. |
| **E2** | **Online error tracking from realised service.** Each predicted per-slot rate is compared with the ACKed (realised) rate, in 2 dB expected-SNR bins. This gives the relative bias *b* and the standard error σ of the observed/predicted ratio. |
| **E3** | **Cold-start gate.** No override until the bin has at least 20 s of observations. |
| **E4** | **Uncertainty-band robustness check.** The band is [ℓ, h] = [max(min(1, 1+b) − 2σ, 0.05), max(1, 1+b) + 2σ]. A switch must stay beneficial with every sampled rate path scaled by ℓ and by h. |
| **E5** | **Paired-sample confidence switching.** There are 8 sampled rate paths with common random numbers. The layer switches only if the paired mean gain over the incumbent exceeds one standard error and the switch passes E4. |
| **E6** | **Compute-budgeted anytime MPC.** All modes are screened at scales 1 and ℓ with a 1-image lookahead. Then a full 4-image MPC runs with queue simulation, under a 50 ms anytime budget; if the budget runs out, the incumbent is kept. |

**Empirical finding that the novelty claim rests on:**
- With E2–E5, a nominal rate model gives the same return as the empirical ACK-calibrated model: H3b on TD is −0.10
  [−1.95, +1.76].
- Without gate and band, the nominal planner is much worse: H3 is +18.6 in favour of the calibrated arm.

**Important for positioning:** the matched-nominal arm is *not* feedback-free.
- ACK feedback still enters through the E2 tracker and the E4 band. Its median bias is about −0.35, so the band
  absorbs the nominal model's over-prediction.
- The finding is therefore "error feedback through a conservative band can replace calibrating the rate model". This
  is close in spirit to RobustMPC's error-normalised throughput and to OLLA's ACK-driven offset (see §1c, §1d).

## 1. Verified references and their overlap

**Overlap codes:** ● = same idea; ◐ = related or partial; blank = none.

### 1a. MPC with online model-error estimation, and learning-based, robust, stochastic or conformal MPC

| # | Reference (verified) | What it does | Overlap with E1–E6 |
|---|---|---|---|
| A1 | L. Hewing, J. Kabzan, M. N. Zeilinger, "Cautious Model Predictive Control Using Gaussian Process Regression," *IEEE Trans. Control Syst. Technol.* 28(6), 2020. DOI 10.1109/TCST.2019.2949757 (arXiv:1705.10702) | GP model of the residual (nominal-model error); MPC with chance constraints that use the GP variance | E2 ● (learns error from data, with variance); E4 ◐ (uncertainty-tightened constraints, not a scaled-band check); E6 ◐ (real-time, 20 ms sampling) |
| A2 | L. Hewing, K. P. Wabersich, M. Menner, M. N. Zeilinger, "Learning-Based Model Predictive Control: Toward Safe Learning in Control," *Annu. Rev. Control Robot. Auton. Syst.* 3:269–296, 2020. DOI 10.1146/annurev-control-090419-075625 (DOI seen in one index) | Survey: learning the model, learning the controller design, and MPC as a safety layer for learning controllers | E1 ◐ (MPC as a safety filter for a learned policy); E2 ● (model learning from data); E4 ◐ |
| A3 | A. Aswani, H. Gonzalez, S. S. Sastry, C. Tomlin, "Provably safe and robust learning-based model predictive control," *Automatica* 49(5):1216–1226, 2013. DOI 10.1016/j.automatica.2013.02.003 | LBMPC: a learned model is used for performance, while a nominal model plus a robust tube keeps safety | E1 ◐ (separates performance from safety); E2 ● (online learned model correction); E4 ● (robustness to a model-error set) |
| A4 | D. Q. Mayne, M. M. Seron, S. V. Raković, "Robust model predictive control of constrained linear systems with bounded disturbances," *Automatica* 41(2):219–224, 2005. DOI 10.1016/j.automatica.2004.08.019 | Tube MPC: a nominal trajectory plus a disturbance-invariant tube | E4 ● (decisions must hold over a bounded uncertainty set) |
| A5 | M. Lorenzen, M. Cannon, F. Allgöwer, "Robust MPC with recursive model update," *Automatica* 103:461–471, 2019. DOI 10.1016/j.automatica.2019.02.023 | Adaptive robust MPC: the parameter set is refined online from observed trajectories, and robust constraints use the shrinking set | E2 ● (online update of the uncertainty set from realised data); E4 ● (robustness over the estimated set) |
| A6 | A. Mesbah, "Stochastic Model Predictive Control: An Overview and Perspectives for Future Research," *IEEE Control Syst. Mag.* 36(6), 2016. DOI 10.1109/MCS.2016.2602087 | Survey of SMPC: sample-based and chance-constrained MPC | E5 ◐ (sampled-scenario evaluation); E4 ◐ |
| A7 | G. Schildbach, L. Fagiano, C. Frei, M. Morari, "The scenario approach for stochastic model predictive control with bounds on closed-loop constraint violations," *Automatica* 50(12), 2014. DOI 10.1016/j.automatica.2014.10.035 (arXiv:1307.5640) | Scenario MPC: optimises over sampled disturbance paths, with violation bounds | E5 ◐ (decisions from a finite set of sampled paths; our 8 paths are far below scenario-theory sample sizes) |
| A8 | L. Lindemann, M. Cleaveland, G. Shim, G. J. Pappas, "Safe Planning in Dynamic Environments using Conformal Prediction," *IEEE Robot. Autom. Lett.* 8(8), 2023 (arXiv:2210.10254; IEEE DOI not confirmed) | Conformal prediction regions on predictor errors, used inside MPC for probabilistic safety | E2 ● (prediction-error calibration from data); E4 ● (calibrated uncertainty region inside the planner, with a coverage guarantee our ±2σ band lacks) |
| A9 | I. Gibbs, E. Candès, "Adaptive Conformal Inference Under Distribution Shift," *NeurIPS* 34, 2021 (arXiv:2106.00170) | Online re-calibration of the interval width under drift; long-run coverage guarantee | E2 ● (online error tracking with drift); E4 ◐ (band width adapted online) |
| A10 | T. Li, R. Yang, G. Qu, G. Shi, C. Yu, A. Wierman, S. Low, "Robustness and Consistency in Linear Quadratic Control with Untrusted Predictions," *Proc. ACM Meas. Anal. Comput. Syst.* 6(1), Art. 18, 2022 (SIGMETRICS'22). DOI 10.1145/3508038 | λ-confident control: a trust parameter blends a prediction-based policy with a robust one; a self-tuning λ comes from observed prediction error | E1 ● (trust in predictions vs a robust default); E2 ● (online error drives trust); E5 ◐ (confidence-weighted, not test-based) |
| A11 | P. O. M. Scokaert, D. Q. Mayne, J. B. Rawlings, "Suboptimal model predictive control (feasibility implies stability)," *IEEE Trans. Autom. Control* 44(3):648–654, 1999. DOI 10.1109/9.751369 | Shows that feasible but suboptimal MPC (early termination) can stabilise | E6 ● (anytime or suboptimal MPC under limited computation) |
| A12 | M. Diehl, H. G. Bock, J. P. Schlöder, "A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control," *SIAM J. Control Optim.* 43(5):1714–1736, 2005. DOI 10.1137/S0363012902400713 | Real-time iteration: one Newton-type step per sample for real-time NMPC | E6 ● (compute-budgeted MPC) |

### 1b. Layers that override, filter or shield a learned or heuristic policy

| # | Reference (verified) | What it does | Overlap with E1–E6 |
|---|---|---|---|
| B1 | K. P. Wabersich, M. N. Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems," *Automatica* 129:109597, 2021. DOI 10.1016/j.automatica.2021.109597 | An MPC-based filter that passes the RL input through unless it would violate constraints, otherwise applies a minimal modification | E1 ● (policy-agnostic MPC layer that keeps the proposed action when acceptable); E4 ◐ (robust to model uncertainty); differs: safety, not performance gain |
| B2 | K. P. Wabersich, A. J. Taylor, J. J. Choi, K. Sreenath, C. J. Tomlin, A. D. Ames, M. N. Zeilinger, "Data-Driven Safety Filters: Hamilton-Jacobi Reachability, Control Barrier Functions, and Predictive Methods for Uncertain Systems," *IEEE Control Syst. Mag.* 43(5):137–177, 2023. DOI 10.1109/MCS.2023.3291885 (only the first three authors were seen in the result) | Tutorial unifying safety filters for uncertain, data-driven systems | E1 ● ; E2 ◐ ; E4 ◐ |
| B3 | M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, U. Topcu, "Safe Reinforcement Learning via Shielding," *AAAI* 2018. DOI 10.1609/aaai.v32i1.11797 | A shield monitors the RL agent's actions and corrects them only when a temporal-logic specification would be violated | E1 ● (override only when needed; agent-agnostic) |
| B4 | R. Cheng, G. Orosz, R. M. Murray, J. W. Burdick, "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks," *AAAI* 33, 2019. DOI 10.1609/aaai.v33i01.33013387 | A CBF-based layer minimally corrects RL actions; GP model of the unknown dynamics | E1 ● ; E2 ◐ (learned uncertainty model) |
| B5 | A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, P. Tabuada, "Control Barrier Functions: Theory and Applications," *ECC* 2019, pp. 3420–3431. DOI 10.23919/ECC.2019.8796030 | CBF tutorial; QP safety filters around nominal controllers | E1 ◐ |
| B6 | L. Sha, "Using simplicity to control complexity," *IEEE Software* 18(4):20–28, 2001. DOI 10.1109/MS.2001.936213 | Simplex architecture: a simple trusted controller guards a complex one | E1 ● (switching between a complex and a simple controller) |
| B7 | D. T. Phan, R. Grosu, N. Jansen, N. Paoletti, S. A. Smolka, S. D. Stoller, "Neural Simplex Architecture," *NFM* 2020, LNCS 12229, pp. 97–114. DOI 10.1007/978-3-030-55754-6_6 | Simplex for neural controllers, with *reverse switching* back to the learned controller and online retraining | E1 ● ; E3 ◐ (conditions for when to hand control back or forth) |
| B8 | T. Johannink et al., "Residual Reinforcement Learning for Robot Control," *ICRA* 2019, pp. 6023–6029. DOI 10.1109/ICRA.2019.8794127 | RL learns a residual on top of a conventional controller | E1 ◐ (layering on an incumbent; the opposite direction: learning on top of a classical controller) |
| B9 | W. Saunders, G. Sastry, A. Stuhlmüller, O. Evans, "Trial without Error: Towards Safe Reinforcement Learning via Human Intervention," *AAMAS* 2018 (extended abstract; arXiv:1707.05173) | A learned "blocker" imitates human interventions that override unsafe RL actions | E1 ● (selective intervention on an RL policy) |
| B10 | N. H. Rotman, M. Schapira, A. Tamar, "Online Safety Assurance for Learning-Augmented Systems," *HotNets* 2020, pp. 88–95. DOI 10.1145/3422604.3425940 (arXiv:2010.03625). Full text checked. | In ABR, falls back from Pensieve to buffer-based when uncertainty (OC-SVM novelty, policy-ensemble KL or value-ensemble spread) exceeds a threshold for *l* consecutive steps | E1 ● (incumbent vs fallback chosen online from uncertainty, same networking domain); E3 ◐ (persistence requirement before switching); E5 ◐ (threshold test on uncertainty, not on paired gain). **Closest networking precedent for "selective".** Direction is reversed: OSAP's default is the heuristic, ours is the incumbent. |
| B11 | T. Lykouris, S. Vassilvitskii, "Competitive Caching with Machine Learned Advice," *J. ACM* 68(4), Art. 24, 2021. DOI 10.1145/3447579 (ICML 2018) | Learning-augmented online algorithms: use predictions when they are good, with bounded worst case when they are bad | E1 ◐ (robustness/consistency framing for trusting an advisor) |
| B12 | S. Abbasloo, C.-Y. Yen, H. J. Chao, "Classic Meets Modern: a Pragmatic Learning-Based Congestion Control for the Internet," *SIGCOMM* 2020. DOI 10.1145/3387514.3405892 | Orca: DRL modulates a classic TCP controller (hybrid, two time scales) | E1 ◐ (hybrid of learned and classic control in networking) |

### 1c. Throughput-prediction MPC for adaptive bitrate (ABR) streaming

| # | Reference (verified) | What it does | Overlap with E1–E6 |
|---|---|---|---|
| C1 | X. Yin, A. Jindal, V. Sekar, B. Sinopoli, "A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP," *SIGCOMM* 2015, pp. 325–338. DOI 10.1145/2785956.2787486 | MPC over a 5-chunk horizon with throughput prediction and buffer. **RobustMPC** normalises the throughput prediction by the *maximum prediction error over the past 5 chunks*. **FastMPC** precomputes decisions offline (rule confirmed via the Pensieve text, C5). | E2 ● (online tracking of prediction error vs realised throughput); E4 ● (conservative, error-scaled throughput in the planner); E6 ● (FastMPC = compute-bounded MPC). **The closest prior art overall.** |
| C2 | Y. Sun et al., "CS2P: Improving Video Bitrate Selection and Adaptation with Data-Driven Throughput Prediction," *SIGCOMM* 2016, pp. 272–285. DOI 10.1145/2934872.2934898 | Data-driven throughput prediction (session clustering plus HMM) feeding MPC | E2 ◐ (empirical rate model replaces the nominal one, like our calibrated arm); E3 ◐ (initial prediction for new sessions from similar sessions) |
| C3 | Z. Akhtar, Y. S. Nam, R. Govindan, S. Rao, J. Chen, E. Katz-Bassett, B. Ribeiro, J. Zhan, H. Zhang, "Oboe: Auto-tuning Video ABR Algorithms to Network Conditions," *SIGCOMM* 2018. DOI 10.1145/3230543.3230558 | Detects network-state changes online and switches ABR parameters (including MPC's) to precomputed best values | E2 ◐ (online state estimation); E1 ◐ (layer that retunes any ABR) |
| C4 | F. Y. Yan, H. Ayers, C. Zhu, S. Fouladi, J. Hong, K. Zhang, P. Levis, K. Winstein, "Learning in situ: a randomized experiment in video streaming," *NSDI* 2020, pp. 495–511 | Fugu: MPC with a learned *probabilistic* transmission-time predictor trained in situ; finds that simple buffer-based control is hard to beat | E2 ◐ (an empirical predictor of realised service time); E4 ◐ (planning under a predicted distribution) |
| C5 | H. Mao, R. Netravali, M. Alizadeh, "Neural Adaptive Video Streaming with Pensieve," *SIGCOMM* 2017, pp. 197–210. DOI 10.1145/3098822.3098843 | DRL ABR; reports that RobustMPC's conservative error-normalised estimate is what makes MPC competitive | Context for E4: shows that *error-aware conservatism* matters more than the plain predictor, an observation that parallels our H3/H3b result |
| C6 | T. Huang, C. Zhou, R.-X. Zhang, C. Wu, X. Yao, L. Sun, "Stick: A Harmonious Fusion of Buffer-based and Learning-based Approach for Adaptive Streaming," *IEEE INFOCOM* 2020 (DOI not confirmed) | DRL tunes BBA bounds; a learned **Trigger** decides when to invoke the learned model, cutting overhead by 24–61% | E1 ◐ (invoke the expensive learned component only when needed); E6 ◐ (computation saving by selective invocation) |
| C7 | N. Kan, C. Li, C. Yang, W. Dai, J. Zou, H. Xiong, "Uncertainty-aware robust adaptive video streaming with Bayesian neural network and model predictive control," *NOSSDAV* 2021, pp. 17–24 (DOI not confirmed) | BayesMPC: a BNN throughput predictor gives uncertainty, and MPC optimises robustly using it | E2 ● ; E4 ● (uncertainty-aware robust MPC for rate decisions) |
| C8 | Z. Zhong, M. Liu, L. Yang, Y. Wang, Y. Xu, J.-N. Hwang, "Video Streaming with Kairos: An MPC-Based ABR with Streaming-Aware Throughput Prediction," arXiv:2503.14271, 2025 (preprint) | Multi-percentile throughput prediction; buffer-aware choice of the percentile fed to MPC | E4 ● (choosing a conservative quantile of the rate distribution inside MPC) |
| C9 | W. Li, J. Huang, W. Lyu, J. Wang, P. Patras, "VAMP: A Predictive Approach to Audio/Video Bitrate Adaptation Over Wireless Networks," *IEEE Wireless Commun. Lett.*, 2021/22. DOI 10.1109/LWC.2021.3134491 | Predictive-control ABR over wireless, published as a *letter* | Venue precedent: an MPC-style rate controller as a WCL letter; E6 ◐ |

### 1d. Link adaptation from ACK feedback (OLLA) and uncertainty-aware wireless or UAV control

| # | Reference (verified) | What it does | Overlap with E1–E6 |
|---|---|---|---|
| D1 | K. I. Pedersen, G. Monghal, I. Z. Kovács, T. E. Kolding, A. Pokhariyal, F. Frederiksen, P. Mogensen, "Frequency Domain Scheduling for OFDMA with Limited and Noisy Channel Feedback," *IEEE VTC-Fall* 2007, pp. 1792–1796 (DOI not confirmed) | LTE scheduling with noisy CQI; reports the effect of outer-loop link adaptation (OLLA) | E2 ● (OLLA: an ACK/NACK-driven additive offset that corrects the biased SINR-to-rate mapping) |
| D2 | A. Durán, M. Toril, F. Ruiz, A. Mendo, "Self-Optimization Algorithm for Outer Loop Link Adaptation in LTE," *IEEE Commun. Lett.*, 2015. DOI 10.1109/LCOMM.2015.2477084 | Speeds up OLLA convergence by initialising the offset from histograms of long connections | E2 ● ; **E3 ●** (the cold-start problem of an ACK-driven correction, handled by initialisation instead of gating) |
| D3 | F. Blanquez-Casado, G. Gomez, M. C. Aguayo-Torres, J. T. Entrambasaguas, "eOLLA: an enhanced outer loop link adaptation for cellular networks," *EURASIP J. Wireless Commun. Netw.* 2016:20. DOI 10.1186/s13638-016-0518-3 (abstract not retrieved) | Enhanced OLLA variant | E2 ● |
| D4 | S. H. Y. Wong, H. Yang, S. Lu, V. Bharghavan, "Robust rate adaptation for 802.11 wireless networks," *MobiCom* 2006, pp. 146–157. DOI 10.1145/1161089.1161107 | RRAA: rate adaptation from short-term loss ratios over windows of frames | E2 ● (rate choice from realised per-window delivery statistics) |
| D5 | V. Saxena, J. Jaldén, J. E. Gonzalez, M. Bengtsson, H. M. Tullberg, I. Stoica, "Contextual Multi-Armed Bandits for Link Adaptation in Cellular Networks," *NetAI@SIGCOMM* 2019, pp. 44–49. DOI 10.1145/3341216.3342212 | Bandit LA that learns from ACKs and improves on OLLA's offline model | E2 ● (learning the rate model from ACK outcomes) |
| D6 | R. Wiesmayr, L. Maggi, S. Cammerer, J. Hoydis, F. Aït Aoudia, A. Keller, "SALAD: Self-Adaptive Link Adaptation," arXiv:2510.05784, 2025 (preprint) | Estimates SINR from ACK/NACK only; picks the MCS by hypothesis testing; feedback-adjusted BLER target | E2 ● ; E5 ◐ (statistical test before acting) |
| D7 | K. M. Cohen, S. Park, O. Simeone, S. Shamai (Shitz), "Calibrating AI Models for Wireless Communications via Conformal Prediction," *IEEE Trans. Mach. Learn. Commun. Netw.*, 2023 (arXiv:2212.07775; IEEE Xplore doc. 10262367; DOI not confirmed) | (Online) conformal calibration of wireless predictors, including channel prediction | E2 ● ; E4 ◐ (a calibrated set in place of a heuristic ±2σ band) |
| D8 | B. Li, J. Luo, T. Charalambous, N. Pappas, "Model Predictive Communication for Timely Status Updates in Low-Altitude Networks," arXiv:2604.20610, 2026 (preprint) | UAV MPC with predicted channels for timing, power and channel allocation under freshness constraints | Application overlap: MPC for UAV communication control; no error-aware selective override |

### 1e. UAV, semantic, task-oriented or edge image and video transmission with mode or compression selection

| # | Reference (verified) | What it does | Overlap with E1–E6 |
|---|---|---|---|
| E1r | C. Huang, X. Chen, G. Chen, P. Xiao, G. Y. Li, W. Huang, "Deep Reinforcement Learning-Based Resource Allocation for Hybrid Bit and Generative Semantic Communications in Space-Air-Ground Integrated Networks," *IEEE JSAC* 43(12), 2025. DOI 10.1109/JSAC.2025.3623157 (arXiv:2412.05647) | DRL selects bit vs generative semantic modes and resources under a delay/quality metric | Application overlap (DRL mode selection = our incumbent class); no override layer |
| E2r | P. Si, J. Zhao, K.-Y. Lam, Q. Yang, "UAV-assisted Semantic Communication with Hybrid Action Reinforcement Learning," *IEEE GLOBECOM* 2023. DOI 10.1109/GLOBECOM54140.2023.10437643 | Hybrid-action RL for semantic model scale, channel, power and trajectory in UAV uplink collection | Application overlap (DRL incumbent) |
| E3r | L. Zeng, H. Chen, D. Feng, X. Zhang, X. Chen, "A3D: Adaptive, Accurate, and Autonomous Navigation for Edge-Assisted Drones," accepted to *IEEE/ACM Trans. Netw.* (arXiv:2307.09880; final DOI not confirmed) | DRL picks execution location, input resolution and image compression ratio for latency and accuracy | Application overlap (compression-mode selection under latency) |
| E4r | A. Coletta, F. Giorgi, G. Maselli, M. Prata, D. Silvestri, J. Ashdown, F. Restuccia, "A²-UAV: Application-Aware Content and Network Optimization of Edge-Assisted UAV Systems," *IEEE INFOCOM* 2023 (arXiv:2301.06363; DOI not confirmed) | Joint routing, compression and task assignment for vision tasks at the edge, with task deadlines | Application overlap (compression vs task success under deadlines) |
| E5r | C. Wang, S. Zhang, Y. Chen, Z. Qian, J. Wu, M. Xiao, "Joint Configuration Adaptation and Bandwidth Allocation for Edge-based Real-time Video Analytics," *IEEE INFOCOM* 2020. DOI 10.1109/INFOCOM41043.2020.9155524 | Online (Lyapunov) choice of resolution and frame rate per stream under bandwidth variation | Application overlap (per-item configuration under uncertain bandwidth) |
| E6r | J. Jiang, G. Ananthanarayanan, P. Bodík, S. Sen, I. Stoica, "Chameleon: Scalable Adaptation of Video Analytics," *SIGCOMM* 2018. DOI 10.1145/3230543.3230574 | Periodic re-profiling of video-analytics configurations | Application overlap; E6 ◐ (cost-bounded configuration search) |
| E7r | B. Zhang, X. Jin, S. Ratnasamy, J. Wawrzynek, E. A. Lee, "AWStream: Adaptive Wide-Area Streaming Analytics," *SIGCOMM* 2018, pp. 236–252. DOI 10.1145/3230543.3230554 | Profiles the accuracy-bandwidth trade-off; adapts the configuration to the measured WAN bandwidth | E2 ◐ (adaptation from measured delivered bandwidth) |
| E8r | N. Mehrabi et al., "Adaptive Data Transport Mechanism for UAV Surveillance Missions in Lossy Environments," arXiv:2410.10843, 2024 (preprint) | DRL selects image patches to transmit for detection | Application overlap only |

### 1f. Statistical method (used by E5)

| # | Reference (verified) | What it does | Overlap |
|---|---|---|---|
| S1 | P. Glasserman, D. D. Yao, "Some Guidelines and Guarantees for Common Random Numbers," *Management Science* 38(6):884–908, 1992. DOI 10.1287/mnsc.38.6.884 | Common random numbers (CRN) for variance-reduced comparison of stochastic systems | E5 ● (paired comparison over shared sampled paths is textbook CRN) |

**Total:** 50 references; some DOIs are unconfirmed (see §4).

## 2. Verdict per element

| Element | Verdict | Citing references | Notes |
|---|---|---|---|
| **E1** Incumbent-agnostic override layer | **Clearly prior art** (concept) | B1, B2, B3, B4, B6, B7, B9, B10, A2, A10; networking: B10, C6, B12 | Simplex, safety filters and shields keep the proposed action unless needed. Ours overrides for *expected-return gain*, not safety, and defaults to the incumbent. That is a framing difference, not a new mechanism. |
| **E2** Online error tracking from ACKed service | **Clearly prior art** | C1 (RobustMPC past-error normalisation), D1–D6 (OLLA and ACK-driven LA), A1, A5, A8, A9, A10, C7 | Binning by expected SNR and tracking bias plus SE of the ratio is an engineering variant. The per-SNR-bin bias correction is essentially OLLA's offset, made per bin. |
| **E3** Cold-start gate (≥ 20 s per bin) | **Clearly prior art / standard practice** | D2 (OLLA cold start), B10 (*l* consecutive OOD steps before switching), B7 (switching conditions), C2 (cold-start prediction) | A minimum-sample rule before trusting a statistic is not a contribution. Report it as a design detail. |
| **E4** Uncertainty band with ℓ/h robustness check | **Clearly prior art** (robust or conservative planning on an estimated set) | C1, C7, C8, A3, A4, A5, A1, A8 | Our band is a heuristic ±2σ multiplicative interval with no coverage or feasibility guarantee. Conformal (A8, A9, D7) or set-membership (A5) bands are the principled versions. |
| **E5** Paired-sample confidence switching (gain > 1 SE, CRN) | **Partially novel in this application** (simple statistics) | S1 (CRN), A7 (scenario sampling), B10 (threshold-based switching), A10 (confidence-weighted trust), D6 (hypothesis-test MCS) | No searched paper gates an override of a learned or heuristic incumbent with a paired-CRN gain-vs-SE test. The test itself is elementary, though, and 8 paths is a weak test. Claim it only as a design choice, validated by ablation. |
| **E6** Compute-budgeted anytime MPC (screen, then full) | **Clearly prior art** | C1 (FastMPC), A11, A12, C6 (selective invocation to save compute) | A wall-clock budget also makes the arms not bit-reproducible (already noted in E7); disclose this. |
| **Combination E1–E6** for multi-UAV image codec/mode selection under deadlines, with ACK-only feedback | **Partially novel in this application** | Closest: C1, B10, B1, D1/D2, A10; application: E1r–E5r | No searched work combines an incumbent-agnostic, test-gated MPC override with ACK-error tracking for UAV image mode selection. This is an application and integration contribution, not an algorithmic one. |
| **Empirical finding:** nominal ≈ calibrated under gate+band (H3b n.s.); gate+band recovers the whole nominal-vs-calibrated gap (H3 +18.6) | **Possibly novel as an evaluation finding** | Related observations: C5 (RobustMPC's conservatism is what matters), C4 (simple control is hard to beat), A10 | This is the most defensible "new" content. Frame it as an ablation result in this simulator, not a general law. |

## 3. Assessment

### 3.1 Defensible contribution (ICT Express letter or similar)

1. **Application and integration.** A drop-in, incumbent-agnostic, error-aware selective MPC layer for deadline-constrained
   multi-UAV image offloading. It uses only ACK feedback and runs within a 50 ms decision budget.
   - It improves a tuned heuristic incumbent under a pre-registered test: H1 +7.28 [+5.05, +9.51] on TD, replicated on
     F/V/TD_all.
   - Wording should be "we adapt and combine known ingredients (RobustMPC-style error-aware conservatism, OLLA-style
     ACK error tracking, safety-filter-style selective override)".
2. **Ablation insight, the strongest part.** Inside this control structure, a nominal rate model is as good as an
   empirically calibrated one (H3b −0.10 [−1.95, +1.76], not rejected; not evidence of equivalence).
   - Removing the gate and band costs about 18.6 return points.
   - So, in this setting, the value lies in *how prediction error is fed back into the decision rule*, not in rate-model
     fidelity.
   - This parallels Pensieve's observation about RobustMPC (C5) and should be presented as consistent with it.
3. **Honest baselines.** The "RobustMPC-style" arms are our own re-implementations.
   - The band variant is near-inactive. The no-band variant is much worse.
   - These results are about the two implemented variants, not about RobustMPC in general (as the pre-registration
     already requires).

### 3.2 Claims to avoid

- "First" or "novel" for MPC with online model-error estimation, robust or uncertainty-aware MPC, or prediction-error-aware
  ABR-style control. Blocked by C1, C7, A1, A3, A5, A8.
- "Novel override, shield or safety-filter layer for DRL". Blocked by B1, B3, B4, B6, B7, B10.
- Any **guarantee** language: safety, coverage, recursive feasibility, stability, "robust to all rate errors".
  - The ±2σ band is heuristic.
  - 8 sampled paths do not give scenario-approach bounds (A7).
  - There is no coverage property, unlike conformal methods (A8, A9).
- "The ACK-calibrated rate model improves performance". H3b is not rejected on TD; only F is CI > 0.
- "The method works without feedback" or "the nominal model suffices". The matched-nominal arm still uses ACK feedback
  through the tracker and band.
- "Outperforms RobustMPC". Only our two RobustMPC-*style* variants were tested.
- "Task-oriented" or "semantic" improvement. The objective is return; recall gains are small (+0.75 seed-level, and the
  video-cluster CI includes 0). The nominal planner has the highest recall.
- "Real-time on UAV hardware". The 50 ms budget was measured on the simulation host.
- "DRL is unnecessary" or general DRL-vs-MPC conclusions beyond the evaluated incumbents and simulator.

### 3.3 References that must be cited and discussed (6)

1. **C1, Yin et al., SIGCOMM 2015 (RobustMPC/FastMPC).** Error-normalised conservative rate prediction inside MPC, and
   compute-bounded MPC. This is the direct precedent for E2+E4+E6.
2. **B10, Rotman, Schapira, Tamar, HotNets 2020 (OSAP).** Online, uncertainty-triggered switching between a DRL policy
   and a heuristic in a networking controller. This is the direct precedent for "selective" (E1/E3/E5).
3. **B1, Wabersich & Zeilinger, Automatica 2021 (predictive safety filter).** The canonical MPC layer that keeps a
   learned policy's action unless it must intervene (E1).
4. **D2, Durán et al., IEEE Commun. Lett. 2015 (with OLLA, D1).** ACK-driven correction of a nominal SINR-to-rate
   mapping, including its cold-start issue (E2/E3).
5. **A10, Li et al., SIGMETRICS 2022 (λ-confident control).** Trusting predictions in proportion to observed error, with a
   robust default (E1/E2/E5).
6. **A5, Lorenzen et al., Automatica 2019, or A8, Lindemann et al., RA-L 2023.** Principled online-estimated uncertainty
   sets inside MPC. Cite one to show awareness that our band is the heuristic counterpart (E4).

**Application context:** also cite at least E1r (Huang et al., JSAC 2025) and one of E3r/E4r, so the UAV
mode-selection setting is placed in its literature.

## 4. Items not verified or deliberately excluded

- **Not confirmed (DOIs):** IEEE DOIs of A8, C6, D1, D7 and E4r; the ACM DOI of C7; the final ToN DOI of E3r. Do not
  print these DOIs until checked on IEEE Xplore or the ACM DL.
- **Not retrieved:** the abstract of D3 (eOLLA), so its mechanism is not described here.
- **Excluded:** BOLA, DeepMPC (ICIP 2020) and the classic OLLA paper by Nakamura et al. (2002). They were not verified in
  this pass.
- **Excluded:** several 2025–2026 UAV semantic-communication papers listed in the 19/11 audit
  (`pdf/SwarmBridge_PaperA_Novelty_Audit_2026-11-19.md`). They were not re-verified here.
