# Axis 2 notes: VoI, task-oriented multi-source scheduling, beliefs and coordination (2026-11-11)

Scope: the theory and method side of the candidate problem. Several UAVs with overlapping views decide what to contribute to a BS's shared perception under limited resources. One UAV's send changes what the others should send. U2U links are optional.

Not redone here: CCSRA, GOSC (basics), GOST, AoII, DSSE (`r0-notes/collaborative-swarm.md`); Glimpse, DDS, AoIS (`r0-notes/r1b-feedback-video.md`).

Conventions:
- Depth tags:
  - F = full read.
  - P = method, system model and evaluation read from pdftotext; the rest skimmed.
  - A = abstract, search snippet, aggregator page, or my prior knowledge (marked [mem]).
- p.N = PDF page counted from the pdftotext form feeds.
- PDFs are in `scratchpad/papers_axis2/` (session temp; not in the repo).
- I never write "first". "Not found" means not found in the sources checked.

---

## 0. Headline findings

1. **GOSC already ran the experiment R1b wanted, in multi-agent form, and found that the precise receiver/peer model adds little.** (Re-read: Rezazadeh et al., arXiv 2609.39477.)
   - A decision-aware theory-of-mind (ToM) VoI value was significantly better than the throughput or AoI values in only 1 of 15 matched-resource comparisons (p.8, sec. V-C, Table VI).
   - An exact set-valued scheduler (joint ToM value, enumeration) gave no improvement in 1,800 paired missions (p.9).
   - A single message changed a teammate's next target in only 6-12% of 914 sampled decisions (p.9).
   - An oracle ToM model that knows the teammates' private evidence was 3.1 [-4.3, 10.7] slots faster, which is not significant (p.11).
   - What mattered: (a) the value is coarsely right (Spearman >= 0.89 is enough, but content-independent values fail in search, p.8-9); (b) a price-of-delivery threshold that can say "no"; (c) the common-knowledge broadcast.
   - With 2 agents or abundant budget (B >= 200), "report everything" beats GOSC (p.11).
   - This replicates both R1b failure modes: the oracle is not an upper bound, and gains appear only in a binding-budget regime.
2. **For one sender, the receiver's state is almost entirely a function of what was delivered.**
   - GOSC Proposition 1 (p.4): with exactly-once fusion and an error-free broadcast, agent i's belief = shared common knowledge + i's own undelivered records. A sender therefore knows the shared component "for free" through ACKs.
   - The broadcast carries information the sender cannot reconstruct only when OTHER agents contribute to the shared state. A second source is BS-side computation the sender cannot replicate (in R1b: YOLO11s outputs on FRAMEs).
   - So a single-UAV R1b could only ever gain from the second source (consistent with +1.24 F1 for C1). The multi-UAV setting is where receiver-state feedback is non-redundant by construction.
3. **The overlapping-coverage remote-estimation theory is new (Sep 2026) but structurally thin.** Sun and Pappas (MobiHoc '26):
   - "overlap" = several sensors can send the SAME exact state of an independent Markov source;
   - the scheduler is centralized with full information;
   - there is no complementarity and no image data.
   - Correlated-source AoI work (Shisher et al.; Zakeri et al.; Kalør and Popovski; Tripathi and Modiano) models correlation as "sensor n carries source m's state with probability p_mn". None I checked derives correlation from real overlapping imagery or object identity.
4. **Multi-view task-oriented edge inference (Shao, Mao, Zhang, TWC) handles cross-device redundancy in two ways:** (i) offline, through distributed-IB coding; (ii) online, through server-side stopping plus attention-selected retransmission (VDDIB-SR). Its own footnote says the feedback signalling cost is "nonnegligible" for large T and is not modelled (p.11, footnote 4). Data: MNIST/CIFAR/TinyImageNet views, WRGBD, ModelNet40. These are static objects, not scenes.
5. **Submodular coordination theory directly prices "who must talk to whom".** Xu, Garimella and Tzoumas (T-RO 2025): f(A_RAG) >= 1/(1+κ_f) · [f(OPT) − Σ_i coin_i(N_i)].
   - coin_i = how much agent i's action overlaps with its NON-neighbours' actions (p.8-9).
   - For camera-footprint coverage, coin_i is computable a priori from geometry; it is 0 when non-neighbours are farther than 2·r_s (Example 1, p.9).
   - This is a ready-made theory for "coordinate only with UAVs whose footprints overlap yours".

---

## 1. (a) Goal-oriented, multi-source scheduling, remote estimation and freshness with correlated or overlapping sources

### 1.1 Sun and Pappas, "Resource-Constrained Semantic-Aware Remote Estimation with Overlapping Sensor Coverage", MobiHoc '26 (arXiv 2609.07563). Depth P (p.1-6)
- **Setting:** M finite-state Markov sources that evolve INDEPENDENTLY. K sensors with a bipartite coverage graph (a source can be seen by several sensors). Shared TDMA uplink: at most one (source, sensor) edge per slot. Per-sensor success q_k, delay d_k in {0,1}, global plus per-sensor transmission-frequency budgets (p.2-3).
- **How correlation enters:** only as alternative paths. Each covering sensor delivers the identical exact state, so the choice is about link reliability, delay and budget. There is no complementarity and no partial observation.
- **Objective:** cost of actuation error (CAE), an asymmetric, state-dependent mismatch cost (eq. 5, p.3).
- **Assumption 2.1 (p.3):** the scheduler knows all source states and receiver estimates. ACKs are "immediate and error-free". Coordination and feedback overhead are NOT counted (centralized full information).
- **Theory:**
  - the K+1 constraints are linearly dependent, so there are only K effective prices;
  - the optimum is a mixture of at most K+1 deterministic policies (Thm 3.5);
  - the Lagrangian value is piecewise-linear and concave;
  - projected dual subgradient ascent reaches a duality gap of 3.7% after 40 iterations (p.6).
- **Evaluation:** small synthetic instances. No baselines that matter for us. Scalable policies are "future work" (p.6).
- **Use for SB:** a formal anchor for "pick a source AND which UAV reports it". It shows the theory community has not yet handled complementary or partial overlap.

### 1.2 Shisher, Tripathi, Chiang, Brinton, "AoI-based Scheduling of Correlated Sources for Timely Inference", arXiv 2509.01926v2 (Dec 2025; ICC 2025 abridged). Depth P (p.1-11)
- **Setting:** M sources over a channel that admits N < M per slot, reliable. The receiver infers M targets. The scheduler is signal-agnostic: it uses AoI only, never the content (p.2-3).
- **How correlation enters:** the penalty of target m depends on the AoI of ALL sources. The result is a restless multi-armed bandit (RMAB) with non-separable penalties.
- **Theory:**
  - Lemma 2 (p.5): an information-theoretic lower bound f_m(Δ_m) under a Markov-chain condition.
  - Theorem 1 (p.6): the decomposition gap is bounded by the correlation parameter ε_m,n. The approximation is tight when correlation is LOW.
  - Maximum Gain First (MGF) gain-index policy; Theorem 2: the optimality gap scales with ε and M.
  - Online-MGF learns the penalties.
  - For 2 sources, a cyclic policy is optimal (Thm 4, appendix).
- **Evaluation:** synthetic Gauss-Markov. Source 1 carries the other sources' states with probability p (p.10). Baselines: max-age-first (MAF), max expected error, randomized, EMAM.
- **Overhead:** none modelled. The paper names distributed scheduling and signal-aware scheduling as future work (p.11).
- **Use:** a cheap, content-free baseline with correlation awareness. Its approximation is worst in exactly the high-overlap regime we care about.

### 1.3 Zakeri, Moltafet, Codreanu, "Goal-Oriented Remote Tracking Through Correlated Observations in Pull-based Communications", arXiv 2503.12962v3 (letter, 2025). Depth P (p.1-6)
- **Setting:** 2 independent binary Markov sources and 2 sensors. Sensor i also carries source j's state with probability ρ_ij ("inter-sensor communications or overlapping monitoring regions", p.2). The monitor PULLS one sensor per slot over an error-prone link.
- **Method:** a POMDP turned into a belief MDP, with the belief expressed via AoI, truncated, and solved by relative value iteration (RVI). The estimator is minimum-mean-distortion, not "last sample". The authors stress that the estimator choice changes the optimal policy (p.1-2, citing their [7]).
- **Results (synthetic):** beats max-age-first and age-optimal. Cost falls as correlation rises, especially for real-time error (Fig. 3, p.5).
- **Overhead:** pull commands are not costed.
- **Use:** a pull-based design removes the need for the sender to know the receiver's state at all. The receiver, which holds the state, decides whom to ask.

### 1.4 Older correlated-source AoI (context only)
- **Kalør and Popovski**, arXiv 1811.06453. Depth A. Independent sources, each observed by several sensors with Bernoulli observation probabilities. Known-parameter and learned schedulers exploit the common observations and give "large" AoI reductions.
- **Tripathi and Modiano**, MobiHoc '22 "Optimizing AoI with correlated sources". Depth A. Same probabilistic model.
- **Erbayat, Maatouk, Zou, Subramaniam**, MobiHoc '24, arXiv 2404.08638. Depth A. Multi-process, multi-sensor, queueing error analysis.
- **Luo and Pappas**, arXiv 2403.16855. Depth A. Multiple Markov sources with CAE on one lossy transmitter. The optimum is a mixture of 2 deterministic policies. Drift-plus-penalty (DPP) and RL variants. "Continuous transmission is inefficient."
- **Common thread:** correlation = "probability a packet carries another source's state". Not found: correlation derived from geometry, occlusion or object identity.

### 1.5 Uysal et al., "Semantic Communications in Networked Systems: A Data Significance Perspective" (`pdf/`; IEEE Network 2022 [mem]). Depth P (grep)
- A vision paper. It defines VoI as the benefit of a sample minus its transmission cost, with the inverted-pendulum example where VoI is positive only at a few instants (p.4, Fig. 3). It also covers query-AoI for pull-based use (p.4-5).
- It calls for semantic-aware feedback. Fig. 5: with constrained feedback, a BS that gives feedback only to selected users beats per-packet equal feedback (p.6). This supports the idea that feedback itself should be scheduled by value.
- It has no multi-source correlation model.

### 1.6 Other freshness metrics (pointers only, depth A)
- Version age / gossip: Yates "Age of Gossip" (version age on a complete graph grows ~log n); Kaswan, Mitra, Srivastava, Ulukus survey, arXiv 2312.16163. Relevant only if U2U relaying of BS state is considered.
- Whittle index for functions of age: Tripathi and Modiano, arXiv 1908.10438. Channel-aware AoI: Dedhia and Moharir.
- "Age of actuation": not checked.

### 1.7 UAV-specific semantic tracking (beyond GOST and Zhou-AoIS)
- **Tang, Feng, Quek**, arXiv 2411.06136 (2024). Depth A. A leader and follower UAVs over MIMO fading channels. Drift-plus-penalty gives a closed-form decentralized semantic transmit/control policy that adapts to tracking-error cost and the local channel. It is about control-loop communication, not perception redundancy.
- **Tang, Stavrou, Kountouris**, arXiv 2602.04971. Depth A. Multi-sensor scheduling with semantic over-the-air aggregation for Kalman remote estimation. Finite-horizon DP; the Q-function structure is "semantic" (it depends on error covariance and channel).
- **Lai, Tsai, Han**, arXiv 2603.24712 (Globecom '26 submission). Depth A. "EPIC": stale peer observations are projected into a predicted belief so that coordination does not depend on jitter. Weighted-coverage efficiency +10.5% under silence. A predict-through-staleness idea, not redundancy scheduling.

---

## 2. (b) Distributed VoI, redundant observations, submodular selection, multi-view edge inference

### 2.1 Shao, Mao, Zhang, "Task-Oriented Communication for Multi-Device Cooperative Edge Inference", IEEE TWC (arXiv 2109.00172v3). Depth P (p.1-12)
- **Setting:** K devices see different views of the SAME object and send features to an edge server. The task is classification or recognition.
- **How correlation enters:**
  - (i) IB-based extraction removes task-irrelevant correlated information (Remark 1, p.4).
  - (ii) Distributed deterministic IB (DIB/DDIB) with variational approximation (VDDIB) controls bits.
  - (iii) **Selective retransmission (VDDIB-SR), sec. V, p.7-8:** the server outputs a confidence after each attempt and stops when it exceeds a threshold. An attention module picks WHICH devices retransmit more feature chunks, given what has already been received.
- **Comm model:** bit budgets, ideal link. The feedback signalling (stop decision and per-device activation) is NOT costed; footnote 4 (p.11) admits it is "nonnegligible" when T is large.
- **Data:**
  - MNIST, CIFAR-10, TinyImageNet split into views;
  - WRGBD (3 cameras);
  - ModelNet40 (12 views).
- **Results:** e.g. ModelNet40 91.31% at 240 bits vs 87.87-90.03% for baselines (Table IV, p.10). Latency 560 ms -> 2.5 ms vs data-oriented (p.11).
- **Use:** the closest "server-side, receiver-driven who-sends-more" mechanism with real image views. It uses static objects only, no time dimension, and learned gating.

### 2.2 Shao, Zhang, Zhang, "Task-Oriented Communication for Edge Video Analytics", arXiv 2211.14049v3 (2024). Depth A
- Deterministic IB plus a temporal entropy model. Multi-camera pedestrian occupancy (Wildtrack/MultiviewX family, per the repo). About 125-320 KB per setting. It removes spatial and temporal redundancy by coding, not by scheduling.

### 2.3 Fang et al., "Prioritized Information Bottleneck ... for Edge Video Analytics" (PIB), IEEE/ACM ToN (arXiv 2409.00146v3). Depth P-light (p.1-6, p.12)
- Multi-camera edge video with a priority from an MLP over SNR, delay and object count.
- Camera and edge-server selection is posed as a combinatorial multi-armed bandit (CMAB). The reward is the marginal MODA gain Δ = MODA(K∪{k}) − MODA(K) "to incorporate submodularity" (p.6, eq. 10).
- FDMA uplink, interference densities 10-100 devices per 100 m² (p.12). Wildtrack.
- Claims +17.8% MODA and −82.65% communication vs TOCOM-TEM and codecs.
- I did not check whether the coordination and gate signalling are costed.

### 2.4 Palena, Cerquitelli, Chiasserini, "Edge-device Collaborative Computing for Multi-view Classification", arXiv 2409.15973 (2024). Depth P (p.3-11)
- **Mechanism:** the BS-side "Context Manager" sends each node a context, i.e. the pooled view embedding (or mean colour histogram) from the PREVIOUS period. A node transmits only if similarity(view, context) < θ (p.6-9).
- This is a receiver-state broadcast used as a novelty filter, across several sources.
- **Overhead:** counts all messages, headers and control packets per inference (p.11). Context dissemination appears to be included.
- **Data:** ModelNet40, 12 views (p.10).
- **Results:** 18-74% less transmitted data vs centralized, at an accuracy cost that depends on θ. The discard ratio is independent of the number of sources (p.11).
- **Caution:** the context is one period stale, and the objects are static shapes.

### 2.5 Mazandarani, Shokrnezhad, Taleb, Günlü, "Exploiting Overlapping Fields of View for Redundancy-Aware Uplink Transmission in Vehicular 6G", IEEE MECOM 2026 (arXiv 2606.31715v3). Depth P+ (all 6 pages, figures not inspected)
- **Problem (P), p.2-3:** choose which patch, which channel and which slot per user. Objective = number of users whose BS reconstruction passes an SSIM/LPIPS threshold. Constraints: V2V vs V2I channel groups and collision cliques.
- **Method (p.3-5):**
  - Phase 1: V2V observation sharing for D_OS slots, prioritized by a geometric FoV overlap prior.
  - Phase 2: pairwise patch similarity (SSIM after view transformation), then non-redundant, object-containing patches are sent to the BS.
- **Coordination overhead IS counted** as V2V slots taken from the frame. Performance is unimodal in D_OS: too short and redundancy is undiscovered (behaves like SOTA); too long and V2I time is lost (p.5-6).
- **Gain is largest in the moderate-resource regime** (p.5).
- **Data:** CARLA, 40 vehicles, 10 s, 10 fps, 1280x720, 10x10 patches.
- **Caution:** "Ours UB" assumes PERFECT view transformation; the LB has no transformation. The metric is pixel/perceptual, not task-level. There is no detector-based mission metric.

### 2.6 Xu, Garimella, Tzoumas, "Communication- and Computation-Efficient Distributed Submodular Optimization in Robot Mesh Networks" (RAG), IEEE T-RO 2025 (arXiv 2407.10382v3). Depth P (p.1-9)
- **Setting:** monotone (doubly) submodular coverage or information objective. Each robot chooses its action using only information from and about its neighbours. Real r2r rates of 0.25 Mbps (XBee) and 100 Mbps (p.5).
- **Theory:**
  - Theorem 1 (p.9): f(A_RAG) >= 1/(1+κ_f) [f(OPT) − Σ_i coin_f,i(N_i)], with κ_f the curvature.
  - coin_f,i = f(a_i) − f(a_i | actions of non-neighbours), i.e. the overlap with non-neighbours. For entropy objectives it equals mutual information (p.8).
  - Fully connected gives >= 1/(1+κ) >= 1/2.
  - Decision time is linear in network size, vs cubic for near-optimal methods.
  - Sparser networks lose approximation only sublinearly.
- **Evaluation:** AirSim road detection with 15 and 45 drones; coverage.
- **Overhead:** communication delays are part of the decision-time model.
- **Use:** gives a principled "who needs to coordinate" rule for UAVs with overlapping footprints, with a bound.

### 2.7 Classic submodular sensor and robot selection (depth A [mem] unless noted)
- Greedy for monotone submodular under a cardinality constraint: 1 − 1/e (Nemhauser, Wolsey, Fisher 1978). Under a knapsack constraint: density greedy plus best single element gives (1 − 1/e)/2; partial enumeration recovers 1 − 1/e (Sviridenko 2004).
- Sequential greedy over agents with a partition matroid (each UAV picks one action/message set): 1/2 (Fisher, Nemhauser, Wolsey 1978); cited as SG in Xu p.4.
- Krause and Guestrin: near-optimal sensor placement and VoI is submodular under conditional-independence assumptions. Without them, VoI is NOT submodular in general. This caveat matters for detection, because complementarity (two partial views jointly confirm an object) breaks diminishing returns.
- Corah and Michael, "Scalable Distributed Planning for Multi-Robot, Multi-Target Tracking", IROS 2021 (arXiv 2107.08550). Depth A. Parallel planning that ignores some robots' decisions; a bound that uses target independence to quantify redundancy; close to sequential greedy with 2-8 steps for 96 robots.

### 2.8 Team decision and remote estimation with multiple sensors
- **Zhang, Vasconcelos, Cui, Mitra**, "Distributed remote estimation over the collision channel with and without local communication", arXiv 2005.11438. Depth P (p.1-11).
  - n sensors with i.i.d. observations; collision channel of capacity k < n.
  - The MSE under a symmetric threshold is quasi-convex in the threshold. The decentralized threshold is "reasonably close" to a centralized lower bound (Fig. 3, p.9-10).
  - Local (graph) communication is used only to LEARN distribution parameters by consensus or quantile estimation, not to coordinate per-slot decisions.
  - Builds on the Vasconcelos and Martins TAC 2017 team-optimal threshold policies (depth A).
  - **Lesson:** with independent sources, per-slot peer-state knowledge buys little. A correctly tuned local threshold is near-centralized.
- Leong, Dey, Quevedo, TAC 2017 (depth A): for variance-based scheduling with drops, a threshold is optimal for a scalar single sensor; counterexamples exist for vectors. Sending estimates beats sending measurements under drops.

---

## 3. (c) Partial observability, shared beliefs, when communication helps

### 3.1 GOSC re-check on common knowledge and shared uplink (`pdf/`, arXiv 2609.39477). Depth P (p.1-11)
- **Common knowledge:**
  - The edge fuses decoded packets and broadcasts the result (p.3). The default broadcast is error-free; V-F3 studies loss and delay.
  - Prop. 1 decomposition (p.4). With a lossy broadcast, agent i uses its last received snapshot. Delivered messages are never re-sent (HARQ).
- **Costing (p.4, sec. III-D):** the total cost includes uplink data, scheduling requests, the broadcast payload plus overhead per broadcast, 1 bit of HARQ per uplink transmission, and DL grants at 2 bit per channel use. Under total cost, GOSC still needs 1.17-16.6x fewer channel uses than periodic, except Search at 16-bit overhead: 0.88 (Table IX, p.10).
- **Shared uplink (IV-F, p.6):**
  - The BS broadcasts a price μ_t (8 bits per slot).
  - Each agent applies its decoupled rule at λ + μ_t and sends a 16-bit request (channel uses plus value density).
  - The BS grants in decreasing value density while requests fit, and updates μ multiplicatively.
  - Baselines send an 8-bit buffer-status report and receive max-min water-filling. Each grant costs 32 DL bits.
  - A central exact network knapsack "used more control signalling than it saved" (p.6).
  - Result (V-F2, p.10): rescue ×4.8 [4.4, 5.2] fewer uplink channel uses (×4.6 total). Search is significant only with the largest shared budget.
- **Theorems:**
  - Thm 1: send iff value > price of a successful delivery; more valuable messages get more robust rates.
  - Thm 2: the Lagrangian relaxation is exact when the price keeps the budget slack, which held in 95.2-95.8% of slots (p.6).
  - Prop. 2: a value error matters only near the threshold, and the loss is bounded by the price of the reallocated channel uses (p.6).
- **Downlink impairments (p.11):** GOSC is faster than periodic at broadcast loss 0.3 and 0.5. With delay up to 10 slots there is no significant difference, and everyone slows down.
- **Limits:**
  - Grid world with Pd/Pfa cells. Records are cell-indexed, so "already known" is exact. There is no association or identity uncertainty.
  - The value is computed per message, not as a submodular set (the set-valued exact value did not help).

### 3.2 Classic Dec-POMDP execution-time communication (depth A)
- **ACE-PJB-COMM**, Roth, Simmons, Veloso, AAMAS 2005. Plan centrally assuming free communication. At run time, communicate local history only if the expected team utility gain exceeds the cost. Summarized in Kwak et al., "Teamwork and Coordination under Model Uncertainty in DEC-POMDPs" (AAAI-WS 2010, p.2; read P).
  - MAOP-COMM (Wu, Zilberstein, Chen 2009) communicates on "history inconsistency".
  - STC (Kwak et al.) uses BDI trigger points plus a pruned joint-belief tree, and is robust to model error.
- **Becker, Carlin, Lesser, Zilberstein**, "Analyzing myopic approaches for multi-agent communication", Comput. Intell. 2009. The myopic VoC assumption (communicate now or never) can cause poor behaviour; they give a relaxed version. Same point as R1b's per-tick heuristic trigger.
- **Liu and Zhang**, ICML 2023 (arXiv 2308.08705). Information sharing (common information) makes POSG planning and learning quasi-efficient. Formal support for "a shared or broadcast state is what makes decentralized decisions tractable".

### 3.3 Learned communication gating (MARL) (depth A unless noted)
- IC3Net (Singh et al., ICLR 2019, gate per agent) [mem]. ATOC (Jiang and Lu, NeurIPS 2018, attention-chosen initiators) [mem]. TarMAC (Das et al., ICML 2019, targeted attention) [mem]. SchedNet (Kim et al., ICLR 2019: learned weights, top-k agents get the shared medium) [mem]. Gated-ACML (Mao et al., AAAI 2020, prune messages, arXiv 1912.05304). When2com and Who2com (covered in R0).
- **I2C** (Ding, Huang, Lu, NeurIPS 2020, arXiv 2006.06455). Agent i talks to j only if KL(P(a_i | a_−i, o) || P(a_i | a_−ij, o)) is large, i.e. only when the other agent CHANGES its action distribution. Learned as a prior network. This is the "communicate only if it changes the action" criterion.
- **MUTE** (Zuo et al., arXiv 2607.03473, Jul 2026; via an aggregator page).
  - Message value = counterfactual change in global Q when the message is masked. 80-90% bandwidth reduction on Hallway, Traffic Junction and SMAC/SMACv2.
  - It explicitly argues that information-theoretic proxies (action-distribution change) can select messages that change actions without improving return, "may even result in negative return impact if the information is misleading".
  - This is exactly R1b's D failure (more accurate belief -> TOK content that added false positives).
- **ET-MAPG / AET-MAPG** (Siddique, Sinha, Cao, arXiv 2509.20338): joint control plus event-trigger policy, attention for whom/what, >50% less communication.
- **Priority-driven control and communication** (Guo et al., Aalto, arXiv 2605.10482): learned continuous priorities; top-L agents get slots. This avoids a hybrid action space. Whether the priority-exchange cost is counted: not checked.
- **ToM2C** (ICLR 2022) and **Thomas, Strinati, Saad** (ToM pragmatic SemCom, arXiv 2311.18224): cited by GOSC; depth A.
- **Common limitation:** benchmark games (SMAC, MPE, Traffic Junction). Ideal channels. Message bits rarely priced against a wireless budget. No perception from real images.

---

## 4. (d) Multi-UAV semantic tracking (additions to R0)
- **GOST** re-skim (TMC 25(10), Oct 2026). Depth P. AoI per UAV in the reward (eq. near p.8). ARIMA, Kalman and GAT inference fills stale twin data. MADDPG. Point targets. No cross-UAV redundancy term found (grep "redundan", "overlap": none).
- **Zhou et al. AoIS** (JWCN 2026) and **Han et al. AoIS** (arXiv 2508.12248): already in R1b notes. Per-UAV freshness × mismatch; no cross-UAV coupling.
- **Tang, Feng, Quek 2411.06136, Lai et al. 2603.24712:** see 1.7.
- Not found: a multi-UAV semantic tracking paper whose objective couples UAVs through shared object identity (the value of UAV i's report depends on whether UAV j already reported the same object).

---

## GAPS (candid)

1. **Correlation models in theory are probabilistic stand-ins.** "Sensor n carries source m's state with probability p" (Kalør, Tripathi, Zakeri, Shisher) or identical alternative paths (Sun and Pappas).
   - Not found: scheduling theory where redundancy comes from geometric footprint overlap plus object identity, with complementary partial views (occlusion, resolution).
   - Diminishing-returns (submodular) value is assumed (PIB, RAG). Not found: a test of whether detection-level value from real overlapping drone views is actually submodular. Complementarity (two weak detections jointly confirm) would break it.
2. **Coordination cost is rarely counted.** Counted only in GOSC (price, request, grant and broadcast bits), Palena (control packets) and Mazandarani (V2V slots).
   - Not counted in Sun and Pappas, Shisher, Zakeri, Shao VDDIB-SR (admitted), PIB (not checked), or most MARL gating.
   - Not found: any study that sweeps coordination overhead against the redundancy level on real multi-drone imagery.
3. **Association uncertainty is ignored.** GOSC's "already known" is exact (cell index). Image-based redundancy needs cross-view association, which is noisy; CCSRA uses embedding similarity.
   - Not found: a scheduler that treats association error as a value error. GOSC Prop. 2 could bound its effect.
4. **Task metric and data.** Multi-view task-oriented inference uses static objects (ModelNet40, WRGBD) or pixel metrics (Mazandarani SSIM/LPIPS, CARLA). Not found: on-time unique-object recall at the BS under deadline and contention with real overlapping drone video.
5. **Evidence against the core hypothesis.** GOSC finds precise peer modelling (ToM) unnecessary once the scheduler is price-aware. Zhang/Vasconcelos find decentralized thresholds near-centralized for independent sources.
   - A Q1 claim that "cooperation intelligence" helps must therefore show a regime where independent price-aware thresholds plus BS-side dedup clearly lose. Likely: high overlap, binding budget, moderate resources.
   - That regime was not found demonstrated with real imagery in the sources checked. It may be narrow (Mazandarani: moderate-resource regime only; GOSC: not for 2 agents or B >= 200).
6. **Sources not read in full:** I2C, MUTE (aggregator only), Becker 2009, Roth 2005 (secondary), Kalør, Tripathi-Modiano, Erbayat, Corah, the IC3Net/ATOC/TarMAC/SchedNet family ([mem]). PIB coordination costing was not checked.

---

## Mechanisms worth testing (non-RL-first)

Ordered by expected value for a multi-UAV -> BS paper. Each item gives the mechanism, the supporting theory, and the cheapest falsification test.

1. **Price-coordinated marginal-value uplink against a broadcast "known set" (GOSC IV-F plus a submodular object-coverage value).**
   - **How it works:**
     - The BS broadcasts a compact known-object set K_t (tracked objects with uncertainty, versioned and costed) plus a price μ_t.
     - Each UAV scores each candidate report r (object crop, token, or ROI) by its marginal value v(r | K_t). Example: expected reduction in BS miss or uncertainty, weighted by association confidence that r is NOT already in K_t.
     - The UAV sends iff v > price of a successful delivery (GOSC Thm 1), with a value-density request. The BS grants by density.
   - **Theory:**
     - Coverage of unique objects with per-object uncertainty is monotone submodular. Density greedy under a knapsack gives (1 − 1/e)/2; GOSC Thm 2 gives exactness when slack.
     - GOSC Prop. 2 says value errors (including association errors) cost at most the price of the reallocated channel uses.
   - **Kill test:** compare against (a) independent per-UAV thresholds tuned to the same budget (Zhang/Vasconcelos style), and (b) "send all, BS-side NMS/dedup". Plot the gap vs. overlap ratio and budget. If the gap is < noise at the realistic overlap of the chosen dataset, stop.
2. **Neighbourhood-limited sequential greedy over U2U (RAG/SG) with geometric coin.**
   - **How it works:** only UAVs whose footprints overlap (coin_i > 0, computable from pose and FOV) run a short sequential-greedy round over U2U. Each announces its chosen object IDs or footprint cells; the others pick by marginal gain. Non-overlapping UAVs act alone.
   - **Theory:** SG gives 1/2 for a partition matroid. RAG Thm 1 gives f >= 1/(1+κ)[OPT − Σ coin_i]. coin_i is a computable a-priori upper bound (Xu Example 1). Mazandarani shows the sharing-time vs uplink-time trade-off is unimodal, so the U2U budget must be optimized, not assumed free.
   - **Kill test:** coordinated SG vs mechanism 1 (BS-mediated, no U2U). If BS mediation matches it, U2U adds nothing.
3. **BS-driven pull of a second view for uncertain objects (pull-based, receiver holds the state).**
   - **How it works:** the BS, which knows its own belief exactly with no feedback cost to model, pulls the UAV whose view most reduces uncertainty on low-confidence tracks. It stops when confidence passes a threshold (Shao VDDIB-SR stopping and attention, Zakeri pull POMDP).
   - **Theory:**
     - Zakeri: belief-MDP optimal pull policy for 2 sensors.
     - Luo-Pappas and Sun-Pappas: the CMDP optimum is a mixture of at most K+1 deterministic policies, so randomized thresholds suffice.
     - Adaptive submodularity (Golovin and Krause [mem]): adaptive greedy for active diagnosis/verification has a (1 − 1/e) guarantee under adaptive monotone submodularity.
   - **Note:** this sidesteps the R1b problem of the sender's view of the receiver. Measure pull-command DL cost and latency explicitly.

Cheap baselines that must be in any study: Shisher MGF / max-age-first (signal-agnostic, correlation-aware); Palena context-similarity filter (BS broadcasts pooled embedding, UAV sends if dissimilar); periodic-tuned (GOSC style); send-all plus BS dedup.

---

## Lessons for when receiver/peer-state knowledge helps (tied to R1b)

R1b failure modes:
- **F1:** the trigger never skipped (TOK at 100% of ticks at 1x, because low-confidence tracks always looked new).
- **F2:** the oracle belief (D) was worse than ACK-only (A1), because the BS map had ~14% precision and better belief made TOK content add false positives.
- **F3:** at 2x/4x, CPU, not uplink, was binding.

1. **For a single sender, feedback is mostly redundant with ACKs** (GOSC Prop. 1). The receiver state = delivered messages + receiver-side processing. Feedback can only add the receiver-side processing part (R1b: YOLO11s outputs). That bounds the possible gain before any experiment.
   - In multi-UAV, the broadcast carries other senders' contributions, which no ACK can provide. Test the multi-UAV case only where this term is large (high overlap).
2. **Knowledge helps only if the decision can change: the action set must include "no", and the budget must bind.**
   - GOSC: the gain vanishes with 2 agents or B >= 200; the price threshold makes "no" the default.
   - Mazandarani: gains only in the moderate-resource regime.
   - Zhang/Vasconcelos: a tuned local threshold is already near-centralized.
   - R1b F1 and F3 are both "the decision could not change". Pre-register a regime check: the fraction of SKIP/no-send decisions under the oracle must be substantial (e.g. > 30%) before any feedback condition is scored.
3. **The value must be priced against delivery cost, not compared with a fixed relevance threshold.** GOSC Thm 1: send iff value > price of a successful delivery. R1b's g threshold measured "is there something new", not "is it worth its airtime". Low-confidence detections made g always positive.
4. **An oracle belief is an upper bound only if the value function is aligned with the scored metric AND the receiver's estimator.**
   - MUTE: information proxies can change actions without improving return, or reduce it.
   - Zakeri: the estimator choice changes the optimal policy.
   - GOSC: the ToM value is not better than throughput.
   - R1b D used an accurate belief of a poor (low-precision) BS map to decide TOK content, which raised false positives. Before using a belief condition as a bound, verify monotonicity on TUNE: does improving belief accuracy, with the value fixed, raise the metric? Otherwise call it an "oracle-belief condition", not an upper bound.
5. **Coarse but content-aware value suffices; content-free value can fail** (GOSC Spearman >= 0.89 OK; content-independent fails in search). Do not spend effort on precise peer models (ToM, Dec-POMDP beliefs) before a price-aware coarse value shows headroom.
6. **Myopic per-tick triggers mis-value communication** (Becker et al. 2009). Under a sequential worker, a FRAME now blocks the next tick, so the value of sending depends on future opportunities. Index policies (Whittle/gain index, Shisher MGF) or a price with lookahead are the standard fixes, not RL.
7. **Count coordination honestly; it can erase the gain.**
   - GOSC: the central exact knapsack used more signalling than it saved; Search with 16-bit overhead lost (0.88).
   - Mazandarani: unimodal sharing time.
   - Any multi-UAV claim must report uplink + downlink + U2U bits against the same budget.
