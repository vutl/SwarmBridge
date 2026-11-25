# Paper A full-data: post-hoc analyses on stored results (25/11)

**Status: POST-HOC.** These analyses were requested after the consolidated report (bc4d2736) and were not
pre-registered. They use only stored rows and delivery lists: no simulation, no training, no LaTeX change. They do not
change any confirmatory verdict.

**Script and outputs:**
- Script: `swarmbridge/experiments/uav_semcom/fd_posthoc.py` (commands `verify`, `decomp`, `r5`).
- Outputs: `runs/uav_semcom/paperA_fd/posthoc/{run_verify.log, decomp.log, decomp.json, r5.log, r5.json}`.

**Requested items:**
1. Return decomposition: why ng has higher return but lower recall than nc.
2. PPO + calibrated MPC vs H\* + calibrated MPC.
3. Video-cluster bootstrap for the R5 recall contrasts.
4. Fail-fast coverage validation.

## 1. Fail-fast coverage and integrity (`verify`)

Every check stops at the **first** failure with exit code 1. The checks:
- the manifest / profile sha256;
- every evaluation file parses;
- exact registered seed lists, arm sets and checkpoint ids;
- one row and one delivery list per seed;
- 52 K tasks per episode;
- finite returns;
- the return identity of section 2.1, on every episode;
- the integrity counters: wrong task count, differing images across arms per seed, deliveries outside the pool;
- for the R5 pairing, identical seeds **and identical image multisets per seed** between Phase 3 and the Phase 2 H\*
  rows;
- the Phase 4 reuse of the Phase 2 seeds.

**Result:** ALL OK, **27,600 / 27,600 evaluation episodes** (Phase 2 2,400 + the 25,200 new ones); every integrity
counter is 0.

**Negative test:** a deliberately wrong seed list for one file stops `verify` at that file with exit code 1.

The registered `fd_next_supplement.py coverage` is left unchanged. It reports problems but always exits 0, and it does
not cover Phase 2.

## 2. Return decomposition: nc vs ng

### 2.1 The return identity

The return identity holds for every one of the 27,600 episodes (checked in `verify`):

> return = SCE + viol + drop + uncapt + energy + time

| Term | Definition |
|---|---|
| SCE | Sum of the 0.5 quality term + 0.5 deadline term over delivered images |
| viol | −1 per delivered image that is late or below Qmin |
| drop | −1 per dropped image |
| uncapt | −1 per never-captured image (0 in all runs) |
| energy | −0.02 per kJ |
| time | −0.3 per UAV-second of mission (a UAV stays active until its queue is empty) |

- The time term is the largest single term: about −120 per episode, against a return of about −40.
- SCE cannot be split into its quality and deadline parts: the per-image Qmin and deadline were not stored.

### 2.2 Reference setting (ablation seeds, TD; paired over 100 environment seeds)

Contrast ng − nc = **+18.05 [+13.79, +22.32]**:

| Component | ng − nc | Levels nc / ng |
|---|---|---|
| SCE (quality + deadline terms) | **+9.74\*** | 107.3 / 117.1 |
| Violations (−1 each) | **−15.71\*** (15.7 more below-Qmin deliveries) | −13.7 / −29.4 |
| Drops (−1 each) | **+15.83\*** (15.8 fewer drops) | −21.0 / −5.2 |
| Time (0.3 per UAV-second) | **+8.10\*** (mission 5.4 s shorter per UAV) | −128.1 / −120.0 |
| Energy | +0.09\* | −1.5 / −1.4 |
| Compliant images (on time, PSNR ≥ Qmin) | +0.12 (n.s.) | 225.3 / 225.4 |
| Late deliveries | 0 | 0 / 0 |

**Mode choice on compliant images:**
- nc: mean payload 303 kbit, PSNR 32.07 dB; modes avif_20 29%, MBT q3 20%, avif_50 18%, avif_5 17%.
- ng: 194 kbit, PSNR 30.32 dB; avif_20 77%.

**Recall gap ng − nc = −1.81 points (40.50 → 38.69), split by paired image occurrence:**

| Part | ng − nc (points) | Images per episode |
|---|---|---|
| Images compliant in both arms (same image, ng sends a smaller mode) | **−1.16\*** | 209 (124 with a different mode) |
| Compliant only under ng | +2.82\* | 16.5 |
| Compliant only under nc | −3.47\* | 16.4 |

Detection yield per compliant image (TP / GT): 47.09% (nc) vs 45.39% (ng), **−1.70\***.

### 2.3 What explains it

**ng chooses smaller payloads than nc** (about 200 vs 300 kbit). The band rejects switches whose advantage is not
robust to the rate error, so ng mostly keeps the incumbent's avif_20. Smaller payloads have three effects on return:
- **Fewer drops.** About 16 fewer images per episode miss their deadline; each drop costs −1.
- **Shorter missions.** UAVs empty their queues sooner, which saves about 8 return points through the 0.3 per
  UAV-second term.
- **Higher SCE**, despite the lower PSNR.

**The price is about 16 more images delivered on time but below Qmin.** In the return, these cost the same −1 as a
drop, so drops and violations roughly cancel. The **net return gain therefore comes from SCE (+9.7) and mission time
(+8.1).**

**Recall falls** for two reasons:
- **The same images are sent at lower PSNR.** This lowers the detector's yield on images that are compliant in both
  arms. It accounts for **about two thirds** of the gap.
- **The set of compliant images changes.** The images that become compliant only under ng have fewer detected objects
  than those that are compliant only under nc. This accounts for about one third.

**In short:**
- The return rewards fast delivery and short missions, and charges the same for a drop and for a below-Qmin delivery.
- Detection recall keeps increasing with payload and PSNR.
- The arms that raise return do so by sending less data, and that costs detections.

### 2.4 Is the mechanism general? (`decomp.log`, 53 contrasts)

**ng − nc**, in every pool and configuration (TD / V / F, Phase 2 TD / F / V / TD_all, Phase 4 × 6 configurations ×
TD / F, Phase 3 × 3 families):
- ng's compliant-image payload is lower in every pair: 138–349 kbit for ng vs 213–476 kbit for nc.
- Detection yield is lower: −0.2 to −4.0 points.
- Mission time is shorter: −2.7 to −6.4 s per UAV.
- There are fewer drops: +5 to +30 return points.
- In the settings with more channels per UAV (N3x2, N5x3, N5x4), violations are unchanged and compliant images
  increase (+4.8 to +6.6). There the return gain comes from drops and time, and the recall loss is small (−0.2 to −0.8
  yield).

**Same direction for the other return-raising steps:**
- band − nc;
- gate − nc;
- cal − ng in every pool and configuration: payload −8 to −27 kbit, yield −0.5 to −1.1 points. This is why calibration
  lowers recall;
- learned incumbents. The old PPO-LA checkpoint (zero-shot) sends 234 kbit vs H\*'s 175. It has **higher recall than
  H\*** (+3.1, see section 4) but lower return (−4.9). The PPO retrained on the fd_v1 return sends 165 kbit, and has
  lower recall (−2.0).

**ng − H\*** (TD) is the only return gain that does not come from smaller payloads:
- ng's payload is slightly larger (194 vs 174 kbit), with +8.3 compliant images and 7.9 fewer violations.
- Recall rises by only +1.6, because the yield per compliant image is −0.24.

### 2.5 Consequences for the paper

- **The return / recall disagreement is explained by the shape of the objective, not by noise.**
  - The returns used for every confirmatory test value delivery speed and mission time.
  - A below-Qmin delivery costs as much as a drop.
  - Detection keeps improving with bits.
  - The controllers that optimise this return (the MPC objective and the RL reward) traded detections for time in
    every contrast measured here.
- This is an evaluation finding: **a wireless return of the form used in the SemCom resource-allocation literature
  (Huang JSAC 2025 / Li JSAC 2026-style SCE utility with mission-time cost) is not a proxy for detection recall at the
  margin where these controllers operate.**
- It is post-hoc. It is measured in one simulator with assumed weights (0.5 / 0.5 SCE, 0.3 per UAV-second, 0.02 per
  kJ). The paper should state the weights and that recall was never optimised.

## 3. PPO + calibrated MPC vs H\* + calibrated MPC (R5 completion)

Phase 3 retrained ppofd (3 training seeds averaged) vs the Phase 2 H\* rows, on the same seeds and images. Paired
95% t-CI over 100 environment seeds.

| Pool | Return X+cal − H\*+cal | Per training seed | Recall (seed-level) | Recall (video cluster) |
|---|---|---|---|---|
| TD | **+0.27 [−1.58, +2.12]** | +1.70 / −0.62 / −0.27 | −0.73\* | −0.73 [−1.27, −0.13]\* |
| F | +1.10 [−1.09, +3.28] | +2.99\* / +0.55 / −0.24 | −0.50\* | −0.51 [−1.27, +0.41] |
| V | +0.32 [−1.30, +1.93] | +4.51\* / −1.71 / −1.84 | +0.04 | +0.05 [−0.54, +0.66] |
| TD_all | +1.15 [−0.94, +3.23] | +1.44 / +0.11 / +1.88 | −0.97\* | −0.97 [−1.42, −0.65]\* |

- **No detected return difference in any pool.** The CIs are about ±2, so this is not evidence of equivalence.
- Recall is lower with PPO in TD and TD_all, under both CI methods.
- The same result as A-X9 (−0.14 n.s.), now on the full-data protocol with retrained PPO.
- QR-DQN + cal and DSAC + cal stay below H\* + cal on return in every pool: QR-DQN −2.7 to −6.2\*, DSAC −8.1 to
  −10.2\*.

## 4. Video-cluster bootstrap for R5 recall (B = 2000, rng 0, as in the registered recall analysis)

Retrained, TD (61 sequences); X − H\* in percentage points:

| X | Alone: seed / cluster | Matched (+ng): seed / cluster | Calibrated (+cal): seed / cluster | Unguarded (+nc): seed / cluster |
|---|---|---|---|---|
| PPO | −2.02\* / −2.01 [−2.72, −1.31]\* | −1.18\* / −1.18 [−1.74, −0.56]\* | −0.73\* / −0.73 [−1.27, −0.13]\* | +0.07 / +0.10 [−0.21, +0.48] |
| QR-DQN | −4.97\* / −4.96 [−5.75, −4.13]\* | −1.39\* / −1.39 [−2.24, −0.36]\* | −0.96\* / −0.96 [−1.80, +0.00] | +0.24 / +0.26 [−0.06, +0.67] |
| DSAC | −9.57\* / −9.57 [−10.29, −8.88]\* | −4.14\* / −4.14 [−4.88, −3.24]\* | −4.02\* / −4.01 [−4.69, −3.14]\* | +0.01 / +0.04 [−0.29, +0.44] |

**Where the cluster bootstrap changes the seed-level verdict** (the seed-level CI excludes 0, the cluster CI does
not):
- **F** (18 sequences): PPO + ng, PPO + cal, QR-DQN + cal.
- **V:** QR-DQN + ng **+0.61** (seed level, above H\*) and QR-DQN + cal +0.50.
- **TD:** QR-DQN + cal.
- **Zero-shot TD:** DSAC + ng and DSAC + cal.

**Every learned-alone contrast stays significant under the cluster bootstrap in every pool.**

### Correction to the R5 statement

The consolidated report said: "no learned incumbent beats H\* on return or recall".
- **This holds for the retrained incumbents** (cluster level). The one seed-level positive, QR-DQN + ng on V, +0.61, is
  not significant at cluster level.
- **It does not hold for the zero-shot PPO-LA checkpoint.** Its recall is **above** H\*, with return below H\*:

| Contrast (zero-shot PPO-LA − H\*, TD) | Return | Recall |
|---|---|---|
| Alone | −4.85\* | **+3.06 [+1.63, +4.75]\*** |
| Matched (+ng) | −4.64\* | **+2.46\*** |
| Calibrated (+cal) | −0.23 (n.s.) | **+2.98\*** |

- This is the same return / recall mechanism as in section 2: PPO-LA sends larger payloads.

**Corrected statement:** no learned incumbent beats H\* on return. On recall, the retrained incumbents are at or below
H\*. The zero-shot PPO-LA is above H\* on recall and below it on return.

**Unguarded planner:** with nc, every retrained incumbent reaches the same recall as H\* + nc (all |Δ| ≤ 0.35,
n.s.). The zero-shot checkpoints are slightly above it: PPO-LA +0.72\*, QR-DQN +0.45\*, DSAC +0.30 (n.s.). The unguarded planner overrides the incumbent so often that the incumbent hardly matters for recall.
