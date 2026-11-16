# Paper A: Method section (draft v2, review #35)

Status: manuscript draft of the Method section. It describes the controller exactly as implemented and evaluated in
A-E5 to A-E9:
- `swarmbridge/experiments/uav_semcom/a9_mpc.py` (`CalRates`, `RateErrTracker`, `MPCMode`, `Deadline`, `Feedback`);
- `a10_hybrid.py` (variant `ahyb`).

No new mechanism is added. The corrections made in v2 are listed in the changelog at the end.

---

## III. System model and problem

### A. Scope

- $K_{\mathrm{U}}$ UAVs fly survey paths and share $M$ orthogonal uplink channels of bandwidth $B/M$ to one ground base
  station (BS). Time is slotted with slot length $\Delta = 1$ s.
- In every slot a **fixed slot policy** $\pi^{\mathrm{slot}}$ chooses, for each UAV, its movement, channel and transmit
  power. $\pi^{\mathrm{slot}}$ is the slot part of the regime's tuned heuristic, and the PPO baseline uses the same one.
- The controller studied here chooses **only the compression/semantic mode** $m \in \mathcal{M}$ ($|\mathcal{M}| = 14$:
  captions with text-to-image, thumbnails, generative image-to-image, AVIF, a learned codec, JPEG 2000) of each
  captured image.
- This is therefore **single-decision-maker mode control**, not multi-agent learning and not joint trajectory, power or
  resource optimisation.

### B. Image life cycle

Image $i$ is captured by UAV $k_i$ at time $t_i$ and has a deadline $D_i$ and a PSNR floor $Q_i$, both drawn per image.
It then goes through these steps:

1. A probe encode runs on the UAV encoder. When the probe finishes, a regression predicts the per-mode payload
   $\hat b_i(m)$ and PSNR $\hat q_i(m)$.
2. At the next decision epoch the controller fixes the mode $m_i$.
3. The serial encoder takes $\tau^{\mathrm{enc}}(m_i)$. The serial FIFO transmitter then sends $b_i(m_i)$ bits at the
   realised link rate.
4. The BS decodes in $\tau^{\mathrm{dec}}(m_i)$.

The delivery delay is $d_i$: the time from capture to the end of decoding. An image that can no longer be delivered by
$t_i + D_i$ is discarded at that moment, so no further airtime is spent on it.

### C. Utility and return (as implemented in the A5 environment)

The utility of a delivered image is

$$
u_i = \xi_q \,\mathrm{clip}\!\left(\frac{q_i(m_i) - Q_i}{q_{\mathrm{span}}}, -1, 1\right)
    + \xi_d \,\frac{[D_i - d_i]^+}{D_i}
    - \mathbb{1}\{q_i(m_i) < Q_i \;\lor\; d_i > D_i\},
$$

with $\xi_q = \xi_d = 0.5$ and $q_{\mathrm{span}} = 15$ dB.

- $q_i(m)$ is the **realised PSNR** of the decoded image.
- A floor violation is an additive penalty of 1 on top of the (then negative) quality term. It does not replace the
  utility.
- An image that is **discarded** at its deadline gets exactly $u_i = -1$.

The episode return is

$$
G = \sum_i u_i
  \;-\; w_E \,\frac{E}{1000}
  \;-\; w_T \,\Delta \sum_t n_{\mathrm{act}}(t)
  \;-\; N_{\mathrm{uncapt}},
$$

where:
- $E$ is the propulsion plus transmit energy in J, and $w_E = 0.02$;
- $n_{\mathrm{act}}(t)$ is the number of UAVs whose sortie is still running, and $w_T = 0.3$ in both regimes used here;
- $N_{\mathrm{uncapt}}$ counts capture points not reached at the horizon or at battery exhaustion.

The mode affects the energy and time terms only through airtime and sortie end. The planner (Section V) does not model
them.

**Quality metric.** The quality term is **PSNR-based**. The controller optimises PSNR-based delivery utility under
deadlines. This is not downstream task accuracy, and no task-accuracy claim is made for Paper A.

### D. Information available at a mode decision

All information is causal:
- the predicted $(\hat b_i(m), \hat q_i(m))$;
- the UAV's queue: already-decided images, remaining bits, encoder and transmitter backlog;
- time and position;
- statistical CSI at the UAV's position: path gain, LoS probability, and the Rician/Rayleigh fading *distribution*;
- the throughput reports that have *arrived* (Section IV).

The realised future fading is never read; this is tested.

### E. Two PHY regimes

- **ref:** Shannon-rate links.
- **phy_mcs_blind:** LTE-CQI MCS with interference-unaware link adaptation and BLER 0.1.

---

## IV. Method

### A. Overview

At every mode decision for image $i$ of UAV $k$:

1. The frozen PPO mode-only policy proposes the **incumbent** mode $a$.
2. The controller departs from $a$ only if a sampling MPC finds a mode whose advantage over $a$ is **resolvable** under
   the nominal rate model and under the rate model's historical error.

The rate model is an empirical, report-driven model (Sections IV-B and IV-C). The planning work is limited by a
cheap screen (IV-F) and a cooperative time budget (IV-G).

### B. Calibrated empirical rate model

**Throughput reports.**
- After slot $t$ ends, the sender receives one report $y_{k,t}$ for every UAV $k$ that transmitted in that slot.
- The report is the mean link rate over UAV $k$'s transmission segments $s$ in the slot, **averaged within the slot by
  airtime**:

$$
y_{k,t} = \frac{\sum_{s} r_s |s|}{\sum_s |s|}.
$$

- Each report then carries the **same weight** $\Delta$, whatever the airtime in the slot. We call this *slot
  weighting*.
- The earlier variant weighted every segment by its airtime *across* slots. That variant is biased low: a queue that
  empties early in a good-fading slot contributes little airtime at the high rate, while a bad-fading slot contributes
  a full slot at the low rate.
- Whether a UAV transmits in slot $t$ is decided before the slot's fading is drawn. Slot weighting therefore samples the
  per-slot rate distribution without this bias.

**Binning.**
- A report is stored in bin $\beta = \lfloor \overline{\mathrm{SNR}}_k(t)/2\,\mathrm{dB} \rfloor$.
- $\overline{\mathrm{SNR}}_k(t) = p_{\max}\,\mathbb{E}[g(\mathbf{x}_k(t))]/(N_0 B)$ is the UAV's expected SNR at the
  slot start: statistical CSI at maximum power.

**Sampling.**
- To plan for UAV $k$, the controller looks up the bin of its current expected SNR.
- It pools that bin with its neighbours in the order $\beta, \beta \pm 1, \beta \pm 2$ until the pooled reports cover at
  least $20$ s.
- It then draws an $N \times S$ matrix of per-slot rates i.i.d. from the pooled reports ($S = 40$ slots).
- Each draw is multiplied by the channel share $\sigma_k = M/\#\{\text{busy UAVs}\}$ and floored at 1 bit/s.

**Fallback.** If the pooled reports cover less than 20 s, the controller draws samples from the statistical fading
model, maps them through the Shannon rate, and multiplies by

$$
\rho = \frac{\sum y\,\Delta + c_0}{\sum y^{\mathrm{Sh}}\,\Delta + c_0}, \qquad c_0 = 5 \cdot 10^{6},
$$

the pooled observed-to-Shannon ratio, shrunk towards 1.

**Feedback model.**
- Report $y_{k,t}$ becomes usable $D_{\mathrm{fb}}$ slots after slot $t$ ends.
- It is lost with probability $P_{\mathrm{fb}}$, independently across UAVs and slots.
- Reports still in flight at the end of an episode are discarded.
- The rate model and the error tracker (IV-C) see the same arrivals and losses.
- The ideal case is $D_{\mathrm{fb}} = 0$, $P_{\mathrm{fb}} = 0$.
- This feedback model applies **only to the planner's rate model**. PPO and the slot policy observe the queue state
  exactly as the environment updates it.

**Warm start (protocol "pre").**
- The rate model and the error tracker are first filled on 10 separate calibration episodes:
  - the regime's heuristic controller drives them;
  - they use the validation image pool;
  - their environment seeds are disjoint from all test seeds.
- The resulting state is frozen and used as the start state of **every** test episode.
- Within an episode the state is updated online. Test episodes do not share what they observe.

### C. Prequential error tracker and sensitivity interval

**Prediction.** At the start of each slot $t$, the tracker records for each UAV the model's predicted mean rate
$\pi_{k,t}$. This is the mean of the pool that the planner would sample from, or the $\rho$-scaled Shannon Monte-Carlo
mean when the pool is too small. The prediction is always made **before** the slot's report exists.

**Update.** When $y_{k,t}$ arrives, the tracker adds to bin $\beta$ (the slot-start bin):
- $(y_{k,t}\Delta,\ \pi_{k,t}\Delta,\ \Delta)$;
- the ratio $r_{k,t} = y_{k,t}/\max(\pi_{k,t}, 1)$.

**Statistics.** With the same bin pooling as above, the tracker returns:
- the relative bias $b = \sum y\Delta / \sum \pi\Delta - 1$;
- the standard error $\mathrm{se} = \mathrm{sd}(r)/\sqrt{n_r}$ of the mean ratio.

If the pooled reports cover less than 20 s, the controller returns $a$.

**Sensitivity interval.** With multiplier $\kappa$:

$$
\ell = \max\!\big(\min(1, 1+b) - \kappa\,\mathrm{se},\ 0.05\big), \qquad
h = \max(1, 1+b) + \kappa\,\mathrm{se}.
$$

- The interval always contains the nominal scale 1 and the bias-corrected scale $1+b$.
- **It is a heuristic uncertainty range, not a confidence interval.**
  - The reports are serially correlated.
  - They are pooled across bins and across calibration and test episodes.
  - The ratio's standard error ignores that dependence.
- No coverage guarantee is claimed.

### D. Sampling MPC over modes

**Inputs.** The evaluation takes:
- a rate matrix $\mathbf{R} \in \mathbb{R}_{+}^{N \times S}$;
- a candidate mode $m$ for image $i$.

**Per-sample simulation.** For each sample path $n$, a fluid model of the UAV's encoder and FIFO transmitter runs on the
slot grid:
- First the already-decided images ahead of $i$ are served (their remaining bits).
- Then image $i$ itself: it becomes ready at $\max(t, \text{encoder free}) + \tau^{\mathrm{enc}}(m)$, and transmission
  ends at $T_n(m)$, where cumulative capacity reaches its $\hat b_i(m)$ bits.
- If $T_n(m) + \tau^{\mathrm{dec}}(m) > t_i + D_i$, the sample scores $-1$ and the transmitter is occupied until the
  expiry time. Otherwise the sample scores $u_i$ evaluated at $\hat q_i(m)$ and $d = T_n(m) + \tau^{\mathrm{dec}}(m) -
  t_i$.

**Causal rollout.**
- Two groups of later images follow:
  - the $L$ images already captured and still awaiting a mode, in capture order. Each uses its own predictions if its
    probe has finished, otherwise the train-split per-mode means.
  - $H$ statistical future frames, arriving every frame interval. Each has the train-split mean per-mode
    $(\bar b(m), \bar q(m))$ and the regime's mean deadline and floor, and needs a probe before encoding.
- Each later image receives **one** mode for all samples: the mode with the highest sample-mean utility given the
  per-sample backlog.
- The rollout is therefore nonanticipative: future modes do not depend on the sampled path.

**Value.** The value $V_n(m; \mathbf{R})$ is the sum of the per-sample utilities of image $i$ and of the rolled-out
images. The planner's objective has three parts:
- the utility of image $i$ and of the rolled-out images, with **predicted** PSNR;
- no energy or mission-time terms;
- no other UAVs' images, except through the channel share.

**Stacked evaluation.**
- To evaluate the same paths at several rate scales $s_1, \dots, s_G$, the matrices $s_g\mathbf{R}$ are concatenated
  along the sample axis and evaluated in one pass.
- The rollout picks one mode per scale group, so every group equals a separate evaluation; this is tested.
- The per-mode cost hardly depends on the number of rows, so $G$ scales cost about one evaluation.

### E. Decision-aware switch rule

**Setup.**
- $\mathcal{C} \ni a$ is the evaluated candidate set.
- $\mathbf{R}$ is the rate matrix of this step.
- The scales are $\mathcal{S} = \{1, \ell, h\}$, evaluated stacked on the **same** $\mathbf{R}$, so comparisons are
  paired across modes and scales.

**Candidate.** Let

$$
c^\star = \arg\max_{m \in \mathcal{C}} \frac{1}{N}\sum_n V_n(m; \mathbf{R}).
$$

**Rule.** With $\delta_n(s) = V_n(c^\star; s\mathbf{R}) - V_n(a; s\mathbf{R})$, the controller switches to $c^\star$
only if $c^\star \neq a$ and, **for every** $s \in \mathcal{S}$,

$$
\bar\delta(s) \;\ge\; Z\, \frac{\mathrm{sd}_n(\delta(s))}{\sqrt{N}}.
$$

Otherwise it keeps $a$. In words: the planner's preferred mode must beat PPO's mode by more than $Z$ paired standard
errors of the sampling noise, at the nominal rates and at both ends of the sensitivity interval.

### F. Selective screening

The full evaluation is expensive: its cost scales with the horizon times the number of modes. The controller therefore
screens first.

**1. Screen.**
- Draw a **fresh** rate matrix $\mathbf{R}'$; it is independent of the $\mathbf{R}$ of the full stage.
- Evaluate all modes with horizon $H_1 = 1$, stacked at scales $\{1, \ell\}$.
- This gives the mean values $\bar V^{(1)}(m)$ and $\bar V^{(\ell)}(m)$.

**2. Candidates.**
- $c_{\mathrm{nom}} = \arg\max_{m \ne a} \bar V^{(1)}(m)$.
- $c_{\mathrm{pes}} = \arg\max_{m \ne a} \bar V^{(\ell)}(m)$. If it coincides with $c_{\mathrm{nom}}$, the second-best
  nominal mode is used instead.

**3. Trigger.**

$$
f_{\mathrm{nom}} = \big[\bar V^{(1)}(c_{\mathrm{nom}}) > \bar V^{(1)}(a)\big], \qquad
f_{\mathrm{pes}} = \big[\bar V^{(\ell)}(c_{\mathrm{pes}}) > \bar V^{(\ell)}(a)\big].
$$

- Variant *any* (main, calibrated model): plan if $f_{\mathrm{nom}} \lor f_{\mathrm{pes}}$.
- Variant *pess* (strongly biased model, e.g. Shannon under MCS): plan if $f_{\mathrm{pes}}$.
- If the trigger is false, return $a$: no full planning, and **neither** screen found a mode better than $a$.

**4. Full planning.**
- Draw a **new** rate matrix $\mathbf{R}$.
- Evaluate $\mathcal{C} = \{a, c_{\mathrm{nom}}, c_{\mathrm{pes}}\}$ with horizon $H$ at $\{1, \ell, h\}$ (one stacked
  pass), then apply IV-E.

Within each stage, all modes and scales share one set of base samples (paired). Only between screen and full stage are
the samples redrawn.

### G. Cooperative decision budget

- **Clock.** The decision clock starts before PPO inference. The budget is $B_{\mathrm{dec}}$.
- **Check.** Before every unit of work, the controller checks
  $\text{elapsed} + \text{units} \times \hat c_{\mathrm{stage}} + \text{margin} \le B_{\mathrm{dec}}$. A unit of work is
  building a sample context, or evaluating one mode, which covers one image plus its rolled-out images.
- **Per-unit cost $\hat c_{\mathrm{stage}}$.** It is learnt online per stage (context, screen, full): the 95th
  percentile of the last 300 measured per-unit times, with a 1 ms prior until 20 measurements exist. The cost of the
  robustness check is reserved before any candidate is evaluated.
- **Fallbacks.**

| Abort | Result |
|---|---|
| no time to start the screen | $a$ |
| screen interrupted | $a$ |
| not even $a$, one candidate and the check fit | $a$ |
| full stage truncated | best of the candidates already evaluated, still subject to IV-E |
| check interrupted | $a$ |

- **Not hard real-time.** Work already started is never pre-empted, so the budget is **cooperative**. Overruns are
  measured and reported.

### H. Algorithm 1

```text
Algorithm 1  Decision-aware selective PPO-MPC mode control (one decision)
Input: image i of UAV k; PPO policy; rate model CalRates; tracker; parameters N, H, H1, Z, kappa, B_dec
 1: start clock
 2: a <- PPO(obs)                                         # incumbent
 3: if tracker has < 20 s of reports in k's SNR bin pool: return a
 4: (b, se) <- tracker.stats(k)
 5: lo <- max(min(1, 1+b) - kappa*se, 0.05);  hi <- max(1, 1+b) + kappa*se
 6: if not budget_ok(context): return a
 7: R1 <- CalRates.sample(k, N x S) * share_k             # screening samples
 8: for m in modes:                                       # one stacked pass at scales {1, lo}, horizon H1
 9:     if not budget_ok(screen unit): return a
10:     Vnom[m], Vlo[m] <- mean V_H1(m; R1), mean V_H1(m; lo*R1)
11: c_nom <- argmax_{m != a} Vnom[m]
12: c_pes <- argmax_{m != a} Vlo[m];  if c_pes = c_nom: c_pes <- second-best nominal mode
13: fire_nom <- Vnom[c_nom] > Vnom[a];  fire_pes <- Vlo[c_pes] > Vlo[a]
14: if not (fire_nom or fire_pes): return a               # trig=any; trig=pess uses fire_pes only
15: R2 <- CalRates.sample(k, N x S) * share_k             # new samples for the full stage
16: E <- {}
17: for m in [a, c_nom, c_pes]:                           # one stacked pass at scales {1, lo, hi}, horizon H
18:     if not budget_ok(units for m, reserving the check):
19:         if |E| <= 1: return a else break              # skip / truncate
20:     V[m] <- per-sample values V_H(m; [R2, lo*R2, hi*R2]); E <- E + {m}
21: c <- argmax_{m in E} mean V[m] at scale 1
22: if c != a and for all s in {1, lo, hi}: mean(V[c]-V[a] at s) >= Z * sd/sqrt(N): return c
23: return a
Between decisions, at every slot boundary: CalRates.update, tracker.update (arrived reports only).
```

### I. Parameters (main configuration)

| | Value |
|---|---|
| samples $N$ / horizon $H$ / screen horizon $H_1$ | 8 / 4 / 1 |
| switch margin $Z$ / interval multiplier $\kappa$ | 1 / 2 |
| slots per path $S$, slot length $\Delta$ | 40, 1 s |
| SNR bin width, minimum pooled reports | 2 dB, 20 s |
| decision budget $B_{\mathrm{dec}}$ / margin | 50 ms / 3 ms |
| warm start | 10 calibration episodes, heuristic, validation pool |
| PPO | A5 mode-only, 3 best-validation checkpoints, unchanged |

Code name: `ahyb:I:cals:8:4:causal:1:2:1:pre:cand=np:trig=any:stack=1:dl=50`. The biased-model variant uses
`shannon` and `trig=pess`.

### J. Assumptions and limitations to state in the paper

1. **Fixed slot policy.** The slot policy is fixed and shared with the baseline. The contribution is mode control.
2. **PSNR quality.** The quality term is PSNR-based (Section III-C).
3. **Warm start.**
   - The method relies on a pre-calibrated rate model.
   - Without the warm start, the gain over PPO drops from about +2 to +0.65 (phy_mcs_blind) and +0.96 (ref) (A-E9).
   - A stale or mismatched warm start has not been tested.
4. **Feedback.**
   - Report delay and loss affect only the planner's rate model.
   - The queue observations of PPO and the slot policy are immediate.
   - Robustness to delayed queue or channel state is not claimed.
5. **Heuristic interval.** The sensitivity interval carries no coverage guarantee.
6. **Cooperative budget.**
   - The budget is cooperative.
   - Isolated overruns of 105 to 139 ms were observed in about 9k decisions per variant.
7. **Planner objective.** The planner ignores energy and mission-time costs and other UAVs' queues, except through the
   channel share.
8. **No retraining.** PPO is not retrained.

### K. Evidence (exploratory; details in the result reports)

| Step | Finding |
|---|---|
| A-E5 | Across-slot airtime weighting is biased low. Slot weighting fixes it. |
| A-E6 | Decision-aware rule plus warm start: +2.82 (phy_mcs_blind) / +2.23 (ref) over PPO with the calibrated model. With a biased model (Shannon under MCS): +1.30, versus −8.67 without the rule. |
| A-E7 | The two-stage selective controller keeps most of the gain at about half the cost. MPC cost scales with horizon × modes, not with $N$. |
| A-E8 | With the 50 ms cooperative budget: p99 40–44 ms. np-any +2.86, about equal to unrestricted planning (+2.82). The pessimistic candidate helps under the biased model (+1.90 vs +0.71, development seeds). |
| A-E9 | Fresh environment seeds 8500–8519; CI over seeds, with the 3 checkpoints as paired repeats. np-any: +2.18 [+1.30, +3.06] (phy_mcs_blind), +1.91 [+1.42, +2.39] (ref). np-any is not better than top-2 or unrestricted planning: its value is equal return at lower cost. With report delay of 1–3 slots or loss of 30–60 %, no degradation is resolved (all CIs include 0) under the warm start. Shannon np-pess beats unrestricted planning (+0.90, CI > 0) but not top-2 (+0.39, CI includes 0). |

---

### Changelog v1 → v2 (review #35)

1. **Trigger *any*.** Full planning runs if **either** screen finds a mode better than $a$. The controller returns $a$
   only if **neither** does (IV-F, Algorithm 1 lines 13–14). The v1 pseudo-code was ambiguous.
2. **Rate samples.** Screen and full stage draw separate rate matrices $\mathbf{R}'$ and $\mathbf{R}$. Pairing holds only
   within a stage, across modes and scales.
3. **Notation.** $K_{\mathrm{U}}$ is the number of UAVs and $\kappa$ the interval multiplier. v1 used $K$ for both.
4. **Utility.** The A5 utility is written out (III-C):
   - a floor violation or late delivery is an additive −1 penalty on top of the utility terms;
   - a discarded image is exactly −1;
   - the energy, mission-time and uncaptured-frame terms of the return are listed.
5. **Reports.** A report is the within-slot airtime-averaged rate of one transmitting (UAV, slot). Reports are weighted
   equally across slots, as distinct from the earlier across-slot airtime weighting.
6. **Scope.** The slot policy is fixed and shared with PPO. The method is mode-only control, not MARL or joint
   trajectory/power optimisation.
7. **Interval.** The sensitivity interval is a heuristic range, not a confidence interval.
8. **Quality metric.** The quality metric is PSNR-based, not task accuracy.
