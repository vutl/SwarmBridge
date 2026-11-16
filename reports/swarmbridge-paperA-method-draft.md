# Paper A: Method section (working draft, review #34)

Status: draft for the paper. It describes the controller as it is implemented and evaluated in A-E5 to A-E9
(`swarmbridge/experiments/uav_semcom/a9_mpc.py`, `a10_hybrid.py`, variant `ahyb`). Numbers in the evidence table come
from the result reports. Nothing here adds a new mechanism.

## 1. Setting

- K UAVs survey their sectors and share M orthogonal uplink channels to one base station. Time is slotted (1 s).
- **Slot decisions.** At every slot a fixed slot policy decides movement, channel and power. This is the slot part of
  the regime's tuned heuristic, the same one that drives the PPO baseline.
- **Mode decisions.** Each captured image waits for a *mode decision*: one of 14 compression/semantic modes (captions,
  thumbnails, generative i2i, AVIF, learned codec, JPEG2000).
- **Controller scope.** The controller studied here chooses **only the mode**.
- **Information available at a decision.** Everything is causal:
  - the image's predicted per-mode (bits, PSNR), from a probe-based regression;
  - the UAV's queue (decided images, remaining bits, encoder and transmitter backlog);
  - time and position;
  - statistical CSI at the UAV's position (path gain, LoS probability, fading distribution);
  - throughput reports for past transmissions (Section 3).
  
  Realised future fading is never read (tested).
- **Utility.** Per-image utility is the A5 reward: a quality term plus a timeliness term, and −1 for a deadline miss or
  a quality-floor violation. The episode return is the sum.

## 2. Controller overview

At every mode decision for image *x* of UAV *k*:

1. **Incumbent.** The trained PPO mode-only policy (frozen checkpoint) proposes mode *a*.
2. **Rate model and its error.** A calibrated empirical rate model gives per-slot rate samples for UAV *k*. A
   prequential error tracker gives the model's relative bias *b* and the standard error *se* of its observed/predicted
   ratio in the UAV's SNR bin. If the tracker has too little data, the controller uses *a*.
3. **Sensitivity interval.**
   - lo = max(min(1, 1+b) − K·se, 0.05)
   - hi = max(1, 1+b) + K·se
4. **Screen.** A cheap short-horizon evaluation (H1 = 1) of all modes runs on the same sampled rate paths at two scales,
   the nominal scale 1 and the pessimistic scale lo, in one stacked pass. The candidate set is {*a*, best nominal mode,
   best pessimistic mode}. Planning continues only if some mode beats *a* in either screen.
5. **Full planning.** The full sampling MPC (horizon H) evaluates the candidates.
6. **Decision-aware switch rule.** The controller switches from *a* to the best candidate *c* only if the paired
   per-sample gain V_c − V_a is resolvable (mean ≥ Z standard errors) under the nominal rates **and** under the same
   rate paths scaled to lo and hi. Otherwise it keeps *a*.
7. **Cooperative deadline.** The whole decision has a budget of B ms, including PPO inference. Before each unit of work,
   the controller checks that the remaining budget covers the work's learnt cost plus a margin. If it does not, the
   controller falls back to *a*, or to the best candidate already evaluated and checked.

## 3. Calibrated rate model (CalRates, slot-weighted)

- **Reports.** At each slot boundary, the sender obtains one throughput report per transmitting UAV for the slot that
  just ended: the airtime-weighted mean link rate in that slot.
- **Weighting.** Each report has weight one slot. Airtime weighting is biased low, because queues empty early in
  good-fading slots (A-E5).
- **Binning.** Reports are stored in 2 dB bins of the UAV's expected SNR at the slot start (statistical CSI).
- **Sampling.** The planner samples per-slot rates from the current bin. Neighbouring bins are pooled until at least
  20 s of data are available.
- **Fallback.** Without enough data, the planner uses Shannon-rate samples scaled by the pooled observed/Shannon ratio ρ.
- **Feedback model (A-E9).** A report becomes usable D slots after its slot ends and is lost with probability P.
  Losses are independent across UAVs and slots. The calibrator and the tracker see the same losses. D = 0, P = 0 is
  the ideal case.
- **Warm start ("pre").** The calibrator and tracker are first filled on 10 separate calibration episodes (heuristic
  controller, validation image pool, seeds 9000–9009). That state is frozen as the starting state of every test
  episode. Online updates follow within an episode, and test episodes are independent.

## 4. Prequential error tracker (RateErrTracker)

- At the start of each slot, the tracker stores the planner model's predicted mean rate for each UAV.
- When that slot's report arrives, the tracker adds (observed × w, predicted × w, w) and the observed/predicted ratio to
  the UAV's SNR bin.
- Outputs: relative bias b = Σ observed·w / Σ predicted·w − 1, and *se* = the standard error of the mean ratio. Bins
  are pooled as in the calibrator.
- The prediction for a slot is always made before that slot's report is seen.

## 5. Sampling MPC over modes

- **Rate paths.** N rate sample paths over S = 40 slots, multiplied by the channel share M / #busy UAVs. They drive a
  fluid FIFO transmitter model that includes the queue ahead and the serial encoder.
- **Value of mode m.** The utility of image *x* with mode m, plus a **causal rollout**: each later queued image, and each
  of H statistical future frames, gets one mode per image for all samples (the mode with the best mean utility given
  the backlog). The rollout is nonanticipative: future modes do not depend on the sampled path.
- **Future frames.** They arrive at the survey frame interval and use the train-split mean per-mode (bits, PSNR) and the
  regime's mean deadline and quality floor.
- **Paired gains.** V_m is a vector over samples. Gains between modes are paired over the same samples.
- **Stacked evaluation.** The nominal and scaled rate paths are concatenated on the sample axis, and the causal rollout
  picks one mode per group. Each group equals a separate evaluation (tested). The robustness check therefore costs
  about one evaluation, not three.

## 6. Selective screening

- Screen: horizon H1 = 1, the same N samples, stacked at scales {1, lo}.
- Candidates: the best mode ≠ *a* at scale 1, and the best mode ≠ *a* at scale lo (if it is the same mode, the
  second-best nominal one).
- Triggers:
  - *any*: some mode beats *a* at either scale (default with the calibrated model);
  - *pess*: some mode beats *a* at scale lo (the variant for a strongly biased model, e.g. Shannon under MCS).
- If the trigger does not fire, the controller returns *a* without full planning.

## 7. Cooperative deadline

- **Clock.** Starts before PPO inference.
- **Work units.** One unit is one mode evaluation of one image plus its later and future images.
- **Cost estimate.** Per-unit cost is learnt online per stage (screen, full, check): the 95th percentile of the last 300
  measurements, with a 1 ms prior until 20 measurements exist.
- **Check.** Before each step: elapsed + units × estimated cost + margin ≤ B. The robustness check's cost is reserved
  before any candidate is evaluated.
- **Fallbacks.**

  | Abort | What happens |
  |---|---|
  | skip_screen | not enough budget to start the screen → *a* |
  | int_screen | screen interrupted → *a* |
  | skip_full | not even the incumbent, one candidate and the check fit → *a* |
  | trunc_full | stop after the candidates evaluated so far → best of those, still checked |
  | int_check | check interrupted → *a* |

- **No hard real-time guarantee.** A running evaluation cannot be pre-empted. Overruns are measured and reported.

## 8. Parameters (main configuration)

| | value |
|---|---|
| samples N / horizon H / screen horizon H1 | 8 / 4 / 1 |
| switch margin Z / interval width K | 1 / 2 |
| slots S, slot length | 40, 1 s |
| SNR bin, minimum data per bin | 2 dB, 20 s |
| budget B / margin | 50 ms / 3 ms |
| rollout | causal (nonanticipative) |
| PPO | A5 mode-only, 3 best-validation checkpoints, unchanged |

Code name: `ahyb:I:cals:8:4:causal:1:2:1:pre:cand=np:trig=any:stack=1:dl=50`. The Shannon variant uses `trig=pess`.

## 9. Algorithm (pseudo-code)

```
decide(x, k):
  start clock;  a <- PPO(obs)
  if tracker(k) has < 20 s data: return a
  (b, se) <- tracker.stats(k);  lo, hi <- interval(b, se, K)
  if not budget.ok(ctx): return a
  R <- sample rate paths (calibrated model, N x S)
  screen <- V_H1(m; [R, lo*R]) for all m          # one stacked pass, interruptible -> a
  c_nom, c_pes <- argmax_{m != a} screen[1], screen[lo]
  if no mode beats a in screen[1] or screen[lo]: return a
  C <- {a, c_nom, c_pes}
  V <- V_H(m; [R, lo*R, hi*R]) for m in C (incumbent first; budget-checked, truncation keeps the evaluated set)
  c <- argmax_{m in C} mean V(m; R)
  if c != a and resolvable(V_c - V_a) at R, lo*R and hi*R: return c
  return a
```

## 10. Evidence so far (exploratory; details in the result reports)

| Step | Finding |
|---|---|
| A-E5 | Airtime-weighted calibration is biased low; slot weighting fixes it. |
| A-E6 | Decision-aware gate + warm start: +2.82 (phy_mcs_blind) / +2.23 (ref) over PPO with the calibrated model. For a biased model (Shannon under MCS): +1.30 vs −8.67 without the gate. |
| A-E7 | Two-stage selective planning keeps most of the gain at about half the cost; MPC cost scales with horizon × modes, not N. |
| A-E8 | 50 ms cooperative budget: p99 40–44 ms, max 47.5 ms (one 103 ms outlier in ref). np-any +2.86 (≈ unrestricted +2.82; the difference is not meaningful). The pessimistic candidate helps under a biased model (Shannon +1.90 vs +0.71). |
| A-E9 | Fresh environment seeds (8500–8519), seed-level CI: np-any +2.18 [+1.30, +3.06] (phy_mcs_blind), +1.91 [+1.42, +2.39] (ref). Report delay of 1–3 slots and loss of 30–60 %: no significant degradation (all CIs include 0). This robustness comes from the warm start: without it (cold), the gain falls to +0.65 / +0.96. Shannon np-pess beats the full planner (+0.90, CI > 0) but not top-2 (+0.39, CI includes 0). |

## 11. Non-claims

- Not hard real-time.
- Not joint UAV trajectory/power control: the slot policy is fixed.
- No PPO retraining.
- The simulator's queue state reflects realised throughput immediately. The feedback model affects only the planner's
  rate model.
- Statistics are reported at the environment-seed level, with the PPO checkpoints as paired repeats.
