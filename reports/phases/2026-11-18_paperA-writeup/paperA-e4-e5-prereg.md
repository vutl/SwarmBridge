# Paper A: E4 (network-size sweep) and E5 (feedback delay and loss), pre-registration

**Written 2026-11-21, before any E4/E5 test episode was run.**

The only E4/E5 runs so far are smoke runs: one seed (9900) per configuration, to check that the code runs and to time it.

## Why

The reviewers (ChatGPT; [chat_with_chatgpt](../../chat_with_chatgpt)) asked whether the calibrated MPC only helps at 5 UAVs and 2 channels.

A second question is whether it still helps when the rate reports arrive late or get lost. The manuscript currently assumes ideal, immediate feedback.

## What stays fixed (no change after seeing results)

- **Method:** exactly as in the manuscript:
  - SNR-binned `CalRates`, cold start each episode;
  - selective MPC with N = 8 paths, H = 4 images, 50 ms anytime budget;
  - switch only if the mean paired gain is > 1 SE.
- **PHY:** nominal LA. That is scheduler-aware LA, persistent interference, CQI delay of 2 blocks, 1 dB error, OLLA with a 10% BLER target.
- **Scheduler:** frozen `ja~JA:stat:R:a1:m3@pacedB3`.
- **Bandwidth:** total fixed at 1 MHz, split into M channels.
- **Incumbent:** H\*, re-tuned per network configuration on the train seeds 0–29 (same 22-rule family as R1/E1b).
  - The test seeds are not used for selection.
  - PPO is not used, because its observation and action sizes depend on K.
- **Seeds:** 13000–13099, test pool, fresh (never used before). One seed set for all arms, so every comparison is paired.

## E4 configurations

| Name | UAVs K | Channels M | K/M | Images / episode |
|---|---|---|---|---|
| N3x2 | 3 | 2 | 1.5 | 156 |
| N5x2 | 5 | 2 | 2.5 | 260 (manuscript configuration) |
| N7x2 | 7 | 2 | 3.5 | 364 |
| N5x3 | 5 | 3 | 1.67 | 260 |
| N5x4 | 5 | 4 | 1.25 | 260 |

**Arms per configuration:**
- H\*;
- H\* + calibrated MPC (`+mpc`);
- + nominal MPC (`+mpcnc`);
- + RobustMPC-style (`+mpcrh`);
- + pooled calibration without SNR bins (`+mpcpl`).

## E5 configurations (on N5x2, same seeds, paired with E4)

Calibrated MPC whose rate reports arrive d slots late (1 slot = 1 s) or are lost with probability p:

| Arm | d | p |
|---|---|---|
| `+mpcfb1l0` | 1 | 0 |
| `+mpcfb3l0` | 3 | 0 |
| `+mpcfb10l0` | 10 | 0 |
| `+mpcfb1l30` | 1 | 30% |
| `+mpcfb3l30` | 3 | 30% |

The `a9_mpc.Feedback` model applies to the rate model only.

## Hypotheses and decision rules (paired 95% t-CI over the 100 seeds)

**H1, the method helps the incumbent:** calibrated MPC − H\* > 0.

**H2, it beats the history-only predictor:** calibrated − RobustMPC-style > 0.

**H3, it beats the nominal model:** calibrated − nominal > 0.

**How each configuration counts:**
- A configuration counts for a hypothesis only if the CI lies entirely above 0.
- If H1 and H2 both pass in at least 4 of the 5 configurations, the paper may say "the gain persisted across the evaluated network sizes (K = 3–7, M = 2–4)" and must name any failing configuration.
- Otherwise the claim is restricted to the passing configurations, and the paper states where the method does not help.

**E5:**
- Report the calibrated MPC with delayed or lossy reports, minus the same with immediate reports, and minus nominal MPC.
- The paper may say the method "tolerates report delay up to d slots / loss of p" only for settings whose CI against nominal MPC stays entirely above 0.

## Secondary metrics (reported, not used for the decision)

- Return per image (return / number of images), so that different K can be compared.
- Qmin-compliant detector recall.
- Drop rate, deadline-miss rate and quality-floor violation rate.
- Decision time:
  - scheduler p99;
  - integrated p99;
  - fraction of slots over 50 ms.

## Known before running

- Smoke runs (seed 9900) show the JA scheduler costs:
  - about 32 ms per slot at K = 7, M = 2 (3^7 assignments);
  - about 36 ms per slot at K = 5, M = 4 (5^5 assignments).
- With the MPC added, about half of the slots exceeded 50 ms in those two configurations. This is reported as a scalability cost.
- The budget is not changed to hide it.

## Addendum 2026-11-22 (written while E4 was running, before any E4/E5/E6 test result was seen)

### E5 fix and rerun

ChatGPT's review of commit a7b1831b found a bug in the `+mpcfb` arms.

- **The bug:** only the calibrator received delayed or lost reports. The error tracker, which gates planning and sets the rate band (`trk.stats`), still saw immediate reports.
- **Fix:** the tracker now gets its own `Feedback` with the same delay, loss and seed.
- **Tests:** two added to `test_a9_mpc.py`:
  - a 10-slot delay shifts the calibrator and tracker counts by exactly 10 slots;
  - `ax2_eval` passes the same schedule to both.
- **Rerun:** the five E5 arms are rerun as `e5.json`, same seeds and same per-network H\*. The `+mpcfb` arms inside `e4.json` are superseded and not reported.
- **E4 is not affected,** because none of its arms use feedback delay.
- **Scope:** E5 delays and drops only the throughput reports used for planning. ACKs for ARQ and the queue observation of the scheduler are unchanged.

### E6 total-bandwidth sweep (ChatGPT GO)

- **Configurations:** K = 5, M = 2, B_total ∈ {0.5, 2} MHz. The 1 MHz point is N5x2 of E4: same seeds 13000–13099, paired.
- **Incumbent:** H\* re-tuned per bandwidth on the train seeds.
- **Arms:** same as E4.
- **Hypotheses:** H1–H3 as in E4.
- **Rule:** the paper may say "persisted across total bandwidths 0.5–2 MHz" only if H1 and H2 pass at all three bandwidths. Otherwise it names where the method fails.
- **Limitation:** this is a one-way sweep. It does not separate the interaction between bandwidth and contention, which would need a two-way K/M × bandwidth design.

### Caveat on E4 fairness

E4 re-tunes only the mode heuristic. The JA scheduler (selected at K5M2) stays frozen for every K/M. The comparison between MPC variants is fair because they share the same scheduler, but the whole stack is not re-optimised per configuration.
