# Paper A: E7 RL-family check, pre-registration

**Written 2026-11-22, before any E7 training or test run.**
- The only runs so far are smoke runs: 3000 training steps, plus one evaluation episode on seed 9900, to check the code.
- The user approved the bounded check.

## Why

The reviewer pointed out that only PPO-family agents were ever trained in Paper A.

**The main question is RQ-B:** does the online-calibrated MPC layer improve mode policies from different RL families? This tests the incumbent-agnostic claim of the method.

**The secondary question is RQ-A:** does a different RL family change the learned-vs-heuristic picture?

**This is not a rescue of a DRL claim.** The A-X9 protocol said "no SAC / DSAC / MAPPO to rescue the DRL framing". E7 has a different purpose, and its results are reported whatever they are.

## Agents (same task, environment and information as PPO-LA of A-X9)

**Task:** mode-only selection, 14 modes, on regime `K5_la`.
- Scheduler-aware LA, persistent interference.
- Causal LA observation features (`obs_la`, 464-dim observation).
- Slots are set by the frozen JA scheduler (`JA:stat:R:a1:m3`) on the `pacedB3` movement rule.
- TRAIN pool. Training budget: 2M RL decisions.

**Training protocol, same as PPO-LA:**
- Checkpoint and validation every 200k decisions.
- Validation on the VAL pool, seeds `C.PPO_VAL_SEEDS`.
- Selected checkpoint = best mean validation return; ties go to the earlier one.
- Discount 0.99 per simulated second, semi-MDP: the bootstrap factor is γ^dt (`offpolicy_rl.boot`).
- MLP [256, 256].

**QR-DQN** (Dabney et al. 2018):
- 50 quantiles, quantile-Huber loss with κ = 1.
- Target network, Polyak τ = 0.005.
- ε-greedy: ε goes from 1 to 0.05 over the first 10% of steps.
- Adam optimiser, batch 256.
- Replay buffer of 250k transitions; training starts after 20k transitions; 1 gradient step per 4 transitions.

**Discrete SAC** (Christodoulou 2019):
- Categorical actor and twin Q heads.
- Learned temperature with target entropy = r · log 14.
- Adam, learning rate 3e-4; batch, buffer, warm-up and update ratio as for QR-DQN.
- Evaluation uses the argmax of the actor.

**Implementation:** `offpolicy_rl.py` and `offpolicy_train.py`. No downloaded code is used.

**Unit tests** (`test_offpolicy_rl.py`, all pass):
- the γ^dt bootstrap factor;
- both agents learn a contextual bandit;
- the evaluation wrapper and action masks behave as expected.

## Hyper-parameter selection (validation only, fixed before training)

- **T1, seed 0, two settings per algorithm:**
  - QR-DQN learning rate ∈ {1e-4, 5e-4};
  - Discrete SAC entropy ratio r ∈ {0.3, 0.6}.
  - The setting with the higher best-checkpoint validation return is chosen.
- **T2:** seeds 1 and 2 with the chosen setting. Seed 0 of the chosen setting completes the three training seeds.
- **Fairness note:** PPO-LA had no sweep (it uses the A4 hyper-parameters). This gives the new families a slight advantage, and is stated in the report.
- **After T1 nothing changes:** no other hyper-parameters, budgets or algorithms are tried, whatever the test results.

## Evaluation (`paperA_r2.py e7`)

**PHY:** nominal LA, same as E1 `L` and E4 `N5x2`. Incumbent H\* = fixed mbt2018-mean q3, the H\* tuned on the train seeds.

**Pools and seeds:**
- Test pool: seeds 13000–13099, shared with E4, which gives a pairing check against E4's N5x2 arms.
- Test-dev pool (held-out content): fresh seeds 14000–14099.

**Arms:**
- H\*, H\* + calibrated MPC, H\* + nominal MPC.
- For each learned incumbent X ∈ {PPO-LA (A-X9 checkpoints), QR-DQN, Discrete SAC}: X, X + calibrated MPC, X + nominal MPC.
- Learned arms are averaged over their 3 training seeds per environment seed.

## Hypotheses and decision rules (paired 95% t-CI over environment seeds)

**RQ-B (decides the wording of the claim)**

For each learned incumbent X:
- B1: X + calibrated MPC − X > 0;
- B2: X + calibrated MPC − X + nominal MPC > 0.

The paper may say "the calibrated layer improved incumbents from three RL families (on-policy actor-critic, distributional value-based, off-policy maximum-entropy actor-critic)" only if all six test-pool CIs lie above 0. Otherwise the failures are named. Test-dev is reported as a replication.

**RQ-A (descriptive, no claim beyond the trained agents)**

Contrasts reported:
- X − H\*;
- X + MPC − H\* + MPC;
- X − PPO-LA.

**Wording rules:**
- Results are stated "with the evaluated training budget and settings".
- No statement that RL in general is unsuitable or superior.
- If a new family beats H\* + MPC with CI > 0 on both pools, that is reported as a positive finding, and the paper's learned-vs-heuristic statement changes accordingly.

**Also reported:**
- Qmin-compliant recall;
- validation learning curves;
- spread across training seeds;
- decision time per mode decision.

## Compute

- Up to 8 training runs of 2M decisions; two run at a time with 2 environment workers each.
- Training starts only after E4–E6 finish, so their latency numbers are not disturbed.

## Addendum 2026-11-22 (ChatGPT review of 61f2c489; written before any E7 training started)

This is the pre-registration of the **main** E7 run. The smoke tests listed above were run before it and are reported publicly.

### Queue fixes (operational only; the experiment design is unchanged)

- E7 now starts only if E4, E5 and E6 **finished successfully**:
  - each log ends with `EXIT 0`;
  - each result file holds the expected number of arms, with 100 seeds per arm.
- The queue now raises an error if the E7 evaluation exits non-zero or its result files are incomplete. It no longer only prints the exit code.

### Naming

- The class is `offpolicy_rl.DiscreteSAC`; `DSAC` remains as an alias for old checkpoints.
- The paper says "Discrete SAC", not DSAC. It is not Distributional SAC, and it is not a reproduction of Huang et al.

### Reporting (added, no new runs)

- Every learned incumbent is also reported **per training seed**:
  - the return of each seed;
  - each seed's B1/B2 contrasts with their environment-seed CIs.
- The registered CI (the 3 training seeds averaged per environment seed, then a paired t-CI over 100 environment seeds) measures environment variability only. Training-seed spread is reported next to it, not folded into it.
- When the MPC lifts a weak policy, the absolute return of X + MPC is also compared with H\* + MPC. A rescued weak policy does not make the final system better.

### Test-dev status

The test-dev pool was already used in earlier SwarmBridge rounds (E3, step C). The fresh *seeds* 14000–14099 do not make the content "never seen". It is held-out content, not an untouched dataset.

### Stop rule (restated)

After E7, the Paper A scope is frozen whatever the RL results. There will be no further algorithm round to rescue a claim.

## Operational note 2026-11-22 12:30 (no design change)

- **What happened:** the first E7 launch died at about 02:00, after 400k of 2M steps of the two T1 QR-DQN runs.
  - The job was tied to the Claude Code session. When the session ended, `wsl.exe` was closed and WSL shut down.
  - The partial logs are kept in `runs/uav_semcom/paperA_rl/interrupted_2026-11-22/`.
  - The partial runs are not used: an off-policy run cannot be resumed exactly, because the replay buffer and optimiser state are not saved.
- **The queue now:**
  - skips runs that are already complete;
  - moves interrupted runs aside and restarts them from scratch.
- **Launch:** the job is now started detached from the session (Windows `Start-Process`), so it survives the session ending.
- **Unchanged:** seeds, hyper-parameters and selection rules.
