# A-E11 dev selection (written before the final run on seeds 8600–8619)

- Dev seeds 8000–8019 (development seeds of A-E5…A-E8), `--procs 6` (WSL has 6 CPUs).
- `invalid_oversubscribed/`: first dev run with `--procs 14` on 6 CPUs. 87–94 % of decisions hit the
  cooperative 50 ms budget (abort_frac), so every controller collapsed to ≈ PPO. Execution error, not a result; kept
  only as a record.
- Pre-declared rule (ae11_summary.py docstring): within each family, keep the configuration with the highest
  worst-scenario mean gain over PPO (maximin over 9 scenarios).
  - recency {win=30, win=100}: worst +0.02 (win=30) vs −3.53 (win=100) → **win=30**
  - split {tcold=30, sd=trust}: worst −0.31 (tcold=30) vs −0.78 (sd=trust) → **tcold=30**
- Final run (seeds 8600–8619, never used before): primary = win=30, tcold=30, against ppo, cold, pre.
  sd=trust is also run and reported, but as a secondary, non-selected variant (it had the better dev mean, +1.39 vs
  +0.91, and no matched-case cost; the rule ranks it second).
