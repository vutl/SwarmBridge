#!/bin/bash
# A5 final evaluation (test pool + test-dev pool, seeds 7000..7099, used once; paired across policies).
# Waits until every PPO run of the CPU queue has finished. Resumable (each output skipped if present).
#   1. heuristics, every regime, test pool;
#   2. heuristics, selected regimes, test-dev pool;
#   3. A4 PPO models (td joint / td mode-only, seeds 0-2) ZERO-SHOT on every K = 3 regime, test pool;
#   4. A5 PPO (selected regimes, joint + mode-only, seeds 0-2): best-val checkpoint (pre-registered) and last
#      checkpoint, test pool; best-val joint also with action replacement (16 combos, heuristic = the regime's tuned
#      reuse heuristic: its slot part for move / channel / power, its mode part for modes); best-val on test-dev.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work/a5
WA4=/home/uservutl/work
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
R3=$($PY -c "import json; print(json.load(open('$A5/selected.json'))['R3'])")
SEL="ref K5 $R3"
for s in 0 1 2; do for r in $SEL; do for part in joint mode; do
  until [ -f $A5/train/${r}_${part}_s$s/train.json ]; do sleep 120; done
done; done; done
mkdir -p $A5/final
REGS="ref K2 K4 K5 v10 v15 env_suburban env_dense env_highrise fade_k3 fade_k0 dl_tight dl_loose q_low q_high wt0 wt0.1 wt1 phy_gap3 phy_mcs phy_mcs_blind"
K3=$(echo $REGS | tr ' ' '\n' | grep -v -E '^K[245]$' | tr '\n' ',' | sed 's/,$//')
POL="T:fixed,H:greedy,T:base,T:budget,T:reuse,H:pacedB1.5+budget1.5,H:greedy+budget1+reuse,T:reuse@oracle"
ev() { local out=$1; shift; [ -f $out ] || $PY a5_eval.py eval $out "$@" > ${out%.json}.log 2>&1 || { echo "EVAL FAILED $out"; exit 1; }; echo "done $out $(date)"; }
ev $A5/final/test_heur.json --regimes $(echo $REGS | tr ' ' ,) --pool test --seeds 7000:7100 --policies $POL
ev $A5/final/testdev_heur.json --regimes $(echo $SEL | tr ' ' ,) --pool testdev --seeds 7000:7100 --policies $POL
ZS=""
for s in 0 1 2; do
  ZS="$ZS,P:a4_joint_s$s=$WA4/a4td_v20_joint_s$s/ppo_s$s.zip:joint:pacedB1.5:PPPP"
  ZS="$ZS,P:a4_mode_s$s=$WA4/a4td_v20_mode_s$s/ppo_s$s.zip:mode:pacedB1.5:PPPP"
done
ev $A5/final/test_a4_zeroshot.json --regimes $K3 --pool test --seeds 7000:7100 --policies ${ZS#,}
best() { $PY -c "import json; print(json.load(open('$A5/train/$1/train.json'))['best_steps'])"; }
for r in $SEL; do
  P=""; PD=""
  for s in 0 1 2; do
    for part in joint mode; do
      N=${r}_${part}_s$s; B=$(best $N)
      P="$P,P:${part}_best_s$s=$W/$N/ckpt_$B.zip:$part:T~reuse:PPPP,P:${part}_last_s$s=$W/$N/ckpt_2000000.zip:$part:T~reuse:PPPP"
      PD="$PD,P:${part}_best_s$s=$W/$N/ckpt_$B.zip:$part:T~reuse:PPPP"
    done
  done
  ev $A5/final/test_ppo_$r.json --regimes $r --pool test --seeds 7000:7100 --policies ${P#,}
  ev $A5/final/testdev_ppo_$r.json --regimes $r --pool testdev --seeds 7000:7100 --policies ${PD#,}
  M=""
  for s in 0 1 2; do
    N=${r}_joint_s$s; B=$(best $N)
    for c in PPPH PPHP PHPP HPPP PPHH PHPH HPPH PHHP HPHP HHPP PHHH HPHH HHPH HHHP HHHH; do
      M="$M,P:joint_best_s${s}_$c=$W/$N/ckpt_$B.zip:joint:T~reuse:$c"
    done
  done
  ev $A5/final/test_replace_$r.json --regimes $r --pool test --seeds 7000:7100 --policies ${M#,}
done
echo A5_FINAL_DONE
