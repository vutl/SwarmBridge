#!/bin/bash
# A5.5 CPU queue (reports/uav-semcom-a5.5.md section 0). Resumable: each output is skipped if present.
#   1. tests; 2. tune the self-calibrating family (train, 21 regimes); 3. CONFIRMATORY evaluation, fresh env seeds
#   8000..8099: heuristics (21 regimes, test pool; 3 regimes, test-dev), A5 PPO checkpoints (3 regimes, both pools);
#   4. EXPLORATORY: new heuristics on the A5 test seeds 7000..7099 (comparable with the A5 tables);
#   5. video cluster bootstrap with re-simulation (B = 50, 20 env seeds per replicate), 3 regimes x 2 pools.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
O=$REPO/runs/uav_semcom/a55
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work/a5
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
mkdir -p $O/final $O/boot $O/tune
$PY -m pytest -q test_a55.py test_a5.py > $O/tests.log 2>&1 || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }
echo "tests ok $(date)"
REGS="ref K2 K4 K5 v10 v15 env_suburban env_dense env_highrise fade_k3 fade_k0 dl_tight dl_loose q_low q_high wt0 wt0.1 wt1 phy_gap3 phy_mcs phy_mcs_blind"
SEL="ref K5 phy_mcs_blind"
need=""
for r in $REGS; do [ -f $O/tune/$r.json ] || need="$need,$r"; done
if [ -n "$need" ]; then
  $PY a55_eval.py tune --regimes ${need#,} > $O/tune.log 2>&1 || { echo "TUNE FAILED"; tail -20 $O/tune.log; exit 1; }
fi
echo "tune done $(date)"
ev() { local out=$1; shift; [ -f $out ] || $PY a55_eval.py "$@" > ${out%.json}.log 2>&1 || { echo "FAILED $out"; tail -20 ${out%.json}.log; exit 1; }; echo "done $out $(date)"; }
HP="T5:fixed,T5:base,T5:budget,T5:reuse,T5:best,T:SC,T:HSTAR,H:greedy+budget1+reuse"
ev $O/final/test_heur.json eval $O/final/test_heur.json --regimes $(echo $REGS | tr ' ' ,) --pool test --policies $HP
ev $O/final/testdev_heur.json eval $O/final/testdev_heur.json --regimes $(echo $SEL | tr ' ' ,) --pool testdev --policies $HP
best() { $PY -c "import json; print(json.load(open('$A5/train/$1/train.json'))['best_steps'])"; }
ppo_specs() {   # $1 regime -> PPO policy specs (best-val checkpoints of A5)
  local r=$1 P=""
  for s in 0 1 2; do
    P="$P,P:mode_best_s$s=$W/${r}_mode_s$s/ckpt_$(best ${r}_mode_s$s).zip:mode:T~best:PPPP"
    P="$P,P:joint_best_s$s=$W/${r}_joint_s$s/ckpt_$(best ${r}_joint_s$s).zip:joint:T~best:PPPP"
    if [ $r = K5 ]; then
      B=$(best K5_joint_long_s$s); F=$W/K5_joint_long_s$s/ckpt_$B.zip; [ -f $F ] || F=$W/K5_joint_s$s/ckpt_$B.zip
      P="$P,P:joint_long_best_s$s=$F:joint:T~best:PPPP"
    fi
  done
  echo ${P#,}
}
for r in $SEL; do
  for pool in test testdev; do
    ev $O/final/${pool}_ppo_$r.json eval $O/final/${pool}_ppo_$r.json --regimes $r --pool $pool --policies $(ppo_specs $r)
  done
done
ev $O/final/expl_test_heur_7000.json eval $O/final/expl_test_heur_7000.json --regimes $(echo $REGS | tr ' ' ,) --pool test \
  --seeds 7000:7100 --exploratory --policies T5:best,T:SC,T:HSTAR
ev $O/final/expl_testdev_heur_7000.json eval $O/final/expl_testdev_heur_7000.json --regimes $(echo $SEL | tr ' ' ,) \
  --pool testdev --seeds 7000:7100 --exploratory --policies T5:best,T:SC,T:HSTAR
for r in $SEL; do
  for pool in test testdev; do
    ev $O/boot/${pool}_$r.json boot $O/boot/${pool}_$r.json --regimes $r --pool $pool --policies T5:best,T:SC,$(ppo_specs $r)
  done
done
echo A55_DONE
