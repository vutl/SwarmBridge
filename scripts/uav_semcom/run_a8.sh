# A8 mismatch ladder (reports/swarmbridge-a8-b1-preregistration.md). CPU only, resumable.
set -e
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
O=$REPO/runs/uav_semcom/a8
A5=$REPO/runs/uav_semcom/a5
W=/home/uservutl/work/a5
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
mkdir -p $O
$PY -m pytest -q -p no:cacheprovider test_a5.py test_a55.py | tail -1
NEW="phy_mcs_l75,phy_mcs_l50,phy_mcs_l25"
ALL="phy_mcs phy_mcs_l75 phy_mcs_l50 phy_mcs_l25 phy_mcs_blind"
[ -f $A5/tune/phy_mcs_l25.json ] || $PY a5_eval.py tune --regimes $NEW --procs 5 > $O/tune_a5.log 2>&1
[ -f $REPO/runs/uav_semcom/a55/tune/phy_mcs_l25.json ] || $PY a55_eval.py tune --regimes $NEW --procs 5 > $O/tune_a55.log 2>&1
echo "tune done $(date)"; tail -3 $O/tune_a55.log
best() { $PY -c "import json; print(json.load(open('$A5/train/$1/train.json'))['best_steps'])"; }
P=""
for s in 0 1 2; do P="$P,P:mode_best_s$s=$W/phy_mcs_blind_mode_s$s/ckpt_$(best phy_mcs_blind_mode_s$s).zip:mode:pacedB2+budget1:PPPP"; done
POL="T5:best,T:SC,T:HSTAR${P}"
for pool in test testdev; do
  [ -f $O/eval_$pool.json ] || $PY a55_eval.py eval $O/eval_$pool.json --regimes $(echo $ALL | tr ' ' ,) --pool $pool --policies $POL --procs 5 > $O/eval_$pool.log 2>&1
  echo "eval $pool done $(date)"
done
for pool in test testdev; do
  for r in $ALL; do
    [ -f $O/boot_${pool}_$r.json ] || $PY a55_eval.py boot $O/boot_${pool}_$r.json --regimes $r --pool $pool --policies $POL --procs 5 > $O/boot_${pool}_$r.log 2>&1
    echo "boot $pool $r done $(date)"
  done
done
echo A8_DONE
