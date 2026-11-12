#!/usr/bin/env bash
# H1-CROP pilot 0B (EXPL) launcher, WSL env ~/envs/semcom-a. Approved sequence (reviews #8-#11):
#   smoke  : smoke_expl precompute (2 seqs x 10 epochs) + alignment overlays of 52 / 68 -> STOP for visual check
#   pilot  : pilot_expl precompute (11 seqs, 459 epochs) + eval -> result JSON
# Usage: bash h1_pilot0b_run.sh smoke|pilot "RIGHTS NOTE"
# Outputs stay outside git in ~/work/h1_pilot0b/. GPU steps hold the shared lock (gpulock_run.py).
set -euo pipefail
STAGE=$1; NOTE=$2
R=/mnt/c/Users/USER/Documents/SwarmBridge
D=$R/swarmbridge/experiments/uav_semcom
PY=/home/uservutl/envs/semcom-a/bin/python
ZIP=$R/data/mdmt/Multi-Drone-Multi-Object-Detection-and-Tracking.zip
W=/home/uservutl/work/h1_pilot0b
ON=/home/uservutl/work/r1a/yolo11n_visdrone_640/weights/best.pt
SV=/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt
LOCK="$PY $R/swarmbridge/experiments/uav3d_comm/gpulock_run.py $R --"
cd "$D"
$PY -m pytest -q -p no:cacheprovider test_h1_pilot0b.py test_h1_gate0a.py test_h1_gate0a_audit.py | tail -1
case "$STAGE" in
  smoke)
    $PY h1_pilot0b_align_check.py "$ZIP" $W/align --seqs 52,68 --rights-note "$NOTE"
    $LOCK $PY h1_pilot0b_precompute.py h1_pilot0b_config.json $W/smoke_expl --source mdmt --zip "$ZIP" \
        --stage smoke_expl --rights-note "$NOTE" --onboard $ON --server $SV
    echo "SMOKE DONE: inspect $W/align/*.jpg before running the pilot (no criterion is read at this stage)";;
  pilot)
    $LOCK $PY h1_pilot0b_precompute.py h1_pilot0b_config.json $W/pilot_expl --source mdmt --zip "$ZIP" \
        --stage pilot_expl --rights-note "$NOTE" --onboard $ON --server $SV
    $PY h1_pilot0b_eval.py h1_pilot0b_config.json $W/pilot_expl $W/pilot_expl_result.json --zip "$ZIP" --procs 5;;
  *) echo "stage must be smoke or pilot"; exit 2;;
esac
