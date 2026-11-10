#!/bin/bash
# G-L2 0.5: on-board features of every manifest image (GPU through the shared lock + CPU timing).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
OUT=$REPO/runs/uav_semcom/g_l2
W=/home/uservutl/work/g_l2
VAL=$REPO/data/visdrone/VisDrone2019-DET-val/images
TD=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev/images
WEI=/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt
N=${1:-0}
mkdir -p $OUT $W
cd $REPO/swarmbridge/experiments/uav_semcom
if [ "$N" != "0" ]; then
  $PY $LOCK $REPO -- $PY g2_features.py $W/smoke_features --weights $WEI --val-dir $VAL --testdev-dir $TD --n $N --cpu-n 2 > $W/smoke_features.log 2>&1 || { echo SMOKE FAILED; tail -20 $W/smoke_features.log; exit 1; }
  tail -3 $W/smoke_features.log; head -c 900 $W/smoke_features.jsonl; exit 0
fi
$PY $LOCK $REPO -- $PY g2_features.py $OUT/features --weights $WEI --val-dir $VAL --testdev-dir $TD > $W/features.log 2>&1 || { echo FEATURES FAILED; tail -20 $W/features.log; exit 1; }
tail -2 $W/features.log
echo G2_FEATURES_DONE
