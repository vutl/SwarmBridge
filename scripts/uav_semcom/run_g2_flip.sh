#!/bin/bash
# G-L2 0.4: flipped-image re-measurement of every (train/val image, codec mode); one GPU job through the shared lock.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
OUT=$REPO/runs/uav_semcom/g_l2/flip
W=/home/uservutl/work/g_l2
VAL=$REPO/data/visdrone/VisDrone2019-DET-val
WEI=/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt
X=$REPO/swarmbridge/experiments/uav_semcom
N=${1:-0}
mkdir -p $OUT $W
cd $X
if [ "$N" != "0" ]; then
  $PY $LOCK $REPO -- $PY g2_flip.py $W/smoke_flip.jsonl --image-dir $VAL/images --ann-dir $VAL/annotations --weights $WEI --n $N > $W/smoke_flip.log 2>&1 || { echo "SMOKE FAILED"; tail -20 $W/smoke_flip.log; exit 1; }
  echo smoke ok; exit 0
fi
$PY $LOCK $REPO -- $PY g2_flip.py $OUT/per_image_flip.jsonl --image-dir $VAL/images --ann-dir $VAL/annotations --weights $WEI > $W/flip.log 2>&1 || { echo "FLIP FAILED"; tail -20 $W/flip.log; exit 1; }
sha256sum $WEI > $OUT/sha256.txt
echo G2_FLIP_DONE
