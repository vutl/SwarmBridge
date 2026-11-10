#!/bin/bash
# R1a measurements: onboard YOLO11n detections, all tiles x {avif_50, avif_80} + BS detections (GPU, shared lock),
# then timing on an idle machine. Arg: N (smoke: first N images) or empty (all).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
W=/home/uservutl/work/r1a
ON=$W/yolo11n_visdrone_640/weights/best.pt
SV=/home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt
VAL=$REPO/data/visdrone/VisDrone2019-DET-val/images
TD=/home/uservutl/data/visdrone_dl/VisDrone2019-DET-test-dev/images
N=${1:-0}
OUT=$W/measure; [ "$N" != "0" ] && OUT=$W/smoke_measure
cd $REPO/swarmbridge/experiments/uav_semcom
$PY $LOCK $REPO -- $PY r1a_measure.py $OUT --onboard $ON --server $SV --val-dir $VAL --testdev-dir $TD --n $N > $OUT.log 2>&1 || { echo "MEASURE FAILED"; tail -20 $OUT.log; exit 1; }
$PY $LOCK $REPO -- $PY r1a_measure.py $OUT --timing --onboard $ON --server $SV --val-dir $VAL --testdev-dir $TD --n-time $([ "$N" != "0" ] && echo 3 || echo 30) >> $OUT.log 2>&1 || { echo "TIMING FAILED"; tail -20 $OUT.log; exit 1; }
tail -3 $OUT.log
sha256sum $ON $SV > $OUT/sha256.txt
echo R1A_MEASURE_DONE
