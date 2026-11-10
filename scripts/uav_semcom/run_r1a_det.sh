#!/bin/bash
# R1a: train the onboard YOLO11n (one GPU job through the shared lock).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
LOCK=$REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py
W=/home/uservutl/work/r1a
mkdir -p $W
cd $W
$PY $LOCK $REPO -- $PY $REPO/swarmbridge/experiments/uav_semcom/r1a_det.py /home/uservutl/data/visdrone_yolo $W > $W/det_train.log 2>&1 || { echo "TRAIN FAILED"; tail -30 $W/det_train.log; exit 1; }
tail -5 $W/yolo11n_visdrone_640/results.csv
sha256sum $W/yolo11n_visdrone_640/weights/best.pt
echo R1A_DET_DONE
