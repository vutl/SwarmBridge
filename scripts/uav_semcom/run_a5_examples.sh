#!/bin/bash
# A5.2 qualitative example figure, after the GPU queue (shared lock).
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
PY=/home/uservutl/envs/semcom-a/bin/python
until [ -f $REPO/runs/uav_semcom/a5/fidelity/coco_testdev.json ]; do sleep 60; done
sleep 30
cd /home/uservutl/work
$PY $REPO/swarmbridge/experiments/uav3d_comm/gpulock_run.py $REPO -- $PY $REPO/swarmbridge/experiments/uav_semcom/a5_examples.py \
  /home/uservutl/work/a5_det/yolo11s_visdrone_960/weights/best.pt $REPO/data/visdrone/VisDrone2019-DET-val
echo EXAMPLES_DONE
