#!/bin/bash
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom
while pgrep -f "paperA_r2.py e4" > /dev/null; do sleep 30; done
/home/uservutl/envs/semcom-a/bin/python paperA_r2.py e5 --procs 6 > /mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_r2/e5.log 2>&1
echo EXIT $? >> /mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_r2/e5.log
