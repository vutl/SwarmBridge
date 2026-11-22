#!/bin/bash
cd /mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom
/home/uservutl/envs/semcom-a/bin/python paperA_e7_queue.py >> /mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_rl/queue.log 2>&1
echo EXIT $? >> /mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom/paperA_rl/queue.log
