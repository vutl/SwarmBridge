#!/bin/bash
# A3 batch 1: mission-time cost sweep (w_time 0.1, 0.3) and two more PPO seeds at w_time 0. CPU only, sequential.
S=/mnt/c/Users/USER/Documents/SwarmBridge/scripts/uav_semcom/run_a3.sh
bash $S /home/uservutl/work/a3_wt0.3_s0 3e6 30 0 '{"w_time": 0.3}'
bash $S /home/uservutl/work/a3_wt0.1_s0 3e6 30 0 '{"w_time": 0.1}'
bash $S /home/uservutl/work/a3_s1 3e6 30 1
bash $S /home/uservutl/work/a3_s2 3e6 30 2
echo BATCH_DONE
