#!/bin/bash
# POST-HOC diagnostic (after the A4 channel analysis): tune the channel-reuse heuristic family (mode_heur.reuse over the
# base + budget families) on train seeds at 20 m/s, then evaluate every member on the fresh seeds 5000..5099.
REPO=/mnt/c/Users/USER/Documents/SwarmBridge
N=$REPO/runs/uav_semcom/a4_native
W=/home/uservutl/work
export CUDA_VISIBLE_DEVICES=""
cd $REPO/swarmbridge/experiments/uav_semcom
/home/uservutl/envs/semcom-a/bin/python -m pytest -q -p no:cacheprovider test_env_survey.py 2>&1 | tail -2
MODES=cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25
nice -n 10 /home/uservutl/envs/semcom-a/bin/python tune_ev.py $N/tune_reuse_v20_wt0.3.json --family reuse --procs 2 \
  --env survey --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES \
  --cfg '{"mode_info":"predicted","v_max":20,"w_time":0.3}' | tail -4
nice -n 10 /home/uservutl/envs/semcom-a/bin/python final_a4.py $W/final_reuse.json --heur-family reuse --slot-heur pacedB1.5 --procs 2 \
  --profiles $N/a11/profile.jsonl $N/a12/profile.jsonl --cpu-file $N/a11/cpu_learned.jsonl --modes $MODES \
  --cfg '{"mode_info":"predicted","v_max":20,"w_time":0.3}' > $W/final_reuse.log 2>&1
cp $W/final_reuse.json $N/final/reuse_heur.json; echo REUSE_DONE
