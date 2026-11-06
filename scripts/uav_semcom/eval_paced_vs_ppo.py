import sys; sys.path.insert(0, '/mnt/c/Users/USER/Documents/SwarmBridge/swarmbridge/experiments/uav_semcom')
import json, numpy as np, env as E, policies as P
R='/mnt/c/Users/USER/Documents/SwarmBridge/runs/uav_semcom'
MODES='cap_t2i,thumb64_up,cap_thumb64_i2i_t0.6,cap_thumb128_i2i_t0.3,avif_5,avif_20,avif_50,avif_80,mbt2018-mean_q1,mbt2018-mean_q3,mbt2018-mean_q5,mbt2018-mean_q8,jpeg2000_100,jpeg2000_25'.split(',')
tab=E.ModeTable([R+'/a11/profile.jsonl',R+'/a12/profile.jsonl'],MODES,R+'/a11/cpu_learned.jsonl'); tr,te=tab.split()
seeds=list(range(1000,1030))
ppo=json.load(open('/home/uservutl/work/a3_s0/eval_s0.json'))['results']['ppo']
pu=np.array([m['mean_util'] for m in ppo])
for p in [P.greedy(True), P.paced(0), P.paced(1), P.paced(2), P.paced(0, False)]:
    ms=[P.run_episode(E.UavSemcomEnv(tab,te,{}),p,s) for s in seeds]
    u=np.array([m['mean_util'] for m in ms]); d=pu-u
    h=2.045*d.std(ddof=1)/np.sqrt(len(d))
    print(f"{p.__name__:16s} util {u.mean():+.3f} ok {np.mean([m['ok_rate'] for m in ms]):.2f} E {np.mean([m['energy_kj'] for m in ms]):.0f}kJ | ppo - this {d.mean():+.3f} [{d.mean()-h:+.3f},{d.mean()+h:+.3f}]")
