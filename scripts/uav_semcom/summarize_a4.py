import json,sys,os,numpy as np
R='runs/uav_semcom/a4_native/'
parts=sys.argv[1].split(',') if len(sys.argv)>1 else ['joint']
def ci(x): return x.mean(), 2.045*x.std(ddof=1)/np.sqrt(len(x))
for v in ['20','10']:
    T=json.load(open(f'{R}tune_v{v}_wt0.3.json')); H=T['tuned']; h=np.array(T['rows'][H]['test_return_all'])
    best_test=max(T['rows'],key=lambda n:T['rows'][n]['test_return'])
    print(f"== v{v}: tuned {H} test {h.mean():+.2f}; best-on-test {best_test} {T['rows'][best_test]['test_return']:+.2f}")
    for part in parts:
        means=[]
        for s in range(3):
            f=f'{R}v{v}_{part}_s{s}/eval_s{s}.json'
            if not os.path.exists(f): continue
            d=json.load(open(f)); res=d['results']; p=np.array([m['return'] for m in res['ppo']])
            means.append(p.mean())
            m,c=ci(p-h)
            g=lambda k: np.nanmean([x[k] for x in res['ppo']])
            hb=lambda k: None
            print(f"  {part} s{s}: ppo {p.mean():+.2f}  -tuned {m:+.2f} [{m-c:+.2f},{m+c:+.2f}]  util {g('mean_util'):.3f} ok {g('ok_rate'):.3f} delay {g('mean_delay'):.2f} psnr {g('mean_psnr'):.2f} mission {g('mission_s'):.1f} E {g('energy_kj'):.1f}  val_last {d['log'][-1]['val_return']:+.2f} curve {[round(r['val_return'],1) for r in d['log']][::2]}")
        if len(means)>1: print(f"  {part}: mean {np.mean(means):+.2f} sd-between-seeds {np.std(means,ddof=1):.2f}")
    r=T['rows'][H]; print(f"  tuned {H}: util {r['test_util']:.3f} ok {r['test_ok']:.3f} mission {r['test_mission']:.1f}")
