"""Paired multi-group frontier; network and power remain scenario assumptions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import core as C
from feasibility import times,service


def nondominated(quality,latency,tol=1e-9):
    q=np.asarray(quality);t=np.asarray(latency)
    return np.array([not any(q[j]>=q[i]-tol and t[j]<=t[i]+tol and
                    (q[j]>q[i]+tol or t[j]<t[i]-tol) for j in range(len(q)) if j!=i) for i in range(len(q))])


def bootstrap(delta,groups,seed=20261106,draws=2000):
    """Image-weighted cluster bootstrap; group membership fixed before scoring."""
    rng=np.random.default_rng(seed);ids=sorted(set(groups));a=np.asarray(delta)
    sums=np.array([a[np.array(groups)==g].sum() for g in ids]);sizes=np.array([sum(x==g for x in groups) for g in ids])
    ix=rng.integers(len(ids),size=(draws,len(ids)))
    z=sums[ix].sum(1)/sizes[ix].sum(1)
    return dict(mean=float(a.mean()),pointwise_cluster_bootstrap95=np.percentile(z,[2.5,97.5]).tolist(),groups=len(ids))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo');args=ap.parse_args();repo=Path(args.repo)
    old=repo/'runs/active_recovery/gate_a';out=repo/'runs/active_recovery/paired_cpu'
    measured=json.loads((out/'measurements.json').read_text());quality=json.loads((out/'quality.json').read_text())
    C.save(out/'timing_samples.json',dict(codec=measured['codec'],models={m:{n:[{k:v for k,v in r.items() if k!='records'} for r in reps] for n,reps in images.items()} for m,images in measured['models'].items()},config_sha256=measured['config_sha256']))
    previous=json.loads((old/'summary.json').read_text());split=json.loads((old/'split.json').read_text())
    rows=[r for r in split['images'] if r['split']=='dev'];names=[r['name'] for r in rows];groups=[r['group'] for r in rows]
    assert len(names)==216 and len(set(groups))==30 and set(measured['codec'])==set(names)
    modes=['nano_boxes','large_boxes','large_jpeg70','large_original'];primary=modes[:3]
    cached={m:json.loads((old/(m+'.json')).read_text())['images'] for m in modes}
    for m in modes:assert C.sha(old/(m+'.json'))==previous['prediction_hashes'][m]
    # Frozen analysis recipe derived from the prior sensitivity, not a blind test.
    cfg=dict(source_config_sha256=C.sha(out/'config.json'),rate_mbps=[.1,.3,.5,1,1.5,3,10,30,100],
             deadlines_s=[.1,.25,.5,1,2,5,10],original_deadlines_s=[2,5,10],
             compute_budgets_s=[.05,.1,.5,2,None],overhead_fraction=.1,common_s=.02,
             group_stability_fraction=2/3,original_grid_consecutive_rates_required=2,
             note='Development sensitivity, not preregistered independent hardware validation; native CPU timing first, no artificial slowdown in primary frontier.',
             energy_assumptions_source='runs/active_recovery/offloading/config.json',
             no_new_GPU_or_training=True)
    C.save(out/'analysis_config.json',cfg)
    def med(records,key):return float(np.median([r[key] for r in records]))
    inputs={};stage={};repeat_drift={};cpu_quality_drift={}
    for mode in modes:
        f1=[];bits=[];compute=[];remote=[];st=[]
        for name in names:
            codec=measured['codec'][name]
            if mode.endswith('boxes'):
                model=measured['models'][mode][name]
                f1.append(quality[mode+'_repeat0']['per_image'][name]['f1'])
                bits.append(model[0]['bytes']*8)
                compute.append(float(np.median([c['source_decode_ms']+c['resize_ms']+c['source_bgr_ms']+r['predict_ms']+r['encode_box_ms'] for c,r in zip(codec,model)]))/1000)
                remote.append(med(model,'decode_box_ms')/1000)
                st.append(med(model,'predict_ms'))
            elif mode=='large_jpeg70':
                for c in codec:assert c['jpeg']['70']['bytes']==cached[mode][name]['payload_bytes'], 'JPEG byte count changed: cached quality no longer paired'
                f1.append(previous['scores'][mode]['per_image'][name]['f1']);bits.append(codec[0]['jpeg']['70']['bytes']*8)
                compute.append(float(np.median([c['source_decode_ms']+c['resize_ms']+c['jpeg']['70']['encode_ms'] for c in codec]))/1000)
                remote.append(float(np.median([c['jpeg']['70']['decode_ms']+c['jpeg']['70']['bgr_ms'] for c in codec]))/1000+cached[mode][name]['processing_ms']/1000)
            else:
                f1.append(previous['scores'][mode]['per_image'][name]['f1']);bits.append(cached[mode][name]['payload_bytes']*8)
                compute.append(0.)
                remote.append(float(np.median([c['source_decode_ms']+c['resize_ms']+c['source_bgr_ms'] for c in codec]))/1000+cached[mode][name]['processing_ms']/1000)
        inputs[mode]=[np.array(v) for v in [f1,bits,compute,remote]]
        stage[mode]=dict(mean_F1=float(np.mean(f1)),mean_bits=float(np.mean(bits)),onboard_compute_ms_p50_p95=np.percentile(np.array(compute)*1000,[50,95]).tolist(),receiver_compute_ms_p50_p95=np.percentile(np.array(remote)*1000,[50,95]).tolist())
        if st:stage[mode]['detector_only_ms_p50_p95']=np.percentile(st,[50,95]).tolist()
    for mode in primary[:2]:
        drift=np.array([abs(m[1]['predict_ms']-m[0]['predict_ms'])/np.median([m[0]['predict_ms'],m[1]['predict_ms']]) for m in measured['models'][mode].values()])
        repeat_drift[mode]=dict(median_absolute_relative_difference=float(np.median(drift)),p95_absolute_relative_difference=float(np.percentile(drift,95)))
        deltas=[quality[mode+'_repeat1']['per_image'][n]['f1']-quality[mode+'_repeat0']['per_image'][n]['f1'] for n in names]
        cpu_quality_drift[mode]=dict(max_abs_F1_change=float(np.max(np.abs(deltas))),mean_F1_change=float(np.mean(deltas)))
    # Small replayable input table, avoiding full detector boxes in git.
    C.save(out/'paired_inputs.json',dict(names=names,groups=groups,
            modes={m:dict(zip(['f1','bits','onboard_s','receiver_s'],[a.tolist() for a in data])) for m,data in inputs.items()},
            source_measurements_sha256=C.sha(out/'measurements.json'),source_quality_sha256=C.sha(out/'quality.json')))
    group_indices={g:np.flatnonzero(np.array(groups)==g) for g in sorted(set(groups))}
    def evaluate(rate,deadline,budget=np.inf):
        return {m:service(*inputs[m],rate,deadline,compute_budget_s=budget,overhead_fraction=.1,common_s=.02) for m in modes}
    frontier=[];positive_task=[];group_sweep=[]
    for rate in cfg['rate_mbps']:
        for deadline in cfg['deadlines_s']:
            full=evaluate(rate,deadline)
            flags=nondominated([full[m]['mean_F1'] for m in primary],[full[m]['latency_p50_s'] for m in primary])
            all_nd=[];win=[]
            for g,ix in group_indices.items():
                result={m:service(*[a[ix] for a in inputs[m]],rate,deadline,overhead_fraction=.1,common_s=.02) for m in primary}
                nd=nondominated([result[m]['mean_F1'] for m in primary],[result[m]['latency_p50_s'] for m in primary])
                all_nd.append(bool(nd[2]));win.append(result[primary[2]]['mean_F1']>max(result[m]['mean_F1'] for m in primary[:2])+1e-9)
                group_sweep.append(dict(group=g,images=len(ix),rate_mbps=rate,deadline_s=deadline,jpeg_nondominated=bool(nd[2]),**{m:result[m] for m in primary}))
            item=dict(rate_mbps=rate,deadline_s=deadline,group_fraction_JPEG_nondominated=float(np.mean(all_nd)),group_fraction_JPEG_task_wins=float(np.mean(win)),
                      JPEG_nondominated=bool(flags[2]),results=full)
            frontier.append(item)
            if full['large_jpeg70']['mean_F1']-max(full[m]['mean_F1'] for m in primary[:2])>=.02 and np.mean(win)>=2/3:positive_task.append(dict(rate_mbps=rate,deadline_s=deadline))
    original_stable=[x for x in frontier if x['deadline_s'] in cfg['original_deadlines_s'] and x['JPEG_nondominated'] and x['group_fraction_JPEG_nondominated']>=2/3]
    stable_region=any(sum(any(x['rate_mbps']==r and x['deadline_s']==d for x in original_stable) for r in cfg['rate_mbps'][i:i+2])==2 for d in cfg['original_deadlines_s'] for i in range(len(cfg['rate_mbps'])-1))
    # Repeat-specific robustness: no new inference, and no averaging away timing drift.
    repeat_stability=[]
    for repeat in [0,1]:
        one={m:[a.copy() for a in v] for m,v in inputs.items()}
        for j,name in enumerate(names):
            c=measured['codec'][name][repeat]
            for mode in primary:
                if mode.endswith('boxes'):
                    r=measured['models'][mode][name][repeat]
                    one[mode][0][j]=quality[mode+'_repeat'+str(repeat)]['per_image'][name]['f1']
                    one[mode][1][j]=r['bytes']*8
                    one[mode][2][j]=(c['source_decode_ms']+c['resize_ms']+c['source_bgr_ms']+r['predict_ms']+r['encode_box_ms'])/1000
                    one[mode][3][j]=r['decode_box_ms']/1000
                else:
                    one[mode][2][j]=(c['source_decode_ms']+c['resize_ms']+c['jpeg']['70']['encode_ms'])/1000
                    one[mode][3][j]=(c['jpeg']['70']['decode_ms']+c['jpeg']['70']['bgr_ms'])/1000+cached[mode][name]['processing_ms']/1000
        cells=[]
        for rate in cfg['rate_mbps']:
            for deadline in cfg['original_deadlines_s']:
                flags=[]
                for ix in group_indices.values():
                    v=[service(*[a[ix] for a in one[m]],rate,deadline,overhead_fraction=.1,common_s=.02) for m in primary]
                    flags.append(nondominated([r['mean_F1'] for r in v],[r['latency_p50_s'] for r in v])[2])
                if np.mean(flags)>=2/3:cells.append(dict(rate_mbps=rate,deadline_s=deadline,group_fraction=float(np.mean(flags))))
        ok=any(all(any(c['rate_mbps']==r and c['deadline_s']==d for c in cells) for r in cfg['rate_mbps'][i:i+2]) for d in cfg['original_deadlines_s'] for i in range(len(cfg['rate_mbps'])-1))
        repeat_stability.append(dict(repeat=repeat,stable=ok,cells=cells))
    # Original-file offload is an additional control, not silently excluded from review.
    extended=[]
    for rate in cfg['rate_mbps']:
        full=evaluate(rate,2)
        nd=nondominated([full[m]['mean_F1'] for m in modes],[full[m]['latency_p50_s'] for m in modes])
        extended.append(dict(rate_mbps=rate,deadline_s=2,nondominated={m:bool(v) for m,v in zip(modes,nd)}))
    # F1 ignores confidence < .25: give box controls that exact payload pruning,
    # plus optimistic zero serialization cost, without rerunning any model.
    pruned={m:[a.copy() for a in inputs[m]] for m in primary}
    for mode in primary[:2]:
        for j,name in enumerate(names):
            recs=measured['models'][mode][name]
            n=sum(b[4]>=.25 for b in recs[0]['records'])
            pruned[mode][1][j]=8*(C.HEADER.size+C.RECORD.size*n)
            pruned[mode][2][j]-=med(recs,'encode_box_ms')/1000
            pruned[mode][3][j]=0.
    pruned_cells=[]
    for rate in cfg['rate_mbps']:
        for deadline in cfg['original_deadlines_s']:
            flags=[]
            for ix in group_indices.values():
                v=[service(*[a[ix] for a in pruned[m]],rate,deadline,overhead_fraction=.1,common_s=.02) for m in primary]
                flags.append(nondominated([r['mean_F1'] for r in v],[r['latency_p50_s'] for r in v])[2])
            if np.mean(flags)>=2/3:pruned_cells.append(dict(rate_mbps=rate,deadline_s=deadline,group_fraction=float(np.mean(flags))))
    # Pointwise paired CIs describe the fixed 3 Mbps / 2 s reference, not a grid search p-value.
    reference=evaluate(3,2);intervals={}
    q=inputs['large_jpeg70'];qt=times(*q[1:],3,overhead_fraction=.1,common_s=.02)[0]
    for mode in ['nano_boxes','large_boxes']:
        p=inputs[mode];pt=times(*p[1:],3,overhead_fraction=.1,common_s=.02)[0]
        intervals[mode]=dict(Q70_minus_mode_F1=bootstrap(q[0]-p[0],groups),Q70_minus_mode_latency_s=bootstrap(qt-pt,groups))
    # Sweep genuine native timings plus externally imposed compute/energy constraints.
    timing=[]
    for budget in cfg['compute_budgets_s']:
        for rate in cfg['rate_mbps']:
            for deadline in cfg['deadlines_s']:
                timing.append(dict(compute_budget_s=budget,rate_mbps=rate,deadline_s=deadline,results=evaluate(rate,deadline,np.inf if budget is None else budget)))
    C.save(out/'timing_frontier.json',timing);C.save(out/'group_frontier.json',group_sweep)
    energy_cfg=json.loads((repo/'runs/active_recovery/offloading/config.json').read_text())['energy'];energy=[]
    for horizon in energy_cfg['platform_horizons']:
        for power in energy_cfg['compute_to_radio_power']:
            for idle in energy_cfg['idle_to_radio_power']:
                for budget in energy_cfg['normalized_budgets_radio_s']:
                    for rate in [1,3,10]:
                        result={m:service(*inputs[m],rate,2,compute_to_radio_power=power,idle_to_radio_power=idle,energy_budget_radio_s=np.inf if budget is None else budget,platform_horizon_s=2 if horizon=='fixed_2s_mission' else None,overhead_fraction=.1,common_s=.02) for m in primary}
                        energy.append(dict(platform_horizon=horizon,power_ratio=power,idle_ratio=idle,energy_budget_radio_s=budget,rate_mbps=rate,results=result))
    C.save(out/'energy_frontier.json',energy)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['tab:blue','tab:orange','tab:green']
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,rate in zip(axs,[1,3,10]):
        for mode,color in zip(primary,colors):
            p=inputs[mode];lat=times(*p[1:],rate,overhead_fraction=.1,common_s=.02)[0]
            ax.scatter([np.median(lat[ix]) for ix in group_indices.values()],[p[0][ix].mean() for ix in group_indices.values()],alpha=.3,color=color,s=18)
            ax.scatter(np.median(lat),p[0].mean(),s=90,marker='X',color=color,label=mode)
        ax.set(xlabel='Completion latency p50 (s)',ylabel='Mean F1 before deadline filtering',title=f'{rate} Mbps; 30 groups + aggregate')
    axs[-1].legend(fontsize=8);fig.suptitle('Paired CPU timings; stage-summed reference, 10% bytes + 20 ms assumed radio latency')
    fig.savefig(out/'paired_frontier.png',dpi=160);plt.close(fig)
    limits=['216 dev images and two CPU repeats; this is not an onboard device/power measurement.',
            'CPU results are rescored; receiver GPU predictions/timing are reused and immutable. No new GPU inference.',
            'Latency is a sum of separately measured stages plus assumed constant goodput/overhead, not a network end-to-end measurement.',
            'Image source is already stored JPEG; camera encoding, sensing and disk I/O are outside the starting point.',
            'CPU four threads on workstation; no claimed embedded equivalence or guaranteed thermal/power behavior.',
            'Filename groups are sequence proxies, not verified independent flights.',
            'Frontier uses mean F1 and p50 latency; group-wise results, p95 and deadline delivery are reported separately.',
            'Cluster CIs are pointwise descriptive on reused dev, not grid-wide significance tests or a blind test.',
            'Shorter deadlines and imposed compute budgets remain exploratory. Primary stability uses native timings and original 2/5/10s deadlines.',
            'Power ratios unchanged from sensitivity; no measured joules and no claim of semantic recovery or RL benefit.']
    summary=dict(images=len(names),groups=len(group_indices),repeats=2,stage_summary=stage,repeat_drift=repeat_drift,cpu_quality_repeat_drift=cpu_quality_drift,
                 reference_3Mbps_2s=reference,paired_cluster_intervals=intervals,original_grid_stable_tradeoff=stable_region,
                 repeat_stability=repeat_stability,extended_frontier_with_original_file=extended,
                 F1_pruned_box_optimistic_control=dict(note='Prune decoded confidence <0.25, same F1 by definition; zero serialization cost favors box controls, not an additional timing measurement',stable_cells=pruned_cells),
                 original_grid_stable_cells=[{k:x[k] for k in ['rate_mbps','deadline_s','group_fraction_JPEG_nondominated']} for x in original_stable],
                 exploratory_task_gain_cells=positive_task,limits=limits,
                 verdict='Stable conditional tradeoff on workstation reference' if stable_region else 'Stable tradeoff not demonstrated on original grid',
                 gate_A_deployment_status='inconclusive_without_onboard_compute_power_and_radio_target',new_GPU_inference=0,training_steps=0,
                 source_hashes={p.name:C.sha(p) for p in [out/'measurements.json',out/'quality.json',old/'summary.json']})
    C.save(out/'summary.json',summary)
    lines=['# Gate A: paired CPU across 30 dev groups','',f'216 images, two CPU timing repeats/model/image, 30 filename groups. No training or holdout scoring.','',
           '| Mode | Mean F1 | UAV compute p50/p95 ms | Reference total at 3 Mbps, 2s p50/p95 s | On-time fraction |','|---|---:|---:|---:|---:|']
    for mode in modes:
        st=stage[mode];r=reference[mode];a,b=st['onboard_compute_ms_p50_p95']
        lines.append('| %s | %.4f | %.1f / %.1f | %.3f / %.3f | %.3f |'%(mode,st['mean_F1'],a,b,r['latency_p50_s'],r['latency_p95_s'],r['delivery_fraction']))
    lines+=['','**Original-grid stable tradeoff:** '+str(stable_region)+'. Requires JPEG to remain nondominated in quality/p50-latency on >=2/3 groups, at two adjacent tested goodputs and an original deadline. This is descriptive development evidence.', '',
            '| Goodput Mbps | Deadline s | Fraction of groups with nondominated JPEG |','|---|---:|---:|']
    for cell in original_stable:lines.append('| %g | %g | %.3f |'%(cell['rate_mbps'],cell['deadline_s'],cell['group_fraction_JPEG_nondominated']))
    lines+=['','Repeat-specific stability (same criterion, without averaging timings): '+str([r['stable'] for r in repeat_stability])+'.',
            'Timing repeat absolute relative difference p50/p95: '+str(repeat_drift)+'. CPU F1 is unchanged between repeats.',
            'Original-file offload is also retained as a fourth control in summary.json; the primary three-mode criterion alone does not prove Q70 beats every codec.',
            'Nondominated means a quality-latency tradeoff, not higher accuracy than local large or an RL benefit.',
            'Additional strong box control: discard scores <0.25 (no F1 change) and assume zero serialization cost. Stable cells: '+str(pruned_cells)+'. This is a favorable bound for boxes, not an extra hardware measurement.']
    lines+=['','Paired cluster-bootstrap 95% intervals (fixed 3 Mbps / 2s; pointwise, not blind confirmatory):','']
    for mode,d in intervals.items():
        for metric,v in d.items():lines.append('- %s, %s: %.4f [%.4f, %.4f].'%(mode,metric,v['mean'],*v['pointwise_cluster_bootstrap95']))
    lines+=['','![Paired frontier](paired_frontier.png)','',
            '**Decision:** '+summary['verdict']+'. Keep offloading only as a conditional candidate; actual UAV feasibility and semantic correction remain unresolved. No mobility or RL opened.', '',
            'Limits:',*['- '+x for x in limits], '',
            'Reproduce measurement: `bash scripts/active_recovery/run_paired_cpu.sh` (CPU-only, resumable).',
            'Reproduce analysis: `python swarmbridge/experiments/active_recovery/analyze_paired.py REPO`.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=summary['verdict'],reference=reference,stable_cells=len(original_stable),intervals=intervals),indent=2))


if __name__=='__main__':main()
