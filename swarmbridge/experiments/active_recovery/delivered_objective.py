"""Maximize fixed-mode delivered F1 at each rate/deadline, not a weighted reward.

Existing predictions and paired timing only. Offline best-fixed comparisons
must not be mistaken for GT-aware per-image scheduling or a deployable policy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import core as C
from feasibility import times

VISIBLE=['nano_boxes','large_boxes','large_jpeg70','large_original','large_jpeg10']
BOX=['nano_boxes','large_boxes','nano_boxes_pruned_bound','large_boxes_pruned_bound']
OFF=['large_jpeg70','large_original','large_jpeg10']


def winners(scores,tolerance=1e-4):
    best=max(scores.values())
    return [m for m,v in scores.items() if best-v<=tolerance]


def delivered(f1,bits,onboard,receiver,rate,deadline):
    completion,_=times(bits,onboard,receiver,rate,overhead_fraction=.1,common_s=.02)
    ontime=completion<=deadline
    return np.asarray(f1)*ontime,completion,ontime


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo');args=ap.parse_args();repo=Path(args.repo)
    parent=repo/'runs/active_recovery';paired=parent/'paired_cpu';old=parent/'gate_a';out=parent/'delivered_objective';out.mkdir(parents=True,exist_ok=True)
    table=json.loads((paired/'paired_inputs.json').read_text());samples=json.loads((paired/'timing_samples.json').read_text())
    score=json.loads((old/'summary.json').read_text());q10=json.loads((old/'large_jpeg10.json').read_text())['images']
    assert C.sha(old/'large_jpeg10.json')==score['prediction_hashes']['large_jpeg10']
    names,groups=table['names'],table['groups'];assert len(names)==216 and len(set(groups))==30
    inputs={m:[np.array(d[k],float) for k in ['f1','bits','onboard_s','receiver_s']] for m,d in table['modes'].items()}
    q10_data=[[],[],[],[]]
    for name in names:
        c=samples['codec'][name]
        assert all(x['jpeg']['10']['bytes']==q10[name]['payload_bytes'] for x in c)
        q10_data[0].append(score['scores']['large_jpeg10']['per_image'][name]['f1']);q10_data[1].append(q10[name]['payload_bytes']*8)
        q10_data[2].append(np.median([x['source_decode_ms']+x['resize_ms']+x['jpeg']['10']['encode_ms'] for x in c])/1000)
        q10_data[3].append(np.median([x['jpeg']['10']['decode_ms']+x['jpeg']['10']['bgr_ms'] for x in c])/1000+q10[name]['processing_ms']/1000)
    inputs['large_jpeg10']=[np.array(a) for a in q10_data]
    # Preserve the previously used optimistic F1-specific box payload control.
    measured=json.loads((paired/'measurements.json').read_text())
    for m in BOX[:2]:
        control=[a.copy() for a in inputs[m]]
        for i,n in enumerate(names):
            records=measured['models'][m][n]
            count=sum(b[4]>=.25 for b in records[0]['records'])
            control[1][i]=8*(C.HEADER.size+C.RECORD.size*count)
            control[2][i]-=np.median([r['encode_box_ms'] for r in records])/1000
            control[3][i]=0
        inputs[m+'_pruned_bound']=control
    cfg=dict(rate_mbps=[.1,.3,.5,1,1.5,3,10,30,100],deadline_s=[.1,.25,.5,1,2,5,10],original_deadline_s=[2,5,10],
             F1_tie_tolerance=1e-4,minimum_descriptive_gain=.02,group_positive_fraction_threshold=2/3,
             overhead_fraction=.1,common_s=.02,compute_budget_imposed=False,weighted_reward=False,
             objective='maximum mean delivered F1 of a fixed mode over the dev set at each rate/deadline',
             completion_time='unclipped stage-summed completion estimate, reported independently',
             power_assumptions_changed=False,original_file_always_included=True,
             note='Development analysis, not blind confirmation. No per-image GT mode selection.')
    C.save(out/'config.json',cfg)
    C.save(out/'inputs.json',dict(names=names,groups=groups,modes={m:{k:a.tolist() for k,a in zip(['f1','bits','onboard_s','receiver_s'],v)} for m,v in inputs.items()}))
    group_ids=sorted(set(groups));ixs={g:np.flatnonzero(np.array(groups)==g) for g in group_ids}
    rng=np.random.default_rng(20261106);draws=rng.integers(len(group_ids),size=(2000,len(group_ids)))
    sizes=np.array([len(ixs[g]) for g in group_ids]);cells=[];flat=[];group_rows=[]
    for deadline in cfg['deadline_s']:
        for rate in cfg['rate_mbps']:
            values={};metrics={}
            for m,v in inputs.items():
                val,completion,on=delivered(*v,rate,deadline);values[m]=val
                metrics[m]=dict(delivered_F1=float(val.mean()),on_time_fraction=float(on.mean()),
                                completion_p50_s=float(np.median(completion)),completion_p95_s=float(np.percentile(completion,95)),
                                completion_mean_s=float(completion.mean()))
                flat.append(dict(rate_mbps=rate,deadline_s=deadline,mode=m,**metrics[m]))
            means={m:r['delivered_F1'] for m,r in metrics.items()}
            box=max(means[m] for m in BOX);off=max(means[m] for m in OFF);gap=off-box
            group_gaps=[]
            for g,ix in ixs.items():
                gm={m:float(v[ix].mean()) for m,v in values.items()};gd=max(gm[m] for m in OFF)-max(gm[m] for m in BOX)
                group_gaps.append(gd);group_rows.append(dict(group=g,images=len(ix),rate_mbps=rate,deadline_s=deadline,best_box_F1=max(gm[m] for m in BOX),best_offload_F1=max(gm[m] for m in OFF),delta=gd))
            sums={m:np.array([v[ixs[g]].sum() for g in group_ids]) for m,v in values.items()}
            bm={m:s[draws].sum(1)/sizes[draws].sum(1) for m,s in sums.items()}
            boot=np.max([bm[m] for m in OFF],axis=0)-np.max([bm[m] for m in BOX],axis=0)
            cell=dict(rate_mbps=rate,deadline_s=deadline,original_deadline=deadline in cfg['original_deadline_s'],
                      best_fixed_modes=winners(means),best_fixed_F1=max(means.values()),
                      best_box_modes=winners({m:means[m] for m in BOX}),best_offload_modes=winners({m:means[m] for m in OFF}),
                      best_offload_minus_best_box=gap,
                      group_fraction_offload_strictly_better=float(np.mean(np.array(group_gaps)>cfg['F1_tie_tolerance'])),
                      pointwise_reselected_family_gap_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),metrics=metrics)
            cells.append(cell)
    with (out/'per_mode.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    C.save(out/'group_results.json',group_rows)
    original=[c for c in cells if c['original_deadline']]
    useful=[c for c in cells if c['best_offload_minus_best_box']>=.02 and c['group_fraction_offload_strictly_better']>=2/3]
    original_gains=[c for c in original if c['best_offload_minus_best_box']>cfg['F1_tie_tolerance']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rates=cfg['rate_mbps'];deadlines=cfg['deadline_s']
    delta=np.array([[next(c for c in cells if c['rate_mbps']==r and c['deadline_s']==d)['best_offload_minus_best_box'] for r in rates] for d in deadlines])
    best=np.array([[next(c for c in cells if c['rate_mbps']==r and c['deadline_s']==d)['best_fixed_F1'] for r in rates] for d in deadlines])
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,data,title,cmap,vmin,vmax in zip(axs,[delta,best],['Best fixed offload minus best fixed boxes','Best fixed delivered F1 (all modes)'],['RdBu','viridis'],[-.6,0],[.6,.6]):
        im=ax.imshow(data,origin='lower',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set(xticks=range(len(rates)),xticklabels=rates,yticks=range(len(deadlines)),yticklabels=deadlines,xlabel='Effective service rate (Mbps)',ylabel='Deadline (s)',title=title)
        for i in range(len(deadlines)):
            for j in range(len(rates)):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=7,color='black' if ax==axs[0] else 'white')
        fig.colorbar(im,ax=ax)
    fig.suptitle('216 paired dev images; native CPU timings; 10% framing + 20 ms assumed link latency\nOriginal deadlines: 2/5/10s; shorter deadlines exploratory; no per-image GT policy')
    fig.savefig(out/'delivered_F1.png',dpi=160);plt.close(fig)
    sources=[paired/'paired_inputs.json',paired/'timing_samples.json',paired/'measurements.json',old/'summary.json',old/'large_jpeg10.json']
    result=dict(cells=cells,original_deadline_cells=len(original),original_deadline_offload_strict_gain_cells=len(original_gains),
                original_deadline_max_gap=max(c['best_offload_minus_best_box'] for c in original),
                gate_B_opened=False,mission_deadlines_revised=False,
                decision='No task-utility gain demonstrated on original deadlines; tighter-deadline task gains are exploratory, not authorization to change mission requirements.',
                descriptive_useful_cells=[{k:c[k] for k in ['rate_mbps','deadline_s','original_deadline','best_offload_modes','best_offload_minus_best_box','group_fraction_offload_strictly_better']} for c in useful],
                no_inference_or_training=True,gpu_seconds=0,holdout_scored=False,
                source_hashes={str(p.relative_to(repo)):C.sha(p) for p in sources},
                limits=['Native workstation CPU plus reused GPU reference, not UAV/network end-to-end measurement.',
                        'Rate/deadline cells are parameter combinations, not independent trials; identical results at 2/5/10s do not provide three independent confirmations.',
                        'All results are reused development images and sequence proxies; deadlines below 2s are exploratory.',
                        'Family optimum is best fixed mode on the dataset, not GT-guided selection per image.',
                        '1e-4 F1 tie tolerance avoids treating CPU/GPU rounding as a substantive advantage.',
                        'Pruned box control keeps F1 but may change AP; zero serialization is an optimistic bound for boxes.',
                        'Completion times include late packets without clipping; no arbitrary weighted reward.',
                        'Power assumptions unchanged; no new energy measurement or correctness claim for generative recovery.',
                        'Bootstrap reselects the best fixed mode within each family per group resample; pointwise exploratory intervals, no multiple-testing claim.',
                        'Gap to original-image F1 is a reference difference, not a mathematical upper bound on every correction algorithm.'])
    C.save(out/'summary.json',result)
    lines=['# Delivered F1 objective — all existing modes','',
           'No new detector, codec measurement, GPU, training, dataset or holdout evaluation. Native paired CPU stages from 216 images / 30 filename groups.','',
           f"**Original deadlines 2/5/10s:** {len(original_gains)}/{len(original)} cells have an offloading advantage above 1e-4 F1. "
           f"Maximum difference: {result['original_deadline_max_gap']:+.6f}. Completion-time savings alone do not increase this objective. "
           'These cells are parameter combinations, not independent trials; repeated deadline rows are not extra evidence.', '',
           'All source-image/JPEG choices are included, plus optimistic pruned-box F1 controls. F1 is zero on a late result. '
           'Tied modes within 1e-4 are reported together; completion time is a separate quantity, not a reward term.', '',
           '| Mbps | Deadline s | Best fixed modes (including ties) | Best F1 | Offload minus box |','|---:|---:|---|---:|---:|']
    for c in cells:
        if c['deadline_s'] in [0.5,1,2] and c['rate_mbps'] in [1,3,10,30]:
            lines.append('| %g | %g | %s | %.4f | %+.4f |'%(c['rate_mbps'],c['deadline_s'],', '.join(c['best_fixed_modes']),c['best_fixed_F1'],c['best_offload_minus_best_box']))
    lines+=['','## Completion time at 3 Mbps / 2 s (reported independently)','',
            '| Mode | Delivered F1 | On-time fraction | Completion p50 / p95 s |','|---|---:|---:|---:|']
    ref=next(c for c in cells if c['rate_mbps']==3 and c['deadline_s']==2)
    for m in VISIBLE:
        r=ref['metrics'][m];lines.append('| %s | %.4f | %.3f | %.3f / %.3f |'%(m,r['delivered_F1'],r['on_time_fraction'],r['completion_p50_s'],r['completion_p95_s']))
    lines+=['','## Exploratory task-gain region','',
            'Descriptive filter: gain >=0.02 delivered F1 over the strongest box controls and positive in >=2/3 groups. No blind-test or publication claim.','',
            '| Mbps | Deadline s | Best offload | F1 gain | Positive group fraction |','|---:|---:|---|---:|---:|']
    for c in useful:lines.append('| %g | %g | %s | %+.4f | %.3f |'%(c['rate_mbps'],c['deadline_s'],', '.join(c['best_offload_modes']),c['best_offload_minus_best_box'],c['group_fraction_offload_strictly_better']))
    lines+=['','![Delivered F1](delivered_F1.png)','',
            '**Decision:** A useful latency/quality frontier is not by itself a task-utility advantage. '
            'Offloading has task-gain cells only under exploratory tighter deadlines in this grid. '
            'Do not open correction for the original objective merely because JPEG is faster. No correction, mobility or RL is executed here.', '',
            'Limits:',*['- '+s for s in result['limits']], '',
            'Reproduce: `python swarmbridge/experiments/active_recovery/delivered_objective.py REPO`. Full per-mode completion times and delivered scores: per_mode.csv; all winners: summary.json.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(original_gain_cells=len(original_gains),original_max_gap=result['original_deadline_max_gap'],useful_cells=result['descriptive_useful_cells']),indent=2))


if __name__=='__main__':main()
