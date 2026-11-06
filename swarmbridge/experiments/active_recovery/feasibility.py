"""CPU-only conditional offloading analysis of existing predictions and timings.

No detector imports, CUDA, new images or channels. A reference scenario is not
an onboard measurement. Ground-truth F1 is used only for offline evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import core as C

MODES = ['nano_boxes','large_boxes','large_original','large_jpeg10','large_jpeg70']


def times(bits, onboard_s, remote_s, rate_mbps, overhead_fraction=.1, common_s=.02, offload_extra_s=0):
    tx=np.asarray(bits)*(1+overhead_fraction)/(rate_mbps*1e6)
    return np.asarray(onboard_s)+np.asarray(remote_s)+tx+common_s+offload_extra_s,tx


def service(f1, bits, onboard_s, remote_s, rate_mbps, deadline_s,
            compute_budget_s=np.inf, energy_budget_radio_s=np.inf,
            compute_to_radio_power=1., idle_to_radio_power=0., platform_horizon_s=None, **kwargs):
    elapsed,tx=times(bits,onboard_s,remote_s,rate_mbps,**kwargs)
    # Pcompute and Pradio are incremental powers above a common platform baseline.
    # This is energy until result completion, excluding station energy.
    horizon=elapsed if platform_horizon_s is None else platform_horizon_s
    energy=compute_to_radio_power*np.asarray(onboard_s)+tx+idle_to_radio_power*horizon
    feasible=(np.asarray(onboard_s)<=compute_budget_s)&(elapsed<=deadline_s)&(energy<=energy_budget_radio_s)
    return dict(mean_F1=float(np.mean(np.asarray(f1)*feasible)),delivery_fraction=float(np.mean(feasible)),
                latency_p50_s=float(np.median(elapsed)),latency_p95_s=float(np.percentile(elapsed,95)),
                mean_normalized_energy_radio_s=float(np.mean(energy)),mean_compute_s=float(np.mean(onboard_s)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo');args=ap.parse_args()
    start=time.monotonic();repo=Path(args.repo);src=repo/'runs/active_recovery/gate_a';out=repo/'runs/active_recovery/offloading';out.mkdir(parents=True,exist_ok=True)
    score=json.loads((src/'summary.json').read_text())
    predictions={m:json.loads((src/(m+'.json')).read_text())['images'] for m in MODES}
    for m in MODES:assert C.sha(src/(m+'.json'))==score['prediction_hashes'][m]
    names=list(predictions['large_original'])
    assert all(list(predictions[m])==names for m in MODES)
    cpu={m:json.loads((src/(m+'.cpu.json')).read_text()) for m in ['yolov8n.pt','yolov8l.pt']}
    paired_count=len(cpu['yolov8l.pt']['samples_ms']);assert paired_count==len(cpu['yolov8n.pt']['samples_ms'])
    paired=names[:paired_count]
    config=dict(kind='conditional sensitivity, not an onboard experiment',rate_mbps=[.1,.3,.5,1,1.5,3,10,30,100],
                deadlines_s=[.1,.25,.5,1,2,5,10],original_protocol_deadlines_s=[2,5,10],
                additional_deadlines_are_exploratory=True,compute_budgets_s=[.05,.1,.5,2,None],
                latency_multipliers=[.25,1,4],extra_offload_s=[0,.02,.1],
                payload_overhead_fraction=.1,common_latency_s=.02,
                energy=dict(compute_to_radio_power=[.25,1,4,16],idle_to_radio_power=[0,25,75],normalized_budgets_radio_s=[.1,.5,1,2,5,10,None],platform_horizons=['until_result','fixed_2s_mission'],
                            definition='E_UAV/Pradio = (Pcompute/Pradio)*t_compute + t_tx + (Pbaseline/Pradio)*t_until_result; incremental powers, station excluded',
                            note='Power ratios and budgets are arbitrary sensitivity coordinates, not measured watts or joules.'),
                timestamp='2026-11-06',gpu_or_inference_runs=0)
    C.save(out/'config.json',config)
    def inputs(mode,device='cpu_reference',multiplier=1.,subset=names,paired_cpu=False):
        p=predictions[mode]
        bits=np.array([p[n]['payload_bytes']*8 for n in subset],float)
        f1=np.array([score['scores'][mode]['per_image'][n]['f1'] for n in subset])
        if mode.endswith('boxes'):
            model='yolov8n.pt' if mode=='nano_boxes' else 'yolov8l.pt'
            if device=='gpu_reference':compute=np.array([p[n]['processing_ms']/1000 for n in subset])
            elif paired_cpu:
                lookup=dict(zip(paired,cpu[model]['samples_ms']));compute=np.array([lookup[n]/1000 for n in subset])
            else:compute=np.full(len(subset),cpu[model]['p50_ms']/1000)
            compute*=multiplier;remote=np.zeros(len(subset))
        else:
            # JPEG timing combines encoder+decoder. Charging both to UAV is a
            # conservative allocation; no original-file stat I/O is charged.
            compute=np.array([p[n]['codec_ms']/1000 if 'jpeg' in mode else 0 for n in subset])*multiplier
            remote=np.array([p[n]['processing_ms']/1000 for n in subset])
        return f1,bits,compute,remote
    # Continuous break-even rates, using each image's real bits and remote timing.
    boundaries={}
    _,bL,cL,_=inputs('large_boxes')
    for mode in ['large_original','large_jpeg70']:
        _,b,c,r=inputs(mode);slack=cL-c-r
        threshold=(b-bL)/slack/1e6
        valid=slack>0
        boundaries[mode]=dict(ideal_per_image_rate_mbps_p50=float(np.median(threshold[valid])),
                             ideal_per_image_rate_mbps_p90=float(np.percentile(threshold[valid],90)),
                             payload_overhead10pct_rate_mbps_p50=float(np.median(threshold[valid])*1.1),
                             finite_fraction=float(np.mean(valid)))
    # Mean-payload/median-processing approximation corresponding to the quoted 0.97 Mbps.
    q=score['scores']['large_jpeg70'];b=score['scores']['large_boxes'];tL=cpu['yolov8l.pt']['p50_ms']/1000
    tq=(np.median([p['codec_ms'] for p in predictions['large_jpeg70'].values()])+q['processing_ms_p50'])/1000
    approximate=(q['payload_bits_mean']-b['payload_bits_mean'])/(tL-tq)/1e6
    fields=['sample_set','onboard_device','latency_multiplier','compute_budget_s','extra_offload_s','rate_mbps','deadline_s','mode','mean_F1','delivery_fraction','latency_p50_s','latency_p95_s','mean_normalized_energy_radio_s','mean_compute_s']
    with (out/'timing_sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for label,subset,paired_cpu in [('dev216_reference_cpu_median',names,False),('paired8_cpu_samples_one_group',paired,True)]:
            devices=['cpu_reference','gpu_reference'] if not paired_cpu else ['cpu_reference']
            for device in devices:
                for mult in config['latency_multipliers']:
                    for budget in config['compute_budgets_s']:
                        for extra in config['extra_offload_s']:
                            for rate in config['rate_mbps']:
                                for deadline in config['deadlines_s']:
                                    for mode in MODES:
                                        metrics=service(*inputs(mode,device,mult,subset,paired_cpu),rate,deadline,
                                                compute_budget_s=np.inf if budget is None else budget,offload_extra_s=extra if not mode.endswith('boxes') else 0)
                                        writer.writerow(dict(sample_set=label,onboard_device=device,latency_multiplier=mult,compute_budget_s=budget,extra_offload_s=extra,rate_mbps=rate,deadline_s=deadline,mode=mode,**metrics))
    examples=[]
    for label,device,rate,deadline,budget in [
            ('CPU reference, original deadlines','cpu_reference',3,2,np.inf),
            ('CPU reference, exploratory 0.5s','cpu_reference',3,.5,np.inf),
            ('CPU reference, exploratory 0.5s, faster link','cpu_reference',10,.5,np.inf),
            ('CPU reference, 100ms compute allowance','cpu_reference',3,2,.1),
            ('GPU onboard reference','gpu_reference',3,2,np.inf)]:
        rows={m:service(*inputs(m,device),rate,deadline,compute_budget_s=budget) for m in MODES}
        examples.append(dict(label=label,rate_mbps=rate,deadline_s=deadline,compute_budget_s=None if np.isinf(budget) else budget,results=rows))
    # Energy analysis is intentionally separated from the timing sweep.
    energy_rows=[]
    for horizon in config['energy']['platform_horizons']:
        for k in config['energy']['compute_to_radio_power']:
            for idle in config['energy']['idle_to_radio_power']:
                for budget in config['energy']['normalized_budgets_radio_s']:
                    for rate in [1,3,10]:
                        for mode in MODES:
                            r=service(*inputs(mode),rate,2,compute_to_radio_power=k,idle_to_radio_power=idle,energy_budget_radio_s=np.inf if budget is None else budget,platform_horizon_s=2 if horizon=='fixed_2s_mission' else None)
                            energy_rows.append(dict(platform_horizon=horizon,compute_to_radio_power=k,idle_to_radio_power=idle,energy_budget_radio_s=budget,rate_mbps=rate,deadline_s=2,mode=mode,**r))
    C.save(out/'energy_sweep.json',energy_rows)
    ratio_at3={}
    for mode in ['large_original','large_jpeg70']:
        _,bm,cm,rm=inputs(mode)
        ratio_at3[mode]=float(np.mean((bm-bL)*1.1/(3e6))/np.mean(cL-cm))
    paired_example={m:service(*inputs(m,subset=paired,paired_cpu=True),3,2) for m in MODES}
    # Standalone figure: fixed-mode diagnostics, never GT-guided online actions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,device,budget,title in zip(axs,['cpu_reference','cpu_reference','gpu_reference'],[np.inf,.1,np.inf],['CPU reference, no compute cap','CPU reference, 100 ms compute cap','GPU onboard reference, no cap']):
        a=np.zeros((len(config['deadlines_s']),len(config['rate_mbps'])))
        for i,d in enumerate(config['deadlines_s']):
            for j,rate in enumerate(config['rate_mbps']):
                box=max(service(*inputs(m,device),rate,d,compute_budget_s=budget)['mean_F1'] for m in MODES[:2])
                off=service(*inputs('large_original',device),rate,d,compute_budget_s=budget)['mean_F1']
                a[i,j]=off-box
        im=ax.imshow(a,origin='lower',aspect='auto',vmin=-.6,vmax=.6,cmap='RdBu')
        ax.set(xticks=range(len(config['rate_mbps'])),xticklabels=config['rate_mbps'],yticks=range(len(config['deadlines_s'])),yticklabels=config['deadlines_s'],xlabel='Effective goodput (Mbps)',ylabel='Deadline (s)',title=title)
    fig.colorbar(im,ax=axs,label='Delivered F1 difference\n(original offload minus best box)')
    fig.suptitle('Conditional workstation sensitivity; 10% byte overhead + 20 ms latency; no UAV hardware validation')
    fig.savefig(out/'feasibility.png',dpi=160);plt.close(fig)
    report=dict(status='offloading_has_a_conditional_feasible_region; real_onboard_feasibility_unresolved',
                mean_payload_median_timing_ideal_Q70_break_even_mbps=approximate,boundaries=boundaries,
                energy_compute_to_radio_power_threshold_at3Mbps_no_platform_baseline=ratio_at3,examples=examples,paired8_at3Mbps_2s=paired_example,
                paired_cpu_images=paired,paired_cpu_groups=sorted({n.split('_')[0] for n in paired}),
                source_hashes={p.name:C.sha(p) for p in [src/'summary.json',src/'yolov8l.pt.cpu.json',src/'yolov8n.pt.cpu.json']},
                analysis_seconds=time.monotonic()-start,inference_runs=0,gpu_seconds=0,
                limits=['No new detector, hardware, channel or energy measurement.',
                        'Dev216 CPU scenarios assign median from eight CPU samples; not 216 paired CPU measurements.',
                        'Paired8 contains only one filename group; cannot support flight/generalization statistics.',
                        'JPEG encode+decode charged to UAV together, conservative; source-file stat I/O excluded.',
                        'Resize, box serialization timing, response, queue, beam/pilot and protocol overhead not independently measured; nuisance overhead is assumed.',
                        'Effective goodput is not PHY bitrate; no mobility or channel policy simulated.',
                        'Incremental power ratios are sensitivity parameters; no measured joules or full-system energy claim.',
                        'Energy is required energy of a completed attempt; failed/deadline-truncated attempt consumption is not modeled. Platform duration is separately swept as until-result vs fixed 2s.',
                        'Best fixed diagnostics use dev labels for offline assessment, not online mode selection.',
                        'Exploratory short deadlines and compute caps are not original-protocol or blind test evidence.',
                        'No correction, semantic recovery, mobility advantage or RL necessity established.'])
    C.save(out/'summary.json',report)
    lines=['# Offloading: conditional feasibility from existing results','',
           'No new inference, GPU, training, data or channel trace. Gate A is **inconclusive for deployment**, not proof that offloading is useless.','',
           f'Q70 ideal break-even with local large: **{approximate:.3f} Mbps** (mean bytes / median processing approximation).',
           'Per-image break-even with local large:', '',
           '| Offload | Ideal rate p50/p90 Mbps | With 10% byte overhead, p50 Mbps |','|---|---:|---:|']
    for m,v in boundaries.items():lines.append('| %s | %.3f / %.3f | %.3f |'%(m,v['ideal_per_image_rate_mbps_p50'],v['ideal_per_image_rate_mbps_p90'],v['payload_overhead10pct_rate_mbps_p50']))
    lines+=['','Reference scenarios use 10% byte overhead and 20 ms common latency. Missed deadlines/compute budgets get F1=0, no prior receiver result.','',
            '| Scenario | Mbps | nano delivered F1 | large-box F1 | original offload F1 | Q70 offload F1 |','|---|---:|---:|---:|---:|---:|']
    for e in examples:
        r=e['results'];lines.append('| %s | %g | %.4f | %.4f | %.4f | %.4f |'%(e['label'],e['rate_mbps'],*[r[m]['mean_F1'] for m in ['nano_boxes','large_boxes','large_original','large_jpeg70']]))
    lines+=['','At 3 Mbps and a 2s deadline, CPU-reference median total latency:', '',
            '| Mode | Total p50 / p95 s |','|---|---:|']
    for mode,r in examples[0]['results'].items():lines.append('| %s | %.3f / %.3f |'%(mode,r['latency_p50_s'],r['latency_p95_s']))
    lines+=['','Paired check on the **eight CPU-timed images (one filename group)**, same 3 Mbps / 2s assumptions:', '',
            '| Mode | Delivered mean F1 | Total p50 / p95 s |','|---|---:|---:|']
    for mode,r in paired_example.items():lines.append('| %s | %.4f | %.3f / %.3f |'%(mode,r['mean_F1'],r['latency_p50_s'],r['latency_p95_s']))
    lines+=['','On this paired subset, original-image offload is slower than local large at 3 Mbps, while Q70 is faster with some quality loss. '
            'This illustrates why average payload and eight reference CPU timings do not establish per-image or cross-scene feasibility.']
    lines+=['','At 3 Mbps, no platform baseline, offloading uses less UAV-normalized energy than local large if incremental Pcompute/Pradio exceeds:',
            *['- %s: %.3f'%(m,k) for m,k in ratio_at3.items()], '',
            'These are assumed-power thresholds, not power measurements. Hover/platform energy and energy budgets are separate sensitivity axes in energy_sweep.json.', '',
            '![Conditional feasibility](feasibility.png)', '',
            'Conclusion: CPU-limited and/or tight-deadline scenarios can justify offloading. GPU-capable onboard inference can reverse the result. '
            'Actual onboard compute/power and radio conditions remain unresolved. This justifies retaining offloading as a candidate, not opening correction or RL.', '',
            'Limitations:',*['- '+x for x in report['limits']], '',
            'Reproduce: `python swarmbridge/experiments/active_recovery/feasibility.py REPO` in the active-recovery WSL environment.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(ideal_Q70_mbps=approximate,examples=examples,energy_thresholds=ratio_at3),indent=2))


if __name__=='__main__':main()
