"""Real-data frozen-model Gate A; no training, no mobility, no hidden GT input."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import core as C


def array(records):
    return np.array(records,float).reshape(-1,6)


def score(data, records, rows):
    """COCO AP on three classes with toolkit-style spatial ignore prefilter."""
    dataset=dict(images=[],annotations=[],categories=[dict(id=c,name=n) for c,n in zip(C.VIS_CLASSES,['car','bus','truck'])],info={})
    detections=[]; per_image={}
    for image_id,r in enumerate(rows):
        name=r['name'];w,h=r['size'];p=array(records[name]['records'])
        ann=np.loadtxt(data/'annotations'/(Path(name).stem+'.txt'),delimiter=',',ndmin=2)
        integ=C.ignored_integral(ann,w,h)
        aa=ann.copy();aa[:,2:4]+=aa[:,:2]
        ann=ann[C.outside_ignored(aa[:,:4],integ)]
        p=p[C.outside_ignored(p[:,:4],integ)]
        gt=[];ig=[]
        dataset['images'].append(dict(id=image_id,width=w,height=h))
        for a in ann:
            cls=int(a[5])
            if cls not in C.VIS_CLASSES:continue
            x,y,bw,bh=a[:4];ignored=int(a[4]==0)
            dataset['annotations'].append(dict(id=len(dataset['annotations'])+1,image_id=image_id,category_id=cls,bbox=[x,y,bw,bh],area=bw*bh,iscrowd=ignored))
            (ig if ignored else gt).append([x,y,x+bw,y+bh,cls])
        for b in p:
            x,y,xe,ye,conf,cls=b
            detections.append(dict(image_id=image_id,category_id=int(cls),bbox=[x,y,xe-x,ye-y],score=conf))
        per_image[name]=C.image_f1(gt,p.tolist(),ig)
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=dataset;coco.createIndex()
        if not detections: raise RuntimeError('No detections: check model/class mapping before interpreting gate')
        dt=coco.loadRes(detections);ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,100,500]
        ev.evaluate();ev.accumulate()
    precision=ev.eval['precision'][:,:, :,0,2]
    ap=lambda x:float(x[x>=0].mean())
    return dict(AP50=ap(precision[0]),AP50_95=ap(precision),mean_image_F1=float(np.mean([r['f1'] for r in per_image.values()])),
                per_class_AP50={str(c):ap(precision[0,:,k]) for k,c in enumerate(ev.params.catIds)},
                per_image=per_image,evaluator='pycocotools on 3 classes, maxDets500, VisDrone MATLAB-style ignored-region prefilter; not official 10-class score')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo');ap.add_argument('--cpu-samples',type=int,default=8)
    args=ap.parse_args();repo=Path(args.repo);out=repo/'runs/active_recovery/gate_a';out.mkdir(parents=True,exist_ok=True)
    data=repo/'data/visdrone/VisDrone2019-DET-val'
    names=sorted((data/'images').glob('*.jpg'));assert len(names)==548
    groups=sorted({p.stem.split('_')[0] for p in names},key=lambda g:__import__('hashlib').sha256(('20261106:'+g).encode()).hexdigest())
    ca=round(.2*len(groups));de=round(.4*len(groups))
    sets=dict(calibration=set(groups[:ca]),dev=set(groups[ca:ca+de]),holdout=set(groups[ca+de:]))
    manifest=[]
    for p in names:
        with Image.open(p) as im:w,h=im.size
        group=p.stem.split('_')[0];split=next(k for k,v in sets.items() if group in v)
        manifest.append(dict(name=p.name,group=group,split=split,size=[w,h],image_sha256=C.sha(p),annotation_sha256=C.sha(data/'annotations'/(p.stem+'.txt'))))
    C.save(out/'split.json',dict(seed=20261106,grouping='filename prefix before first underscore; conservative sequence proxy, not verified flight independence',groups={k:sorted(v) for k,v in sets.items()},images=manifest))
    rows=[r for r in manifest if r['split']=='dev']
    config=dict(split_sha256=C.sha(out/'split.json'),inference=dict(imgsz=1280,conf=.01,iou=.7,max_det=500,classes=[2,5,7],batch=1,half=False),
                cpu_threads=4,cpu_samples=args.cpu_samples,onboard_budget='unknown; workstation CPU and GPU are reference scenarios, not measured UAV hardware',
                phases=['nano_boxes','large_boxes','large_original','large_jpeg10','large_jpeg70'],holdout_evaluated=False)
    C.save(out/'config.json',config)
    print('LOCKED_SPLIT', {s:sum(r['split']==s for r in manifest) for s in sets},'groups',len(groups),flush=True)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats();job_start=time.monotonic()
    opts=config['inference'].copy();opts.pop('batch');opts.update(verbose=False,device=0)
    results={};costs={}
    for model_name,modes in [('yolov8n.pt',['nano_boxes']),('yolov8l.pt',['large_boxes','large_original','large_jpeg10','large_jpeg70'])]:
        weight=repo/'external/active_recovery'/model_name
        model=YOLO(str(weight))
        first=np.array(Image.open(data/'images'/rows[0]['name']).convert('RGB'))[:,:,::-1].copy()
        for _ in range(3):model.predict(first,**opts)
        for mode in modes:
            path=out/(mode+'.json')
            if path.exists():
                old=json.loads(path.read_text())
                assert old['config_sha256']==C.sha(out/'config.json') and old['weight_sha256']==C.sha(weight)
                results[mode]=old['images'];print('SKIP',mode,flush=True);continue
            records={}
            for n,r in enumerate(rows):
                original=Image.open(data/'images'/r['name']).convert('RGB');w,h=original.size
                scale=min(1,1280/max(w,h));canvas=original.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
                t0=time.perf_counter();payload_bytes=(data/'images'/r['name']).stat().st_size if mode=='large_original' else 0
                if 'jpeg' in mode:
                    buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=10 if mode.endswith('10') else 70,subsampling=2)
                    packet=buf.getvalue();payload_bytes=len(packet)
                    canvas=Image.open(io.BytesIO(packet)).convert('RGB')
                codec_ms=1000*(time.perf_counter()-t0)
                pixels=np.array(canvas)[:,:,::-1].copy()
                torch.cuda.synchronize();t0=time.perf_counter()
                pred=model.predict(pixels,**opts)[0]
                torch.cuda.synchronize();latency=1000*(time.perf_counter()-t0)
                b=pred.boxes.data.detach().cpu().numpy()
                scale_x,scale_y=canvas.width/w,canvas.height/h
                detected=[[float(x1/scale_x),float(y1/scale_y),float(x2/scale_x),float(y2/scale_y),float(p),C.COCO_TO_VIS[int(c)]] for x1,y1,x2,y2,p,c in b]
                if mode.endswith('boxes'):
                    packet=C.encode_boxes(detected,w,h,n);payload_bytes=len(packet)
                    dst=out/'packets'/mode/(Path(r['name']).stem+'.bin');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(packet)
                    detected=C.decode_boxes(packet)
                records[r['name']]=dict(records=detected,payload_bytes=payload_bytes,codec_ms=codec_ms,processing_ms=latency)
                if n%25==0:print('INFER',mode,n,len(rows),flush=True)
            C.save(path,dict(config_sha256=C.sha(out/'config.json'),weight_sha256=C.sha(weight),images=records))
            results[mode]=records
        del model;torch.cuda.empty_cache()
    # CPU latency of feasible hard-output competitors, independent of GPU inference.
    for model_name in ['yolov8n.pt','yolov8l.pt']:
        cpu_path=out/(model_name+'.cpu.json')
        if cpu_path.exists():costs[model_name]=json.loads(cpu_path.read_text());continue
        model=YOLO(str(repo/'external/active_recovery'/model_name));cpu_opts=dict(opts,device='cpu')
        for _ in range(2):model.predict(first,**cpu_opts)
        samples=[]
        for r in rows[:args.cpu_samples]:
            im=Image.open(data/'images'/r['name']).convert('RGB');w,h=im.size;s=min(1,1280/max(w,h))
            pix=np.array(im.resize((round(w*s),round(h*s)),Image.Resampling.LANCZOS))[:,:,::-1].copy()
            t0=time.perf_counter();model.predict(pix,**cpu_opts);samples.append(1000*(time.perf_counter()-t0))
        costs[model_name]=dict(samples_ms=samples,p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),threads=4,device='workstation WSL CPU reference, not onboard measurement')
        C.save(cpu_path,costs[model_name]);print('CPU_BENCH',model_name,costs[model_name]['p95_ms'],flush=True)
        del model
    scores={}
    for mode,pred in results.items():
        scores[mode]=score(data,pred,rows)
        scores[mode]['payload_bits_mean']=float(np.mean([v['payload_bytes']*8 for v in pred.values()]))
        scores[mode]['processing_ms_p50']=float(np.median([v['processing_ms'] for v in pred.values()]))
        scores[mode]['codec_ms_p50']=float(np.median([v['codec_ms'] for v in pred.values()]))
        print('SCORE',mode,scores[mode]['AP50'],scores[mode]['mean_image_F1'],scores[mode]['payload_bits_mean'],flush=True)
    box=scores['large_boxes'];full=scores['large_original']
    C.save(out/'summary.json',dict(scores=scores,cpu_reference=costs,dev_images=len(rows),dev_groups=len(sets['dev']),
             holdout_evaluated=False,config_sha256=C.sha(out/'config.json'),prediction_hashes={m:C.sha(out/(m+'.json')) for m in results},
             gate=dict(onboard_compute_budget_known=False,passed=False,reason='Gate A cannot pass without independently justified onboard compute budget; same-model hard-output comparison is explicit.',
                       large_box_minus_raw_F1=box['mean_image_F1']-full['mean_image_F1']),
             resources=dict(job_wall_seconds=time.monotonic()-job_start,peak_vram_bytes=torch.cuda.max_memory_allocated()),
             limits=['Single frozen COCO-pretrained pair on aerial domain; no detector training.', 'JPEG/box Gate A only; no generative recovery, correction, mobility or RL claim.', 'COCOeval subset, not official ten-class VisDrone leaderboard result.', 'Filename-prefix groups may not guarantee independent flights.', 'No UAV hardware or power measurement; CPU/GPU latency references only.']))
    print('GATE_A_DONE',flush=True)


if __name__=='__main__':main()
