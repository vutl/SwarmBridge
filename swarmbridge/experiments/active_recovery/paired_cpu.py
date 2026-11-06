"""Measure all dev images on CPU, with separate codec/serialization stages.

Preserves prior Gate A results; no detector training, GPU or holdout scoring.
Raw per-image repeat records permit resuming an interrupted measurement.
"""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

import core as C
from run_gate_a import score


def timed(fn):
    t=time.perf_counter();value=fn()
    return value,1000*(time.perf_counter()-t)


def decode(buf):
    return Image.open(BytesIO(buf)).convert('RGB')


def resize(im):
    s=min(1,1280/max(im.size))
    return im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.LANCZOS)


def bgr(im):
    return np.array(im)[:,:,::-1].copy()


def encode(im,q):
    f=BytesIO();im.save(f,format='JPEG',quality=q,subsampling=2)
    return f.getvalue()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo');ap.add_argument('--repeats',type=int,default=2)
    args=ap.parse_args();repo=Path(args.repo);old=repo/'runs/active_recovery/gate_a';out=repo/'runs/active_recovery/paired_cpu'
    data=repo/'data/visdrone/VisDrone2019-DET-val';out.mkdir(parents=True,exist_ok=True)
    split=json.loads((old/'split.json').read_text());rows=[r for r in split['images'] if r['split']=='dev']
    assert len(rows)==216 and len({r['group'] for r in rows})==30
    assert not torch.cuda.is_available(), 'CPU-only launcher must hide CUDA devices'
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    old_score=json.loads((old/'summary.json').read_text())
    opts=dict(imgsz=1280,conf=.01,iou=.7,max_det=500,classes=[2,5,7],batch=1,device='cpu',quantize=None,verbose=False)
    weights={m:C.sha(repo/'external/active_recovery'/m) for m in ['yolov8n.pt','yolov8l.pt']}
    config=dict(date='2026-11-06',split_sha256=C.sha(old/'split.json'),weights=weights,repeats=args.repeats,
                threads=4,interop_threads=1,options=opts,seed=20261106,
                sample_count=len(rows),group_count=30,starting_point='original source JPEG already stored at UAV',
                exclude='disk I/O, model loading, warmups, sensing/camera JPEG encoding, radio and power',
                old_GPU_predictions_preserved=True,holdout_scored=False,
                environment=dict(platform=platform.platform(),torch=torch.__version__,cpu=next((line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    C.save(out/'config.json',config)
    sig=C.sha(out/'config.json');start=time.monotonic()
    order=np.random.default_rng(20261106).permutation(len(rows)).tolist()
    for r in rows:
        assert C.sha(data/'images'/r['name'])==r['image_sha256']
    print('PAIRED_START',len(rows),'images',30,'groups',args.repeats,'repeats',flush=True)
    # Codec stages use memory-resident source bytes; no cold filesystem read in timing.
    for repeat in range(args.repeats):
        indices=order if repeat%2==0 else list(reversed(order))
        for n,ix in enumerate(indices):
            r=rows[ix];p=out/'raw/codec'/str(repeat)/(Path(r['name']).stem+'.json')
            if p.exists():assert json.loads(p.read_text())['config_sha256']==sig;continue
            buf=(data/'images'/r['name']).read_bytes()
            im,source_decode=timed(lambda:decode(buf))
            canvas,resize_ms=timed(lambda:resize(im))
            _,source_bgr=timed(lambda:bgr(canvas))
            rec=dict(config_sha256=sig,name=r['name'],repeat=repeat,source_decode_ms=source_decode,
                     resize_ms=resize_ms,source_bgr_ms=source_bgr,source_bytes=len(buf),jpeg={})
            for q in [10,70]:
                packet,enc=timed(lambda:encode(canvas,q))
                received,dec=timed(lambda:decode(packet))
                _,conv=timed(lambda:bgr(received))
                expected=old_score['prediction_hashes']['large_jpeg'+str(q)]
                rec['jpeg'][str(q)]=dict(bytes=len(packet),encode_ms=enc,decode_ms=dec,bgr_ms=conv)
            C.save(p,rec)
        print('CODEC_DONE',repeat,flush=True)
    # Alternate model order and reverse image order in repeat two to expose drift.
    for repeat in range(args.repeats):
        indices=order if repeat%2==0 else list(reversed(order))
        for mode,filename in ([('nano_boxes','yolov8n.pt'),('large_boxes','yolov8l.pt')] if repeat%2==0 else [('large_boxes','yolov8l.pt'),('nano_boxes','yolov8n.pt')]):
            model=YOLO(str(repo/'external/active_recovery'/filename))
            for r in rows[:3]:model.predict(bgr(resize(decode((data/'images'/r['name']).read_bytes()))),**opts)
            for n,ix in enumerate(indices):
                r=rows[ix];p=out/'raw'/mode/str(repeat)/(Path(r['name']).stem+'.json')
                if p.exists():assert json.loads(p.read_text())['config_sha256']==sig;continue
                im=decode((data/'images'/r['name']).read_bytes());w,h=im.size
                canvas=resize(im);pixels=bgr(canvas)
                pred,elapsed=timed(lambda:model.predict(pixels,**opts)[0])
                boxes=pred.boxes.data.numpy();sx=canvas.width/w;sy=canvas.height/h
                det=[[float(x1/sx),float(y1/sy),float(x2/sx),float(y2/sy),float(conf),C.COCO_TO_VIS[int(cls)]] for x1,y1,x2,y2,conf,cls in boxes]
                packet,enc=timed(lambda:C.encode_boxes(det,w,h,ix))
                reconstructed,dec=timed(lambda:C.decode_boxes(packet))
                C.save(p,dict(config_sha256=sig,name=r['name'],repeat=repeat,predict_ms=elapsed,encode_box_ms=enc,decode_box_ms=dec,
                             bytes=len(packet),packet_sha256=__import__('hashlib').sha256(packet).hexdigest(),records=reconstructed))
                if n%25==0:print('PAIRED',mode,'repeat',repeat,n,len(rows),flush=True)
            del model
    # Scores are computed from CPU predictions, not copied from GPU labels/output.
    cpu_scores={};quality={};models={}
    for mode in ['nano_boxes','large_boxes']:
        models[mode]={}
        for repeat in range(args.repeats):
            records={r['name']:json.loads((out/'raw'/mode/str(repeat)/(Path(r['name']).stem+'.json')).read_text()) for r in rows}
            q=score(data,records,rows);quality[mode+'_repeat'+str(repeat)]=q
            for name,rec in records.items():models[mode].setdefault(name,[]).append(rec)
        cpu_scores[mode]=quality[mode+'_repeat0']
        print('PAIRED_SCORE',mode,cpu_scores[mode]['mean_image_F1'],flush=True)
    codec={r['name']:[json.loads((out/'raw/codec'/str(rep)/(Path(r['name']).stem+'.json')).read_text()) for rep in range(args.repeats)] for r in rows}
    C.save(out/'measurements.json',dict(config_sha256=sig,codec=codec,models=models))
    C.save(out/'quality.json',quality)
    C.save(out/'execution.json',dict(seconds=time.monotonic()-start,inference_calls=2*len(rows)*args.repeats,
                cpu_only=True,gpu_seconds=0,training_steps=0,dev_images=len(rows),groups=30,holdout_scored=False))
    print('PAIRED_DONE',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
