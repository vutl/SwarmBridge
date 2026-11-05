"""Validation-only Platt calibration of legacy box/feature scores, then fixed NMS.

No temperature, threshold or NMS radius is selected using dev labels. Correctness
labels use the evaluator's causal per-frame greedy 2m matching rule on validation.
This is a postprocessing control, not a new semantic codec.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/"uav3d_comm"))
import core as C
import tilecodec as TC
from evaldet import MiniDB, evaluate, load_nusc_results
from hybrid_merge import tile_of


def logit(s):
    s=np.clip(s,1e-6,1-1e-6)
    return np.log(s/(1-s))


def fit(records, db, tokens):
    xs,ys=[],[]
    for tok in tokens:
        gt=db.gt_xy(tok);taken=np.zeros(len(gt),bool)
        for b in sorted(records.get(tok,[]),key=lambda b:-b["detection_score"]):
            xy=np.array(b["translation"][:2])
            if np.any(np.abs(xy-db.uav0[tok])>=C.HALF):continue
            hit=False
            if len(gt):
                d=np.linalg.norm(gt-xy,axis=1);d[taken]=np.inf;j=int(np.argmin(d))
                hit=d[j]<2.0
                if hit:taken[j]=True
            xs.append(logit(b["detection_score"]));ys.append(float(hit))
    x,y=np.array(xs),np.array(ys)
    def objective(v):
        z=v[0]*x+v[1]
        loss=np.mean(np.logaddexp(0,z)-y*z)+1e-5*(v[0]-1)**2
        dz=expit(z)-y
        return loss,np.array([np.mean(dz*x)+2e-5*(v[0]-1),np.mean(dz)])
    sol=minimize(objective,[1.,0.],jac=True,bounds=[(0,10),(-20,20)],method="L-BFGS-B")
    if not sol.success:raise RuntimeError(str(sol.message))
    return dict(a=float(sol.x[0]),b=float(sol.x[1]),n=len(x),positive_fraction=float(y.mean()),loss=float(sol.fun))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("repo");ap.add_argument("out");args=ap.parse_args()
    repo,root=Path(args.repo),Path(args.out)
    inf=json.loads((root/"inference/inference.json").read_text())
    split=json.loads((root/"prepared/split.json").read_text())
    db=MiniDB(repo/"data/v1.0-mini")
    load=lambda p:json.loads(Path(p).read_text())["results"]
    boxes_val=load(inf["predictions"]["val/kf_baseline"])
    bc=fit(boxes_val,db,split["tokens"]["val"])
    geo=json.loads((root/"prepared/geometry.json").read_text())
    old=repo/"runs/hybrid_comm/m0"
    sent=json.loads((old/"sent_tiles.json").read_text())["sent"]
    boxes_dev=load(inf["predictions"]["dev/kf_baseline"])
    cal,paths={},{}
    for k in [16,64]:
        fv=load(inf["predictions"]["val/legacy_tiles_%d"%k])
        filtered={}
        for tok in split["tokens"]["val"]:
            union=set()
            for j in range(1,5):
                data=(root/"prepared/tiles"/(tok+"_%d_obj.bin"%j)).read_bytes()
                _,idx,_=TC.decode(TC.prefix(data,k));union.update(int(x) for x in idx)
            xy=geo["poses"][tok][:2]
            filtered[tok]=[b for b in fv[tok] if tile_of(b,xy,C.HALF,C.CELL,8,"xy") in union]
        fc=fit(filtered,db,split["tokens"]["val"])
        cal[str(k)]=dict(box=bc,feature=fc)
        fd=load(old/("tiles_obj_%d/results_nusc.json"%k))
        predictions={}
        for tok,s in sent.items():
            union=set().union(*(s["obj"][str(j)][:k] for j in range(1,5)))
            feature=[b for b in fd[tok] if tile_of(b,s["ego_xy"],C.HALF,C.CELL,8,"xy") in union]
            merged=[]
            for recs,c in [(feature,fc),(boxes_dev[tok],bc)]:
                for b in recs:
                    q=dict(b);q["detection_score"]=float(expit(c["a"]*logit(b["detection_score"])+c["b"]))
                    merged.append(q)
            predictions[tok]=C.nms_max(merged)
        p=root/"calibrated"/("m%d"%({16:2,64:3}[k]))/"results_nusc.json"
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(json.dumps(dict(meta={"use_camera":True},results=predictions)).encode())
        paths["calibrated_m%d"%({16:2,64:3}[k])]=str(p)
    C.save_json(root/"calibrated/calibration.json",dict(parameters=cal,validation_scenes=split["scenes"]["val"],
                predictions=paths,hashes={k:C.sha(p) for k,p in paths.items()},
                protocol="Platt, class correctness at 2m, monotone slope >= 0, fixed 2m NMS. No dev fitting."))
    manifest=json.loads((root/"tracking_manifest.json").read_text());manifest.update(paths)
    # New controls not present in the old tracking artifacts.
    reprod=json.loads((root/"reproduction/reproduction.json").read_text())
    for name in ["m1_same_nms","feat_dense_q8"]:manifest[name]=reprod[name]["pred"].replace("\\","/")
    # The old dense diagnostic is tracked separately from q8.
    manifest["feat_dense_float"]=reprod["feat_dense"]["pred"].replace("\\","/")
    C.save_json(root/"tracking_manifest.json",manifest)
    print(json.dumps(cal,indent=2))


if __name__=="__main__":main()
