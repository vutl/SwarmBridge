"""Join prediction hashes, detection, official tracking, costs and the locked gate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from core import DEV, save_json, sha
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"uav3d_comm"))
from evaldet import MiniDB, evaluate, load_nusc_results


def main():
    ap=argparse.ArgumentParser();ap.add_argument("repo");ap.add_argument("out");args=ap.parse_args()
    repo,root=Path(args.repo),Path(args.out)
    manifest=json.loads((root/"tracking_manifest.json").read_text())
    db=MiniDB(repo/"data/v1.0-mini")
    rows={}
    for name,path in manifest.items():
        p=Path(str(path).replace("\\","/"))
        if not p.is_absolute():p=repo/p
        digest=sha(p)
        predictions=load_nusc_results(p)
        row=dict(prediction=p.as_posix(),sha256=digest,
                 detection=evaluate(db,predictions,db.tokens(DEV)),per_scene={})
        for group,scenes in [("all",list(DEV))]+[(s,[s]) for s in DEV]:
            td=root/"tracking"/(name+"__"+group)
            if not (td/"metrics_summary.json").exists():raise RuntimeError("Tracking missing: "+str(td))
            inp=json.loads((td/"input.json").read_text())
            assert inp["det_sha256"]==digest and inp["dt"]=="0.2" and inp["scenes"]==scenes
            m=json.loads((td/"metrics_summary.json").read_text())
            values={k:m[k] for k in ["amota","ids","recall","amotp"]}
            values["mAP"]=evaluate(db,predictions,db.tokens(scenes))["mAP"]
            if group=="all":row["tracking"]=values
            else:row["per_scene"][group]=values
        rows[name]=row
    controls=[k for k in ["kf_baseline","box_m1","joint_m1","m1_same_nms"] if k in rows]
    best=max(controls,key=lambda k:rows[k]["tracking"]["amota"])
    candidates={}
    for k in ["joint_m2","joint_m3","calibrated_m2","calibrated_m3"]:
        diff=rows[k]["tracking"]["amota"]-rows[best]["tracking"]["amota"]
        per={s:rows[k]["per_scene"][s]["amota"]-rows[best]["per_scene"][s]["amota"] for s in DEV}
        candidates[k]=dict(delta_amota=diff,per_scene_delta=per,
                ids_delta=rows[k]["tracking"]["ids"]-rows[best]["tracking"]["ids"],
                pass_gate=bool(diff>=.01 and sum(v>0 for v in per.values())>=2))
    # The gate additionally reports stronger postprocessing alternatives rather
    # than silently discarding them when choosing a useful research direction.
    legacy_alts=[k for k in ["calibrated_m2","calibrated_m3"] if k in rows]
    prep=json.loads((root/"prepared/prepared.json").read_text())
    train=json.loads((root/"matched/result.json").read_text())
    infer=json.loads((root/"inference/inference.json").read_text())
    total=prep["seconds"]+train["gpu_job_seconds"]+infer["gpu_job_seconds"]
    payload_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    report=dict(rows=rows,gate=dict(baseline=best,threshold=.01,required_positive_scenes=2,
                    candidates=candidates,open_wireless=any(c["pass_gate"] for c in candidates.values())),
                best_calibrated_control=max(legacy_alts,key=lambda k:rows[k]["tracking"]["amota"]) if legacy_alts else None,
                training=train,timings=infer["timings"],payload_bits=infer["payload_bits"],
                resource=dict(successful_gpu_phase_seconds=total,artifact_bytes=payload_bytes,
                              adaptation_envelope_seconds=2700),
                limits=["One training seed; three previously-used dev scenes.",
                        "Initial decoder had seen the 17 non-dev scenes before the new split.",
                        "Ideal message arrivals at age 0/1; not a wireless result.",
                        "Dense diagnostic is a reference, not a bound or achievable sparse quality."])
    save_json(root/"summary.json",report)
    lines=["# Matched receiver pilot (Codex)","", "All rows use dt=0.2 and cap=500. Development evidence only.","",
           "| Method | mAP | AMOTA | IDS | scene_11 | scene_13 | scene_15 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k,r in rows.items():
        m=r["tracking"]
        lines.append("| %s | %.4f | %.4f | %s | %s |"%(k,r["detection"]["mAP"],m["amota"],m["ids"],
                    " | ".join("%.4f"%r["per_scene"][s]["amota"] for s in DEV)))
    lines += ["", "Gate baseline: **%s**. Wireless gate: **%s**."%(best,report["gate"]["open_wireless"]),
              "",json.dumps(report["gate"],indent=2),"", "Successful GPU phases: %.1f min; paired adaptation: %.1f min."%(total/60,train["gpu_job_seconds"]/60)]
    (root/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report["gate"],indent=2))


if __name__=="__main__":main()
