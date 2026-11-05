"""Export matched receiver modes, validation references and real payload profiles.

This phase never selects checkpoints with dev labels. Frozen legacy references
on validation support CPU confidence calibration, not additional training.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "uav3d_comm"))
sys.path.insert(0, str(HERE.parent / "uav3d_comm/proto"))
import core as C
import receiver as R
import tilecodec as TC
import boxcodec as BC
import box_proto as BP
from soft_fusion import fuse as soft_fuse
from fusion import load_variant
from ft_tools import setup, run_decode


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("out")
    args = ap.parse_args()
    repo, root = Path(args.repo), Path(args.out)
    out, prep = root / "inference", root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    split = json.loads((prep / "split.json").read_text())
    result = json.loads((root / "matched/result.json").read_text())
    if (out / "inference.json").exists():
        old = json.loads((out / "inference.json").read_text())
        assert old["checkpoint_hashes"] == result["checkpoint_hashes"]
        print("SKIP inference", flush=True); return
    data = R.ReceiverData(prep)
    cfg = "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/lowerbound/default.yaml"
    base = str(repo / "external/uav3d_ckpt/lowerbound.pth")
    infos = str(repo / "runs/uav3d_comm/lowerbound/infos_all.pkl")
    init = str(repo / "runs/uav3d_comm/fusion/ego_s2/decoder.pt")
    manifest, timings = {}, {}
    t_start = time.monotonic()
    payloads = {}
    for arm in ["box", "joint"]:
        dataset, model, ds = setup(cfg, base, infos, "ego", init)
        model.encoders["camera"]["backbone"].cpu(); model.encoders["camera"]["neck"].cpu()
        R.expand_input(ds)
        checkpoint = torch.load(str(root / "matched" / (arm + "_best.pt")), map_location="cuda")
        R.load_state(model, ds, checkpoint["weights"])
        model.heads["object"].test_cfg["post_max_size"] = 500
        model.eval()
        original = list(dataset.data_infos)
        order = {i["token"]: i for i in original}
        configs = [("none",0,False,0), ("m1",0,True,0), ("m1_age1",0,True,1)]
        if arm == "joint": configs += [("m2",16,True,0),("m3",64,True,0),("m2_age1",16,True,1)]
        with torch.no_grad():
            # Warm up without evaluating labels or optimizer updates.
            tok = split["tokens"]["val"][0]
            x, metas, _ = data.build(dict(token=tok,helpers={},k=0,rank="obj"), arm == "joint")
            for _ in range(3): run_decode(model, ds, x, metas)
            torch.cuda.synchronize()
            for group in ["val", "dev"]:
                tokens = [i["token"] for i in original if i["token"] in set(split["tokens"][group])]
                for mode, count, use_helpers, age in configs:
                    if group == "val" and age: continue
                    preds, prof = [], []
                    for n,tok in enumerate(tokens):
                        seq = next(seq for seq in split["frames"].values() if tok in seq)
                        ix = seq.index(tok)
                        helpers = {str(j):dict(token=seq[ix-age],age=age) for j in range(1,5)} if use_helpers and ix >= age else {}
                        ctx = dict(token=tok, helpers=helpers, k=count, rank="obj")
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        x, metas, bits = data.build(ctx, arm == "joint")
                        torch.cuda.synchronize(); t1 = time.perf_counter()
                        pred = run_decode(model, ds, x, metas)
                        torch.cuda.synchronize(); t2 = time.perf_counter()
                        preds.append(pred)
                        prof.append([1000*(t1-t0),1000*(t2-t1)])
                        if group == "dev": payloads.setdefault(arm+"_"+mode,[]).append(bits)
                    dataset.data_infos = [order[t] for t in tokens]
                    dest = out / group / (arm + "_" + mode)
                    dataset.format_results(preds, jsonfile_prefix=str(dest))
                    p = dest / "results_nusc.json"
                    manifest[group+"/"+arm+"_"+mode] = str(p)
                    z = np.array(prof)
                    timings[group+"/"+arm+"_"+mode] = dict(
                        input_decode_scatter_and_transfer_ms=dict(p50=float(np.percentile(z[:,0],50)),p95=float(np.percentile(z[:,0],95))),
                        receiver_decode_ms=dict(p50=float(np.percentile(z[:,1],50)),p95=float(np.percentile(z[:,1],95))))
                    print("INFER",group,arm,mode,"frames",len(preds),flush=True)
                    dataset.data_infos = original
        del model, ds, checkpoint
        torch.cuda.empty_cache()
    # Frozen legacy sparse receiver on VALIDATION only, for score calibration.
    dataset, model, ds = setup(cfg, base, infos, "feat", str(repo / "runs/uav3d_comm/fusion/feat_s1/decoder.pt"))
    model.heads["object"].test_cfg["post_max_size"] = 500
    original = list(dataset.data_infos)
    tokens = [i["token"] for i in original if i["token"] in set(split["tokens"]["val"])]
    with torch.no_grad():
        for count in [16,64]:
            preds=[]
            for tok in tokens:
                ego, metas = data.ego(tok)
                num=ego.cuda().float().clone(); den=torch.ones_like(num[:1])
                for j in range(1,5):
                    packet=TC.prefix(data.packet(tok,str(j),"obj","tile"),count)
                    _,idx,tl=TC.decode(packet); f,m=TC.scatter(idx,tl)
                    num+=torch.from_numpy(f).cuda();den+=torch.from_numpy(m[None]).cuda()
                preds.append(run_decode(model,ds,num/den,metas))
            dataset.data_infos = [i for i in original if i["token"] in set(tokens)]
            dest=out / "val" / ("legacy_tiles_%d" % count)
            dataset.format_results(preds,jsonfile_prefix=str(dest))
            manifest["val/legacy_tiles_%d" % count] = str(dest/"results_nusc.json")
    # KF-PV baseline made from exactly the same decoded packets as receiver inputs.
    A0=load_variant(prep / "agents/agent0/results_nusc.json")
    for group in ["val","dev"]:
        pred={}
        for tok in split["tokens"][group]:
            own=A0[tok]; sets=[(0,own["xy"],own["score"],BP.records(own,tok))]
            for j in range(1,5):
                _,b=BC.decode(data.packet(tok,str(j),"obj","box"))
                sets.append((j,b["xy"],b["score"],BP.records(b,tok)))
            b=soft_fuse(sets,"noisyor",return_records=True)
            b.sort(key=lambda x:-x["detection_score"])
            pred[tok]=[dict(x,sample_token=tok) for x in b[:500]]
        p=out/group/"kf_baseline/results_nusc.json"
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(json.dumps(dict(meta=BP.META,results=pred)).encode())
        manifest[group+"/kf_baseline"]=str(p)
    C.save_json(out / "inference.json",dict(checkpoint_hashes=result["checkpoint_hashes"],predictions=manifest,
                timings=timings,payload_bits={k:dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95))) for k,v in payloads.items()},
                gpu_job_seconds=time.monotonic()-t_start,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                hashes={k:C.sha(p) for k,p in manifest.items()}))
    C.save_json(root/"tracking_manifest.json",{k.split("/",1)[1]:v for k,v in manifest.items() if k.startswith("dev/")})
    print("INFERENCE_DONE", flush=True)


if __name__ == "__main__": main()
