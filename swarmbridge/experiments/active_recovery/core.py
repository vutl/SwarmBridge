"""Image-evidence pilot: serialized boxes and explicit VisDrone-style metrics."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

VIS_CLASSES = (4, 9, 6)
COCO_TO_VIS = {2: 4, 5: 9, 7: 6}
HEADER = struct.Struct("<4sIHHH")
RECORD = struct.Struct("<B5H")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(p)


def encode_boxes(records, width, height, image_id):
    out = [HEADER.pack(b"SBR1", image_id, width, height, len(records))]
    for b in records:
        xy = np.array(b[:4]) / [width, height, width, height]
        q = np.rint(np.clip(xy, 0, 1) * 65535).astype(int)
        out.append(RECORD.pack(int(b[5]), *q, int(round(np.clip(b[4], 0, 1) * 65535))))
    return b"".join(out)


def decode_boxes(packet):
    magic, image_id, width, height, n = HEADER.unpack_from(packet)
    assert magic == b"SBR1" and len(packet) == HEADER.size + n * RECORD.size
    result = []
    for i in range(n):
        cls, x1, y1, x2, y2, score = RECORD.unpack_from(packet, HEADER.size+i*RECORD.size)
        result.append([x1*width/65535, y1*height/65535, x2*width/65535, y2*height/65535, score/65535, cls])
    return result


def ignored_integral(annotations, width, height):
    """MATLAB toolkit dropObjectsInIgr/createIntImg conventions, including endpoints."""
    rects = annotations[annotations[:, 5] == 0, :4]
    if not len(rects): return None
    mask = np.zeros((height, width), np.int32)
    for rect in rects:
        x,y,w,h = np.maximum(1, rect.astype(int))
        mask[y-1:min(height,y+h), x-1:min(width,x+w)] = 1
    return mask.cumsum(0, dtype=np.int64).cumsum(1)


def outside_ignored(xyxy, integral):
    if integral is None: return np.ones(len(xyxy), bool)
    h,w = integral.shape
    box = np.array(xyxy, dtype=float).reshape(-1,4)
    xywh = box.copy(); xywh[:,2:] -= xywh[:,:2]
    pos = np.maximum(1, np.floor(xywh+0.5).astype(int))
    x=np.clip(pos[:,0],1,w)-1; y=np.clip(pos[:,1],1,h)-1
    xe=np.minimum(w,pos[:,0].clip(1,w)+pos[:,2])-1
    ye=np.minimum(h,pos[:,1].clip(1,h)+pos[:,3])-1
    area=integral[y,x]+integral[ye,xe]-integral[y,xe]-integral[ye,x]
    return area/(pos[:,2]*pos[:,3]) < .5


def overlap(a,b,ioa=False):
    a=np.array(a,float).reshape(-1,4);b=np.array(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(1)
    bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return inter/np.maximum(aa[:,None] if ioa else aa[:,None]+bb[None,:]-inter,1e-12)


def image_f1(gt, pred, ignored=None):
    """Score-order matching at IoU .5 and confidence .25; ignored objects use IoA."""
    pred=sorted([b for b in pred if b[4]>=.25],key=lambda b:-b[4])
    used=set();tp=fp=0
    for b in pred:
        candidates=[i for i,g in enumerate(gt) if i not in used and g[4]==b[5]]
        if candidates:
            ovs=overlap([b[:4]],[gt[i][:4] for i in candidates])[0]
            best=int(np.argmax(ovs))
            if ovs[best]>=.5:
                used.add(candidates[best]);tp+=1;continue
        ig=[g[:4] for g in (ignored or []) if g[4]==b[5]]
        if ig and overlap([b[:4]],ig,ioa=True).max()>=.5: continue
        fp+=1
    fn=len(gt)-tp
    return dict(f1=2*tp/max(2*tp+fp+fn,1),tp=tp,fp=fp,fn=fn,recall=tp/max(len(gt),1))
