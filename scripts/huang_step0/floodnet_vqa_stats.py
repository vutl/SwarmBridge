import json, sys, collections, os
d = sys.argv[1]; w = sys.argv[2]
def load(n): return list(json.load(open(os.path.join(d,n),encoding='utf-8')).values())
S = {n: load(n+'.json') for n in ['train','valid','test']}
for n,rows in S.items():
    imgs = {r['Image_ID'] for r in rows}
    print(f"== {n}: {len(rows)} QA, {len(imgs)} images, keys={sorted(rows[0].keys())}")
    print("  per type:", dict(collections.Counter(r['Question_Type'] for r in rows)))
    print("  unique questions:", len({r['Question'] for r in rows}))
tr = S['train']
bytype = collections.defaultdict(list)
for r in tr: bytype[r['Question_Type']].append(r)
tot_major = 0
for t,rows in bytype.items():
    qs = collections.Counter(r['Question'] for r in rows)
    print(f"\n### {t}: n={len(rows)} templates={len(qs)}")
    for q,c in qs.most_common(): 
        ans = collections.Counter(str(r['Ground_Truth']) for r in rows if r['Question']==q)
        top = ans.most_common(1)[0]
        print(f"  [{c}] {q!r}  vocab={len(ans)} major={top} ({top[1]/c:.2f})  top5={ans.most_common(5)}")
    ans = collections.Counter(str(r['Ground_Truth']) for r in rows)
    top = ans.most_common(1)[0]
    pq = sum(collections.Counter(str(r['Ground_Truth']) for r in rows if r['Question']==q).most_common(1)[0][1] for q in qs)
    tot_major += pq
    print(f"  TYPE vocab={len(ans)} type-majority={top} acc={top[1]/len(rows):.3f}  per-question-majority acc={pq/len(rows):.3f}")
    print("  answers:", sorted(ans.items(), key=lambda x:-x[1])[:25])
print(f"\nOverall per-question-majority acc (train, in-sample) = {tot_major/len(tr):.3f}")
# per image question count
pi = collections.Counter(r['Image_ID'] for r in tr); print("questions/image dist:", sorted(collections.Counter(pi.values()).items()))
# mask overlap
def L(f): return {x.strip().split('.')[0].replace('_lab','') for x in open(os.path.join(w,f)) if x.strip()}
masks = L('list_1-UE3OjkRqoAjFmCxscIynx9tNdMN1F-G.txt') | L('list_1mU3b16Lbsl-tqe0N0yzN5mbwSCBebSel.txt')
print("\nTrack1 train masks:", len(masks))
for n,rows in S.items():
    ids = {r['Image_ID'].split('.')[0] for r in rows}
    ov = ids & masks
    print(f"  {n}: VQA images with Track1 train mask = {len(ov)}/{len(ids)}; QA pairs on them = {sum(1 for r in rows if r['Image_ID'].split('.')[0] in masks)}")
