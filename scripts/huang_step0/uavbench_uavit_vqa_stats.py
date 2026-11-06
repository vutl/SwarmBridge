"""Stats on stratified UAVIT-1M VQA-block sample. Usage: python -I uavit_vqa_stats.py <sample_json>"""
import sys, json, re
from collections import Counter, defaultdict
d = json.load(open(sys.argv[1], encoding="utf-8"))
d = [x for x in d if x["tag"] == "vqa"]
src = Counter(x["image"].split("/")[0] for x in d)
print("n_vqa_sampled", len(d), "sources", src.most_common())
def norm(q):
    q = re.sub(r"\[[a-z_]+\]\s*", "", q); q = q.split("\n")[0]
    return re.sub(r"\d+(\.\d+)?", "<N>", q.lower()).strip()
by = defaultdict(list)
for x in d:
    by[x["image"].split("/")[0]].append((norm(x["q"]), x["a"].strip().lower().rstrip(".")))
for s, l in by.items():
    qs = Counter(q for q, _ in l); an = Counter(a for _, a in l)
    print(f"\n== {s}: n={len(l)} uniq_q={len(qs)} ans_vocab={len(an)} maj={an.most_common(1)[0]}")
    for q, c in qs.most_common(6): print("   ", c, q[:110])
