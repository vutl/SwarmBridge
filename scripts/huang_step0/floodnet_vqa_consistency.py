import json, sys, collections
rows = list(json.load(open(sys.argv[1],encoding='utf-8')).values())
img = collections.defaultdict(dict)
for r in rows: img[r['Image_ID']][r['Question']] = (r['Question_Type'], str(r['Ground_Truth']))
types_per_img = collections.Counter(tuple(sorted({v[0] for v in q.values()})) for q in img.values())
print("type combos per image:", types_per_img.most_common())
# overall condition vs other qs
ov = collections.Counter(); rc=collections.Counter(); yn=collections.Counter()
for q in img.values():
    o = q.get('What is the overall condition of the given image?',(None,None))[1]
    road = [v[1] for k,v in q.items() if v[0]=='Condition_Recognition' and 'road' in k]
    ynv = [(k,v[1]) for k,v in q.items() if v[0]=='Yes_No']
    cc = [(k,v[1]) for k,v in q.items() if v[0]=='Complex_Counting']
    for r in road: rc[(o,r)]+=1
    for k,a in ynv: yn[(o,'nonfl' if 'non' in k else 'fl',a)]+=1
    for k,a in cc: ov[(o,'nonfl' if 'non' in k else 'fl')]+=1
print("overall x road cond:", rc); print("overall x yesno:", yn); print("overall x complex-count kind:", ov)
