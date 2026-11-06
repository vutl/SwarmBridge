"""Probe UAVIT-1M ordering via HF datasets-server /rows (no bulk download).
Usage: python -I uavit_probe.py <out_json> map|sample <start> <end> <step> <len>"""
import sys, json, re, urllib.request, time
B = "https://datasets-server.huggingface.co/rows?dataset=ZhanYang-nwpu/UAVIT-1M&config=default&split=train"
def rows(off, n):
    for k in range(4):
        try:
            with urllib.request.urlopen(f"{B}&offset={off}&length={n}", timeout=60) as r:
                return json.load(r)["rows"]
        except Exception as e:
            time.sleep(2 + 3 * k)
    return []
out, mode, s, e, st, ln = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
res = []
for off in range(s, e, st):
    for r in rows(off, ln):
        c = r["row"]["conversations"]
        q = c[0]["value"]; a = c[1]["value"] if len(c) > 1 else ""
        tag = re.search(r"\[([a-z_]+)\]", q)
        res.append({"idx": r["row_idx"], "image": r["row"]["image"], "tag": tag.group(1) if tag else None,
                    "q": q.replace("<image>", "").strip(), "a": a, "nturn": len(c)})
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=True)
if mode == "map":
    for x in res:
        print(x["idx"], x["tag"], x["image"].split("/")[0], x["nturn"])
