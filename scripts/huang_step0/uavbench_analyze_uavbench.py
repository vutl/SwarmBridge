"""Question-diversity audit of UAVBench QA-style annotation files (no images).
Usage: python -I analyze_uavbench.py <meta_dir>
"""
import sys, os, json, re
from collections import Counter, defaultdict

META = sys.argv[1]
GROUPS = {
    "vqa": ["vqa_altitu.json", "vqa_altituint_Estima.json", "vqa_altituone_Estima.json",
            "vqa_compa.json", "vqa_pres.json", "vqa_weath.json", "vqa_weathUAVDTS.json"],
    "rvqa": ["rvqa_UAVDTS.json"],
    "count": ["count_AUAIR.json", "count_DroneVehicle.json", "count_HazyDet.json", "count_HazyDethazy.json",
              "count_RDDTS.json", "count_UAVDTM.json", "count_VisDrone.json"],
    "vg": ["vg_RefDrone.json", "vg_UAVDTS.json"],
}

def load(fn):
    with open(os.path.join(META, fn), encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for d in data:
        conv = d["conversations"]
        q = conv[0]["value"]; a = conv[1]["value"]
        q = q.replace("<image>", "").strip()
        q = re.sub(r"^\[[a-z_]+\]\s*", "", q)
        q = re.split(r"\n(Answer|answer)", q)[0].strip()
        q = re.sub(r"\s*Answer (an integer|one number|with).*$", "", q).strip()
        out.append((d["image"], q, a.strip().rstrip(".").lower()))
    return out

COLORS = r"\b(red|white|black|blue|green|yellow|gr[ae]y|silver|orange|brown|purple|pink|dark|light)\b"
def qtype(q, group):
    s = q.lower()
    if group == "vg":
        return "grounding(referring expr)"
    if re.search(r"\b(altitude|elevat\w*|height|meters?|\d+m\b|high above|above (the )?ground|from the ground|flying above|flying below|how high)", s):
        return "altitude(metadata)"
    if re.search(r"\b(weather|fog\w*|haz[ey]\w*|mist\w*|visibility|air quality|rain|snow|dark|day\w*|night\w*|illuminat\w*|lighting|bright|sunny|cloud)", s):
        return "weather/illumination"
    if re.search(r"\b(how many|count|number of)\b", s):
        return "counting"
    if re.search(r"\bcolou?r\b", s):
        return "color/attribute"
    if re.search(r"\b(more|less|larger|smaller|similar|bigger|than|compare|occup|area|dominat)\b", s):
        return "area comparison"
    if re.search(r"\b(where|left|right|top|bottom|position|locat|cent(er|re))\b", s):
        return "location/position"
    if re.search(r"\b(next to|near|beside|behind|in front|between|above|below)\b", s):
        return "spatial relation"
    if re.search(r"\b(scene|land|type of area|what kind of place)\b", s):
        return "scene/land-use"
    if re.search(r"\b(text|written|sign|letters|number plate)\b", s):
        return "OCR/text"
    if re.search(r"^(is|are|does|do|can|was|were|has|have)\b|\bpresent\b|\bcontain", s):
        return "existence/yes-no"
    if re.search(r"^(what|which)\b.*\b(type|kind|category|class|vehicle|object)\b", s):
        return "object category"
    if re.search(r"\b(describe|why|how|explain)\b", s):
        return "open-ended"
    return "other"

OBJ_WORDS = None
def template(q):
    t = q.lower()
    t = re.sub(r"\{<\d+><\d+><\d+><\d+>\}", "<BOX>", t)
    t = re.sub(r"\d+(\.\d+)?", "<N>", t)
    return t

def main():
    report = {}
    alltypes = Counter(); alltot = 0
    for g, files in GROUPS.items():
        rows = []
        perfile = {}
        for fn in files:
            r = load(fn); rows += r; perfile[fn] = len(r)
        imgs = Counter(r[0] for r in rows)
        qpi = Counter(imgs.values())
        uq = set(r[1] for r in rows)
        types = defaultdict(list)
        for im, q, a in rows:
            types[qtype(q, g)].append((q, a))
        # templates: replace object nouns found in count/pres questions by <OBJ>
        tpl = Counter(template(q) for q in uq)
        g_rep = {"files": perfile, "n_qa": len(rows), "n_images": len(imgs),
                 "q_per_image": dict(sorted(qpi.items())[:12]),
                 "q_per_image_mean": round(len(rows) / max(1, len(imgs)), 2),
                 "unique_questions": len(uq), "templates_after_num_box_norm": len(tpl),
                 "types": {}}
        for t, lst in sorted(types.items(), key=lambda x: -len(x[1])):
            ans = Counter(a for _, a in lst)
            maj, majn = ans.most_common(1)[0]
            g_rep["types"][t] = {"n": len(lst), "frac": round(len(lst) / len(rows), 4),
                                 "answer_vocab": len(ans), "majority": maj,
                                 "majority_acc": round(majn / len(lst), 4),
                                 "unique_q": len(set(q for q, _ in lst)),
                                 "top_answers": ans.most_common(8),
                                 "examples": list(dict.fromkeys(q for q, _ in lst))[:6]}
            alltypes[(g, t)] += len(lst)
        alltot += len(rows)
        report[g] = g_rep
    # object vocab for counting/presence/rvqa/vg
    objs = Counter()
    for fn in GROUPS["count"]:
        for _, q, _a in load(fn):
            m = re.search(r"(?:number of|how many|count(?: the)?(?: number of)?)\s+(?:the\s+)?([a-z][a-z \-]*?)(?:\s+(?:are|is|in|can|visible|present|appear|there|does|do)\b|[?.]|$)", q.lower())
            if m: objs[("count", m.group(1).strip())] += 1
    for _, q, _a in load("vqa_pres.json"):
        m = re.search(r"(?:contain|there an?|there any|a|an|any|of|see)\s+([a-z][a-z \-]*?)\s*(?:\?|$| in | present| visible)", q.lower())
        if m: objs[("pres", m.group(1).strip())] += 1
    for _, q, _a in load("rvqa_UAVDTS.json"):
        m = re.search(r"the ([a-z]+) <box>", template(q))
        if m: objs[("rvqa", m.group(1))] += 1
    report["object_vocab"] = {k: sorted(set(o for (gg, o) in objs if gg == k)) for k in ("count", "pres", "rvqa")}
    report["all_type_counts"] = {f"{g}|{t}": n for (g, t), n in alltypes.most_common()}
    report["all_total"] = alltot
    # VG expression attributes
    vg = []
    for fn in GROUPS["vg"]:
        vg += [q for _, q, _ in load(fn)]
    feat = Counter()
    for q in vg:
        s = q.lower()
        feat["has_color"] += bool(re.search(COLORS, s))
        feat["has_abs_position"] += bool(re.search(r"\b(left|right|top|bottom|middle|center|centre|corner|upper|lower|edge)\b", s))
        feat["has_relation_to_other_obj"] += bool(re.search(r"\b(next to|near|beside|behind|in front of|between|adjacent|closest|nearest|parked (on|in|along|by)|on the road|along|across)\b", s))
        feat["has_action_state"] += bool(re.search(r"\b(parked|moving|driving|turning|waiting|stopped|walking|crossing)\b", s))
        feat["has_ordinal_or_size"] += bool(re.search(r"\b(first|second|third|largest|smallest|biggest|leftmost|rightmost|topmost)\b", s))
    report["vg_features"] = {k: round(v / len(vg), 4) for k, v in feat.items()}
    # per-file (generator) stats with object-normalised templates
    OBJ = sorted(set(report["object_vocab"]["count"]) | set(o for o in report["object_vocab"]["pres"] if not o.startswith("you see")), key=len, reverse=True)
    objre = re.compile(r"\b(" + "|".join(re.escape(o) for o in OBJ) + r")s?\b")
    COLR = re.compile(COLORS + r"|\bcyan\b")
    pf = {}
    for g, files in GROUPS.items():
        for fn in files:
            r = load(fn); ans = Counter(a for _, _, a in r)
            tp = set(COLR.sub("<C>", objre.sub("<OBJ>", template(q))) for _, q, _ in r)
            pf[fn] = {"n": len(r), "imgs": len(set(x[0] for x in r)), "uq": len(set(q for _, q, _ in r)),
                      "tpl_norm": len(tp), "ans_vocab": len(ans),
                      "maj_acc": round(ans.most_common(1)[0][1] / len(r), 4)}
    report["per_file"] = pf
    report["vg_n"] = len(vg)
    report["vg_unique"] = len(set(vg))
    report["vg_examples"] = vg[:3] + vg[-3:]
    json.dump(report, sys.stdout, indent=1, ensure_ascii=True)

main()
