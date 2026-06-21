#!/usr/bin/env python3
"""Recompute evaluation_metrics.csv from TTL outputs and JSONL files using all 4 GT scenarios.
F1 logic matches inference scripts exactly (full URI comparison, no filtering)."""

import os, json, glob
from rdflib import Graph, URIRef, RDF

BASE = "/Users/alicelan/cornercase"
GT_BASE = f"{BASE}/ground_truth"
OUT_BASE = f"{BASE}/outputs"
METRICS_BASE = f"{BASE}/metrics"

SCENARIOS = [
    "bus_obscuring_car_second_car",
    "bus_stop_near_playground",
    "spawn_police_car_chase_1",
    "spawn_police_car_chase_2",
]
WEATHERS = ["day", "night", "fog"]

RDF_TYPE = str(RDF.type)

# Hallucination: same ALLOWED_CLASSES/PROPERTIES as all inference scripts
ALLOWED_CLASSES = {
    "http://cornercase.org/avcco#Vehicle",
    "http://cornercase.org/avcco#OcclusionEvent",
    "http://cornercase.org/avcco#SensorBlindSpotCase",
    "http://www.w3.org/ns/prov#Activity",
    "http://www.w3.org/ns/prov#Agent",
}
ALLOWED_PROPERTIES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/ns/prov#wasGeneratedBy",
    "http://cornercase.org/avcco#hasOccluder",
    "http://cornercase.org/avcco#hasOccludedEntity",
    "http://cornercase.org/avcco#hasTriggerEvent",
    "http://cornercase.org/avcco#hasActor",
    "http://cornercase.org/avcco#hasObstacle",
    "http://cornercase.org/avcco#hasConfidenceScore",
}

def load_gt_graphs():
    """Load all GT graphs indexed by (scenario, weather)."""
    gt = {}
    for sc in SCENARIOS:
        for wx in WEATHERS:
            path = f"{GT_BASE}/{sc}/{wx}/ground_truth.ttl"
            if os.path.exists(path):
                g = Graph()
                g.parse(path, format="turtle")
                gt[(sc, wx)] = g
    return gt

def compute_f1_vs_gt(pred_g, gt_g):
    """Exact same logic as inference scripts: full URI set comparison."""
    if gt_g is None:
        return None, None, None
    pred_types = {str(o) for s, p, o in pred_g if str(p) == RDF_TYPE}
    gt_types   = {str(o) for s, p, o in gt_g   if str(p) == RDF_TYPE}
    if not gt_types:
        return None, None, None
    tp = len(pred_types & gt_types)
    precision = tp / len(pred_types) if pred_types else 0.0
    recall    = tp / len(gt_types)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)

def compute_hallucination(pred_g):
    """Same logic as inference scripts."""
    total = hallucinated = 0
    for s, p, o in pred_g:
        p_str = str(p)
        total += 1
        if p_str == RDF_TYPE:
            if str(o) not in ALLOWED_CLASSES:
                hallucinated += 1
        else:
            if p_str not in ALLOWED_PROPERTIES:
                hallucinated += 1
    return round(hallucinated / total, 4) if total > 0 else 0.0

def scenario_from_path(fpath, model_dir):
    """Extract scenario from file path."""
    base = f"{OUT_BASE}/{model_dir}/"
    rel = fpath[len(base):]
    parts = rel.split(os.sep)
    return parts[0] if parts else ""

def weather_from_path(fpath, model_dir):
    """Extract weather from file path."""
    base = f"{OUT_BASE}/{model_dir}/"
    rel = fpath[len(base):]
    parts = rel.split(os.sep)
    return parts[1] if len(parts) > 1 else ""

def scan_model(model_dir, gt_graphs):
    """Scan model output directory computing F1, hallucination per file."""
    results = {"f1": [], "precision": [], "recall": [], "hall": []}
    model_path = f"{OUT_BASE}/{model_dir}"
    if not os.path.exists(model_path):
        return results
    for sc in SCENARIOS:
        sc_path = f"{model_path}/{sc}"
        if not os.path.exists(sc_path):
            continue
        for wx in WEATHERS:
            gt_g = gt_graphs.get((sc, wx))
            wx_path = f"{sc_path}/{wx}"
            if not os.path.exists(wx_path):
                continue
            for root, dirs, files in os.walk(wx_path):
                for fn in files:
                    if not fn.endswith(".ttl") or "_loop" in fn:
                        continue
                    fpath = os.path.join(root, fn)
                    try:
                        g = Graph()
                        g.parse(fpath, format="turtle")
                        if len(g) == 0:
                            continue
                    except Exception:
                        continue
                    if gt_g is not None:
                        p, r, f1 = compute_f1_vs_gt(g, gt_g)
                        if f1 is not None:
                            results["f1"].append(f1)
                            results["precision"].append(p)
                            results["recall"].append(r)
                    hall = compute_hallucination(g)
                    results["hall"].append(hall)
    return results

def load_jsonl_stats(dirs, bs_filter=None):
    """Return status counts from JSONL files."""
    counts = {"success": 0, "parse_error": 0, "api_error": 0, "timeout_error": 0, "total": 0}
    for d in dirs:
        seen = set()
        all_files = glob.glob(f"{d}/**/*.jsonl", recursive=True)
        for fpath in all_files:
            if fpath in seen:
                continue
            seen.add(fpath)
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("status") == "smoke_test":
                        continue
                    bs = r.get("batch_size")
                    if bs_filter is not None and bs != bs_filter:
                        continue
                    counts["total"] += 1
                    s = r.get("status", "")
                    if s in counts:
                        counts[s] += 1
    return counts

def build_qwen_bs_map():
    """Map (scenario, weather, loop, vehicle) → batch_size from Qwen7B JSONL."""
    bs_map = {}
    qwen_dir = f"{METRICS_BASE}/Qwen 2.5 VL 7B metrics"
    for fpath in glob.glob(f"{qwen_dir}/*.jsonl"):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "smoke_test":
                    continue
                # vehicle in JSONL is like "ego", "flee", "A", "B", "C"
                # filename is vehicle_ego_observations.ttl → vehicle = "ego"
                key = (r.get("scenario",""), r.get("weather",""),
                       str(r.get("loop","")), r.get("vehicle",""))
                bs = r.get("batch_size")
                if bs and key not in bs_map:
                    bs_map[key] = bs
    return bs_map

def scan_qwen_by_bs(target_bs, gt_graphs, bs_map):
    """Scan Qwen7B outputs filtering by batch size using bs_map."""
    results = {"f1": [], "precision": [], "recall": [], "hall": []}
    model_path = f"{OUT_BASE}/qwen7B"
    if not os.path.exists(model_path):
        return results
    for sc in SCENARIOS:
        sc_path = f"{model_path}/{sc}"
        if not os.path.exists(sc_path):
            continue
        for wx in WEATHERS:
            gt_g = gt_graphs.get((sc, wx))
            wx_path = f"{sc_path}/{wx}"
            if not os.path.exists(wx_path):
                continue
            if not os.path.isdir(wx_path):
                continue
            for loop_dir in os.listdir(wx_path):
                loop_path = f"{wx_path}/{loop_dir}"
                if not os.path.isdir(loop_path):
                    continue
                for fn in os.listdir(loop_path):
                    if not fn.endswith(".ttl") or "_loop" in fn:
                        continue
                    # derive vehicle: "vehicle_ego_observations.ttl" → "ego"
                    vehicle = fn.replace("vehicle_", "").replace("_observations.ttl", "")
                    key = (sc, wx, loop_dir, vehicle)
                    bs = bs_map.get(key)
                    if bs != target_bs:
                        continue
                    fpath = f"{loop_path}/{fn}"
                    try:
                        g = Graph()
                        g.parse(fpath, format="turtle")
                        if len(g) == 0:
                            continue
                    except Exception:
                        continue
                    if gt_g is not None:
                        p, r, f1 = compute_f1_vs_gt(g, gt_g)
                        if f1 is not None:
                            results["f1"].append(f1)
                            results["precision"].append(p)
                            results["recall"].append(r)
                    hall = compute_hallucination(g)
                    results["hall"].append(hall)
    return results

def avg(lst):
    return sum(lst) / len(lst) if lst else None

def fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None else "N/A"

def rdf_rate(c):
    denom = c["success"] + c["parse_error"]
    return c["success"] / denom if denom > 0 else 0.0

# ─── Load GT ─────────────────────────────────────────────────────────────────
print("Loading ground truth graphs...")
gt_graphs = load_gt_graphs()
print(f"  {len(gt_graphs)} GT graphs loaded.")

# ─── Scan TTL outputs ─────────────────────────────────────────────────────────
print("\nScanning TTL output files...")
ttl = {}
for label, dir_ in [("gpt","gpt"), ("smolvlm","smolvlm"), ("glm","glm"), ("moondream","moondream")]:
    r = scan_model(dir_, gt_graphs)
    ttl[label] = {"p": avg(r["precision"]), "r": avg(r["recall"]),
                  "f1": avg(r["f1"]), "hall": avg(r["hall"]), "n": len(r["f1"])}
    print(f"  {label}: n={len(r['f1'])}, P={fmt(ttl[label]['p'])}, R={fmt(ttl[label]['r'])}, "
          f"F1={fmt(ttl[label]['f1'])}, Hall={fmt(ttl[label]['hall'])}")

print("\nBuilding Qwen7B bs_map...")
bs_map = build_qwen_bs_map()
print(f"  {len(bs_map)} entries")

for bs in [1, 3, 5]:
    r = scan_qwen_by_bs(bs, gt_graphs, bs_map)
    key = f"qwen_bs{bs}"
    ttl[key] = {"p": avg(r["precision"]), "r": avg(r["recall"]),
                "f1": avg(r["f1"]), "hall": avg(r["hall"]), "n": len(r["f1"])}
    print(f"  Qwen7B BS{bs}: n={len(r['f1'])}, P={fmt(ttl[key]['p'])}, "
          f"R={fmt(ttl[key]['r'])}, F1={fmt(ttl[key]['f1'])}, Hall={fmt(ttl[key]['hall'])}")

# ─── Load JSONL stats ─────────────────────────────────────────────────────────
print("\nLoading JSONL stats...")
gpt_c  = load_jsonl_stats([f"{METRICS_BASE}/GPT metrics"])
qw1_c  = load_jsonl_stats([f"{METRICS_BASE}/Qwen 2.5 VL 7B metrics"], bs_filter=1)
qw3_c  = load_jsonl_stats([f"{METRICS_BASE}/Qwen 2.5 VL 7B metrics"], bs_filter=3)
qw5_c  = load_jsonl_stats([f"{METRICS_BASE}/Qwen 2.5 VL 7B metrics"], bs_filter=5)
glm_c  = load_jsonl_stats([f"{METRICS_BASE}/glm"])
moon_c = load_jsonl_stats([f"{METRICS_BASE}/moondream"])
smol_c = load_jsonl_stats([f"{METRICS_BASE}/smolvlm"])

# ─── Write CSV ────────────────────────────────────────────────────────────────
def row(model, deploy, bs, c, tk):
    t = ttl.get(tk, {})
    p = t.get("p"); r_ = t.get("r"); f = t.get("f1"); h = t.get("hall")
    rdf = rdf_rate(c)
    comp = (1 - h) if h is not None else None
    return (f"{model},{deploy},{bs},{c['total']},{c['success']},{c['parse_error']},"
            f"{c['api_error']},{c['timeout_error']},{rdf:.4f},"
            f"{fmt(p)},{fmt(r_)},{fmt(f)},{fmt(h)},{fmt(comp)}\n")

header = ("Model,Deployment,Batch Size,Total Calls,Success,Parse Error,"
          "API / Server Error,Timeout Error,RDF Success Rate,"
          "Avg Precision,Avg Recall,Avg F1,Avg Hallucination Rate,Avg Ontology Compliance\n")

out_path = f"{BASE}/results/evaluation_metrics.csv"
with open(out_path, "w") as f:
    f.write(header)
    f.write(row("GPT-5.5",  "Cloud", 1, gpt_c,  "gpt"))
    f.write(row("Qwen7B",   "Local", 1, qw1_c,  "qwen_bs1"))
    f.write(row("Qwen7B",   "Local", 3, qw3_c,  "qwen_bs3"))
    f.write(row("Qwen7B",   "Local", 5, qw5_c,  "qwen_bs5"))
    f.write(row("GLM",      "Local", 1, glm_c,  "glm"))
    f.write(row("Moondream","Local", 1, moon_c, "moondream"))
    f.write(row("SmolVLM",  "Local", 1, smol_c, "smolvlm"))

print(f"\nWrote {out_path}")

# ─── Final table ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL EVALUATION METRICS TABLE")
print("="*80)
print(f"{'Model':<14} {'BS':>2}  {'RDF%':>7}  {'P':>7}  {'R':>7}  {'F1':>7}  {'Hall%':>7}  {'Comp%':>7}")
print("-"*70)
rows_tbl = [
    ("GPT-5.5",  1, gpt_c,  "gpt"),
    ("Qwen7B",   1, qw1_c,  "qwen_bs1"),
    ("Qwen7B",   3, qw3_c,  "qwen_bs3"),
    ("Qwen7B",   5, qw5_c,  "qwen_bs5"),
    ("GLM",      1, glm_c,  "glm"),
    ("Moondream",1, moon_c, "moondream"),
    ("SmolVLM",  1, smol_c, "smolvlm"),
]
def pct(v):
    return f"{v*100:.2f}%" if v is not None else "N/A"

for model, bs, c, tk in rows_tbl:
    t = ttl.get(tk, {})
    rdf = rdf_rate(c)
    p = t.get("p"); r_ = t.get("r"); f = t.get("f1"); h = t.get("hall")
    comp = (1 - h) if h is not None else None
    print(f"{model:<14} {bs:>2}  {pct(rdf):>8}  {pct(p):>8}  {pct(r_):>8}  {pct(f):>8}  {pct(h):>8}  {pct(comp):>8}")
