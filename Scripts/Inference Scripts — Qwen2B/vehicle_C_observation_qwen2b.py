import os
import sys
import re
import glob
import json
import time
import random
import psutil
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datetime import datetime, timezone
from rdflib import Graph, RDF

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CORNERCASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(CORNERCASE_DIR, "Utilities & Support"))
import ground_truth as gt_module

# ── Metric helpers (identical to Qwen7B) ────────────────────────────────────
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
GT_SCENARIOS = {"bus_obscuring_car_second_car"}

def get_gt_graph(scenario, weather):
    if scenario not in GT_SCENARIOS:
        return None
    try:
        return gt_module.create_comprehensive_gt(weather)
    except Exception:
        return None

def compute_f1_vs_gt(predicted_ttl, gt_graph):
    if gt_graph is None:
        return None, None, None
    try:
        pred_graph = Graph()
        pred_graph.parse(data=predicted_ttl, format="turtle")
    except Exception:
        return 0.0, 0.0, 0.0
    RDF_TYPE = str(RDF.type)
    pred_types = {str(o) for s, p, o in pred_graph if str(p) == RDF_TYPE}
    gt_types   = {str(o) for s, p, o in gt_graph   if str(p) == RDF_TYPE}
    if not gt_types:
        return None, None, None
    tp = len(pred_types & gt_types)
    precision = tp / len(pred_types) if pred_types else 0.0
    recall    = tp / len(gt_types)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)

def compute_hallucination_rate(predicted_ttl):
    try:
        pred_graph = Graph()
        pred_graph.parse(data=predicted_ttl, format="turtle")
    except Exception:
        return None
    total = hallucinated = 0
    RDF_TYPE = str(RDF.type)
    for s, p, o in pred_graph:
        total += 1
        p_str = str(p)
        if p_str == RDF_TYPE:
            if str(o) not in ALLOWED_CLASSES:
                hallucinated += 1
        else:
            if p_str not in ALLOWED_PROPERTIES:
                hallucinated += 1
    return round(hallucinated / total, 4) if total > 0 else 0.0

def compute_compliance_rate(predicted_ttl):
    h = compute_hallucination_rate(predicted_ttl)
    return round(1.0 - h, 4) if h is not None else None

def compute_avg_confidence_score(triples):
    scores = []
    for line in triples.strip().splitlines():
        if "hasConfidenceScore" not in line:
            continue
        parts = line.strip().rstrip(";.").split()
        for i, tok in enumerate(parts):
            if "hasConfidenceScore" in tok and i + 1 < len(parts):
                try:
                    scores.append(float(parts[i + 1].split("^^")[0].strip('"')))
                except ValueError:
                    pass
    return sum(scores) / len(scores) if scores else None

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen3-VL-2B"
VEHICLE       = "C"
BATCH_SIZE    = 1
PROGRESS_EVERY = 10
SCENARIOS     = ["bus_obscuring_car_second_car"]
RGB_FOLDER    = "second_car_rgb"
ACTIVITY_NAME = "vehicleC_activity"

DATASET_DIR = os.path.join(CORNERCASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(CORNERCASE_DIR, "outputs", "qwen2b")
METRICS_DIR = os.path.join(CORNERCASE_DIR, "metrics", "qwen2b")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
metrics_file = os.path.join(METRICS_DIR, f"_metrics_{random.randint(10**18, 10**19)}.jsonl")

prefixes = """\
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
"""

prompt = f"""Output Turtle RDF only.

Check whether this image shows a clear occlusion-related blind-spot hazard.

@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex: <http://cornercase.org/instances#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

Use only:
Classes: avcco:Vehicle, avcco:OcclusionEvent, avcco:SensorBlindSpotCase
Properties: prov:wasGeneratedBy, avcco:hasOccluder, avcco:hasOccludedEntity, avcco:hasTriggerEvent, avcco:hasActor, avcco:hasObstacle, avcco:hasConfidenceScore

Rules:
- No explanation.
- Create only relevant vehicles.
- Create avcco:OcclusionEvent only if one object clearly blocks another.
- Create avcco:SensorBlindSpotCase only if the occlusion creates a clear dangerous blind spot.
- If uncertain, omit it.
- Do not invent classes or properties.

Always include:
ex:{ACTIVITY_NAME} a prov:Activity .

Example format only. Do not copy it unless supported by the image:
ex:{ACTIVITY_NAME} a prov:Activity .
ex:veh1 a avcco:Vehicle .
ex:veh2 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh2 ;
    avcco:hasOccludedEntity ex:veh1 ;
    prov:wasGeneratedBy ex:{ACTIVITY_NAME} ;
    avcco:hasConfidenceScore "0.90"^^xsd:float .
ex:cc1 a avcco:SensorBlindSpotCase ;
    avcco:hasActor ex:veh1 ;
    avcco:hasObstacle ex:veh2 ;
    avcco:hasTriggerEvent ex:occ1 ;
    prov:wasGeneratedBy ex:{ACTIVITY_NAME} ;
    avcco:hasConfidenceScore "0.85"^^xsd:float .
"""

# ── Load model once ──────────────────────────────────────────────────────────
print("Loading Qwen3-VL-2B-Instruct model...")
_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    trust_remote_code=True,
)
_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
_device = next(_model.parameters()).device
print(f"Model loaded on: {_device}")

# ── Inference function ────────────────────────────────────────────────────────
def get_triples_from_model(image_path, prompt_text):
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]
    }]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(_device)
    input_len = inputs["input_ids"].shape[1]

    _proc = psutil.Process(os.getpid())
    _cpu_start    = time.process_time()
    _ram_before   = _proc.memory_info().rss / (1024 * 1024)
    t0            = time.time()

    success = True
    error   = None
    raw_output = ""
    try:
        with torch.no_grad():
            generated_ids = _model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
            )
        raw_output = _processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
    except Exception as e:
        success = False
        error   = f"{type(e).__name__}: {e}"

    latency_s   = time.time() - t0
    cpu_time_s  = round(time.process_time() - _cpu_start, 4)
    ram_peak_mb = round(_proc.memory_info().rss / (1024 * 1024), 2)

    call_metrics = {
        "batch_size": BATCH_SIZE, "latency_s": latency_s,
        "cpu_time_s": cpu_time_s, "ram_peak_mb": ram_peak_mb,
        "ram_delta_mb": round(ram_peak_mb - _ram_before, 2),
        "input_bytes": len(text.encode("utf-8")),
        "output_bytes": len(raw_output.encode("utf-8")),
        "prompt_tokens": None, "completion_tokens": None,
        "success": success, "error": error,
    }

    print("=== RAW OUTPUT ==="); print(raw_output)

    if not success:
        return "", call_metrics

    if "```turtle" in raw_output:
        m = re.search(r"```turtle(.+?)```", raw_output, re.DOTALL)
        return (m.group(1).strip() if m else raw_output.strip()), call_metrics
    elif "```" in raw_output:
        m = re.search(r"```(.+?)```", raw_output, re.DOTALL)
        return (m.group(1).strip() if m else raw_output.strip()), call_metrics
    return raw_output.strip(), call_metrics

# ── Resume helpers ────────────────────────────────────────────────────────────
def completed_loops(vehicle, scenario, weather):
    max_loop = 0
    for fpath in glob.glob(os.path.join(METRICS_DIR, "_metrics_*.jsonl")):
        try:
            with open(fpath) as f:
                for line in f:
                    r = json.loads(line.strip())
                    if (r.get("vehicle") == vehicle and
                            r.get("scenario") == scenario and
                            r.get("weather") == weather):
                        max_loop = max(max_loop, r.get("loop", 0))
        except Exception:
            pass
    return max_loop

# ── Main loop ─────────────────────────────────────────────────────────────────
_t_run_start = time.time()

for scenario in SCENARIOS:
    scenario_folder = os.path.join(DATASET_DIR, scenario)
    if not os.path.isdir(scenario_folder):
        print(f"Skipping {scenario}: folder not found.")
        continue

    for weather in sorted(os.listdir(scenario_folder)):
        weather_folder = os.path.join(scenario_folder, weather)
        if not os.path.isdir(weather_folder):
            continue

        rgbs_folder = os.path.join(weather_folder, RGB_FOLDER)
        if not os.path.isdir(rgbs_folder):
            print(f"  Skipping: {RGB_FOLDER} not found in {weather_folder}")
            continue

        rgb_images = sorted(
            glob.glob(os.path.join(rgbs_folder, "*.png")) +
            glob.glob(os.path.join(rgbs_folder, "*.jpg"))
        )
        if not rgb_images:
            continue

        loop = completed_loops(VEHICLE, scenario, weather)
        main_graph = Graph()
        _t_scenario = time.time()
        print(f"\n[START] vehicle={VEHICLE}  {scenario}/{weather}  "
              f"images={len(rgb_images)}  resume_loop={loop}")

        while loop < len(rgb_images):
            print(f"Loop: {loop + 1}/{len(rgb_images)}")
            image_path = rgb_images[loop]

            triples, call_metrics = get_triples_from_model(image_path, prompt)

            if triples and not triples.startswith("@prefix"):
                triples = prefixes + "\n" + triples

            avg_confidence = compute_avg_confidence_score(triples) or 0.0
            temp           = Graph()
            rdf_parse_ok   = False

            if call_metrics["success"] and triples:
                try:
                    temp.parse(data=triples, format="turtle")
                    rdf_parse_ok = True
                except Exception as parse_err:
                    print(f"RDF parse failed: {parse_err}")

            gt_graph = get_gt_graph(scenario, weather)
            if rdf_parse_ok:
                _, _, f1       = compute_f1_vs_gt(triples, gt_graph)
                hallucination  = compute_hallucination_rate(triples)
                compliance     = compute_compliance_rate(triples)
            else:
                f1            = 0.0 if call_metrics["success"] else None
                hallucination = None
                compliance    = None

            if not call_metrics["success"]:
                status = "api_error"
            elif not rdf_parse_ok:
                status = "parse_error"
            else:
                status = "success"

            print(f"Confidence: {avg_confidence}, F1: {f1}, "
                  f"Hallucination: {hallucination}, Status: {status}")

            with open(metrics_file, "a") as mf:
                mf.write(json.dumps({
                    "scenario":           scenario,
                    "weather":            weather,
                    "loop":               loop + 1,
                    "batch_size":         BATCH_SIZE,
                    "model":              MODEL_ID,
                    "vehicle":            VEHICLE,
                    "status":             status,
                    "latency_s":          call_metrics["latency_s"],
                    "cpu_time_s":         call_metrics["cpu_time_s"],
                    "ram_peak_mb":        call_metrics["ram_peak_mb"],
                    "ram_delta_mb":       call_metrics["ram_delta_mb"],
                    "input_bytes":        call_metrics["input_bytes"],
                    "output_bytes":       call_metrics["output_bytes"],
                    "prompt_tokens":      call_metrics["prompt_tokens"],
                    "completion_tokens":  call_metrics["completion_tokens"],
                    "confidence":         avg_confidence,
                    "f1":                 f1,
                    "hallucination_rate": hallucination,
                    "compliance_rate":    compliance,
                    "rdf_parse_ok":       rdf_parse_ok,
                    "error":              call_metrics["error"],
                    "ts":                 datetime.now(timezone.utc).isoformat(),
                }) + "\n")

            _done   = loop + 1
            _is_last = _done >= len(rgb_images)
            if _done % PROGRESS_EVERY == 0 or _is_last:
                elapsed_total = round(time.time() - _t_run_start, 1)
                tag = "[DONE]    " if _is_last else "[PROGRESS]"
                print(f"{tag} vehicle={VEHICLE}  {scenario}/{weather}  "
                      f"loop={_done}/{len(rgb_images)} "
                      f"({min(100.0, round(_done/len(rgb_images)*100,1))}%)  "
                      f"status={status}  f1={f1}  "
                      f"elapsed={round(time.time()-_t_scenario,1)}s  total={elapsed_total}s")

            if not rdf_parse_ok:
                loop += 1
                continue

            main_graph = main_graph + temp

            loop_output_path = os.path.join(OUTPUT_DIR, scenario, weather, str(loop + 1))
            os.makedirs(loop_output_path, exist_ok=True)

            loop_file = os.path.join(loop_output_path, f"vehicle_{VEHICLE}_observations_loop.ttl")
            main_file = os.path.join(loop_output_path, f"vehicle_{VEHICLE}_observations.ttl")

            temp.serialize(destination=loop_file, format="turtle")
            main_graph.serialize(destination=main_file, format="turtle")
            print(f"Saved: {loop_file} ({len(temp)} triples), "
                  f"main: {len(main_graph)} triples")

            loop += 1

print(f"\n[FINISHED] vehicle={VEHICLE}  total={round(time.time()-_t_run_start,1)}s")
