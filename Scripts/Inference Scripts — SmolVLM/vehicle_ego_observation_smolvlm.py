import os
import re
import glob
import json
import time
import random
import psutil
from datetime import datetime, timezone
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from rdflib import Graph, RDF, RDFS, OWL
import weather_classifier_inference as uciclassifier
import ground_truth as gt_module


# === Model setup ===
MODEL_ID   = "HuggingFaceTB/SmolVLM-256M-Instruct"
BATCH_SIZE = 1
PROGRESS_EVERY = 10
VEHICLE    = "ego"

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR    = os.path.join(CORNERCASE_DIR, "metrics_new", "smolvlm")
os.makedirs(METRICS_DIR, exist_ok=True)
metrics_file = os.path.join(METRICS_DIR, f"_metrics_{random.randint(10**18, 10**19)}.jsonl")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading {MODEL_ID} on {device.upper()}...")
_t_load = time.time()
smolvlm_processor = AutoProcessor.from_pretrained(MODEL_ID)
smolvlm_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
).to(device)
print(f"Model loaded in {time.time()-_t_load:.1f}s\n")


# === Ontology extraction ===
def extract_ontology_prompt(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    class_lines = ["Ontology Classes (and Hierarchy):"]
    for s in g.subjects(RDF.type, OWL.Class):
        label = g.value(s, RDFS.label)
        comment = g.value(s, RDFS.comment)
        subclass_of = g.value(s, RDFS.subClassOf)
        class_name = s.split("#")[-1] if "#" in s else s
        superclass = (subclass_of.split("#")[-1] if subclass_of and "#" in subclass_of else subclass_of)
        line = f"- {class_name}"
        if superclass:
            line += f" (subclass of {superclass})"
        if label:
            line += f": {label}"
        if comment:
            line += f"\n  {comment}"
        class_lines.append(line)
    property_lines = ["\nOntology Properties:"]
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        property_lines.append(f"- {s.split('#')[-1]}")
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        property_lines.append(f"- {s.split('#')[-1]}")
    return "\n".join(class_lines + property_lines)


# === Confidence score extraction ===
def compute_avg_confidence_score(triples):
    confidence_scores = []
    current_subject = None
    for line in triples.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("@prefix") or line.startswith("#"):
            continue
        if "hasConfidenceScore" not in line:
            continue
        if line.endswith(";") or line.endswith("."):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                s, p, o = parts
                s = s.strip().split(":")[-1]
                p = p.strip().split("^^")[0].strip('"')
                if s == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(p))
                    except ValueError:
                        pass
            current_subject = parts[0].strip().split(":")[-1]
        else:
            parts = line.split(" ", 1)
            if len(parts) == 2 and current_subject:
                p, o = parts
                p = p.strip().split(":")[-1]
                o = o.strip("<>").strip('"')
                if p == "hasConfidenceScore":
                    try:
                        confidence_scores.append(float(o))
                    except ValueError:
                        pass
    return sum(confidence_scores) / len(confidence_scores) if confidence_scores else None


# === Classifier score ===
def compute_avg_classifier_score(image_paths):
    classifier = uciclassifier.WeatherClassifier()
    class_weights = {"Day": 1.00, "Night": 1.25, "Fog": 1.30}
    total_weighted_score = 0.0
    valid_image_count = 0
    for image_path in image_paths:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                label = classifier.predict_image(image_path=image_path).strip()
                if label not in class_weights:
                    continue
                total_weighted_score += class_weights[label]
                valid_image_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return total_weighted_score / valid_image_count if valid_image_count > 0 else 0.0


# === Allowed vocabulary ===
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
        pred_graph.parse(data=predicted_ttl, format='turtle')
    except Exception:
        return 0.0, 0.0, 0.0
    RDF_TYPE = str(RDF.type)
    pred_types = {str(o) for s, p, o in pred_graph if str(p) == RDF_TYPE}
    gt_types   = {str(o) for s, p, o in gt_graph  if str(p) == RDF_TYPE}
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
        pred_graph.parse(data=predicted_ttl, format='turtle')
    except Exception:
        return None
    total = hallucinated = 0
    RDF_TYPE = str(RDF.type)
    for s, p, o in pred_graph:
        p_str = str(p)
        total += 1
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


# === Resume logic ===
def completed_loops(vehicle, scenario, weather, batch_size, attempt=None):
    max_loop = 0
    search_dirs = [CORNERCASE_DIR, METRICS_DIR]
    for d in search_dirs:
        for fpath in glob.glob(os.path.join(d, "_metrics_*.jsonl")):
            try:
                with open(fpath) as f:
                    for line in f:
                        r = json.loads(line.strip())
                        if (r.get("vehicle") == vehicle and
                                r.get("scenario") == scenario and
                                r.get("weather") == weather and
                                r.get("batch_size") == batch_size and
                                r.get("model") == MODEL_ID and
                                (attempt is None or r.get("attempt") == attempt)):
                            max_loop = max(max_loop, r.get("loop", 0))
            except Exception:
                pass
    return max_loop

def get_attempt_number(vehicle):
    max_attempt = 0
    search_dirs = [CORNERCASE_DIR, METRICS_DIR]
    for d in search_dirs:
        for fpath in glob.glob(os.path.join(d, "_metrics_*.jsonl")):
            try:
                with open(fpath) as f:
                    for line in f:
                        r = json.loads(line.strip())
                        if r.get("vehicle") == vehicle and r.get("model") == MODEL_ID:
                            max_attempt = max(max_attempt, r.get("attempt", 0))
            except Exception:
                pass
    return max_attempt + 1

ATTEMPT = get_attempt_number(VEHICLE)


# === SmolVLM inference ===
def get_triples_from_llm(image_paths, prompt):
    rgb_image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not rgb_image_paths:
        return "", {"batch_size": 0, "latency_s": 0, "cpu_time_s": 0,
                    "ram_peak_mb": 0, "ram_delta_mb": 0, "input_bytes": 0,
                    "output_bytes": 0, "prompt_tokens": None,
                    "completion_tokens": None, "success": False,
                    "error": "no RGB images"}

    image_path = rgb_image_paths[0]
    _proc = psutil.Process(os.getpid())
    _cpu_start = time.process_time()
    _ram_before_mb = _proc.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    raw_output = ""
    success = True
    error = None
    prompt_tokens = None
    completion_tokens = None
    try:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "Output only valid Turtle RDF. No prose. No explanation."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        prompt_text = smolvlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = smolvlm_processor(text=prompt_text, images=[image], return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = smolvlm_model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generated_ids = output_ids[:, prompt_tokens:]
        completion_tokens = len(generated_ids[0])
        raw_output = smolvlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"

    latency_s    = time.time() - t0
    cpu_time_s   = round(time.process_time() - _cpu_start, 4)
    ram_peak_mb  = round(_proc.memory_info().rss / (1024 * 1024), 2)
    ram_delta_mb = round(ram_peak_mb - _ram_before_mb, 2)
    input_bytes  = os.path.getsize(image_path)
    output_bytes = len(raw_output.encode("utf-8")) if raw_output else 0

    call_metrics = {
        "batch_size":        1,
        "latency_s":         latency_s,
        "cpu_time_s":        cpu_time_s,
        "ram_peak_mb":       ram_peak_mb,
        "ram_delta_mb":      ram_delta_mb,
        "input_bytes":       input_bytes,
        "output_bytes":      output_bytes,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "success":           success,
        "error":             error,
    }

    print("=== RAW TTL ===")
    print(raw_output)

    if not success:
        return "", call_metrics

    def _extract(text):
        if "```turtle" in text:
            m = re.search(r'```turtle\s*(.*?)```', text, re.DOTALL)
            text = m.group(1).strip() if m else text
        elif "```" in text:
            m = re.search(r'```\s*(.*?)```', text, re.DOTALL)
            text = m.group(1).strip() if m else text
        text = text.replace("ex/", "ex:")
        m = re.search(r'\bex:', text)
        if m:
            text = text[m.start():]
        kept = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                kept.append(line)
            elif any(tok in s for tok in ["ex:", "avcco:", "prov:", "xsd:", ";", " .", "^^", "@prefix"]):
                kept.append(line)
        text = "\n".join(kept).strip()
        lines = text.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].strip()
            if s:
                if s.endswith(";"):
                    lines[i] = lines[i].rstrip()[:-1].rstrip() + " ."
                elif not s.endswith(".") and (":" in s or '"' in s):
                    lines[i] = lines[i].rstrip() + " ."
                break
        return "\n".join(lines).strip()

    return _extract(raw_output), call_metrics


# === Paths & prompt ===
ttl_path = os.path.join(CORNERCASE_DIR, "avcc_with_reasoning_no_shacl.ttl")
SCENARIOS = ["spawn_police_car_chase_1", "spawn_police_car_chase_2"]

prefixes = """@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

prompt = """\
Look at this image. Output ONLY valid Turtle RDF triples. No explanation.

If a vehicle clearly blocks another vehicle, complete all lines.
If not, output only the first line.

ex:vehicleEgo_activity a prov:Activity .
ex:veh1 a avcco:Vehicle .
ex:veh2 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh2 ;
    avcco:hasOccludedEntity ex:veh1 ;
    prov:wasGeneratedBy ex:vehicleEgo_activity ;
    avcco:hasConfidenceScore "0.85"^^xsd:float .
ex:cc1 a avcco:SensorBlindSpotCase ;
    avcco:hasActor ex:veh1 ;
    avcco:hasObstacle ex:veh2 ;
    avcco:hasTriggerEvent ex:occ1 ;
    prov:wasGeneratedBy ex:vehicleEgo_activity ;
    avcco:hasConfidenceScore "0.80"^^xsd:float ."""

# === Main loop ===
_t_run_start = time.time()

for scenario in SCENARIOS:
    scenario_folder = os.path.join(CORNERCASE_DIR, scenario)
    if not os.path.isdir(scenario_folder):
        print(f"Skipping {scenario}: folder not found.")
        continue

    for weather in sorted(os.listdir(scenario_folder)):
        main_graph = Graph()
        loop = completed_loops(VEHICLE, scenario, weather, BATCH_SIZE, ATTEMPT)

        weather_folder = os.path.join(scenario_folder, weather)
        if not os.path.isdir(weather_folder):
            continue

        print(f"Processing scenario: {scenario}, weather: {weather}, vehicle: {VEHICLE}")

        rgbs_folder = os.path.join(weather_folder, "ego_rgb")
        if not os.path.isdir(rgbs_folder):
            print(f"  Skipping: ego_rgb not found in {weather_folder}")
            continue

        rgb_images = sorted(
            glob.glob(os.path.join(rgbs_folder, "*.png")) +
            glob.glob(os.path.join(rgbs_folder, "*.jpg"))
        )

        _t_scenario = time.time()
        print(f"\n[START] vehicle={VEHICLE}  {scenario}/{weather}  images={len(rgb_images)}  resume_loop={loop}")

        while loop < len(rgb_images):
            print(f"Loop: {loop + 1}/{len(rgb_images)}")

            selected_images = [rgb_images[loop]]
            triples, call_metrics = get_triples_from_llm(selected_images, prompt)

            print("=== TRIPLES ===")
            print(triples)

            if triples and not triples.startswith("@prefix"):
                triples = prefixes + "\n" + triples

            avg_confidence_score = compute_avg_confidence_score(triples) if triples else None
            if avg_confidence_score is None:
                avg_confidence_score = 0.0
            avg_classifier_score = compute_avg_classifier_score(selected_images)
            if avg_classifier_score is None:
                avg_classifier_score = 1.0
            adjusted_score = round(avg_confidence_score / avg_classifier_score, 4) if avg_classifier_score else 0.0

            temp = Graph()
            rdf_parse_ok = False
            if call_metrics["success"] and triples:
                try:
                    temp.parse(data=triples, format='turtle')
                    rdf_parse_ok = True
                except Exception as parse_err:
                    print(f"RDF parse failed: {parse_err}")

            gt_graph = get_gt_graph(scenario, weather)
            if rdf_parse_ok:
                _, _, f1 = compute_f1_vs_gt(triples, gt_graph)
                hallucination_rate = compute_hallucination_rate(triples)
                compliance_rate    = compute_compliance_rate(triples)
            else:
                f1 = 0.0 if call_metrics["success"] else None
                hallucination_rate = None
                compliance_rate    = None

            if not call_metrics["success"]:
                status = "api_error"
            elif not rdf_parse_ok:
                status = "parse_error"
            else:
                status = "success"

            print(f"Confidence: {avg_confidence_score}, Classifier: {avg_classifier_score}, "
                  f"Adjusted: {adjusted_score}, F1: {f1}, "
                  f"Hallucination: {hallucination_rate}, Compliance: {compliance_rate}, "
                  f"Status: {status}")

            with open(metrics_file, "a") as mf:
                mf.write(json.dumps({
                    "scenario":           scenario,
                    "weather":            weather,
                    "loop":               loop + 1,
                    "attempt":            ATTEMPT,
                    "batch_size":         call_metrics["batch_size"],
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
                    "confidence":         avg_confidence_score,
                    "classifier":         avg_classifier_score,
                    "adjusted_score":     adjusted_score,
                    "f1":                 f1,
                    "hallucination_rate": hallucination_rate,
                    "compliance_rate":    compliance_rate,
                    "rdf_parse_ok":       rdf_parse_ok,
                    "success":            call_metrics["success"],
                    "error":              call_metrics["error"],
                    "ts":                 datetime.now(timezone.utc).isoformat(),
                }) + "\n")

            _done = loop + 1
            _is_last = _done >= len(rgb_images)
            if _done % PROGRESS_EVERY == 0 or _is_last:
                _pct = min(100.0, round(_done / len(rgb_images) * 100, 1))
                _tag = "[DONE]    " if _is_last else "[PROGRESS]"
                elapsed_total = round(time.time() - _t_run_start, 1)
                print(f"{_tag} vehicle={VEHICLE}  {scenario}/{weather}  "
                      f"loop={_done}/{len(rgb_images)} ({_pct}%)  status={status}  "
                      f"f1={f1}  elapsed={round(time.time()-_t_scenario,1)}s  total={elapsed_total}s")

            if not rdf_parse_ok:
                loop += 1
                continue

            main_graph = main_graph + temp
            print(f"main graph has {len(main_graph)} triples.")

            loop_output_path = os.path.join(CORNERCASE_DIR, "outputs_new", "smolvlm", scenario, weather, str(loop + 1))
            os.makedirs(loop_output_path, exist_ok=True)

            loop_output_file = os.path.join(loop_output_path, f"vehicle_{VEHICLE}_observations_loop.ttl")
            try:
                temp.serialize(destination=loop_output_file, format='turtle')
                print(f"Loop graph with {len(temp)} triples saved to {loop_output_file}.")
            except Exception as ser_err:
                print(f"[WARN] Loop serialize failed: {ser_err}")

            main_output_file = os.path.join(loop_output_path, f"vehicle_{VEHICLE}_observations.ttl")
            try:
                main_graph.serialize(destination=main_output_file, format='turtle')
                print(f"Main graph with {len(main_graph)} triples saved to {main_output_file}.")
            except Exception as ser_err:
                print(f"[WARN] Main serialize failed: {ser_err}")

            print(f"Adjusted score: {adjusted_score}. Continuing to next loop.")
            loop += 1

total_elapsed = round(time.time() - _t_run_start, 1)
print(f"\n[COMPLETE] vehicle={VEHICLE}  total_elapsed={total_elapsed}s  ({round(total_elapsed/3600,2)}h)")
print(f"Metrics written to: {metrics_file}")
