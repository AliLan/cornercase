"""
Smoke test for Qwen3-VL-2B-Instruct inference pipeline.
Runs one image through the full stack and reports:
  - model load time
  - inference latency
  - raw output
  - RDF parse result
  - triple count
  - hallucination rate
"""

import os
import sys
import re
import time
import psutil
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from rdflib import Graph, RDF

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CORNERCASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, CORNERCASE_DIR)

# Use one image from dataset
TEST_IMAGE = os.path.join(
    CORNERCASE_DIR, "dataset",
    "bus_obscuring_car_second_car", "day", "bus_rgb", "200060.png"
)

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

prefixes = """\
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
"""

prompt = """\
Output Turtle RDF only.

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
ex:vehicleA_activity a prov:Activity .

Example format only. Do not copy it unless supported by the image:
ex:vehicleA_activity a prov:Activity .
ex:veh1 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh1 ;
    avcco:hasOccludedEntity ex:veh2 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "0.90"^^xsd:float .
"""

def sep(title=""):
    print(f"\n{'─'*60}" + (f"  {title}" if title else ""))

def run_smoke_test():
    print("=" * 60)
    print("  Qwen2B Smoke Test")
    print("=" * 60)

    # ── Check image ───────────────────────────────────────────────
    sep("1. Image check")
    if not os.path.exists(TEST_IMAGE):
        print(f"  ERROR: test image not found: {TEST_IMAGE}")
        sys.exit(1)
    img = Image.open(TEST_IMAGE)
    print(f"  Path : {TEST_IMAGE}")
    print(f"  Size : {img.size}  Mode: {img.mode}")

    # ── Load model ────────────────────────────────────────────────
    sep("2. Model loading")
    _proc = psutil.Process(os.getpid())
    ram_before = _proc.memory_info().rss / (1024 * 1024)
    t_load = time.time()

    print("  Loading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True,
    )
    print("  Loading Qwen3VLForConditionalGeneration...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    device = next(model.parameters()).device

    load_time = round(time.time() - t_load, 2)
    ram_after = _proc.memory_info().rss / (1024 * 1024)
    print(f"  Device      : {device}")
    print(f"  Load time   : {load_time}s")
    print(f"  RAM delta   : +{round(ram_after - ram_before, 1)} MB  (total: {round(ram_after, 1)} MB)")

    # ── Inference ─────────────────────────────────────────────────
    sep("3. Inference")
    image = Image.open(TEST_IMAGE).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}")

    cpu_start = time.process_time()
    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
        )
    latency    = round(time.time() - t0, 3)
    cpu_time   = round(time.process_time() - cpu_start, 4)
    ram_peak   = round(_proc.memory_info().rss / (1024 * 1024), 1)
    output_len = generated_ids.shape[1] - input_len

    raw_output = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

    print(f"  Latency     : {latency}s")
    print(f"  CPU time    : {cpu_time}s")
    print(f"  RAM peak    : {ram_peak} MB")
    print(f"  Output tokens: {output_len}")

    # ── Raw output ────────────────────────────────────────────────
    sep("4. Raw model output")
    print(raw_output)

    # ── Extract Turtle ────────────────────────────────────────────
    sep("5. Turtle extraction")
    if "```turtle" in raw_output:
        m = re.search(r"```turtle(.+?)```", raw_output, re.DOTALL)
        triples = m.group(1).strip() if m else raw_output.strip()
        print("  Extracted from ```turtle block")
    elif "```" in raw_output:
        m = re.search(r"```(.+?)```", raw_output, re.DOTALL)
        triples = m.group(1).strip() if m else raw_output.strip()
        print("  Extracted from ``` block")
    else:
        triples = raw_output.strip()
        print("  Using raw output directly")

    if triples and not triples.startswith("@prefix"):
        triples = prefixes + "\n" + triples

    # ── RDF parse ─────────────────────────────────────────────────
    sep("6. RDF parse")
    g = Graph()
    try:
        g.parse(data=triples, format="turtle")
        print(f"  PASS — {len(g)} triples parsed")
        rdf_ok = True
    except Exception as e:
        print(f"  FAIL — {e}")
        rdf_ok = False

    # ── Hallucination check ───────────────────────────────────────
    if rdf_ok:
        sep("7. Hallucination check")
        total = hallucinated = 0
        RDF_TYPE = str(RDF.type)
        for s, p, o in g:
            total += 1
            p_str = str(p)
            if p_str == RDF_TYPE:
                if str(o) not in ALLOWED_CLASSES:
                    hallucinated += 1
                    print(f"  [HALLUCINATED type] {str(o).split('#')[-1]}")
            else:
                if p_str not in ALLOWED_PROPERTIES:
                    hallucinated += 1
                    print(f"  [HALLUCINATED prop] {str(p).split('#')[-1]}")
        hall_rate = round(hallucinated / total, 4) if total > 0 else 0.0
        print(f"  Total triples    : {total}")
        print(f"  Hallucinated     : {hallucinated}")
        print(f"  Hallucination rate: {hall_rate:.2%}")
        print(f"  Compliance rate  : {(1-hall_rate):.2%}")

    # ── Summary ───────────────────────────────────────────────────
    sep()
    print("\n  SMOKE TEST SUMMARY")
    print(f"  {'Image':<22}: {os.path.basename(TEST_IMAGE)}")
    print(f"  {'Model load time':<22}: {load_time}s")
    print(f"  {'Inference latency':<22}: {latency}s")
    print(f"  {'RDF parse':<22}: {'PASS' if rdf_ok else 'FAIL'}")
    if rdf_ok:
        print(f"  {'Triples generated':<22}: {len(g)}")
        print(f"  {'Hallucination rate':<22}: {hall_rate:.2%}")
    print()
    print("  " + ("✓ Smoke test PASSED" if rdf_ok else "✗ Smoke test FAILED — model output not valid RDF"))
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
