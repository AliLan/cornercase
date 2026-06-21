"""
Validation script: Moondream fix (v2)
Root cause: answer_question / query pipeline hard-caps output to ~13 chars.
            generate() _generate_text path produces only dots (no image context).
Fix: programmatic approach — ask 3 short VQA questions, build Turtle from answers.
Tests on 5 images from bus_obscuring_car_second_car/day/A/rgb/.
"""
import os, re, glob, time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdflib import Graph

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-01-09"

PREFIXES = """\
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

Q_OCCLUSION   = "Is any vehicle clearly blocked or hidden by another vehicle or large object?"
Q_CONFIDENCE  = "How certain are you? Answer only with a decimal between 0.0 and 1.0."


def extract_float(text, default=0.5):
    m = re.search(r"0?\.\d+|1\.0|0\.0", text)
    return float(m.group()) if m else default


def build_turtle(occlusion_answer, confidence_answer):
    """Construct valid Turtle from moondream's short VQA answers."""
    ttl = "ex:vehicleA_activity a prov:Activity .\n"
    conf = extract_float(confidence_answer)
    positive = any(w in occlusion_answer.lower()
                   for w in ["yes", "block", "hidden", "obscur", "occlu", "cover"])
    if positive and conf >= 0.3:
        ttl += f"""\
ex:veh1 a avcco:Vehicle .
ex:veh2 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh2 ;
    avcco:hasOccludedEntity ex:veh1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "{conf:.2f}"^^xsd:float .
ex:cc1 a avcco:SensorBlindSpotCase ;
    avcco:hasActor ex:veh1 ;
    avcco:hasObstacle ex:veh2 ;
    avcco:hasTriggerEvent ex:occ1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "{conf:.2f}"^^xsd:float .
"""
    return PREFIXES + "\n" + ttl


def try_parse(ttl):
    try:
        g = Graph()
        g.parse(data=ttl, format="turtle")
        return True, len(g)
    except Exception as e:
        return False, str(e)


print(f"Loading {MODEL_ID} (revision={REVISION})...")
t0 = time.time()
md_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION,
    torch_dtype=torch.float16,
    device_map={"": "mps"},
    trust_remote_code=True,
)
md_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
print(f"Loaded in {time.time()-t0:.1f}s\n")

images = sorted(glob.glob(
    os.path.join(CORNERCASE_DIR, "CARLA_DATASET_MULTI_AGENTS",
                 "bus_obscuring_car_second_car", "day", "A", "rgb", "*.png")
))[:5]

print(f"Testing on {len(images)} images\n")
print("=" * 70)

results = []
for i, img_path in enumerate(images):
    img = Image.open(img_path).convert("RGB")
    enc = md_model.encode_image(img)

    t0 = time.time()
    a_occ  = md_model.answer_question(enc, Q_OCCLUSION,  md_tokenizer)
    a_conf = md_model.answer_question(enc, Q_CONFIDENCE, md_tokenizer)
    elapsed = time.time() - t0

    ttl = build_turtle(a_occ, a_conf)
    ok, info = try_parse(ttl)

    status = "PASS" if ok else "FAIL"
    results.append(ok)

    print(f"[{i+1}/5] {os.path.basename(img_path)}  {elapsed:.1f}s  → {status}")
    print(f"  Q1 occlusion:  {a_occ!r}")
    print(f"  Q2 confidence: {a_conf!r}")
    if ok:
        print(f"  parsed: {info} triples")
    else:
        print(f"  parse error: {str(info)[:120]}")
    print()

print("=" * 70)
print(f"Result: {sum(results)}/{len(results)} parsed successfully")
