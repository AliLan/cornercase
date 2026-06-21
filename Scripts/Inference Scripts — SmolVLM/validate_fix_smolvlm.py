"""
Validation script: SmolVLM-256M fix (v2)
Root cause: model echoes instructions before Turtle; also generates incomplete triples.
Fixes:
  1. Fill-in-the-blank format: show the first line of Turtle, let model complete it
  2. Stripped prompt — no @prefix in body, no verbose instructions
  3. Extractor anchors on first 'ex:' token, discards any preamble
  4. Prepend prefixes before parsing
Tests on 5 images from bus_obscuring_car_second_car/day/A/rgb/.
"""
import os, re, glob, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from rdflib import Graph

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

PREFIXES = """\
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

# Fill-in-the-blank: model sees the first anchor line and continues.
# No @prefix in prompt body, minimal prose.
PROMPT = """\
Look at this image. Output ONLY valid Turtle RDF triples. No explanation.

If a vehicle clearly blocks another vehicle, complete all lines.
If not, output only the first line.

ex:vehicleA_activity a prov:Activity .
ex:veh1 a avcco:Vehicle .
ex:veh2 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh2 ;
    avcco:hasOccludedEntity ex:veh1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "0.85"^^xsd:float .
ex:cc1 a avcco:SensorBlindSpotCase ;
    avcco:hasActor ex:veh1 ;
    avcco:hasObstacle ex:veh2 ;
    avcco:hasTriggerEvent ex:occ1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "0.80"^^xsd:float ."""


def extract_ttl(raw):
    text = raw.strip()
    # Strip code fences
    if "```turtle" in text:
        m = re.search(r"```turtle\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    elif "```" in text:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    # Anchor on first ex: token — discard any echoed instructions before it
    m = re.search(r"\bex:", text)
    if m:
        text = text[m.start():]
    text = text.replace("ex/", "ex:")
    # Drop lines that are clearly prose (no Turtle tokens)
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            kept.append(line)
        elif any(tok in s for tok in ["ex:", "avcco:", "prov:", "xsd:", ";", " .", "^^", "@prefix"]):
            kept.append(line)
    text = "\n".join(kept).strip()
    if text and not text.startswith("@prefix"):
        text = PREFIXES + "\n" + text
    return text


def try_parse(ttl):
    try:
        g = Graph()
        g.parse(data=ttl, format="turtle")
        return True, len(g)
    except Exception as e:
        return False, str(e)


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_ID} on {device.upper()}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, _attn_implementation="eager",
).to(device)
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
    messages = [
        {"role": "system",
         "content": "Output only valid Turtle RDF. No prose. No explanation."},
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[img], return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    n_prompt = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0]
    elapsed = time.time() - t0

    ttl = extract_ttl(raw)
    ok, info = try_parse(ttl)

    status = "PASS" if ok else "FAIL"
    results.append(ok)

    print(f"[{i+1}/5] {os.path.basename(img_path)}  {elapsed:.1f}s  → {status}")
    print(f"  raw ({len(raw)} chars): {raw[:200].replace(chr(10), ' | ')!r}")
    print(f"  extracted TTL (first 3 lines): {chr(10).join(ttl.splitlines()[:3])!r}")
    if ok:
        print(f"  parsed: {info} triples")
    else:
        print(f"  parse error: {str(info)[:100]}")
    print()

print("=" * 70)
print(f"Result: {sum(results)}/{len(results)} parsed successfully")
