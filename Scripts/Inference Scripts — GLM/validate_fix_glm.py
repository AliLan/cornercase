"""
Validation script: GLM-OCR fix
Fixes:
  1. Remove all @prefix declarations from prompt body
     (GLM-OCR reads them as XSD namespace declarations and outputs XML)
  2. Simplify to plain-English instructions with an exact output template
  3. Post-process: strip code fences, prepend prefixes before rdflib.parse()
Tests on 5 images from bus_obscuring_car_second_car/day/A/rgb/.
"""
import os, re, glob, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from rdflib import Graph

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "zai-org/GLM-OCR"

PREFIXES = """\
@prefix avcco: <http://cornercase.org/avcco#> .
@prefix ex:    <http://cornercase.org/instances#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

# FIX: no @prefix lines anywhere in the prompt.
# Plain English instructions + copy-paste Turtle template only.
PROMPT = """\
Output ONLY Turtle RDF lines. No XML. No JSON. No explanation.

Does any vehicle in this image clearly block another vehicle from view?

Always output this first line:
ex:vehicleA_activity a prov:Activity .

Only if a vehicle clearly occludes another, also output these lines (adjust confidence):
ex:veh1 a avcco:Vehicle .
ex:veh2 a avcco:Vehicle .
ex:occ1 a avcco:OcclusionEvent ;
    avcco:hasOccluder ex:veh2 ;
    avcco:hasOccludedEntity ex:veh1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "0.90"^^xsd:float .
ex:cc1 a avcco:SensorBlindSpotCase ;
    avcco:hasActor ex:veh1 ;
    avcco:hasObstacle ex:veh2 ;
    avcco:hasTriggerEvent ex:occ1 ;
    prov:wasGeneratedBy ex:vehicleA_activity ;
    avcco:hasConfidenceScore "0.85"^^xsd:float ."""


def extract_ttl(raw):
    text = raw.strip()
    # Strip code fences
    if "```turtle" in text:
        m = re.search(r"```turtle\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    elif "```" in text:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        text = m.group(1).strip() if m else text
    text = text.replace("ex/", "ex:")
    # Fix common GLM typos in namespace prefix
    text = re.sub(r"\bavco:", "avcco:", text)
    # Anchor on first ex: token — drop prose preamble
    m = re.search(r"\bex:", text)
    if m:
        text = text[m.start():]
    # Keep only Turtle-looking lines
    turtle_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            turtle_lines.append(line)
        elif s.startswith("#") or s.startswith("@"):
            turtle_lines.append(line)
        elif any(tok in s for tok in ["ex:", "avcco:", "prov:", "xsd:", ";", " .", "^^"]):
            turtle_lines.append(line)
    text = "\n".join(turtle_lines).strip()
    # Close truncated triples: last non-empty line missing terminator
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if s:
            if s.endswith(";"):
                # Trailing semicolon — drop the incomplete predicate, close the block
                lines[i] = lines[i].rstrip()[:-1].rstrip() + " ."
            elif not s.endswith(".") and (":" in s or '"' in s):
                # Predicate-object pair without terminator
                lines[i] = lines[i].rstrip() + " ."
            break
    text = "\n".join(lines).strip()
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


print(f"Loading {MODEL_ID}...")
t0 = time.time()
glm_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
glm_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Loaded in {time.time()-t0:.1f}s\n")

images = sorted(glob.glob(
    os.path.join(CORNERCASE_DIR, "CARLA_DATASET_MULTI_AGENTS",
                 "bus_obscuring_car_second_car", "day", "A", "rgb", "*.png")
))[:5]

print(f"Testing on {len(images)} images\n")
print("=" * 70)

results = []
for i, img_path in enumerate(images):
    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = glm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = glm_processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(glm_model.device) for k, v in inputs.items()}
    n_prompt = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = glm_model.generate(
            **inputs, max_new_tokens=768,
            temperature=0.7, repetition_penalty=1.2,
            do_sample=True, top_p=0.9,
        )
    raw = glm_processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0]
    elapsed = time.time() - t0

    ttl = extract_ttl(raw)
    ok, info = try_parse(ttl)

    status = "PASS" if ok else "FAIL"
    results.append(ok)

    print(f"[{i+1}/5] {os.path.basename(img_path)}  {elapsed:.1f}s  → {status}")
    print(f"  raw ({len(raw)} chars): {raw[:200].replace(chr(10),' ')!r}")
    if ok:
        print(f"  parsed: {info} triples")
    else:
        print(f"  parse error: {str(info)[:120]}")
    print()

print("=" * 70)
print(f"Result: {sum(results)}/{len(results)} parsed successfully")
