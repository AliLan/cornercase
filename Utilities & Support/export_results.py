"""
Reads all _metrics_*.jsonl files and exports two aggregated CSVs to results/.

Performance Metrics CSV  — latency, CPU, RAM, success rate per (scenario, model, batch_size, attempt)
Evaluation Metrics CSV   — F1, hallucination, compliance, call rates per (scenario, model, batch_size, attempt)
"""

import csv, glob, json, os
from collections import defaultdict

RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "results")
QWEN_METRICS_DIR  = os.path.join(os.path.dirname(__file__), "Qwen 2.5 VL 7B metrics")
MOON_METRICS_DIR  = os.path.join(os.path.dirname(__file__), "metrics_new", "moondream")
GLM_METRICS_DIR   = os.path.join(os.path.dirname(__file__), "metrics_new", "glm")
SMOL_METRICS_DIR  = os.path.join(os.path.dirname(__file__), "metrics_new", "smolvlm")

JSONL_GLOBS = [
    os.path.join(QWEN_METRICS_DIR, "_metrics_*.jsonl"),
    os.path.join(os.path.dirname(__file__), "_metrics_*.jsonl"),  # root fallback
    os.path.join(MOON_METRICS_DIR, "_metrics_*.jsonl"),
    os.path.join(GLM_METRICS_DIR,  "_metrics_*.jsonl"),
    os.path.join(SMOL_METRICS_DIR, "_metrics_*.jsonl"),
]

EXTRA_FILES = [
    os.path.join(QWEN_METRICS_DIR, "_smoke_test_metrics.jsonl"),
    os.path.join(QWEN_METRICS_DIR, "_demo_metrics.jsonl"),
    os.path.join(os.path.dirname(__file__), "_smoke_test_metrics.jsonl"),
    os.path.join(os.path.dirname(__file__), "_demo_metrics.jsonl"),
]

SKIP_STATUSES = {"smoke_test"}


def load_records():
    paths = []
    for pattern in JSONL_GLOBS:
        paths += glob.glob(pattern)
    paths = list(dict.fromkeys(paths))  # deduplicate while preserving order
    for p in EXTRA_FILES:
        if os.path.exists(p) and p not in paths:
            paths.append(p)

    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("status") not in SKIP_STATUSES:
                            records.append(r)
                    except json.JSONDecodeError:
                        pass
    return records


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        key = (r.get("scenario", ""), r.get("model", ""),
               r.get("batch_size", ""), r.get("attempt", ""))
        groups[key].append(r)
    return groups


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def export_performance(groups, out_path):
    fieldnames = [
        "Scenario", "Model", "Attempt", "Batch Size",
        "Response Time (s)", "Avg Latency (s)", "CPU Time (s)",
        "RAM Usage (MB)", "Success Rate",
    ]
    rows = []
    for (scenario, model, batch_size, attempt), recs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]), str(x[0][3]))):
        total   = len(recs)
        success = sum(1 for r in recs if r.get("status") == "success" or r.get("success") is True)
        rows.append({
            "Scenario":          scenario,
            "Model":             model,
            "Attempt":           attempt,
            "Batch Size":        batch_size,
            "Response Time (s)": _mean([r.get("latency_s") for r in recs]),
            "Avg Latency (s)":   _mean([r.get("latency_s") for r in recs]),
            "CPU Time (s)":      _mean([r.get("cpu_time_s") for r in recs]),
            "RAM Usage (MB)":    _mean([r.get("ram_peak_mb") for r in recs]),
            "Success Rate":      round(success / total, 4) if total else None,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {out_path}")


def export_evaluation(groups, out_path):
    fieldnames = [
        "Scenario", "Model", "Attempt", "Batch Size",
        "Average F1 Score", "Hallucination Rate", "Ontology Compliance",
        "Success Calls Rate", "Failed Calls Rate",
    ]
    rows = []
    for (scenario, model, batch_size, attempt), recs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]), str(x[0][3]))):
        total   = len(recs)
        success = sum(1 for r in recs if r.get("status") == "success" or r.get("success") is True)
        rows.append({
            "Scenario":           scenario,
            "Model":              model,
            "Attempt":            attempt,
            "Batch Size":         batch_size,
            "Average F1 Score":   _mean([r.get("f1")               for r in recs]),
            "Hallucination Rate": _mean([r.get("hallucination_rate") for r in recs]),
            "Ontology Compliance":_mean([r.get("compliance_rate")   for r in recs]),
            "Success Calls Rate": round(success / total, 4) if total else None,
            "Failed Calls Rate":  round((total - success) / total, 4) if total else None,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    records = load_records()
    print(f"Loaded {len(records)} records from JSONL files.")

    groups = aggregate(records)
    print(f"Aggregated into {len(groups)} (scenario, model, batch_size, attempt) groups.\n")

    export_performance(groups, os.path.join(RESULTS_DIR, "performance_metrics.csv"))
    export_evaluation(groups,  os.path.join(RESULTS_DIR, "evaluation_metrics.csv"))

    print("\nDone.")
