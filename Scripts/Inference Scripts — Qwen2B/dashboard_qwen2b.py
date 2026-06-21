"""
Real-time dashboard for Qwen2B inference runs.
Run:  python3 dashboard_qwen2b.py
Refreshes every 10 seconds.
"""

import os
import glob
import json
import time
from datetime import datetime

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CORNERCASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
METRICS_DIR    = os.path.join(CORNERCASE_DIR, "metrics", "qwen2b")

VEHICLES  = ["ego", "flee", "police", "A", "B", "C"]
SCENARIOS = [
    "spawn_police_car_chase_1",
    "spawn_police_car_chase_2",
    "bus_obscuring_car_second_car",
    "bus_stop_near_playground",
]
WEATHERS  = ["day", "night", "fog"]

# Expected total images per vehicle/scenario/weather
# (approximate — read from actual folders if needed)
EXPECTED = {
    ("ego",    "spawn_police_car_chase_1"): 213,
    ("ego",    "spawn_police_car_chase_2"): 211,
    ("flee",   "spawn_police_car_chase_1"): 213,
    ("flee",   "spawn_police_car_chase_2"): 211,
    ("police", "spawn_police_car_chase_1"): 213,
    ("police", "spawn_police_car_chase_2"): 211,
    ("A",      "bus_obscuring_car_second_car"): 49,
    ("A",      "bus_stop_near_playground"):     78,
    ("B",      "bus_obscuring_car_second_car"): 49,
    ("B",      "bus_stop_near_playground"):     78,
    ("C",      "bus_obscuring_car_second_car"): 49,
}

def load_records():
    records = []
    for fpath in glob.glob(os.path.join(METRICS_DIR, "_metrics_*.jsonl")):
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            pass
    return records

def render(records):
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(records)
    success   = sum(1 for r in records if r.get("status") == "success")
    parse_err = sum(1 for r in records if r.get("status") == "parse_error")
    api_err   = sum(1 for r in records if r.get("status") == "api_error")

    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    f1s       = [r["f1"] for r in records if r.get("f1") is not None]
    halls     = [r["hallucination_rate"] for r in records if r.get("hallucination_rate") is not None]

    avg_lat  = sum(latencies) / len(latencies) if latencies else 0
    avg_f1   = sum(f1s) / len(f1s) if f1s else None
    avg_hall = sum(halls) / len(halls) if halls else None
    rdf_rate = success / (success + parse_err) if (success + parse_err) > 0 else 0

    print("=" * 72)
    print(f"  Qwen2B (Qwen3-VL-2B) Dashboard        {now}")
    print("=" * 72)
    print(f"  Total calls : {total:>6}    Success : {success:>6}    Parse Err : {parse_err:>5}")
    print(f"  API Err     : {api_err:>6}    RDF Rate: {rdf_rate*100:>5.1f}%   Avg Latency: {avg_lat:>6.2f}s")
    f1_str   = f"{avg_f1*100:.2f}%"   if avg_f1   is not None else "N/A"
    hall_str = f"{avg_hall*100:.3f}%" if avg_hall  is not None else "N/A"
    print(f"  Avg F1      : {f1_str:>7}    Avg Hallucination: {hall_str}")
    print()

    # Per-vehicle breakdown
    print(f"  {'Vehicle':<8} {'Scenario':<33} {'W':<5} {'Done':>5} {'Tot':>5} {'%':>5}  {'Status mix'}")
    print("  " + "-" * 68)
    for vehicle in VEHICLES:
        v_recs = [r for r in records if r.get("vehicle") == vehicle]
        if not v_recs:
            print(f"  {vehicle:<8}  (no data yet)")
            continue
        for scenario in SCENARIOS:
            s_recs = [r for r in v_recs if r.get("scenario") == scenario]
            if not s_recs:
                continue
            for weather in WEATHERS:
                w_recs = [r for r in s_recs if r.get("weather") == weather]
                if not w_recs:
                    continue
                done    = max(r.get("loop", 0) for r in w_recs)
                total_e = EXPECTED.get((vehicle, scenario), "?")
                pct     = f"{done/total_e*100:.0f}%" if isinstance(total_e, int) and total_e > 0 else "?"
                ok      = sum(1 for r in w_recs if r.get("status") == "success")
                pe      = sum(1 for r in w_recs if r.get("status") == "parse_error")
                sc_short = scenario.replace("spawn_police_car_chase_", "chase_").replace("bus_obscuring_car_second_car", "bus_car").replace("bus_stop_near_playground", "bus_stop")
                print(f"  {vehicle:<8}  {sc_short:<31} {weather:<5} {done:>5} {str(total_e):>5} {pct:>5}  "
                      f"ok={ok} pe={pe}")

    # Recent 5 records
    print()
    print(f"  Recent activity:")
    for r in sorted(records, key=lambda x: x.get("ts",""))[-5:]:
        ts   = r.get("ts","")[-8:-1] if r.get("ts") else "?"
        sc   = (r.get("scenario","") or "")[-12:]
        v    = r.get("vehicle","?")
        loop = r.get("loop","?")
        st   = r.get("status","?")
        lat  = r.get("latency_s", 0)
        f1   = r.get("f1")
        f1s_ = f"{f1:.3f}" if f1 is not None else "N/A"
        print(f"    {ts}  v={v:<6} {sc:<14} loop={loop:<4} {st:<12} lat={lat:.1f}s  f1={f1s_}")

    print()
    print(f"  Metrics dir : {METRICS_DIR}")
    print(f"  Refreshing every 10s  (Ctrl+C to stop)")
    print("=" * 72)

def main():
    print("Starting Qwen2B dashboard...")
    try:
        while True:
            records = load_records()
            render(records)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

if __name__ == "__main__":
    main()
