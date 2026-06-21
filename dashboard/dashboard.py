"""
Cornercase Benchmark Monitor
Live dashboard for vehicle_A/B/C_observation.py background tasks.
Run: python dashboard.py  →  open http://localhost:8050
"""

import glob
import json
import os
import time

import psutil
from dash import Dash, Input, Output, dcc, html

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEHICLES       = ["A", "B", "C"]
REFRESH_MS     = 3000  # auto-refresh interval

# ── Colour palette ────────────────────────────────────────────
COLORS = {
    "bg":       "#0f1117",
    "card":     "#1a1d27",
    "border":   "#2a2d3e",
    "running":  "#00d68f",
    "done":     "#598bff",
    "idle":     "#6b7280",
    "error":    "#ff4d4f",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
    "accent":   "#7c3aed",
}

app = Dash(__name__, title="Cornercase Monitor")
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', sans-serif", "color": COLORS["text"],
           "padding": "24px"},
    children=[
        # ── Header ─────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "28px"},
            children=[
                html.Div([
                    html.H1("Cornercase Benchmark Monitor",
                            style={"margin": 0, "fontSize": "22px", "fontWeight": 700}),
                    html.P("qwen2.5vl:7b  ·  Ollama local inference",
                           style={"margin": "4px 0 0", "color": COLORS["subtext"],
                                  "fontSize": "13px"}),
                ]),
                html.Div(id="last-refresh",
                         style={"color": COLORS["subtext"], "fontSize": "12px",
                                "textAlign": "right"}),
            ]
        ),

        # ── Vehicle cards ───────────────────────────────────────
        html.Div(id="vehicle-cards",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "16px", "marginBottom": "24px"}),

        # ── Global stats bar ────────────────────────────────────
        html.Div(id="global-stats",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "12px", "marginBottom": "24px"}),

        # ── Recent activity table ───────────────────────────────
        html.Div([
            html.H3("Recent Activity",
                    style={"fontSize": "14px", "fontWeight": 600,
                           "color": COLORS["subtext"], "marginBottom": "12px",
                           "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(id="activity-table"),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "padding": "20px", "border": f"1px solid {COLORS['border']}"}),

        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
    ]
)


# ── Data helpers ───────────────────────────────────────────────

def load_all_metrics():
    """Return dict {vehicle: [records]} from all _metrics_*.jsonl files."""
    data = {v: [] for v in VEHICLES}
    for fpath in glob.glob(os.path.join(CORNERCASE_DIR, "_metrics_*.jsonl")):
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    v = rec.get("vehicle")
                    if v in data:
                        data[v].append(rec)
        except Exception:
            pass
    return data


def is_process_running(vehicle):
    """True if vehicle_X_observation.py is in a running python process."""
    target = f"vehicle_{vehicle}_observation.py"
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any(target in arg for arg in (proc.info["cmdline"] or [])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def vehicle_stats(records, vehicle):
    if not records:
        return {"status": "idle", "calls": 0, "success": 0, "parse_errors": 0,
                "api_errors": 0, "avg_latency": None, "batch_sizes": "—",
                "total_images": 0, "scenario_summary": []}

    running = is_process_running(vehicle)
    status  = "running" if running else "done"

    success_recs  = [r for r in records if r.get("status") == "success"]
    parse_errors  = [r for r in records if r.get("status") == "parse_error"]
    api_errors    = [r for r in records if r.get("status") == "api_error"]
    # fall back to boolean for records written before status field existed
    if not success_recs and not parse_errors and not api_errors:
        success_recs = [r for r in records if r.get("success")]
        api_errors   = [r for r in records if not r.get("success")]

    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]

    # Batch sizes used
    batch_sizes = sorted({r.get("batch_size") for r in records if r.get("batch_size") is not None})
    batch_label = ", ".join(str(b) for b in batch_sizes) if batch_sizes else "—"

    # Per-scenario, per-weather max loop keyed by attempt
    scenario_map = {}  # (scenario, weather) → (max_loop, attempt)
    for r in records:
        key = (r.get("scenario", "?"), r.get("weather", "?"))
        cur_loop    = r.get("loop", 0)
        cur_attempt = r.get("attempt", "—")
        prev_loop, _ = scenario_map.get(key, (0, "—"))
        if cur_loop >= prev_loop:
            scenario_map[key] = (cur_loop, cur_attempt)
    scenario_summary = sorted(scenario_map.items())

    current_attempt = max((r.get("attempt", 0) for r in records), default="—")

    total_images = sum(
        r.get("batch_size", 0) for r in records if r.get("status") == "success"
    )

    return {
        "status":           status,
        "calls":            len(records),
        "success":          len(success_recs),
        "parse_errors":     len(parse_errors),
        "api_errors":       len(api_errors),
        "avg_latency":      sum(latencies) / len(latencies) if latencies else None,
        "batch_sizes":      batch_label,
        "total_images":     total_images,
        "scenario_summary": scenario_summary,
        "current_attempt":  current_attempt,
    }


# ── UI builders ───────────────────────────────────────────────

def status_dot(status):
    colors = {"running": COLORS["running"], "done": COLORS["done"], "idle": COLORS["idle"]}
    pulse  = status == "running"
    return html.Span(
        style={
            "display":       "inline-block",
            "width":         "10px", "height": "10px",
            "borderRadius":  "50%",
            "backgroundColor": colors.get(status, COLORS["idle"]),
            "marginRight":   "8px",
            "boxShadow":     f"0 0 6px {colors.get(status)}" if pulse else "none",
        }
    )


def stat_pill(label, value, color=None):
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "alignItems": "center", "padding": "6px 0",
               "borderBottom": f"1px solid {COLORS['border']}"},
        children=[
            html.Span(label, style={"color": COLORS["subtext"], "fontSize": "12px"}),
            html.Span(value, style={"fontSize": "13px", "fontWeight": 600,
                                    "color": color or COLORS["text"]}),
        ]
    )


def build_vehicle_card(vehicle, stats):
    status      = stats["status"]
    status_text = {"running": "Running", "done": "Completed", "idle": "Not started"}
    status_col  = {"running": COLORS["running"], "done": COLORS["done"], "idle": COLORS["idle"]}

    total   = stats["calls"]
    success_rate = f"{stats['success'] / total * 100:.0f}%" if total > 0 else "—"
    avg_lat = f"{stats['avg_latency']:.1f}s" if stats["avg_latency"] is not None else "—"

    # Build scenario/weather/loop rows
    scenario_rows = []
    for (sc, wt), (max_loop, attempt) in stats["scenario_summary"]:
        sc_short = sc.replace("bus_obscuring_car_second_car", "bus_obscuring")
        sc_short = sc_short.replace("bus_stop_near_playground", "bus_stop")
        scenario_rows.append(
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "padding": "4px 0", "fontSize": "11px",
                       "borderBottom": f"1px solid {COLORS['border']}"},
                children=[
                    html.Span(f"{sc_short} / {wt}",
                              style={"color": COLORS["subtext"]}),
                    html.Span(f"attempt {attempt}  ·  loop {max_loop}",
                              style={"color": COLORS["text"], "fontWeight": 600}),
                ]
            )
        )

    return html.Div(
        style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
               "padding": "20px", "border": f"1px solid {COLORS['border']}",
               "borderTop": f"3px solid {status_col[status]}"},
        children=[
            # Title row
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "marginBottom": "16px"},
                children=[
                    html.H2(f"Vehicle {vehicle}",
                            style={"margin": 0, "fontSize": "16px", "fontWeight": 700}),
                    html.Span(
                        [status_dot(status), status_text[status]],
                        style={"fontSize": "12px", "color": status_col[status],
                               "display": "flex", "alignItems": "center"}
                    ),
                ]
            ),
            # Scenario / weather / loop breakdown
            html.Div(
                style={"marginBottom": "10px"},
                children=(scenario_rows if scenario_rows else [
                    html.Span("No runs yet", style={"color": COLORS["subtext"],
                                                    "fontSize": "12px"})
                ])
            ),
            stat_pill("Current attempt",  str(stats["current_attempt"])),
            stat_pill("Batch sizes used", stats["batch_sizes"]),
            stat_pill("Total calls",      str(total)),
            stat_pill("✓ Succeeded",      str(stats["success"]),
                      COLORS["running"] if stats["success"] > 0 else COLORS["text"]),
            stat_pill("⚠ Parse errors",   str(stats["parse_errors"]),
                      "#f59e0b" if stats["parse_errors"] > 0 else COLORS["text"]),
            stat_pill("✗ API errors",     str(stats["api_errors"]),
                      COLORS["error"] if stats["api_errors"] > 0 else COLORS["text"]),
            stat_pill("Images processed", str(stats["total_images"])),
            stat_pill("Avg latency",      avg_lat),
        ]
    )


def build_global_card(label, value, sub=None, color=None):
    return html.Div(
        style={"backgroundColor": COLORS["card"], "borderRadius": "10px",
               "padding": "16px", "border": f"1px solid {COLORS['border']}"},
        children=[
            html.P(label, style={"margin": "0 0 4px", "fontSize": "11px",
                                  "textTransform": "uppercase", "letterSpacing": "0.05em",
                                  "color": COLORS["subtext"]}),
            html.H3(value, style={"margin": 0, "fontSize": "24px", "fontWeight": 700,
                                   "color": color or COLORS["text"]}),
            html.P(sub or "", style={"margin": "4px 0 0", "fontSize": "11px",
                                      "color": COLORS["subtext"]}),
        ]
    )


def build_activity_table(all_records):
    # Collect last 20 records across all vehicles, sorted by time
    combined = []
    for v, recs in all_records.items():
        for r in recs:
            combined.append((r.get("ts", ""), v, r))
    combined.sort(key=lambda x: x[0], reverse=True)
    recent = combined[:20]

    if not recent:
        return html.P("No activity yet.", style={"color": COLORS["subtext"]})

    header = html.Tr([
        html.Th(col, style={"padding": "8px 12px", "color": COLORS["subtext"],
                             "fontSize": "11px", "textTransform": "uppercase",
                             "letterSpacing": "0.05em", "fontWeight": 500,
                             "borderBottom": f"1px solid {COLORS['border']}"})
        for col in ["Vehicle", "Scenario", "Weather", "Attempt", "Loop", "Batch", "Latency", "Tokens In", "Tokens Out", "Status"]
    ])

    rows = []
    for ts, v, r in recent:
        status_val = r.get("status")
        # fall back for old records that only have the boolean field
        if not status_val:
            status_val = "success" if r.get("success") else "api_error"
        _status_display = {"success": "✓ ok", "parse_error": "⚠ parse",
                           "api_error": "✗ api"}.get(status_val, status_val)
        _status_color   = {"success": COLORS["running"], "parse_error": "#f59e0b",
                           "api_error": COLORS["error"]}.get(status_val, COLORS["subtext"])
        status_cell = html.Td(
            _status_display,
            style={"padding": "8px 12px", "fontSize": "12px",
                   "color": _status_color, "fontWeight": 700}
        )
        lat = r.get("latency_s")
        rows.append(html.Tr(
            style={"borderBottom": f"1px solid {COLORS['border']}"},
            children=[
                html.Td(f"Vehicle {v}", style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(r.get("scenario", "—").replace("_", " "),
                        style={"padding": "8px 12px", "fontSize": "12px",
                               "color": COLORS["subtext"], "maxWidth": "160px",
                               "overflow": "hidden", "textOverflow": "ellipsis",
                               "whiteSpace": "nowrap"}),
                html.Td(r.get("weather", "—"), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("attempt", "—")), style={"padding": "8px 12px", "fontSize": "13px",
                        "color": COLORS["accent"]}),
                html.Td(str(r.get("loop", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("batch_size", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(f"{lat:.1f}s" if lat else "—",
                        style={"padding": "8px 12px", "fontSize": "13px",
                               "color": COLORS["running"] if lat and lat < 90 else COLORS["error"]}),
                html.Td(str(r.get("prompt_tokens", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("completion_tokens", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                status_cell,
            ]
        ))

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[html.Thead(header), html.Tbody(rows)]
    )


# ── Callback ───────────────────────────────────────────────────

@app.callback(
    Output("vehicle-cards",  "children"),
    Output("global-stats",   "children"),
    Output("activity-table", "children"),
    Output("last-refresh",   "children"),
    Input("interval", "n_intervals"),
)
def refresh(_):
    all_metrics = load_all_metrics()

    # Vehicle cards
    cards = [
        build_vehicle_card(v, vehicle_stats(all_metrics[v], v))
        for v in VEHICLES
    ]

    # Global stats
    all_recs    = [r for recs in all_metrics.values() for r in recs]
    total_calls = len(all_recs)
    total_ok    = sum(1 for r in all_recs if r.get("success"))
    lats        = [r["latency_s"] for r in all_recs if r.get("latency_s") is not None]
    avg_lat     = sum(lats) / len(lats) if lats else 0
    running_v   = sum(1 for v in VEHICLES if is_process_running(v))

    global_cards = [
        build_global_card("Active Vehicles", str(running_v), "processes running",
                          COLORS["running"] if running_v > 0 else COLORS["idle"]),
        build_global_card("Total API Calls", str(total_calls), "across all vehicles"),
        build_global_card("Success Rate",
                          f"{total_ok / total_calls * 100:.0f}%" if total_calls else "—",
                          f"{total_ok} succeeded / {total_calls - total_ok} failed",
                          COLORS["running"] if total_calls > 0 else COLORS["idle"]),
        build_global_card("Avg Latency",
                          f"{avg_lat:.1f}s" if avg_lat else "—",
                          "per LLM call"),
    ]

    now = time.strftime("%H:%M:%S")
    refresh_label = [
        html.Span("⟳ ", style={"color": COLORS["running"]}),
        f"Last updated {now}  ·  refreshes every {REFRESH_MS // 1000}s"
    ]

    return cards, global_cards, build_activity_table(all_metrics), refresh_label


if __name__ == "__main__":
    print("Dashboard running at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
