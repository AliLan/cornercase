"""
Police Chase Benchmark Monitor
Live dashboard for vehicle_ego/flee/police_observation.py background tasks.
Run: python dashboard_police_chase.py  →  open http://localhost:8054
"""

import glob
import json
import os
import time

import psutil
from dash import Dash, Input, Output, dcc, html

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR    = os.path.join(CORNERCASE_DIR, "Qwen 2.5 VL 7B metrics")
VEHICLES       = ["ego", "flee", "police"]
REFRESH_MS     = 3000

# Total images per (vehicle, scenario) — used for progress bars
TOTAL_IMAGES = {
    ("ego",    "spawn_police_car_chase_1"): 212,
    ("flee",   "spawn_police_car_chase_1"): 210,
    ("police", "spawn_police_car_chase_1"): 210,
    ("ego",    "spawn_police_car_chase_2"): 234,
    ("flee",   "spawn_police_car_chase_2"): 234,
    ("police", "spawn_police_car_chase_2"): 234,
}
GRAND_TOTAL = sum(TOTAL_IMAGES.values())  # 1334

COLORS = {
    "bg":      "#0f1117",
    "card":    "#1a1d27",
    "border":  "#2a2d3e",
    "running": "#00d68f",
    "done":    "#598bff",
    "idle":    "#6b7280",
    "error":   "#ff4d4f",
    "text":    "#e2e8f0",
    "subtext": "#94a3b8",
    "accent":  "#f59e0b",
}

app = Dash(__name__, title="Police Chase Monitor")
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter', 'Segoe UI', sans-serif", "color": COLORS["text"],
           "padding": "24px"},
    children=[
        html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "28px"},
            children=[
                html.Div([
                    html.H1("Police Chase Monitor",
                            style={"margin": 0, "fontSize": "22px", "fontWeight": 700}),
                    html.P("qwen2.5vl:7b  ·  spawn_police_car_chase_1 → chase_2  ·  ego → flee → police",
                           style={"margin": "4px 0 0", "color": COLORS["subtext"],
                                  "fontSize": "13px"}),
                ]),
                html.Div(id="last-refresh",
                         style={"color": COLORS["subtext"], "fontSize": "12px",
                                "textAlign": "right"}),
            ]
        ),

        html.Div(id="vehicle-cards",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "16px", "marginBottom": "24px"}),

        html.Div(id="global-stats",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "12px", "marginBottom": "24px"}),

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


def load_all_metrics():
    data = {v: [] for v in VEHICLES}
    for fpath in glob.glob(os.path.join(METRICS_DIR, "_metrics_*.jsonl")):
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    v = rec.get("vehicle")
                    if v in data:
                        # only police chase scenarios
                        sc = rec.get("scenario", "")
                        if "police_car_chase" in sc:
                            data[v].append(rec)
        except Exception:
            pass
    return data


def is_running(vehicle):
    target = f"vehicle_{vehicle}_observation.py"
    for proc in psutil.process_iter(["cmdline", "name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = proc.info["cmdline"] or []
            if any(arg == target or arg.endswith(f"/{target}") for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def vehicle_stats(records, vehicle):
    if not records:
        return {"status": "idle", "calls": 0, "success": 0, "parse_errors": 0,
                "api_errors": 0, "avg_latency": None, "scenario_summary": [],
                "current_attempt": "—", "completed": 0}

    running = is_running(vehicle)
    status  = "running" if running else "done"

    success_recs = [r for r in records if r.get("status") == "success"]
    parse_errors = [r for r in records if r.get("status") == "parse_error"]
    api_errors   = [r for r in records if r.get("status") == "api_error"]

    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]

    scenario_map = {}
    for r in records:
        key = (r.get("scenario", "?"), r.get("weather", "?"))
        cur_loop    = r.get("loop", 0)
        cur_attempt = r.get("attempt", "—")
        prev_loop, _ = scenario_map.get(key, (0, "—"))
        if cur_loop >= prev_loop:
            scenario_map[key] = (cur_loop, cur_attempt)

    completed = sum(loop for (loop, _) in scenario_map.values())

    return {
        "status":           status,
        "calls":            len(records),
        "success":          len(success_recs),
        "parse_errors":     len(parse_errors),
        "api_errors":       len(api_errors),
        "avg_latency":      sum(latencies) / len(latencies) if latencies else None,
        "scenario_summary": sorted(scenario_map.items()),
        "current_attempt":  max((r.get("attempt", 0) for r in records), default="—"),
        "completed":        completed,
    }


def status_dot(status):
    colors = {"running": COLORS["running"], "done": COLORS["done"], "idle": COLORS["idle"]}
    pulse  = status == "running"
    return html.Span(
        style={
            "display": "inline-block", "width": "10px", "height": "10px",
            "borderRadius": "50%",
            "backgroundColor": colors.get(status, COLORS["idle"]),
            "marginRight": "8px",
            "boxShadow": f"0 0 6px {colors.get(status)}" if pulse else "none",
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


def progress_bar(pct, color):
    return html.Div(
        style={"backgroundColor": COLORS["border"], "borderRadius": "4px",
               "height": "6px", "marginTop": "4px", "overflow": "hidden"},
        children=[html.Div(style={"width": f"{pct:.1f}%", "height": "100%",
                                  "backgroundColor": color, "borderRadius": "4px"})]
    )


def build_vehicle_card(vehicle, stats):
    status      = stats["status"]
    status_text = {"running": "Running", "done": "Completed", "idle": "Not started"}
    status_col  = {"running": COLORS["running"], "done": COLORS["done"], "idle": COLORS["idle"]}

    total        = stats["calls"]
    success_rate = f"{stats['success'] / total * 100:.0f}%" if total > 0 else "—"
    avg_lat      = f"{stats['avg_latency']:.1f}s" if stats["avg_latency"] is not None else "—"

    vehicle_total = sum(
        v for (veh, sc), v in TOTAL_IMAGES.items() if veh == vehicle
    )
    completed = stats["completed"]
    pct_done  = min(100.0, completed / vehicle_total * 100) if vehicle_total else 0.0

    scenario_rows = []
    for (sc, wt), (max_loop, attempt) in stats["scenario_summary"]:
        sc_short = sc.replace("spawn_police_car_", "")
        key = (vehicle, sc)
        sc_total = TOTAL_IMAGES.get(key, "?")
        pct = min(100.0, max_loop / sc_total * 100) if isinstance(sc_total, int) and sc_total else 0
        scenario_rows.append(
            html.Div(
                style={"padding": "4px 0", "fontSize": "11px",
                       "borderBottom": f"1px solid {COLORS['border']}"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between"},
                        children=[
                            html.Span(f"{sc_short} / {wt}",
                                      style={"color": COLORS["subtext"]}),
                            html.Span(f"{max_loop}/{sc_total}  ({pct:.0f}%)",
                                      style={"color": COLORS["text"], "fontWeight": 600}),
                        ]
                    ),
                    progress_bar(pct, status_col[status]),
                ]
            )
        )

    return html.Div(
        style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
               "padding": "20px", "border": f"1px solid {COLORS['border']}",
               "borderTop": f"3px solid {status_col[status]}"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "marginBottom": "8px"},
                children=[
                    html.H2(f"Vehicle: {vehicle}",
                            style={"margin": 0, "fontSize": "16px", "fontWeight": 700}),
                    html.Span(
                        [status_dot(status), status_text[status]],
                        style={"fontSize": "12px", "color": status_col[status],
                               "display": "flex", "alignItems": "center"}
                    ),
                ]
            ),
            html.Div(
                style={"marginBottom": "6px", "fontSize": "12px",
                       "color": COLORS["subtext"]},
                children=f"Overall: {completed}/{vehicle_total} images ({pct_done:.0f}%)"
            ),
            progress_bar(pct_done, status_col[status]),
            html.Div(style={"marginTop": "12px", "marginBottom": "10px"},
                     children=(scenario_rows if scenario_rows else [
                         html.Span("No runs yet", style={"color": COLORS["subtext"],
                                                          "fontSize": "12px"})
                     ])),
            stat_pill("Attempt",       str(stats["current_attempt"])),
            stat_pill("Total calls",   str(total)),
            stat_pill("✓ Succeeded",   str(stats["success"]),
                      COLORS["running"] if stats["success"] > 0 else COLORS["text"]),
            stat_pill("⚠ Parse errors", str(stats["parse_errors"]),
                      "#f59e0b" if stats["parse_errors"] > 0 else COLORS["text"]),
            stat_pill("✗ API errors",  str(stats["api_errors"]),
                      COLORS["error"] if stats["api_errors"] > 0 else COLORS["text"]),
            stat_pill("Avg latency",   avg_lat),
            stat_pill("Success rate",  success_rate),
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
        for col in ["Vehicle", "Scenario", "Weather", "Loop", "Latency",
                    "Tokens In", "Tokens Out", "Hallucination", "Compliance", "Status"]
    ])

    rows = []
    for ts, v, r in recent:
        status_val = r.get("status", "success" if r.get("success") else "api_error")
        _disp  = {"success": "✓ ok", "parse_error": "⚠ parse", "api_error": "✗ api"}.get(status_val, status_val)
        _color = {"success": COLORS["running"], "parse_error": "#f59e0b",
                  "api_error": COLORS["error"]}.get(status_val, COLORS["subtext"])
        lat   = r.get("latency_s")
        hall  = r.get("hallucination_rate")
        comp  = r.get("compliance_rate")
        sc    = r.get("scenario", "—").replace("spawn_police_car_", "")
        rows.append(html.Tr(
            style={"borderBottom": f"1px solid {COLORS['border']}"},
            children=[
                html.Td(v, style={"padding": "8px 12px", "fontSize": "13px",
                                  "fontWeight": 600, "color": COLORS["accent"]}),
                html.Td(sc, style={"padding": "8px 12px", "fontSize": "12px",
                                   "color": COLORS["subtext"]}),
                html.Td(r.get("weather", "—"), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("loop", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(f"{lat:.1f}s" if lat else "—",
                        style={"padding": "8px 12px", "fontSize": "13px",
                               "color": COLORS["running"] if lat and lat < 90 else COLORS["error"]}),
                html.Td(str(r.get("prompt_tokens", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("completion_tokens", "—")), style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(f"{hall:.2f}" if hall is not None else "—",
                        style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(f"{comp:.2f}" if comp is not None else "—",
                        style={"padding": "8px 12px", "fontSize": "13px"}),
                html.Td(_disp, style={"padding": "8px 12px", "fontSize": "12px",
                                      "color": _color, "fontWeight": 700}),
            ]
        ))

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[html.Thead(header), html.Tbody(rows)]
    )


@app.callback(
    Output("vehicle-cards",  "children"),
    Output("global-stats",   "children"),
    Output("activity-table", "children"),
    Output("last-refresh",   "children"),
    Input("interval", "n_intervals"),
)
def refresh(_):
    all_metrics = load_all_metrics()

    cards = [
        build_vehicle_card(v, vehicle_stats(all_metrics[v], v))
        for v in VEHICLES
    ]

    all_recs     = [r for recs in all_metrics.values() for r in recs]
    total_done   = sum(
        max((r.get("loop", 0) for r in recs), default=0)
        for recs in all_metrics.values()
    )
    total_calls  = len(all_recs)
    total_ok     = sum(1 for r in all_recs if r.get("status") == "success")
    lats         = [r["latency_s"] for r in all_recs if r.get("latency_s") is not None]
    avg_lat      = sum(lats) / len(lats) if lats else 0
    running_v    = sum(1 for v in VEHICLES if is_running(v))
    grand_pct    = min(100.0, total_calls / GRAND_TOTAL * 100) if GRAND_TOTAL else 0

    eta_s = (GRAND_TOTAL - total_calls) * avg_lat if avg_lat and total_calls < GRAND_TOTAL else None
    eta_h = f"{eta_s / 3600:.1f}h remaining" if eta_s else "—"

    global_cards = [
        build_global_card("Active Agent", str(running_v), "process running",
                          COLORS["running"] if running_v > 0 else COLORS["idle"]),
        build_global_card("Total Images", f"{total_calls}/{GRAND_TOTAL}",
                          f"{grand_pct:.1f}% complete"),
        build_global_card("Success Rate",
                          f"{total_ok / total_calls * 100:.0f}%" if total_calls else "—",
                          f"{total_ok} ok / {total_calls - total_ok} errors",
                          COLORS["running"] if total_calls > 0 else COLORS["idle"]),
        build_global_card("ETA", eta_h,
                          f"avg {avg_lat:.1f}s/img" if avg_lat else "no data yet"),
    ]

    now = time.strftime("%H:%M:%S")
    refresh_label = [
        html.Span("⟳ ", style={"color": COLORS["running"]}),
        f"Last updated {now}  ·  refreshes every {REFRESH_MS // 1000}s"
    ]

    return cards, global_cards, build_activity_table(all_metrics), refresh_label


if __name__ == "__main__":
    print("Dashboard running at http://localhost:8054")
    app.run(debug=False, host="0.0.0.0", port=8054)
