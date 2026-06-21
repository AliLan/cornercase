"""
SmolVLM Benchmark Monitor
Live dashboard for vehicle_A/B/C_observation_smolvlm.py runs.
Run: python dashboard_moondream.py  →  open http://localhost:8053
Reads ONLY from:
  metrics_new/moondream/_metrics_*.jsonl
  outputs_new/moondream/
"""

import glob
import json
import os
import time

import psutil
from dash import Dash, Input, Output, dcc, html

CORNERCASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR    = os.path.join(CORNERCASE_DIR, "metrics_new", "smolvlm")
OUTPUT_DIR     = os.path.join(CORNERCASE_DIR, "outputs_new", "smolvlm")
VEHICLES       = ["A", "B", "C"]
REFRESH_MS     = 5000

# Image counts per vehicle/scenario/weather (from actual filesystem)
TOTAL_IMAGES = {
    "A": {
        "bus_obscuring_car_second_car": {"fog": 38, "day": 31, "night": 38},
        "bus_stop_near_playground":     {"fog": 89, "day": 89, "night": 89},
    },
    "B": {
        "bus_obscuring_car_second_car": {"fog": 38, "day": 30, "night": 38},
        "bus_stop_near_playground":     {"fog": 88, "day": 88, "night": 88},
    },
    "C": {
        "bus_obscuring_car_second_car": {"fog": 38, "day": 38, "night": 38},
    },
}

def total_images_for_vehicle(v):
    return sum(n for sc in TOTAL_IMAGES.get(v, {}).values() for n in sc.values())

COLORS = {
    "bg":      "#0f1117",
    "card":    "#1a1d27",
    "border":  "#2a2d3e",
    "running": "#00d68f",
    "done":    "#598bff",
    "idle":    "#6b7280",
    "error":   "#ff4d4f",
    "warn":    "#f59e0b",
    "text":    "#e2e8f0",
    "subtext": "#94a3b8",
    "accent":  "#7c3aed",
    "bar_bg":  "#2a2d3e",
    "model":   "#84cc16",
}

app = Dash(__name__, title="SmolVLM Monitor")
app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter','Segoe UI',sans-serif", "color": COLORS["text"],
           "padding": "24px"},
    children=[
        # Header
        html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "28px"},
            children=[
                html.Div([
                    html.H1("SmolVLM Benchmark Monitor",
                            style={"margin": 0, "fontSize": "22px", "fontWeight": 700}),
                    html.P("HuggingFaceTB/SmolVLM-256M-Instruct  ·  HuggingFace MPS local inference",
                           style={"margin": "4px 0 0", "color": COLORS["model"],
                                  "fontSize": "13px"}),
                ]),
                html.Div(id="smol-last-refresh",
                         style={"color": COLORS["subtext"], "fontSize": "12px",
                                "textAlign": "right"}),
            ]
        ),

        # Global progress bar
        html.Div(id="smol-global-bar", style={"marginBottom": "24px"}),

        # Vehicle cards
        html.Div(id="smol-vehicle-cards",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "16px", "marginBottom": "24px"}),

        # File paths panel
        html.Div(id="smol-file-paths",
                 style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                        "padding": "20px", "border": f"1px solid {COLORS['border']}",
                        "marginBottom": "24px"}),

        # Recent activity table
        html.Div([
            html.H3("Recent Activity",
                    style={"fontSize": "14px", "fontWeight": 600,
                           "color": COLORS["subtext"], "marginBottom": "12px",
                           "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(id="smol-activity-table"),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "padding": "20px", "border": f"1px solid {COLORS['border']}"}),

        dcc.Interval(id="smol-interval", interval=REFRESH_MS, n_intervals=0),
    ]
)


# ── Data helpers ───────────────────────────────────────────────

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
                    if rec.get("status") == "smoke_test":
                        continue
                    v = rec.get("vehicle")
                    if v in data:
                        data[v].append(rec)
        except Exception:
            pass
    return data


def is_running(vehicle):
    target = f"vehicle_{vehicle}_observation_smolvlm.py"
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


def latest_output_file(vehicle):
    pattern = os.path.join(OUTPUT_DIR, "*", "*", "*",
                           f"vehicle_{vehicle}_observations_loop.ttl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def latest_metrics_file():
    files = glob.glob(os.path.join(METRICS_DIR, "_metrics_*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def vehicle_summary(records, vehicle):
    running   = is_running(vehicle)
    total_img = total_images_for_vehicle(vehicle)

    # Count completed loops per scenario/weather
    sc_weather_loops = {}  # (scenario, weather) → set of loop numbers
    for r in records:
        key = (r.get("scenario", "?"), r.get("weather", "?"))
        sc_weather_loops.setdefault(key, set()).add(r.get("loop", 0))

    processed = sum(len(loops) for loops in sc_weather_loops.values())

    # Latest record
    latest = None
    if records:
        latest = max(records, key=lambda r: r.get("ts", ""))

    latencies = [r["latency_s"] for r in records if r.get("latency_s") and r["latency_s"] > 0]
    avg_lat   = sum(latencies) / len(latencies) if latencies else None

    # Elapsed time from first record
    timestamps = [r["ts"] for r in records if r.get("ts")]
    elapsed_s  = None
    if timestamps:
        from datetime import datetime, timezone
        first_ts = min(timestamps)
        try:
            t0 = datetime.fromisoformat(first_ts)
            elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
        except Exception:
            pass

    # ETA
    remaining = max(0, total_img - processed)
    eta_s = round(remaining * avg_lat) if avg_lat and remaining > 0 else None

    success = sum(1 for r in records if r.get("status") == "success")
    parse_e = sum(1 for r in records if r.get("status") == "parse_error")
    api_e   = sum(1 for r in records if r.get("status") == "api_error")

    return {
        "running":       running,
        "status":        "running" if running else ("done" if records else "idle"),
        "total_img":     total_img,
        "processed":     processed,
        "remaining":     remaining,
        "pct":           round(processed / total_img * 100, 1) if total_img else 0,
        "avg_lat":       avg_lat,
        "elapsed_s":     elapsed_s,
        "eta_s":         eta_s,
        "latest":        latest,
        "sc_weather":    sc_weather_loops,
        "success":       success,
        "parse_errors":  parse_e,
        "api_errors":    api_e,
    }


# ── UI helpers ─────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def progress_bar(pct, color):
    return html.Div(
        style={"backgroundColor": COLORS["bar_bg"], "borderRadius": "6px",
               "height": "8px", "overflow": "hidden", "marginTop": "6px"},
        children=[html.Div(
            style={"width": f"{pct}%", "backgroundColor": color,
                   "height": "100%", "borderRadius": "6px",
                   "transition": "width 0.4s ease"}
        )]
    )


def stat_row(label, value, color=None):
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between",
               "padding": "5px 0", "borderBottom": f"1px solid {COLORS['border']}"},
        children=[
            html.Span(label, style={"color": COLORS["subtext"], "fontSize": "12px"}),
            html.Span(value, style={"fontSize": "13px", "fontWeight": 600,
                                    "color": color or COLORS["text"]}),
        ]
    )


def build_vehicle_card(vehicle, s):
    status_col  = {"running": COLORS["running"], "done": COLORS["done"], "idle": COLORS["idle"]}
    status_text = {"running": "Running", "done": "Completed", "idle": "Not started"}
    col = status_col[s["status"]]
    pct = s["pct"]

    # Latest scenario/weather/loop
    latest_line = "—"
    if s["latest"]:
        r = s["latest"]
        sc = r.get("scenario","?").replace("bus_obscuring_car_second_car","bus_obscuring").replace("bus_stop_near_playground","bus_stop")
        latest_line = f"{sc} / {r.get('weather','?')} / loop {r.get('loop','?')}"

    # Per scenario/weather breakdown
    breakdown_rows = []
    for (sc, wt), loops in sorted(s["sc_weather"].items()):
        total_sw = TOTAL_IMAGES.get(vehicle, {}).get(sc, {}).get(wt, "?")
        done_sw  = len(loops)
        sc_short = sc.replace("bus_obscuring_car_second_car","bus_obscuring").replace("bus_stop_near_playground","bus_stop")
        breakdown_rows.append(html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "padding": "4px 0", "fontSize": "11px",
                   "borderBottom": f"1px solid {COLORS['border']}"},
            children=[
                html.Span(f"{sc_short}/{wt}", style={"color": COLORS["subtext"]}),
                html.Span(f"{done_sw}/{total_sw}", style={"fontWeight": 600}),
            ]
        ))

    return html.Div(
        style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
               "padding": "20px", "border": f"1px solid {COLORS['border']}",
               "borderTop": f"3px solid {col}"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.H2(f"Vehicle {vehicle}",
                            style={"margin": 0, "fontSize": "16px", "fontWeight": 700}),
                    html.Span(status_text[s["status"]],
                              style={"fontSize": "12px", "color": col, "fontWeight": 600}),
                ]
            ),
            # Progress bar
            html.Div([
                html.Span(f"{pct}% complete",
                          style={"fontSize": "12px", "color": COLORS["subtext"]}),
                html.Span(f"{s['processed']} / {s['total_img']} images",
                          style={"fontSize": "12px", "color": COLORS["text"], "float": "right"}),
            ]),
            progress_bar(pct, col),
            html.Div(style={"marginTop": "14px"}, children=breakdown_rows or [
                html.Span("No runs yet", style={"color": COLORS["subtext"], "fontSize": "12px"})
            ]),
            html.Div(style={"marginTop": "10px"}, children=[
                stat_row("Current",          latest_line),
                stat_row("Elapsed",          fmt_time(s["elapsed_s"]), COLORS["running"]),
                stat_row("ETA",              fmt_time(s["eta_s"]),     COLORS["warn"]),
                stat_row("Avg latency/img",  f"{s['avg_lat']:.1f}s" if s["avg_lat"] else "—"),
                stat_row("✓ Succeeded",      str(s["success"]),        COLORS["running"]),
                stat_row("⚠ Parse errors",  str(s["parse_errors"]),   COLORS["warn"] if s["parse_errors"] else COLORS["text"]),
                stat_row("✗ API errors",    str(s["api_errors"]),     COLORS["error"] if s["api_errors"] else COLORS["text"]),
            ]),
        ]
    )


def build_global_bar(summaries):
    total_all     = sum(s["total_img"]  for s in summaries.values())
    processed_all = sum(s["processed"]  for s in summaries.values())
    pct_all       = round(processed_all / total_all * 100, 1) if total_all else 0

    running_v = [v for v, s in summaries.items() if s["running"]]
    status_txt = f"Running: Vehicle {', '.join(running_v)}" if running_v else (
                 "All done" if all(s["status"] == "done" for s in summaries.values()) else "Idle")

    # Overall ETA: pick the max ETA among running vehicles
    etas = [s["eta_s"] for s in summaries.values() if s["eta_s"] is not None]
    overall_eta = max(etas) if etas else None

    # Elapsed (from earliest first record across all vehicles)
    all_elapsed = [s["elapsed_s"] for s in summaries.values() if s["elapsed_s"]]
    overall_elapsed = max(all_elapsed) if all_elapsed else None

    return html.Div(
        style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
               "padding": "20px", "border": f"1px solid {COLORS['border']}"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "marginBottom": "8px"},
                children=[
                    html.Span("Overall Progress",
                              style={"fontSize": "14px", "fontWeight": 600}),
                    html.Span(f"{processed_all} / {total_all} images  ({pct_all}%)",
                              style={"fontSize": "13px", "color": COLORS["subtext"]}),
                ]
            ),
            progress_bar(pct_all, COLORS["model"]),
            html.Div(
                style={"display": "flex", "gap": "32px", "marginTop": "10px",
                       "fontSize": "12px", "color": COLORS["subtext"]},
                children=[
                    html.Span(status_txt, style={"color": COLORS["running"]}),
                    html.Span(f"Elapsed: {fmt_time(overall_elapsed)}"),
                    html.Span(f"ETA remaining: {fmt_time(overall_eta)}",
                              style={"color": COLORS["warn"]}),
                ]
            )
        ]
    )


def build_file_paths(summaries):
    rows = []
    for v in VEHICLES:
        out_file = latest_output_file(v)
        out_rel  = os.path.relpath(out_file, CORNERCASE_DIR) if out_file else "—"
        rows.append(html.Div(
            style={"display": "flex", "gap": "12px", "padding": "6px 0",
                   "borderBottom": f"1px solid {COLORS['border']}",
                   "fontSize": "12px", "alignItems": "center"},
            children=[
                html.Span(f"Vehicle {v}",
                          style={"color": COLORS["accent"], "fontWeight": 600,
                                 "minWidth": "70px"}),
                html.Span("TTL →", style={"color": COLORS["subtext"], "minWidth": "40px"}),
                html.Code(out_rel, style={"color": COLORS["running"], "fontSize": "11px",
                                          "wordBreak": "break-all"}),
            ]
        ))

    mf = latest_metrics_file()
    mf_rel = os.path.relpath(mf, CORNERCASE_DIR) if mf else "—"
    rows.append(html.Div(
        style={"display": "flex", "gap": "12px", "padding": "8px 0",
               "fontSize": "12px", "alignItems": "center", "marginTop": "4px"},
        children=[
            html.Span("Metrics", style={"color": COLORS["warn"], "fontWeight": 600,
                                        "minWidth": "70px"}),
            html.Span("JSONL →", style={"color": COLORS["subtext"], "minWidth": "40px"}),
            html.Code(mf_rel, style={"color": COLORS["warn"], "fontSize": "11px",
                                     "wordBreak": "break-all"}),
        ]
    ))

    return html.Div([
        html.H3("Latest File Paths",
                style={"fontSize": "14px", "fontWeight": 600,
                       "color": COLORS["subtext"], "marginBottom": "12px",
                       "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        *rows
    ])


def build_activity_table(all_records):
    combined = [(r.get("ts",""), v, r)
                for v, recs in all_records.items() for r in recs]
    combined.sort(key=lambda x: x[0], reverse=True)
    recent = combined[:20]

    if not recent:
        return html.P("No activity yet.", style={"color": COLORS["subtext"]})

    header = html.Tr([
        html.Th(col, style={"padding": "8px 12px", "color": COLORS["subtext"],
                             "fontSize": "11px", "textTransform": "uppercase",
                             "letterSpacing": "0.05em", "fontWeight": 500,
                             "borderBottom": f"1px solid {COLORS['border']}"})
        for col in ["Vehicle", "Scenario", "Weather", "Loop",
                    "Latency", "F1", "Halluc", "Comply", "Status"]
    ])

    rows = []
    for ts, v, r in recent:
        s = r.get("status", "")
        s_display = {"success": "✓ ok", "parse_error": "⚠ parse", "api_error": "✗ api"}.get(s, s)
        s_color   = {"success": COLORS["running"], "parse_error": COLORS["warn"],
                     "api_error": COLORS["error"]}.get(s, COLORS["subtext"])
        lat = r.get("latency_s")
        sc  = r.get("scenario","—").replace("bus_obscuring_car_second_car","bus_obscuring").replace("bus_stop_near_playground","bus_stop")
        f1  = r.get("f1")
        hal = r.get("hallucination_rate")
        com = r.get("compliance_rate")

        rows.append(html.Tr(
            style={"borderBottom": f"1px solid {COLORS['border']}"},
            children=[
                html.Td(f"Veh {v}",   style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(sc,           style={"padding": "7px 12px", "fontSize": "11px",
                                             "color": COLORS["subtext"], "maxWidth": "140px",
                                             "overflow": "hidden", "textOverflow": "ellipsis",
                                             "whiteSpace": "nowrap"}),
                html.Td(r.get("weather","—"), style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(str(r.get("loop","—")), style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(f"{lat:.1f}s" if lat else "—",
                        style={"padding": "7px 12px", "fontSize": "13px",
                               "color": COLORS["running"] if lat and lat < 30 else COLORS["warn"]}),
                html.Td(f"{f1:.2f}"  if f1  is not None else "—",
                        style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(f"{hal:.3f}" if hal is not None else "—",
                        style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(f"{com:.3f}" if com is not None else "—",
                        style={"padding": "7px 12px", "fontSize": "13px"}),
                html.Td(s_display,
                        style={"padding": "7px 12px", "fontSize": "12px",
                               "color": s_color, "fontWeight": 700}),
            ]
        ))

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[html.Thead(header), html.Tbody(rows)]
    )


# ── Callback ───────────────────────────────────────────────────

@app.callback(
    Output("smol-global-bar",    "children"),
    Output("smol-vehicle-cards", "children"),
    Output("smol-file-paths",    "children"),
    Output("smol-activity-table","children"),
    Output("smol-last-refresh",  "children"),
    Input("smol-interval",       "n_intervals"),
)
def refresh(_):
    all_metrics = load_all_metrics()
    summaries   = {v: vehicle_summary(all_metrics[v], v) for v in VEHICLES}

    global_bar   = build_global_bar(summaries)
    cards        = [build_vehicle_card(v, summaries[v]) for v in VEHICLES]
    file_paths   = build_file_paths(summaries)
    activity     = build_activity_table(all_metrics)
    now          = time.strftime("%H:%M:%S")
    refresh_lbl  = [
        html.Span("⟳ ", style={"color": COLORS["model"]}),
        f"Last updated {now}  ·  refreshes every {REFRESH_MS // 1000}s"
    ]

    return global_bar, cards, file_paths, activity, refresh_lbl


if __name__ == "__main__":
    print("SmolVLM Monitor running at http://localhost:8053")
    app.run(debug=False, host="0.0.0.0", port=8053)
