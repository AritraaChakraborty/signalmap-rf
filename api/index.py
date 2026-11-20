"""Flask WSGI application for Vercel deployment and local web dashboard.

Routes all 9 pages (/, /data, /map, /heatmap, /model, /stats, /analysis,
/about, and custom 404) plus JSON/CSV export endpoints.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path for Vercel serverless execution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flask import Flask, Response, jsonify, render_template, request

from core.engine_lite import run_full_analysis_lite
from core.io_utils import (
    dataset_to_csv_string,
    generate_synthetic_rows,
    load_csv_file,
    validate_csv_text,
    write_rows_to_csv,
)
from core.models import DEFAULT_STRONG_THRESHOLD, DEFAULT_WEAK_THRESHOLD

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static"),
)

SAMPLE_CSV_PATH = os.path.join(ROOT_DIR, "data", "sample_survey.csv")


def _load_sample_dataset():
    if os.path.exists(SAMPLE_CSV_PATH):
        return load_csv_file(
            SAMPLE_CSV_PATH,
            source_label="Synthetic Floor Survey (sample_survey.csv)",
            is_synthetic=True,
        )
    # Fallback in read-only environments if CSV was somehow absent
    import csv
    import io

    rows = generate_synthetic_rows(seed=42)
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["point_id", "x", "y", "signal_dbm", "location", "floor", "ap_id"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return validate_csv_text(
        buf.getvalue(),
        source_label="Synthetic Floor Survey (generated)",
        is_synthetic=True,
    )


def _parse_thresholds(args_or_json):
    try:
        weak = float(args_or_json.get("weak_threshold", DEFAULT_WEAK_THRESHOLD))
    except (TypeError, ValueError):
        weak = DEFAULT_WEAK_THRESHOLD
    try:
        strong = float(args_or_json.get("strong_threshold", DEFAULT_STRONG_THRESHOLD))
    except (TypeError, ValueError):
        strong = DEFAULT_STRONG_THRESHOLD
    weak = max(-100.0, min(-55.0, weak))
    strong = max(weak + 5.0, min(-30.0, strong))
    return weak, strong


# --- Page Routes ---

@app.route("/")
def page_home():
    return render_template("index.html", active_page="home")


@app.route("/data")
def page_data():
    return render_template("data.html", active_page="data")


@app.route("/map")
def page_map():
    return render_template("map.html", active_page="map")


@app.route("/heatmap")
def page_heatmap():
    return render_template("heatmap.html", active_page="heatmap")


@app.route("/model")
def page_model():
    return render_template("model.html", active_page="model")


@app.route("/stats")
def page_stats():
    return render_template("stats.html", active_page="stats")


@app.route("/analysis")
def page_analysis():
    return render_template("analysis.html", active_page="analysis")


@app.route("/about")
def page_about():
    return render_template("about.html", active_page="about")


# --- API Routes ---

@app.route("/api/sample", methods=["GET"])
def api_sample():
    weak, strong = _parse_thresholds(request.args)
    dataset, rejected, meta = _load_sample_dataset()
    payload = run_full_analysis_lite(
        dataset, rejected, meta, weak_threshold=weak, strong_threshold=strong
    )
    return jsonify(payload)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True) or {}
    csv_text = body.get("csv_text", "")
    source_label = str(body.get("source_label", "Uploaded CSV")).strip() or "Uploaded CSV"
    is_synthetic = bool(body.get("is_synthetic", False))
    weak, strong = _parse_thresholds(body)

    if not csv_text or not isinstance(csv_text, str):
        dataset, rejected, meta = _load_sample_dataset()
    else:
        try:
            dataset, rejected, meta = validate_csv_text(
                csv_text, source_label=source_label, is_synthetic=is_synthetic
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    try:
        payload = run_full_analysis_lite(
            dataset, rejected, meta, weak_threshold=weak, strong_threshold=strong
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(payload)


@app.route("/api/export/csv", methods=["POST"])
def api_export_csv():
    body = request.get_json(silent=True) or {}
    csv_text = body.get("csv_text", "")
    weak, strong = _parse_thresholds(body)
    if csv_text:
        dataset, _, _ = validate_csv_text(csv_text, source_label="Exported CSV", is_synthetic=False)
    else:
        dataset, _, _ = _load_sample_dataset()

    cleaned_csv = dataset_to_csv_string(dataset, weak_threshold=weak, strong_threshold=strong)
    return Response(
        cleaned_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_wireless_survey.csv"},
    )


@app.errorhandler(404)
def page_not_found(_err):
    return render_template("404.html", active_page=""), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
