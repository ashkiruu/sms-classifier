from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model_service import predict_all
from utils import load_json_report
from history_service import save_analysis_history

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    requirements = [
        "Supervised classification project with separate ensemble and neural network models",
        "Flask-only web integration",
        "User-entered fields for analysis",
        "Separate model outputs with confidence information",
        "Problem, dataset, model, and metrics discussion pages",
    ]
    return render_template("about.html", requirements=requirements)


@app.route("/metrics")
def metrics():
    ensemble_metrics = load_json_report("best_model_test_metrics.json", default={})
    nn_metrics = load_json_report("nn_metrics_summary.json", default={})

    nn_report_path = Path(__file__).resolve().parent / "outputs" / "reports" / "nn_best_model_report.txt"
    if nn_report_path.exists():
        nn_classification_report = nn_report_path.read_text(encoding="utf-8")
    else:
        nn_classification_report = ""

    return render_template(
        "metrics.html",
        ensemble_metrics=ensemble_metrics,
        nn_metrics=nn_metrics,
        nn_classification_report=nn_classification_report,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    sender = (data.get("sender") or "unknown").strip() or "unknown"

    if not message:
        return jsonify({"error": "Message is required."}), 400

    result = predict_all(message=message, sender=sender)
    storage_used = save_analysis_history(
        sender=sender,
        message=message,
        ensemble_result=result["ensemble"],
        nn_result=result["nn"],
    )
    result["history_storage"] = storage_used
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
