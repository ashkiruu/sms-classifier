from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model_service import predict_all
from utils import load_json_report

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
    return render_template("metrics.html", ensemble_metrics=ensemble_metrics, nn_metrics=nn_metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    sender = (data.get("sender") or "unknown").strip() or "unknown"

    if not message:
        return jsonify({"error": "Message is required."}), 400

    result = predict_all(message=message, sender=sender)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
