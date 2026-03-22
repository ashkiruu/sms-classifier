"""Evaluate the trained best model on the external 15-row confidence set."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_dataset, preprocess_dataframe
from utils import DEFAULT_CONFIDENCE_DATA, MODELS_DIR, get_report_path, interpret_label, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Train the model first: python src/train_ensemble.py")

    df = load_dataset(DEFAULT_CONFIDENCE_DATA)
    processed = preprocess_dataframe(df)
    X = processed["clean_text"]
    y = processed["label"]

    model = joblib.load(model_path)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = probs.max(axis=1)

    results = df.copy()
    results["predicted_label"] = preds
    results["predicted_interpretation"] = results["predicted_label"].map(interpret_label)
    results["confidence"] = confidence.round(4)
    results["is_correct"] = results["label"] == results["predicted_label"]

    out_csv = get_report_path("confidence_set_predictions.csv")
    results.to_csv(out_csv, index=False)
    logger.info("Saved confidence-set predictions to %s", out_csv)

    report = classification_report(y, preds, digits=4)
    print(f"Accuracy on confidence set: {accuracy_score(y, preds):.4f}")
    print(f"Macro F1 on confidence set: {f1_score(y, preds, average='macro'):.4f}")
    print("\nClassification report:\n")
    print(report)
    get_report_path("confidence_set_classification_report.txt").write_text(report)

    print("\nPreview of predictions:\n")
    cols = ["message", "label", "predicted_label", "confidence", "predicted_interpretation"]
    print(results[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
