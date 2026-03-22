"""Evaluate the trained best model (Ensemble or NN) on the external confidence set."""
from __future__ import annotations

import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_dataset, preprocess_dataframe
from utils import DEFAULT_CONFIDENCE_DATA, MODELS_DIR, get_report_path, interpret_label, setup_logger

logger = setup_logger(__name__)

def evaluate_ensemble(X, y, df):
    """Evaluation logic for the scikit-learn Ensemble model."""
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Ensemble model not found. Run train_ensemble.py first.")
    
    model = joblib.load(model_path)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = probs.max(axis=1)
    return preds, confidence

def evaluate_nn(X_text, y_true):
    """Evaluation logic for the TensorFlow Neural Network."""
    model_path = MODELS_DIR / "nn_best_model.keras"
    tokenizer_path = MODELS_DIR / "nn_tokenizer.pkl"
    le_path = MODELS_DIR / "nn_label_encoder.pkl"

    if not all(p.exists() for p in [model_path, tokenizer_path, le_path]):
        raise FileNotFoundError("NN files missing. Run train_nn.py first.")

    # Load NN components
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    le = joblib.load(le_path)

    # Preprocess text for NN
    sequences = tokenizer.texts_to_sequences(X_text)
    X_padded = pad_sequences(sequences, maxlen=100, padding='post')

    # Predict
    probs = model.predict(X_padded)
    pred_indices = np.argmax(probs, axis=1)
    
    # Convert indices back to original labels (e.g., 'Normal', 'Spam')
    preds = le.inverse_transform(pred_indices)
    confidence = np.max(probs, axis=1)
    
    return preds, confidence

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence set.")
    parser.add_argument("--model", type=str, choices=["ensemble", "nn"], default="ensemble",
                        help="Choose which model to evaluate (default: ensemble)")
    args = parser.parse_args()

    logger.info(f"Evaluating the {args.model.upper()} model on confidence set...")

    # Load and preprocess the confidence data
    df = load_dataset(DEFAULT_CONFIDENCE_DATA)
    processed = preprocess_dataframe(df)
    X_text = processed["clean_text"]
    y_true = processed["label"]

    # Route to the correct evaluation logic
    if args.model == "nn":
        preds, confidence = evaluate_nn(X_text, y_true)
    else:
        preds, confidence = evaluate_ensemble(X_text, y_true, df)

    # Build results dataframe
    results = df.copy()
    results["predicted_label"] = preds
    results["predicted_interpretation"] = results["predicted_label"].map(interpret_label)
    results["confidence"] = confidence.round(4)
    results["is_correct"] = results["label"] == results["predicted_label"]

    # Save results with model-specific names
    prefix = "nn_" if args.model == "nn" else "ensemble_"
    out_csv = get_report_path(f"{prefix}confidence_set_predictions.csv")
    results.to_csv(out_csv, index=False)
    
    report = classification_report(y_true, preds, digits=4)
    get_report_path(f"{prefix}confidence_set_classification_report.txt").write_text(report)

    # Print Summary
    print(f"\n=== 🎯 {args.model.upper()} Confidence Set Results ===")
    print(f"Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"Macro F1 : {f1_score(y_true, preds, average='macro'):.4f}")
    print("\nClassification report:\n")
    print(report)

    print("\nPreview of predictions:\n")
    cols = ["message", "label", "predicted_label", "confidence", "predicted_interpretation"]
    print(results[cols].head(15).to_string(index=False))

if __name__ == "__main__":
    main()