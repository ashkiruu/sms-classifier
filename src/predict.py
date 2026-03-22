"""Command-line prediction helper for Ensemble and Neural Network models."""
from __future__ import annotations

import argparse
import sys
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_text
from utils import MODELS_DIR, interpret_label, setup_logger

logger = setup_logger(__name__)

def predict_ensemble(cleaned_text: str):
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Ensemble model not found. Train it first: python src/train_ensemble.py")
    
    model = joblib.load(model_path)
    pred = model.predict([cleaned_text])[0]
    probs = model.predict_proba([cleaned_text])[0]
    labels = model.classes_
    return pred, probs, labels

def predict_nn(cleaned_text: str):
    # Paths for NN components
    model_path = MODELS_DIR / "nn_best_model.keras"
    tokenizer_path = MODELS_DIR / "nn_tokenizer.pkl"
    le_path = MODELS_DIR / "nn_label_encoder.pkl"

    if not all(p.exists() for p in [model_path, tokenizer_path, le_path]):
        raise FileNotFoundError("NN components missing. Train it first: python src/train_nn.py")

    # Load components
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    le = joblib.load(le_path)

    # Preprocess for NN
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')

    # Predict
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    labels = le.classes_
    
    return pred_label, probs, labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict SMS category using Ensemble or NN.")
    parser.add_argument("message", nargs="+", help="SMS message to classify")
    parser.add_argument("--model", type=str, choices=["ensemble", "nn"], default="ensemble", 
                        help="Model type to use (default: ensemble)")
    args = parser.parse_args()

    raw_text = " ".join(args.message)
    cleaned = preprocess_text(raw_text)

    try:
        if args.model == "nn":
            pred, probs, labels = predict_nn(cleaned)
        else:
            pred, probs, labels = predict_ensemble(cleaned)

        print(f"\n--- {args.model.upper()} PREDICTION RESULTS ---")
        print(f"Input text : {raw_text}")
        print(f"Prediction : {pred}")
        print(f"Meaning    : {interpret_label(pred)}")
        print("\nConfidence by class:")
        for label, prob in zip(labels, probs):
            print(f"  {label:<8} {prob:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()