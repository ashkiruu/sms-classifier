"""Command-line prediction helper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_text
from utils import MODELS_DIR, interpret_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="+", help="SMS message to classify")
    args = parser.parse_args()

    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Train the model first with: python src/train_ensemble.py")

    model = joblib.load(model_path)
    raw_text = " ".join(args.message)
    cleaned = preprocess_text(raw_text)
    pred = model.predict([cleaned])[0]
    probs = model.predict_proba([cleaned])[0]

    print(f"Prediction: {pred}")
    print(f"Meaning   : {interpret_label(pred)}")
    print("Confidence by class:")
    for label, prob in zip(model.classes_, probs):
        print(f"  {label:<8} {prob:.4f}")


if __name__ == "__main__":
    main()
