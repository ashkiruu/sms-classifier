"""Command-line prediction helper for Ensemble and Neural Network models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_service import predict_all, predict_with_ensemble, predict_with_nn


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict SMS category using Ensemble and/or NN.")
    parser.add_argument("message", nargs="+", help="SMS message to classify")
    parser.add_argument("--sender", type=str, default="unknown", help="Sender or originating source")
    parser.add_argument(
        "--model",
        type=str,
        choices=["ensemble", "nn", "both"],
        default="both",
        help="Model type to use (default: both)",
    )
    args = parser.parse_args()

    raw_text = " ".join(args.message)
    if args.model == "ensemble":
        result = {"ensemble": predict_with_ensemble(raw_text, sender=args.sender)}
    elif args.model == "nn":
        result = {"nn": predict_with_nn(raw_text, sender=args.sender)}
    else:
        result = predict_all(raw_text, sender=args.sender)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
