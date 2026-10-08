"""Shared model loading and inference helpers for Flask and CLI use."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_numpy import NumpyNN
from preprocessing import build_model_input
from utils import MODELS_DIR, confidence_descriptor, interpret_label, setup_logger

logger = setup_logger(__name__)

_ENSEMBLE_MODEL = None
_NN_MODEL = None


def get_ensemble_model():
    global _ENSEMBLE_MODEL
    if _ENSEMBLE_MODEL is None:
        model_path = MODELS_DIR / "ensemble_best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Ensemble model not found: {model_path}")
        _ENSEMBLE_MODEL = joblib.load(model_path)
        logger.info("Loaded ensemble model from %s", model_path)
    return _ENSEMBLE_MODEL


def get_nn_model() -> NumpyNN:
    """Load the neural network as numpy-only weights (no TensorFlow needed at runtime).

    The files are produced from the trained Keras model by `python src/export_nn_numpy.py`.
    """
    global _NN_MODEL
    if _NN_MODEL is None:
        weights_path = MODELS_DIR / "nn_numpy_weights.npz"
        meta_path = MODELS_DIR / "nn_numpy_meta.json"
        if not (weights_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                "Neural network export files are missing. Run `python src/export_nn_numpy.py` after training."
            )
        _NN_MODEL = NumpyNN(weights_path, meta_path)
        logger.info("Loaded neural network weights from %s", weights_path)
    return _NN_MODEL


def _format_result(prediction: str, probabilities: np.ndarray, labels: list[str]) -> dict:
    confidence_scores = {label: round(float(prob), 4) for label, prob in zip(labels, probabilities)}
    top_confidence = float(np.max(probabilities)) if len(probabilities) else 0.0
    return {
        "prediction": prediction,
        "interpretation": interpret_label(prediction),
        "confidence": round(top_confidence, 4),
        "confidence_level": confidence_descriptor(top_confidence),
        "confidence_scores": confidence_scores,
    }


def predict_with_ensemble(message: str, sender: str = "unknown") -> dict:
    model = get_ensemble_model()
    model_input = build_model_input(message=message, sender=sender)
    prediction = model.predict([model_input])[0]
    probabilities = model.predict_proba([model_input])[0]
    labels = model.classes_.tolist()
    result = _format_result(prediction, probabilities, labels)
    result["model_input"] = model_input
    return result


def predict_with_nn(message: str, sender: str = "unknown") -> dict:
    try:
        nn_model = get_nn_model()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "prediction": None,
            "interpretation": "Neural network unavailable.",
            "confidence": None,
            "confidence_level": None,
            "confidence_scores": {},
        }

    def _normalize_numbers(text: str) -> str:
        import re
        return re.sub(r"\d+", "NUM", str(text).lower()).strip()

    def _clean_sender_text(text: str) -> str:
        import re
        text = _normalize_numbers(text)
        text = re.sub(r"[^a-z0-9_@.+\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else "unknown"

    def _clean_message_text(text: str) -> str:
        import re
        text = _normalize_numbers(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else "empty"

    model_input = build_model_input(message=message, sender=sender)

    sender_text = _clean_sender_text(sender)
    message_text = _clean_message_text(message)

    probabilities = nn_model.predict_proba([sender_text], [message_text])[0]

    pred_idx = int(np.argmax(probabilities))
    labels = nn_model.labels
    prediction = labels[pred_idx]
    result = _format_result(prediction, probabilities, labels)
    result.update({"available": True, "model_input": model_input})
    return result


def predict_all(message: str, sender: str = "unknown") -> dict:
    return {
        "sender": sender or "unknown",
        "message": message,
        "ensemble": predict_with_ensemble(message=message, sender=sender),
        "nn": predict_with_nn(message=message, sender=sender),
    }