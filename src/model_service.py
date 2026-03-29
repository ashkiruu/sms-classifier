"""Shared model loading and inference helpers for Flask and CLI use."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import build_model_input
from utils import MODELS_DIR, confidence_descriptor, interpret_label, setup_logger

logger = setup_logger(__name__)

_ENSEMBLE_MODEL = None
_NN_MODEL = None
_NN_TOKENIZER = None
_NN_LABEL_ENCODER = None
_NN_IMPORT_ERROR = None


def get_ensemble_model():
    global _ENSEMBLE_MODEL
    if _ENSEMBLE_MODEL is None:
        model_path = MODELS_DIR / "ensemble_best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Ensemble model not found: {model_path}")
        _ENSEMBLE_MODEL = joblib.load(model_path)
        logger.info("Loaded ensemble model from %s", model_path)
    return _ENSEMBLE_MODEL


def get_nn_components():
    global _NN_MODEL, _NN_TOKENIZER, _NN_LABEL_ENCODER, _NN_IMPORT_ERROR
    if _NN_MODEL is not None and _NN_TOKENIZER is not None and _NN_LABEL_ENCODER is not None:
        return _NN_MODEL, _NN_TOKENIZER, _NN_LABEL_ENCODER
    if _NN_IMPORT_ERROR is not None:
        raise RuntimeError(_NN_IMPORT_ERROR)

    model_path = MODELS_DIR / "nn_best_model.keras"
    tokenizer_path = MODELS_DIR / "nn_tokenizer.pkl"
    label_encoder_path = MODELS_DIR / "nn_label_encoder.pkl"

    if not all(path.exists() for path in [model_path, tokenizer_path, label_encoder_path]):
        raise FileNotFoundError("Neural network files are incomplete. Retrain the NN model first.")

    try:
        import tensorflow as tf
    except Exception as exc:
        _NN_IMPORT_ERROR = f"TensorFlow is not installed or failed to import: {exc}"
        raise RuntimeError(_NN_IMPORT_ERROR) from exc

    _NN_MODEL = tf.keras.models.load_model(model_path)
    _NN_TOKENIZER = joblib.load(tokenizer_path)
    _NN_LABEL_ENCODER = joblib.load(label_encoder_path)
    logger.info("Loaded neural network model from %s", model_path)
    return _NN_MODEL, _NN_TOKENIZER, _NN_LABEL_ENCODER


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
        model, tokenizer, label_encoder = get_nn_components()
        from tensorflow.keras.preprocessing.sequence import pad_sequences
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

    model_input = build_model_input(message=message, sender=sender)
    sequences = tokenizer.texts_to_sequences([model_input])
    max_len = model.input_shape[1] if isinstance(model.input_shape, tuple) else 100
    padded = pad_sequences(sequences, maxlen=max_len, padding="post")
    probabilities = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    labels = label_encoder.classes_.tolist()
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
