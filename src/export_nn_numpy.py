"""Export the trained Keras NN into numpy-only files used by the deployed app.

Run once after (re)training the neural network. Needs TensorFlow, which is only
a training-time dependency (see requirements-train.txt):

    python src/export_nn_numpy.py

Writes models/nn_numpy_weights.npz and models/nn_numpy_meta.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODELS_DIR, setup_logger

logger = setup_logger(__name__)

WEIGHTS_FILE = "nn_numpy_weights.npz"
META_FILE = "nn_numpy_meta.json"


def _tokenizer_config(tokenizer) -> dict:
    return {
        "word_index": {word: int(idx) for word, idx in tokenizer.word_index.items()},
        "num_words": tokenizer.num_words,
        "oov_token": tokenizer.oov_token,
        "lower": bool(tokenizer.lower),
        "split": tokenizer.split,
        "filters": tokenizer.filters,
    }


def export(models_dir: Path = MODELS_DIR) -> None:
    import tensorflow as tf

    model = tf.keras.models.load_model(models_dir / "nn_best_model.keras")
    bundle = joblib.load(models_dir / "nn_tokenizer.pkl")
    label_encoder = joblib.load(models_dir / "nn_label_encoder.pkl")

    bigru = model.get_layer("message_bigru")
    if not (bigru.forward_layer.reset_after and bigru.merge_mode == "concat"):
        raise ValueError("Unsupported GRU layout; nn_numpy.py expects reset_after=True and merge_mode='concat'.")
    fw_kernel, fw_recurrent, fw_bias = bigru.forward_layer.get_weights()
    bw_kernel, bw_recurrent, bw_bias = bigru.backward_layer.get_weights()
    bn = model.get_layer("batch_normalization")
    bn_gamma, bn_beta, bn_mean, bn_var = bn.get_weights()

    weights = {
        "message_embedding": model.get_layer("message_embedding").get_weights()[0],
        "sender_embedding": model.get_layer("sender_embedding").get_weights()[0],
        "gru_fw_kernel": fw_kernel, "gru_fw_recurrent": fw_recurrent, "gru_fw_bias": fw_bias,
        "gru_bw_kernel": bw_kernel, "gru_bw_recurrent": bw_recurrent, "gru_bw_bias": bw_bias,
        "bn_gamma": bn_gamma, "bn_beta": bn_beta, "bn_mean": bn_mean, "bn_var": bn_var,
    }
    for layer in ("message_dense", "sender_dense", "merged_dense_1", "merged_dense_2", "classifier"):
        kernel, bias = model.get_layer(layer).get_weights()
        weights[f"{layer}_kernel"] = kernel
        weights[f"{layer}_bias"] = bias

    meta = {
        "tokenizers": {
            "sender": _tokenizer_config(bundle["sender_tokenizer"]),
            "message": _tokenizer_config(bundle["message_tokenizer"]),
        },
        "sender_max_len": int(bundle["sender_max_len"]),
        "message_max_len": int(bundle["message_max_len"]),
        "labels": label_encoder.classes_.tolist(),
        "batch_norm_epsilon": float(bn.epsilon),
    }

    np.savez_compressed(models_dir / WEIGHTS_FILE, **{k: np.asarray(v, dtype=np.float32) for k, v in weights.items()})
    (models_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s and %s to %s", WEIGHTS_FILE, META_FILE, models_dir)


if __name__ == "__main__":
    export()
