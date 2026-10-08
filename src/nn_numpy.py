"""Numpy-only inference for the trained SMS neural network.

The Keras model (nn_best_model.keras) is a small fixed architecture:
embeddings -> bidirectional GRU -> pooling -> dense layers -> softmax.
Running it with plain numpy keeps TensorFlow/Keras out of the deployed app
(they are far too large for a Vercel function). The weights are exported once
by export_nn_numpy.py, and the output matches Keras to float32 precision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _Tokenizer:
    """Re-implementation of keras Tokenizer.texts_to_sequences (word level)."""

    def __init__(self, cfg: dict):
        self.word_index: dict[str, int] = cfg["word_index"]
        self.num_words = cfg["num_words"]
        self.oov_token = cfg["oov_token"]
        self.lower = cfg["lower"]
        self.split = cfg["split"]
        self.filters = cfg["filters"]
        self.oov_index = self.word_index.get(self.oov_token) if self.oov_token is not None else None
        self._translate = str.maketrans({c: self.split for c in self.filters})

    def encode(self, text: str) -> list[int]:
        text = str(text)
        if self.lower:
            text = text.lower()
        words = [w for w in text.translate(self._translate).split(self.split) if w]
        seq: list[int] = []
        for word in words:
            idx = self.word_index.get(word)
            if idx is not None:
                if self.num_words and idx >= self.num_words:
                    if self.oov_index is not None:
                        seq.append(self.oov_index)
                else:
                    seq.append(idx)
            elif self.oov_index is not None:
                seq.append(self.oov_index)
        return seq


def _pad_post(seq: list[int], maxlen: int) -> list[int]:
    """Keras pad_sequences(padding='post', truncating='post'), value 0."""
    seq = seq[:maxlen]
    return seq + [0] * (maxlen - len(seq))


def _gru(x: np.ndarray, kernel: np.ndarray, recurrent: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Keras GRU (reset_after=True, tanh/sigmoid), returning the full sequence.

    x: (batch, time, features). Gate order in the weights is update(z), reset(r), candidate(h).
    """
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    input_bias, recurrent_bias = bias[0], bias[1]
    projected = x @ kernel + input_bias  # (batch, time, 3*units)
    h = np.zeros((batch, units), dtype=x.dtype)
    out = np.empty((batch, steps, units), dtype=x.dtype)
    for t in range(steps):
        xz, xr, xh = np.split(projected[:, t], 3, axis=-1)
        hz, hr, hh = np.split(h @ recurrent + recurrent_bias, 3, axis=-1)
        z = _sigmoid(xz + hz)
        r = _sigmoid(xr + hr)
        candidate = np.tanh(xh + r * hh)
        h = z * h + (1.0 - z) * candidate
        out[:, t] = h
    return out


class NumpyNN:
    def __init__(self, weights_path: Path, meta_path: Path):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.sender_tokenizer = _Tokenizer(meta["tokenizers"]["sender"])
        self.message_tokenizer = _Tokenizer(meta["tokenizers"]["message"])
        self.sender_max_len = meta["sender_max_len"]
        self.message_max_len = meta["message_max_len"]
        self.labels: list[str] = meta["labels"]
        self.bn_epsilon = meta["batch_norm_epsilon"]
        with np.load(weights_path) as data:
            self.w = {name: data[name] for name in data.files}

    def predict_proba(self, sender_texts: list[str], message_texts: list[str]) -> np.ndarray:
        """Return class probabilities, shape (n, n_classes). Inputs must already be cleaned."""
        w = self.w
        sender_ids = np.array(
            [_pad_post(self.sender_tokenizer.encode(t), self.sender_max_len) for t in sender_texts], dtype=np.int64
        )
        message_ids = np.array(
            [_pad_post(self.message_tokenizer.encode(t), self.message_max_len) for t in message_texts], dtype=np.int64
        )

        # Message branch: embedding -> BiGRU -> max+avg pooling -> dense
        emb = w["message_embedding"][message_ids]
        forward = _gru(emb, w["gru_fw_kernel"], w["gru_fw_recurrent"], w["gru_fw_bias"])
        backward = _gru(emb[:, ::-1], w["gru_bw_kernel"], w["gru_bw_recurrent"], w["gru_bw_bias"])[:, ::-1]
        seq = np.concatenate([forward, backward], axis=-1)
        pooled = np.concatenate([seq.max(axis=1), seq.mean(axis=1)], axis=-1)
        message_feat = np.maximum(pooled @ w["message_dense_kernel"] + w["message_dense_bias"], 0.0)

        # Sender branch: embedding -> avg pooling -> dense
        sender_pooled = w["sender_embedding"][sender_ids].mean(axis=1)
        sender_feat = np.maximum(sender_pooled @ w["sender_dense_kernel"] + w["sender_dense_bias"], 0.0)

        # Merge -> batch norm (inference) -> dense stack -> softmax
        merged = np.concatenate([sender_feat, message_feat], axis=-1)
        merged = (merged - w["bn_mean"]) / np.sqrt(w["bn_var"] + self.bn_epsilon) * w["bn_gamma"] + w["bn_beta"]
        x = np.maximum(merged @ w["merged_dense_1_kernel"] + w["merged_dense_1_bias"], 0.0)
        x = np.maximum(x @ w["merged_dense_2_kernel"] + w["merged_dense_2_bias"], 0.0)
        logits = x @ w["classifier_kernel"] + w["classifier_bias"]
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=-1, keepdims=True)
