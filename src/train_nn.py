"""Train the neural network model for SMS classification.

Upgrade:
- The NN now trains on the same sender-aware combined input used by the ensemble.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_dataset, preprocess_dataframe
from utils import MODELS_DIR, get_report_path, setup_logger

logger = setup_logger(__name__)


def save_history_plot(history, name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title(f"{name} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title(f"{name} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plot_path = get_report_path(f"nn_{name.lower().replace(' ', '_')}_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved training plot to %s", plot_path)


def normalize_numbers(text: str) -> str:
    return re.sub(r"\d+", "NUM", str(text).lower())


def build_and_train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading dataset using sender-aware preprocessing...")
    df = preprocess_dataframe(load_dataset())

    X_raw = [normalize_numbers(t) for t in df["model_input"]]

    max_words = 6000
    max_len = 120
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_raw)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(X_raw), maxlen=max_len, padding="post")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    initializers = {
        "Xavier": tf.keras.initializers.GlorotUniform(seed=42),
        "He": tf.keras.initializers.HeNormal(seed=42),
        "Random": tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42),
    }

    results_list = []
    best_f1 = -1.0
    best_model = None
    best_init_name = ""

    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001)

    for name, init_method in initializers.items():
        tf.keras.backend.clear_session()
        logger.info("Training NN with %s initializer", name)

        inputs = tf.keras.Input(shape=(max_len,))
        x = tf.keras.layers.Embedding(max_words, 64)(inputs)
        x = tf.keras.layers.SpatialDropout1D(0.3)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)

        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Concatenate()([max_pool, avg_pool])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_initializer=init_method,
            kernel_regularizer=l2(0.005),
        )(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(16, activation="relu", kernel_initializer=init_method)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=init_method)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            class_weight=class_weights_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        save_history_plot(history, name)

        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        current_f1 = f1_score(y_test, y_pred, average="macro")
        results_list.append(
            {
                "Initialization": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1_Macro": current_f1,
                "Kappa": cohen_kappa_score(y_test, y_pred),
            }
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model = model
            best_init_name = name

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(get_report_path("nn_initialization_comparison.csv"), index=False)

    best_model_path = MODELS_DIR / "nn_best_model.keras"
    best_model.save(best_model_path)
    joblib.dump(tokenizer, MODELS_DIR / "nn_tokenizer.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "nn_label_encoder.pkl")

    y_final_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_final_pred, target_names=label_encoder.classes_, digits=4)
    get_report_path("nn_best_model_report.txt").write_text(report, encoding="utf-8")

    metrics_summary = {
        "best_initialization": best_init_name,
        "feature_strategy": "sender-aware combined input with number normalization",
        "test_results": results_df.to_dict(orient="records"),
    }
    get_report_path("nn_metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    logger.info("Training complete. Best initializer: %s", best_init_name)
    print("\n=== NN EXPERIMENT RESULTS ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    build_and_train()
