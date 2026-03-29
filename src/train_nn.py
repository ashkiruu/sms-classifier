"""
Train the neural network model for SMS classification.

Laptop-friendly upgrade:
- True dual-input neural network for sender + message
- Lightweight architecture compatible with tf.keras on CPU
- Avoids masking warnings by not using mask_zero=True
- Preserves output/report structure and saved artifact names
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

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


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
    return re.sub(r"\d+", "NUM", str(text).lower()).strip()


def clean_sender_text(text: str) -> str:
    text = normalize_numbers(text)
    text = re.sub(r"[^a-z0-9_@.+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


def clean_message_text(text: str) -> str:
    text = normalize_numbers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty"


def build_dual_sequences(
    sender_texts,
    message_texts,
    sender_max_words=1500,
    message_max_words=6000,
    sender_max_len=8,
    message_max_len=60,
):
    sender_tokenizer = Tokenizer(
        num_words=sender_max_words,
        oov_token="<OOV>",
        lower=True,
        filters=""
    )
    message_tokenizer = Tokenizer(
        num_words=message_max_words,
        oov_token="<OOV>",
        lower=True,
        filters=""
    )

    sender_tokenizer.fit_on_texts(sender_texts)
    message_tokenizer.fit_on_texts(message_texts)

    X_sender = pad_sequences(
        sender_tokenizer.texts_to_sequences(sender_texts),
        maxlen=sender_max_len,
        padding="post",
        truncating="post"
    )

    X_message = pad_sequences(
        message_tokenizer.texts_to_sequences(message_texts),
        maxlen=message_max_len,
        padding="post",
        truncating="post"
    )

    tokenizer_bundle = {
        "sender_tokenizer": sender_tokenizer,
        "message_tokenizer": message_tokenizer,
        "sender_max_len": sender_max_len,
        "message_max_len": message_max_len,
        "sender_max_words": sender_max_words,
        "message_max_words": message_max_words,
    }

    return X_sender, X_message, tokenizer_bundle


def build_model(
    sender_vocab_size: int,
    message_vocab_size: int,
    sender_max_len: int,
    message_max_len: int,
    num_classes: int,
    initializer,
):
    sender_input = tf.keras.Input(shape=(sender_max_len,), name="sender_input")
    message_input = tf.keras.Input(shape=(message_max_len,), name="message_input")

    # Sender branch: lightweight, sender strings are short
    sender_x = tf.keras.layers.Embedding(
        input_dim=sender_vocab_size,
        output_dim=16,
        embeddings_regularizer=l2(1e-5),
        name="sender_embedding"
    )(sender_input)
    sender_x = tf.keras.layers.SpatialDropout1D(0.15)(sender_x)
    sender_x = tf.keras.layers.GlobalAveragePooling1D(name="sender_pool")(sender_x)
    sender_x = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=l2(1e-4),
        name="sender_dense"
    )(sender_x)

    # Message branch: sequence-aware but still laptop-friendly
    message_x = tf.keras.layers.Embedding(
        input_dim=message_vocab_size,
        output_dim=64,
        embeddings_regularizer=l2(1e-5),
        name="message_embedding"
    )(message_input)
    message_x = tf.keras.layers.SpatialDropout1D(0.20)(message_x)

    message_x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            32,
            return_sequences=True,
            dropout=0.20,
            recurrent_dropout=0.0,
            kernel_regularizer=l2(1e-4),
        ),
        name="message_bigru"
    )(message_x)

    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="message_max_pool")(message_x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="message_avg_pool")(message_x)
    message_x = tf.keras.layers.Concatenate(name="message_pool_concat")([max_pool, avg_pool])

    message_x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=l2(1e-4),
        name="message_dense"
    )(message_x)
    message_x = tf.keras.layers.Dropout(0.35)(message_x)

    merged = tf.keras.layers.Concatenate(name="merged_features")([sender_x, message_x])

    # Simple interaction enhancement
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=l2(0.002),
        name="merged_dense_1"
    )(merged)
    merged = tf.keras.layers.Dropout(0.40)(merged)

    merged = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=l2(0.001),
        name="merged_dense_2"
    )(merged)
    merged = tf.keras.layers.Dropout(0.25)(merged)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=initializer,
        name="classifier"
    )(merged)

    model = tf.keras.Model(
        inputs={"sender_input": sender_input, "message_input": message_input},
        outputs=outputs,
        name="dual_input_sms_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_and_train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading dataset using sender-aware preprocessing...")
    df = preprocess_dataframe(load_dataset())

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the processed dataframe.")

    # Use preprocessing outputs directly
    sender_texts = [clean_sender_text(x) for x in df["clean_sender"].fillna("unknown").astype(str)]
    message_texts = [clean_message_text(x) for x in df["clean_text"].fillna("").astype(str)]

    X_sender, X_message, tokenizer_bundle = build_dual_sequences(
        sender_texts=sender_texts,
        message_texts=message_texts,
        sender_max_words=1500,
        message_max_words=6000,
        sender_max_len=8,
        message_max_len=60,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    num_classes = len(label_encoder.classes_)

    X_sender_train, X_sender_test, X_message_train, X_message_test, y_train, y_test = train_test_split(
        X_sender,
        X_message,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))

    initializers = {
        "Xavier": tf.keras.initializers.GlorotUniform(seed=SEED),
        "He": tf.keras.initializers.HeNormal(seed=SEED),
        "Random": tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=SEED),
    }

    results_list = []
    best_f1 = -1.0
    best_model = None
    best_init_name = ""

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )

    sender_vocab_size = min(tokenizer_bundle["sender_max_words"], len(tokenizer_bundle["sender_tokenizer"].word_index) + 1)
    message_vocab_size = min(tokenizer_bundle["message_max_words"], len(tokenizer_bundle["message_tokenizer"].word_index) + 1)

    for name, init_method in initializers.items():
        tf.keras.backend.clear_session()
        logger.info("Training NN with %s initializer", name)

        model = build_model(
            sender_vocab_size=sender_vocab_size,
            message_vocab_size=message_vocab_size,
            sender_max_len=tokenizer_bundle["sender_max_len"],
            message_max_len=tokenizer_bundle["message_max_len"],
            num_classes=num_classes,
            initializer=init_method,
        )

        history = model.fit(
            {"sender_input": X_sender_train, "message_input": X_message_train},
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            class_weight=class_weights_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        save_history_plot(history, name)

        y_pred_probs = model.predict(
            {"sender_input": X_sender_test, "message_input": X_message_test},
            verbose=0
        )
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

    # Keep the same filename, but store both tokenizers in one bundle
    joblib.dump(tokenizer_bundle, MODELS_DIR / "nn_tokenizer.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "nn_label_encoder.pkl")

    y_final_pred = np.argmax(
        best_model.predict(
            {"sender_input": X_sender_test, "message_input": X_message_test},
            verbose=0
        ),
        axis=1
    )

    report = classification_report(
        y_test,
        y_final_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
    get_report_path("nn_best_model_report.txt").write_text(report, encoding="utf-8")

    metrics_summary = {
        "best_initialization": best_init_name,
        "feature_strategy": "dual-input sender+message with sender embedding and BiGRU message encoder",
        "test_results": results_df.to_dict(orient="records"),
    }
    get_report_path("nn_metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8"
    )

    logger.info("Training complete. Best initializer: %s", best_init_name)
    print("\n=== NN EXPERIMENT RESULTS ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    build_and_train()