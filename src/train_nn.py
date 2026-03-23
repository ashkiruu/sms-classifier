import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re # Added for Number Normalization
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.utils import class_weight

# Import your group's shared utilities
from preprocessing import load_dataset, preprocess_dataframe
from utils import MODELS_DIR, get_report_path, setup_logger

logger = setup_logger(__name__)

def save_history_plot(history, name):
    """Generates desktop/mobile friendly training plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title(f'{name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss Plot
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title(f'{name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = get_report_path(f"nn_{name.lower().replace(' ', '_')}_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved training plot to {plot_path}")

def build_and_train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. LOAD & PREPARE DATA 
    # ==========================================
    logger.info("Loading dataset using shared preprocessing...")
    df = preprocess_dataframe(load_dataset())
    
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    
    # --- UPGRADE: Number Normalization ---
    # Replaces all digits with the token 'NUM' to prevent OTP overfitting
    def normalize_numbers(text):
        return re.sub(r'\d+', 'NUM', str(text).lower())

    # Apply the normalization to the raw text before tokenizing
    X_raw = [normalize_numbers(t) for t in df[text_col]]
    
    # Tokenization
    MAX_WORDS = 5000
    MAX_LEN = 100 
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_raw)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(X_raw), maxlen=MAX_LEN, padding='post')

    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    NUM_CLASSES = len(le.classes_)

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.20, stratify=y, random_state=42
    )

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    # ==========================================
    # 2. EXPERIMENT WITH INITIALIZATIONS
    # ==========================================
    initializers = {
        'Xavier': tf.keras.initializers.GlorotUniform(seed=42),
        'Heuristic': tf.keras.initializers.HeNormal(seed=42),
        'Random': tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
    }

    results_list = []
    best_f1 = -1
    best_model = None
    best_init_name = ""

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    for name, init_method in initializers.items():
        tf.keras.backend.clear_session()
        logger.info(f"🚀 Training NN with {name} Initialization")

        # --- IMPROVED HYBRID POOLING ARCHITECTURE ---
        inputs = tf.keras.Input(shape=(MAX_LEN,))

        x = tf.keras.layers.Embedding(MAX_WORDS, 64)(inputs)
        x = tf.keras.layers.SpatialDropout1D(0.3)(x)

        # BiLSTM layers (with dropout to reduce overfitting)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(16, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)

        # 🔥 Combine Max + Average Pooling (CRITICAL FIX)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Concatenate()([max_pool, avg_pool])

        # Dense layers
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_initializer=init_method,
            kernel_regularizer=l2(0.005)
        )(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(
            16,
            activation='relu',
            kernel_initializer=init_method
        )(x)

        outputs = tf.keras.layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            kernel_initializer=init_method
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32,
            validation_split=0.1, 
            class_weight = {k: v * 0.8 for k, v in class_weights_dict.items()},
            callbacks=[early_stop, reduce_lr], 
            verbose=0
        )

        # Save Visual History
        save_history_plot(history, name)

        # Evaluation
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        current_f1 = f1_score(y_test, y_pred, average='macro')
        results_list.append({
            "Initialization": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1_Macro": current_f1,
            "Kappa": cohen_kappa_score(y_test, y_pred)
        })

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model = model
            best_init_name = name

    # ==========================================
    # 3. SAVE MODELS & REPORTS
    # ==========================================
    results_df = pd.DataFrame(results_list)
    comparison_path = get_report_path("nn_initialization_comparison.csv")
    results_df.to_csv(comparison_path, index=False)

    best_model_path = MODELS_DIR / "nn_best_model.keras"
    best_model.save(best_model_path)
    
    joblib.dump(tokenizer, MODELS_DIR / "nn_tokenizer.pkl")
    joblib.dump(le, MODELS_DIR / "nn_label_encoder.pkl")

    y_final_pred = np.argmax(best_model.predict(X_test), axis=1)
    report = classification_report(y_test, y_final_pred, target_names=le.classes_, digits=4)
    get_report_path("nn_best_model_report.txt").write_text(report)

    metrics_summary = {
        "best_initialization": best_init_name,
        "test_results": results_df.to_dict(orient="records")
    }
    get_report_path("nn_metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))

    logger.info(f"✅ Training Complete. Best Init: {best_init_name}")
    print("\n=== 📊 NN EXPERIMENT RESULTS ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    build_and_train()