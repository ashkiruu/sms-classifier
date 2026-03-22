import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

def build_and_train():
    # ==========================================
    # 1. LOAD & PREPARE DATA
    # ==========================================
    print("Loading preprocessed dataset...")
    df = pd.read_csv(r"data\raw\tagalog-sms.csv") 
    
    # Use the cleaned text from your preprocessing step
    # If the column name is 'clean_text', ensure it's used
    text_col = 'text' # Update to 'clean_text' if you saved the output of preprocessing.py
    df[text_col] = df[text_col].astype(str).str.lower()
    
    MAX_WORDS = 5000
    MAX_LEN = 100 # Increased slightly to capture full Tagalog context
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[text_col])
    X = pad_sequences(tokenizer.texts_to_sequences(df[text_col]), maxlen=MAX_LEN, padding='post')

    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    NUM_CLASSES = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # Class Weights are critical for your 2.2% 'Gov' class
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    # ==========================================
    # 2. EXPERIMENT WITH INITIALIZATIONS (Keras 3 Style)
    # ==========================================
    initializers = {
        'Xavier (Glorot)': tf.keras.initializers.GlorotUniform(seed=42),
        'Heuristic (He)': tf.keras.initializers.HeNormal(seed=42),
        'Generic (Random)': tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
    }

    results_list = []
    
    # Patience set to 5 for a more thorough test of the plateaus
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for name, init_method in initializers.items():
        print(f"\n{'='*40}")
        print(f"🚀 Training Model with {name} Initialization")
        print(f"{'='*40}")

        # Build Architecture with an extra layer for better feature extraction
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(MAX_WORDS, 64), # Increased embedding dim to 64
            tf.keras.layers.Bidirectional(tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')), 
            # Note: Bi-directional pooling is a Keras 3 trick for better context
            
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=init_method),
            tf.keras.layers.Dropout(0.4), # Increased dropout to strictly avoid overfitting
            
            tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init_method),
            
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', kernel_initializer=init_method)
        ])

        # Using a slightly lower learning rate for stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        history = model.fit(
            X_train, y_train, 
            epochs=50, # Increased epochs because EarlyStopping will catch the plateau
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights_dict,
            callbacks=[early_stop],
            verbose=1
        )

        # ==========================================
        # 3. METRICS & CONFIDENCE EVALUATION
        # ==========================================
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate Average Confidence
        confidences = np.max(y_pred_probs, axis=1)
        avg_conf = np.mean(confidences)

        results_list.append({
            "Initialization": name,
            "Final Accuracy": f"{accuracy_score(y_test, y_pred)*100:.2f}%",
            "Avg Confidence": f"{avg_conf*100:.2f}%",
            "F1 (Macro)": f"{f1_score(y_test, y_pred, average='macro', zero_division=0)*100:.2f}%",
            "Kappa Score": f"{cohen_kappa_score(y_test, y_pred):.4f}"
        })

    # ==========================================
    # 4. FINAL REPORT
    # ==========================================
    print("\n\n=== 📊 INITIALIZATION EXPERIMENT RESULTS ===")
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    build_and_train()