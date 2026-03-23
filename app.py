from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Integrate your existing logic
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import preprocess_text
from utils import MODELS_DIR, interpret_label

app = Flask(__name__)

# --- Global Model Loading ---
# We load these once when the server starts to keep it fast
ensemble_model = joblib.load(MODELS_DIR / "ensemble_best_model.pkl")
nn_model = tf.keras.models.load_model(MODELS_DIR / "nn_best_model.keras")
nn_tokenizer = joblib.load(MODELS_DIR / "nn_tokenizer.pkl")
nn_le = joblib.load(MODELS_DIR / "nn_label_encoder.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get("message", "")
    model_choice = data.get("model_type", "ensemble")

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = preprocess_text(raw_text)

    if model_choice == "nn":
        # Neural Network Logic
        seq = nn_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post')
        probs = nn_model.predict(padded, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_label = nn_le.inverse_transform([pred_idx])[0]
        labels = nn_le.classes_.tolist()
    else:
        # Ensemble Logic
        pred_label = ensemble_model.predict([cleaned])[0]
        probs = ensemble_model.predict_proba([cleaned])[0]
        labels = ensemble_model.classes_.tolist()

    return jsonify({
        "prediction": pred_label,
        "interpretation": interpret_label(pred_label),
        "confidence_scores": dict(zip(labels, [round(float(p), 4) for p in probs]))
    })

if __name__ == '__main__':
    app.run(debug=True)