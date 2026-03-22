from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from preprocessing import preprocess_text
from utils import MODELS_DIR, interpret_label

st.set_page_config(page_title="SMS Classifier - Ensemble Learning", layout="centered")
st.title("SMS Classifier - Ensemble Learning")
st.caption("Predicts the SMS message type using the trained best ensemble model.")

model_path = MODELS_DIR / "best_model.pkl"
if not model_path.exists():
    st.error("Model not found. Run: python src/train_ensemble.py")
    st.stop()

model = joblib.load(model_path)

text = st.text_area("Enter an SMS message", height=180)
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess_text(text)
        pred = model.predict([cleaned])[0]
        probs = model.predict_proba([cleaned])[0]
        st.subheader(f"Prediction: {pred}")
        st.write(interpret_label(pred))
        df = pd.DataFrame({"label": model.classes_, "confidence": probs}).sort_values("confidence", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
