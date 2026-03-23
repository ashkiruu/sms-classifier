"""
confidence_test.py — Confidence accuracy test on 15 held-out SMS records.

Loads the ensemble and neural network models, runs predictions on the
15 manually curated records, and prints a detailed per-record report
plus an overall accuracy summary.

Usage:
    python src/confidence_test.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make src/ siblings importable when run directly ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_text
from utils import setup_logger, PROJECT_ROOT

logger = setup_logger(__name__)

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT       = PROJECT_ROOT
MODELS_DIR = ROOT / "models"

# ── 15 confidence-test records ────────────────────────────────────────────────
RECORDS = [
    {
        "date":   "23/09/2024 16:17",
        "sender": "BDO Deals",
        "text":   "Earn Peso Points effortlessly! Simply Scan to Pay with BDO Pay in stores and get 50 Peso Points for every 1,000 spent from August 15 to October 31, 2024. Download BDO Pay! T&Cs apply. DTI200125",
        "label":  "ads",
    },
    {
        "date":   "22/09/2024 18:01",
        "sender": "GCash",
        "text":   "Did you request to SEND MONEY to CH**O NI***E E.'s GCash number, 09706864475 with amount of PHP 190.00? If not, DON'T ENTER YOUR OTP ON ANY SITE OR SEND IT TO ANYONE because IT'S A SCAM! If you requested, your OTP is 652025.",
        "label":  "otp",
    },
    {
        "date":   "22/09/2024 12:48",
        "sender": "TNT",
        "text":   "DOBLENG SAYA, DOBLENG PANALO! HANGGANG TODAY NA LANG: sagot na namin ang 2X DATA mo sa TIKTOK SAYA 50 (3 GB + FREE 3 GB)! 2X DATA ka na rin sa: SURFSAYA 30/49/99 ALL DATA 50/99 ALL DATA+ 75/149 TIKTOK SAYA 99/149 GIGA VIDEO/STORIES/GAMES 60/120 DOBLE GIGA VIDEO+/STORIES+ 75/149 TRIPLE DATA VIDEO+/STORIES+ 75/149 UTP+ 30 Kaya load na via https://smrt.ph/GetSmartApp o i-dial ang *123#",
        "label":  "notifs",
    },
    {
        "date":   "05/09/2024 2:31",
        "sender": "TingogTayo",
        "text":   "ISIP Beneficiary Message Part 2 of 2 Plus many more prizes to be given away :) Please be sure to bring the following: 1. Wear you ISIP bracelet for entry 2. Screenshot of your registered ISIP IDs 3. Valid ID (Original with 2 Xerox Copies, eg. validated school ID and any Valid Government ID) 4. Original & 1 Xerox Copy of Barangay Certificate (indicating the following ONLY: that you are an indigent or belong to the indigent group, for the purpose of DSWD financial assistance) See you there!",
        "label":  "gov",
    },
    {
        "date":   "03/09/2024 18:47",
        "sender": "BagongPinas",
        "text":   "Ang iyong One Time Pin ay: 191023",
        "label":  "gov",
    },
    {
        "date":   "22/08/2024 13:00",
        "sender": "6.39621E+11",
        "text":   "Manatili lamang sa bahay,tuturuan kita,araw-araw 1000+,Telegram:@apk552",
        "label":  "spam",
    },
    {
        "date":   "22/08/2024 10:56",
        "sender": "BDO",
        "text":   "APP-GRADE to the new BDO Online app to continue to view your account balances and make transactions. Download the BDO Online app now!",
        "label":  "notifs",
    },
    {
        "date":   "21/08/2024 14:55",
        "sender": "CIMB_Bank",
        "text":   "CIMB MaxSave is now available for a shorter 3-month term. Start saving now for a minimum deposit of ₱5,000! Visit CIMB website to learn more.",
        "label":  "ads",
    },
    {
        "date":   "21/08/2024 0:44",
        "sender": "3404",
        "text":   "Use 868178 for two-factor authentication on Facebook.",
        "label":  "otp",
    },
    {
        "date":   "24/06/2024 1:35",
        "sender": "6.39703E+11",
        "text":   "Spin the Daily Lucky Wheel atGojplaywild.de and win a Samsung Galaxy S23+ 5G worth P42,999! New members grab a 130% bonus up to P1000. Don't miss out!",
        "label":  "spam",
    },
    {
        "date":   "20/06/2024 17:28",
        "sender": "6.3965E+11",
        "text":   "Join the Slot Tournament now! Get 3000p FREE on sign-up + a chance to win a VIVO V30e 256GB! Don't miss out on the excitement! sbetphlucks.eu",
        "label":  "spam",
    },
    {
        "date":   "17/06/2024 14:35",
        "sender": "BDO Deals",
        "text":   "Get up to 1,000 bonus Peso Points with BDO Pay! Simply Scan to Pay to earn 2 bonus Peso Points per transaction. Promo runs from June 17 to July 31, 2024. Visit the BDO website, click Deals and search Bonus Points to learn more. T&Cs apply. DTI195469",
        "label":  "ads",
    },
    {
        "date":   "14/06/2024 21:14",
        "sender": "TNT",
        "text":   "100% Cashback handog ng TNT at Maya! I-download at i-upgrade ang Maya app at bumili ng P100 load for yourself sa Maya. I-claim mo na bago pa mawala: https://official.maya.ph/3xMF/SmartSignUp T&Cs apply. Users who qualify for the promo will receive their reward within 3 business days. DTI192508",
        "label":  "notifs",
    },
    {
        "date":   "21/12/2023 12:39",
        "sender": "NTC",
        "text":   "1/3 This is a public service advisory from the National Telecommunications Commission, Telecommunications Connectivity, Inc. and SMART.",
        "label":  "gov",
    },
    {
        "date":   "27/10/2023 15:48",
        "sender": "GOMO",
        "text":   "GOMO Anniv ain't over yet! Get a FREE UPsize from PICKUP COFFEE tomorrow, October 28! Just show your unique referral code found in the GOMO PH app, under Account > Refer a Friend. Valid in selected PICKUP COFFEE branches nationwide. Visit https://bit.ly/GOMOFAQTopic for more info. No advisories? Text OFF to 2686 for free.",
        "label":  "ads",
    },
]


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_ensemble_model():
    """
    Load the saved ensemble model from models/.

    The model is a VotingClassifier or StackingClassifier whose base
    estimators are full sklearn Pipelines — each pipeline contains its
    own TfidfVectorizer, so no separate vectorizer file is needed.

    Returns:
        Tuple of (model, None) — vectorizer is always None here.

    Raises:
        FileNotFoundError: If ensemble_best_model.pkl is missing.
    """
    import joblib

    model_path = MODELS_DIR / "ensemble_best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found at: {model_path}\n"
            "Ensure 'ensemble_best_model.pkl' is inside models/."
        )

    model = joblib.load(model_path)
    logger.info(
        "Loaded ensemble model (%s) from %s",
        type(model).__name__, model_path,
    )
    return model, None  # vectorizer baked into base Pipeline estimators


def load_nn_model():
    """
    Load the saved neural network model and its tokenizer/vectorizer.

    Returns:
        Tuple of (model, vectorizer_or_tokenizer). Second element may be None.

    Raises:
        FileNotFoundError: If nn_best_model.keras is missing.
    """
    model_path = MODELS_DIR / "nn_best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Neural network model not found at: {model_path}\n"
            "Ensure 'nn_best_model.keras' is inside models/."
        )

    try:
        from tensorflow import keras  # type: ignore
    except ImportError:
        try:
            import keras  # type: ignore
        except ImportError:
            raise ImportError(
                "TensorFlow/Keras is not installed. "
                "Run: pip install tensorflow"
            )

    model = keras.models.load_model(str(model_path))
    logger.info("Loaded neural network model from %s", model_path)

    # Try loading a matching tokenizer or vectorizer
    vectorizer = None
    import joblib
    for vname in ("nn_tokenizer.pkl", "tokenizer.pkl", "vectorizer.pkl", "tfidf.pkl"):
        vpath = MODELS_DIR / vname
        if vpath.exists():
            vectorizer = joblib.load(vpath)
            logger.info("Loaded NN vectorizer/tokenizer from %s", vpath)
            break

    return model, vectorizer


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_ensemble(model, vectorizer, texts: list[str]) -> list[str]:
    """
    Run inference with the ensemble model.

    The ensemble is a VotingClassifier or StackingClassifier whose base
    estimators are full Pipelines (TfidfVectorizer + classifier each).
    It accepts raw text directly — no separate vectorizer needed.

    Args:
        model:      Loaded sklearn VotingClassifier / StackingClassifier.
        vectorizer: Unused for this model (vectorizer is baked into each Pipeline).
        texts:      List of preprocessed text strings.

    Returns:
        List of predicted class label strings.
    """
    predictions = model.predict(texts)
    return list(predictions)


def predict_nn(model, vectorizer, texts: list[str], label_classes: list[str]) -> list[str]:
    """
    Run inference with the neural network model.

    Args:
        model:         Loaded Keras model.
        vectorizer:    Fitted tokenizer or vectorizer, or None.
        texts:         List of preprocessed text strings.
        label_classes: Ordered list of class names matching model output neurons.

    Returns:
        List of predicted class label strings.
    """
    if vectorizer is None:
        raise ValueError(
            "NN model requires a tokenizer/vectorizer but none was found. "
            "Save it as 'models/nn_tokenizer.pkl' or 'models/tokenizer.pkl'."
        )

    # Handle both Keras Tokenizer and sklearn vectorizers
    if hasattr(vectorizer, "texts_to_sequences"):
        # Keras Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
        sequences = vectorizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=model.input_shape[1], padding="post")
    elif hasattr(vectorizer, "transform"):
        # sklearn-style vectorizer (TF-IDF, etc.)
        X = vectorizer.transform(texts).toarray()
    else:
        raise ValueError("Unrecognised vectorizer type: cannot transform texts.")

    probs = model.predict(X, verbose=0)
    indices = np.argmax(probs, axis=1)
    return [label_classes[i] for i in indices]


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(
    records:       list[dict],
    clean_texts:   list[str],
    ensemble_preds: list[str] | None,
    nn_preds:       list[str] | None,
) -> None:
    """
    Print a formatted per-record prediction report and accuracy summary.

    Args:
        records:        Original 15 record dicts with 'text' and 'label' keys.
        clean_texts:    Preprocessed versions of each text.
        ensemble_preds: Ensemble model predictions (or None if unavailable).
        nn_preds:       NN model predictions (or None if unavailable).
    """
    labels      = [r["label"] for r in records]
    n           = len(records)
    col_w       = 10  # column width for label fields

    print("\n" + "═" * 100)
    print("  CONFIDENCE TEST REPORT  —  15 records")
    print("═" * 100)
    print(
        f"  {'#':<4} {'True Label':<{col_w}} "
        f"{'Ensemble':<{col_w}} {'E.Match':<8} "
        f"{'NN':<{col_w}} {'N.Match':<8} "
        f"  Sender"
    )
    print("─" * 100)

    ens_correct = 0
    nn_correct  = 0

    for i, (rec, true_label) in enumerate(zip(records, labels), start=1):
        ens_pred  = ensemble_preds[i - 1] if ensemble_preds else "—"
        nn_pred   = nn_preds[i - 1]       if nn_preds       else "—"
        ens_match = "✓" if ens_pred == true_label else "✗"
        nn_match  = "✓" if nn_pred  == true_label else "✗"

        if ensemble_preds and ens_pred == true_label:
            ens_correct += 1
        if nn_preds and nn_pred == true_label:
            nn_correct  += 1

        sender = rec["sender"][:20]
        print(
            f"  {i:<4} {true_label:<{col_w}} "
            f"{ens_pred:<{col_w}} {ens_match:<8} "
            f"{nn_pred:<{col_w}} {nn_match:<8} "
            f"  {sender}"
        )

    print("═" * 100)
    print("  ACCURACY SUMMARY")
    print("─" * 100)

    if ensemble_preds:
        ens_acc = ens_correct / n * 100
        print(f"  Ensemble model : {ens_correct}/{n} correct  →  {ens_acc:.1f}%")
    else:
        print("  Ensemble model : not loaded")

    if nn_preds:
        nn_acc = nn_correct / n * 100
        print(f"  Neural network : {nn_correct}/{n} correct  →  {nn_acc:.1f}%")
    else:
        print("  Neural network : not loaded")

    if ensemble_preds and nn_preds:
        # Simple majority vote (tie goes to ensemble)
        combined_correct = sum(
            1 for t, e, n_ in zip(labels, ensemble_preds, nn_preds)
            if (e == t and n_ == t) or (e == t) or (n_ == t)
        )
        # Strict agreement accuracy
        agree_correct = sum(
            1 for t, e, n_ in zip(labels, ensemble_preds, nn_preds)
            if e == t and n_ == t
        )
        print(f"  Both correct   : {agree_correct}/{n}  →  {agree_correct/n*100:.1f}%")

    print("═" * 100 + "\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_confidence_test() -> None:
    """
    Preprocess the 15 test records, load both models, run predictions,
    and print the full confidence report.
    """
    # ── 1. Preprocess texts ──────────────────────────────────────────────────
    logger.info("Preprocessing %d test records…", len(RECORDS))
    clean_texts = [
        preprocess_text(r["text"], keep_numbers=True, include_tagalog=True)
        for r in RECORDS
    ]

    # ── 2. Load NN label encoder ─────────────────────────────────────────────
    import joblib
    label_classes = ["ads", "gov", "notifs", "otp", "spam"]  # fallback
    encoder_path  = MODELS_DIR / "nn_label_encoder.pkl"
    if encoder_path.exists():
        encoder       = joblib.load(encoder_path)
        label_classes = list(encoder.classes_)
        logger.info("Loaded label encoder — classes: %s", label_classes)
    else:
        logger.warning("nn_label_encoder.pkl not found, using default class order: %s", label_classes)

    ensemble_preds = None
    nn_preds       = None

    # ── 3. Ensemble model ────────────────────────────────────────────────────
    try:
        ens_model, ens_vec = load_ensemble_model()
        ensemble_preds = predict_ensemble(ens_model, ens_vec, clean_texts)
        logger.info("Ensemble predictions complete.")
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.warning("Ensemble model skipped — reason: %s", exc)

    # ── 3. Neural network model ──────────────────────────────────────────────
    try:
        nn_model, nn_vec = load_nn_model()
        nn_preds = predict_nn(nn_model, nn_vec, clean_texts, label_classes)
        logger.info("Neural network predictions complete.")
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.warning("Neural network model skipped: %s", exc)

    # ── 4. Print report ──────────────────────────────────────────────────────
    print_report(RECORDS, clean_texts, ensemble_preds, nn_preds)


if __name__ == "__main__":
    run_confidence_test()