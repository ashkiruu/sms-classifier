"""Train and compare sender-aware ensemble models for SMS classification.

Key upgrade:
- Uses the sender field as an additional signal by prepending a normalized
  sender token to the cleaned message.
- Uses richer TF-IDF feature unions (word + character n-grams) for the
  strongest linear models.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_dataset, preprocess_dataframe
from utils import MODELS_DIR, get_report_path, setup_logger

logger = setup_logger(__name__)
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def build_union_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=10000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=12000,
                sublinear_tf=True,
            ),
        ),
    ])


def build_base_pipelines():
    cnb = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=10000,
                sublinear_tf=True,
            ),
        ),
        ("clf", ComplementNB(alpha=0.3)),
    ])

    lr = Pipeline([
        ("tfidf", build_union_vectorizer()),
        (
            "clf",
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    sgd = Pipeline([
        ("tfidf", build_union_vectorizer()),
        (
            "clf",
            SGDClassifier(
                loss="modified_huber",
                alpha=1e-5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=2500,
                tol=1e-3,
            ),
        ),
    ])
    return cnb, lr, sgd


def build_soft_voting():
    cnb, lr, sgd = build_base_pipelines()
    return VotingClassifier(
        estimators=[("cnb", cnb), ("lr", lr), ("sgd", sgd)],
        voting="soft",
        weights=[1, 2, 2],
    )


def build_stacking():
    cnb, lr, sgd = build_base_pipelines()
    final_estimator = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=[("cnb", cnb), ("lr", lr), ("sgd", sgd)],
        final_estimator=final_estimator,
        stack_method="predict_proba",
        passthrough=False,
        cv=2,
        n_jobs=None,
    )


def evaluate_model(model, X_eval, y_eval) -> dict:
    preds = model.predict(X_eval)
    return {
        "accuracy": float(accuracy_score(y_eval, preds)),
        "f1_macro": float(f1_score(y_eval, preds, average="macro")),
        "classification_report": classification_report(y_eval, preds, digits=4),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = preprocess_dataframe(load_dataset())
    X = df["model_input"]
    y = df["label"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    stratify_train = y_train_full if y_train_full.value_counts().min() >= 2 else None
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_train,
    )

    models = {
        "soft_voting": build_soft_voting(),
        "stacking": build_stacking(),
    }

    comparison_rows = []
    for name, model in models.items():
        logger.info("Training %s on train_sub split", name)
        fitted = clone(model)
        fitted.fit(X_train_sub, y_train_sub)
        metrics = evaluate_model(fitted, X_val, y_val)
        comparison_rows.append(
            {
                "model": name,
                "validation_accuracy": metrics["accuracy"],
                "validation_f1_macro": metrics["f1_macro"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_f1_macro", "validation_accuracy"], ascending=False
    ).reset_index(drop=True)
    comparison_path = get_report_path("ensemble_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    logger.info("Saved comparison table to %s", comparison_path)

    best_name = comparison.iloc[0]["model"]
    best_model = clone(models[best_name])
    best_model.fit(X_train_full, y_train_full)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train_full, y_train_full)
        model_path = MODELS_DIR / f"{name}_model.pkl"
        joblib.dump(fitted, model_path)
        logger.info("Saved %s to %s", name, model_path)

    best_path = MODELS_DIR / "ensemble_best_model.pkl"
    joblib.dump(best_model, best_path)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    logger.info("Saved best model (%s) to %s", best_name, best_path)

    metrics = {
        "best_model": best_name,
        "feature_strategy": "sender-aware combined text with word+character TF-IDF feature unions",
        "train_sub_rows": int(len(X_train_sub)),
        "validation_rows": int(len(X_val)),
        "train_full_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "validation_comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    get_report_path("best_model_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    get_report_path("best_model_classification_report.txt").write_text(test_metrics["classification_report"], encoding="utf-8")

    print("\nSplit summary:")
    print(f"Train sub rows : {len(X_train_sub)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Train full rows: {len(X_train_full)}")
    print(f"Test rows      : {len(X_test)}")
    print("\nValidation comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['f1_macro']:.4f}")
    print("\nClassification report:\n")
    print(test_metrics["classification_report"])


if __name__ == "__main__":
    main()
