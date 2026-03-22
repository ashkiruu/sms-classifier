"""Text cleaning and dataset loading utilities for the SMS classifier."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    try:
        _ENGLISH_STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        _ENGLISH_STOPWORDS = set()
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
    _ENGLISH_STOPWORDS = set()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, DEFAULT_TRAIN_DATA

logger = setup_logger(__name__)

if not _NLTK_AVAILABLE or not _ENGLISH_STOPWORDS:
    logger.warning(
        "NLTK stopwords unavailable; continuing with Tagalog stopwords only."
    )

TAGALOG_STOPWORDS: set[str] = {
    "ang", "ng", "mga", "na", "sa", "at", "ay", "ko", "mo", "ka",
    "siya", "kami", "kayo", "sila", "namin", "nila", "natin", "ninyo",
    "ito", "iyan", "iyon", "dito", "diyan", "doon", "oo", "hindi",
    "po", "ho", "din", "rin", "nga", "ba", "man", "lang", "lamang",
    "nang", "noon", "ngayon", "bukas", "kahapon", "pag", "para",
    "kung", "kaya", "pero", "dahil", "kasi", "naman", "muna",
    "talaga", "siguro", "yung", "yun", "yon", "si", "ni", "niya",
    "sya", "sana", "bago", "wag", "huwag", "pala", "eh", "ano",
}

REQUIRED_COLUMNS = ["message", "label"]


def load_dataset(filepath: str | Path = DEFAULT_TRAIN_DATA) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    rename_map = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"text", "message", "sms", "body"}:
            rename_map[col] = "message"
        elif low in {"category", "label", "class", "target"}:
            rename_map[col] = "label"
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset must contain columns {REQUIRED_COLUMNS}. Missing: {missing}")

    df = df.copy()
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"notif": "notifs"})
    df = df[df["message"].str.strip() != ""].dropna(subset=["message", "label"]).reset_index(drop=True)
    return df


def clean_text(text: str, keep_numbers: bool = True) -> str:
    text = str(text).strip().lower()
    if keep_numbers:
        text = re.sub(r"[^\w\s]", " ", text)
        text = text.replace("_", " ")
    else:
        text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    return [token for token in text.split() if len(token) > 1]


def remove_stopwords(tokens: list[str], include_tagalog: bool = True, extra_stopwords: Optional[set[str]] = None) -> list[str]:
    stop = set(_ENGLISH_STOPWORDS)
    if include_tagalog:
        stop.update(TAGALOG_STOPWORDS)
    if extra_stopwords:
        stop.update({w.lower() for w in extra_stopwords})
    return [t for t in tokens if t not in stop]


def preprocess_text(text: str, keep_numbers: bool = True, include_tagalog: bool = True, extra_stopwords: Optional[set[str]] = None) -> str:
    cleaned = clean_text(text, keep_numbers=keep_numbers)
    tokens = tokenize_text(cleaned)
    filtered = remove_stopwords(tokens, include_tagalog=include_tagalog, extra_stopwords=extra_stopwords)
    return " ".join(filtered)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "message", keep_numbers: bool = True, include_tagalog: bool = True, extra_stopwords: Optional[set[str]] = None) -> pd.DataFrame:
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame.")
    out = df.copy()
    out["clean_text"] = out[text_column].astype(str).apply(
        lambda x: preprocess_text(
            x,
            keep_numbers=keep_numbers,
            include_tagalog=include_tagalog,
            extra_stopwords=extra_stopwords,
        )
    )
    return out


def split_data(df: pd.DataFrame, text_column: str = "clean_text", target_column: str = "label", test_size: float = 0.2, random_state: int = 42):
    if text_column not in df.columns or target_column not in df.columns:
        raise KeyError(f"Expected columns '{text_column}' and '{target_column}'.")
    y = df[target_column]
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(df[text_column], y, test_size=test_size, random_state=random_state, stratify=stratify)


def run_preprocessing(filepath: str | Path = DEFAULT_TRAIN_DATA) -> None:
    df = load_dataset(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    processed = preprocess_dataframe(df)
    x_train, x_test, y_train, y_test = split_data(processed)
    print("\nSample preprocessing preview:\n")
    preview = processed[["message", "clean_text", "label"]].head(10)
    print(preview.to_string(index=False))
    print("\nTrain/Test split sizes:")
    print(f"X_train: {x_train.shape[0]}")
    print(f"X_test : {x_test.shape[0]}")
    print(f"y_train: {y_train.shape[0]}")
    print(f"y_test : {y_test.shape[0]}")


if __name__ == "__main__":
    run_preprocessing()
