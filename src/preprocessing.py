"""
preprocessing.py — Text cleaning and feature preparation for the SMS Classifier.

Pipeline overview:
    raw text
        └─► clean_text()
                └─► tokenize_text()
                        └─► remove_stopwords()
                                └─► preprocess_text()   (full single-string pipeline)

    DataFrame-level helpers:
        preprocess_dataframe()   — applies preprocess_text() column-wise
        split_data()             — stratified 80/20 train/test split

Dataset format: CSV file (tagalog-sms.csv) with columns 'message' and 'label'.

Usage:
    python src/preprocessing.py
"""

import re
import string
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import nltk
    from nltk.corpus import stopwords
    # Download quietly on first run
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt",     quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# ── Make sure src/ sibling modules are importable when run directly ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import setup_logger, get_data_path

logger = setup_logger(__name__)

# Warn once at import time — not once per row
if not _NLTK_AVAILABLE:
    logger.warning(
        "NLTK is not installed. English stopwords will be skipped. "
        "Fix with: pip install nltk"
    )

# Pre-load English stopwords once at module level (empty set if NLTK missing)
_ENGLISH_STOPWORDS: set[str] = (
    set(stopwords.words("english")) if _NLTK_AVAILABLE else set()
)

# ── Tagalog stopwords (curated; extend as needed) ────────────────────────────
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


# ── 1. Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str, keep_numbers: bool = True) -> str:
    """
    Normalise a raw SMS message.

    Steps applied (in order):
        1. Cast to string and strip leading/trailing whitespace.
        2. Convert to lowercase.
        3. Optionally preserve digit sequences (useful for OTP codes).
        4. Remove all remaining punctuation characters.
        5. Collapse multiple consecutive whitespace characters to a single space.

    Args:
        text:         Raw SMS string.
        keep_numbers: When True, digit characters are preserved so that OTP
                      codes remain intact.  Set to False to strip digits too.

    Returns:
        Cleaned, normalised string.

    Example:
        >>> clean_text("  Hello, World!! Your OTP is 123456.  ")
        'hello world your otp is 123456'
    """
    text = str(text).strip().lower()

    if keep_numbers:
        # Remove punctuation but keep digits and whitespace
        text = re.sub(r"[^\w\s]", " ", text)   # \w keeps [a-z0-9_]
        text = re.sub(r"_", " ", text)          # remove underscore separately
    else:
        # Remove all punctuation AND digits
        text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 2. Tokenisation ──────────────────────────────────────────────────────────

def tokenize_text(text: str) -> list[str]:
    """
    Split a cleaned string into individual word tokens.

    Uses simple whitespace splitting for speed and language-agnosticism.
    Single-character tokens are discarded as they are almost always noise.

    Args:
        text: A cleaned (lowercased, punctuation-removed) string.

    Returns:
        List of token strings.

    Example:
        >>> tokenize_text("hello world your otp is 123456")
        ['hello', 'world', 'your', 'otp', 'is', '123456']
    """
    return [token for token in text.split() if len(token) > 1]


# ── 3. Stopword removal ──────────────────────────────────────────────────────

def remove_stopwords(
    tokens:                  list[str],
    include_tagalog:         bool = True,
    extra_stopwords:         Optional[set[str]] = None,
) -> list[str]:
    """
    Filter stopwords from a token list.

    Combines NLTK English stopwords with a built-in Tagalog list.
    Callers can inject additional domain-specific stopwords via
    *extra_stopwords*.

    Args:
        tokens:          List of word tokens (already lowercased).
        include_tagalog: If True, the built-in Tagalog stopword list is used
                         in addition to English stopwords.
        extra_stopwords: Optional set of additional words to remove.

    Returns:
        Filtered token list with stopwords removed.

    Example:
        >>> remove_stopwords(["hello", "ang", "world", "is", "great"])
        ['hello', 'world', 'great']
    """
    stop = set(_ENGLISH_STOPWORDS)  # copy the pre-loaded set

    if include_tagalog:
        stop.update(TAGALOG_STOPWORDS)

    if extra_stopwords:
        stop.update({w.lower() for w in extra_stopwords})

    return [t for t in tokens if t not in stop]


# ── 4. Full single-string pipeline ───────────────────────────────────────────

def preprocess_text(
    text:            str,
    keep_numbers:    bool = True,
    include_tagalog: bool = True,
    extra_stopwords: Optional[set[str]] = None,
) -> str:
    """
    Apply the complete preprocessing pipeline to a single SMS string.

    Pipeline:
        clean_text → tokenize_text → remove_stopwords → rejoin to string

    Args:
        text:            Raw SMS string.
        keep_numbers:    Passed through to clean_text().
        include_tagalog: Passed through to remove_stopwords().
        extra_stopwords: Passed through to remove_stopwords().

    Returns:
        Preprocessed string ready for vectorisation.

    Example:
        >>> preprocess_text("  Kumusta ka? Your OTP is 987654.  ")
        'kumusta otp 987654'
    """
    cleaned  = clean_text(text, keep_numbers=keep_numbers)
    tokens   = tokenize_text(cleaned)
    filtered = remove_stopwords(
        tokens,
        include_tagalog=include_tagalog,
        extra_stopwords=extra_stopwords,
    )
    return " ".join(filtered)


# ── 5. DataFrame-level preprocessing ────────────────────────────────────────

def preprocess_dataframe(
    df:              pd.DataFrame,
    text_column:     str = "text",
    keep_numbers:    bool = True,
    include_tagalog: bool = True,
    extra_stopwords: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to every row of a DataFrame.

    The original DataFrame is never mutated; a copy with an additional
    *clean_text* column is returned.

    Args:
        df:              Input DataFrame containing a raw text column.
        text_column:     Name of the column holding SMS text (default: 'text').)
        keep_numbers:    Passed through to preprocess_text().
        include_tagalog: Passed through to preprocess_text().
        extra_stopwords: Passed through to preprocess_text().

    Returns:
        New DataFrame identical to *df* plus a 'clean_message' column.

    Raises:
        KeyError: If *text_column* is not present in *df*.
    """
    if text_column not in df.columns:
        raise KeyError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    logger.info("Preprocessing column '%s' (%d rows)…", text_column, len(out))

    out["clean_text"] = out[text_column].apply(
        lambda x: preprocess_text(
            x,
            keep_numbers=keep_numbers,
            include_tagalog=include_tagalog,
            extra_stopwords=extra_stopwords,
        )
    )

    empty_mask = out["clean_text"].str.strip() == ""
    if empty_mask.any():
        logger.warning(
            "%d row(s) became empty after preprocessing — consider reviewing.",
            empty_mask.sum(),
        )

    logger.info("Preprocessing complete.")
    return out


# ── 6. Train / test split ────────────────────────────────────────────────────

def split_data(
    df:            pd.DataFrame,
    target_column: str = "category",
    text_column:   str = "clean_text",
    test_size:     float = 0.20,
    random_state:  int   = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform a stratified 80/20 train/test split.

    Stratification ensures that class proportions are preserved in both
    splits, which is important for imbalanced datasets.

    Args:
        df:            Preprocessed DataFrame (output of preprocess_dataframe).
        target_column: Name of the label column (default: 'category').
        text_column:   Name of the text feature column (default: 'clean_text').
        test_size:     Fraction of data reserved for testing (default: 0.20).
        random_state:  Seed for reproducibility (default: 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pandas Series.

    Raises:
        KeyError: If either *target_column* or *text_column* is absent.
    """
    for col in [text_column, target_column]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    X = df[text_column]
    y = df[target_column]

    # Stratification requires at least 2 samples per class.
    # If any class has only 1 sample, fall back to a non-stratified split
    # and warn the user — this is expected during small confidence tests.
    min_class_count = y.value_counts().min()
    use_stratify = y if min_class_count >= 2 else None

    if use_stratify is None:
        logger.warning(
            "Some classes have only 1 sample — stratification disabled. "
            "This is expected for small confidence-test datasets."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=use_stratify,
        random_state=random_state,
    )

    logger.info(
        "Split → train: %d rows | test: %d rows (%.0f/%.0f)",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )
    logger.info("Train class distribution:\n%s", y_train.value_counts().to_string())
    logger.info("Test  class distribution:\n%s", y_test.value_counts().to_string())

    return X_train, X_test, y_train, y_test


# ── Entry point (smoke test) ─────────────────────────────────────────────────

def run_preprocessing(
    filepath:     Optional[str | Path] = None,
    text_column:  str = "text",
    label_column: str = "category",
) -> None:
    """
    Load raw data, preprocess it, split it, and print a summary.

    Reads the CSV as UTF-8. This function is intended for quick
    verification from the terminal.

    Args:
        filepath:     Path to the raw dataset.  Defaults to
                      data/raw/tagalog-sms.csv.
        text_column:  Column containing SMS text (default: 'text').
        label_column: Column containing class labels (default: 'category').
    """
    import pandas as pd  # local import keeps top-level imports clean

    if filepath is None:
        filepath = get_data_path("tagalog-sms.csv")

    filepath = Path(filepath)
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info("Loaded %d rows from %s", len(df), filepath)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    processed_df = preprocess_dataframe(
        df,
        text_column=text_column,
        keep_numbers=True,
        include_tagalog=True,
    )

    print("\n  Sample preprocessed messages:")
    print(processed_df[[text_column, "clean_text"]].head(10).to_string(index=False))

    X_train, X_test, y_train, y_test = split_data(
        processed_df,
        target_column=label_column,
        text_column="clean_text",
    )

    print(f"\n  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_test  shape : {y_test.shape}")
    logger.info("Preprocessing smoke-test complete.")


if __name__ == "__main__":
    run_preprocessing()