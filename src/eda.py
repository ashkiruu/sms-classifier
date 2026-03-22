"""
eda.py — Exploratory Data Analysis for the Tagalog SMS Classifier.

All functions are read-only with respect to the dataset; no mutations
are made.  Plots are saved to outputs/figures/ and key insights are
printed to stdout.

Dataset format: CSV file (tagalog-sms.csv) with columns 'message' and 'label'.

Usage:
    python src/eda.py
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Make sure src/ sibling modules are importable when run directly ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import setup_logger, get_data_path, get_figure_path, ensure_dirs

logger = setup_logger(__name__)

# ── Shared plot style ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
LABEL_ORDER = ["spam", "gov", "notifs", "otp", "ads"]
CLASS_PALETTE = {
    "spam":   "#E63946",
    "gov":    "#457B9D",
    "notifs": "#2A9D8F",
    "otp":    "#E9C46A",
    "ads":    "#F4A261",
}


# ── 1. Data loading ──────────────────────────────────────────────────────────

def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the SMS dataset from a CSV file.

    Tries UTF-8 first, then falls back to latin-1 (ISO-8859-1) to handle
    files exported from Excel that may contain non-UTF-8 characters.

    Args:
        filepath: Path to the .csv file (string or Path object).

    Returns:
        Raw DataFrame exactly as stored on disk.

    Raises:
        FileNotFoundError: If the file does not exist at *filepath*.
        ValueError:        If the file cannot be parsed as a CSV.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Place 'tagalog-sms.csv' inside data/raw/ and try again."
        )
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info("Loaded dataset from %s", filepath)
        return df
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc


# ── 2. Basic info ────────────────────────────────────────────────────────────

def basic_info(df: pd.DataFrame) -> None:
    """
    Print shape, column names, dtypes, and the first five rows.

    Args:
        df: Raw SMS DataFrame.
    """
    print("\n" + "═" * 60)
    print("  BASIC DATASET INFO")
    print("═" * 60)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    print("\n  Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<25} {dtype}")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print("═" * 60 + "\n")


# ── 3. Missing-value audit ───────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> None:
    """
    Report the count and percentage of missing values per column.

    Args:
        df: Raw SMS DataFrame.
    """
    print("\n" + "═" * 60)
    print("  MISSING VALUE AUDIT")
    print("═" * 60)
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0]

    if report.empty:
        print("  ✓ No missing values detected across all columns.")
    else:
        print(report.to_string())
        logger.warning("Missing values found — review before training.")
    print("═" * 60 + "\n")


# ── 4. Class distribution ────────────────────────────────────────────────────

def class_distribution(
    df: pd.DataFrame,
    label_column: str = "label",
) -> None:
    """
    Print value counts and save a bar chart of class frequencies.

    Args:
        df:           SMS DataFrame with a label column.
        label_column: Name of the target column (default: 'label').
    """
    print("\n" + "═" * 60)
    print("  CLASS DISTRIBUTION")
    print("═" * 60)
    counts = df[label_column].value_counts()
    pct    = (counts / len(df) * 100).round(2)

    for cls in counts.index:
        bar = "█" * int(pct[cls] // 2)
        print(f"  {cls:<8} {counts[cls]:>6,}  ({pct[cls]:>5.1f}%)  {bar}")
    print("═" * 60 + "\n")

    # ── figure ───────────────────────────────────────────────────────────────
    ordered   = [c for c in LABEL_ORDER if c in counts.index]
    palette   = [CLASS_PALETTE.get(c, "#999") for c in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

    # bar chart
    ax = axes[0]
    bars = ax.bar(ordered, [counts[c] for c in ordered], color=palette, edgecolor="white", linewidth=0.6)
    ax.set_title("Absolute Frequency")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, cls in zip(bars, ordered):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{counts[cls]:,}",
            ha="center", va="bottom", fontsize=9,
        )

    # pie chart
    ax2 = axes[1]
    ax2.pie(
        [counts[c] for c in ordered],
        labels=ordered,
        colors=palette,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax2.set_title("Relative Frequency")

    plt.tight_layout()
    out = get_figure_path("class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved class distribution plot → %s", out)


# ── 5. Message-length analysis ───────────────────────────────────────────────

def message_length_analysis(
    df: pd.DataFrame,
    text_column:  str = "message",
    label_column: str = "label",
) -> None:
    """
    Compute and visualise character-length and word-count distributions
    per class.

    Args:
        df:           SMS DataFrame.
        text_column:  Column containing the raw SMS text.
        label_column: Column containing the class labels.
    """
    view = df[[label_column, text_column]].copy()
    view["char_len"] = view[text_column].astype(str).str.len()
    view["word_cnt"] = view[text_column].astype(str).str.split().str.len()

    print("\n" + "═" * 60)
    print("  MESSAGE LENGTH SUMMARY (characters)")
    print("═" * 60)
    print(view.groupby(label_column)["char_len"].describe().round(1).to_string())
    print("\n  MESSAGE LENGTH SUMMARY (words)")
    print(view.groupby(label_column)["word_cnt"].describe().round(1).to_string())
    print("═" * 60 + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("Message Length Analysis", fontsize=14, fontweight="bold")

    for ax, col, title, xlabel in zip(
        axes,
        ["char_len", "word_cnt"],
        ["Character Length by Class", "Word Count by Class"],
        ["Character count", "Word count"],
    ):
        for cls in LABEL_ORDER:
            if cls in view[label_column].unique():
                subset = view.loc[view[label_column] == cls, col]
                sns.kdeplot(
                    subset, ax=ax, label=cls,
                    color=CLASS_PALETTE.get(cls, "#999"),
                    linewidth=2, fill=True, alpha=0.15,
                )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(title="Class")

    plt.tight_layout()
    out = get_figure_path("message_length_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved message-length plot → %s", out)


# ── 6. Top words per class ───────────────────────────────────────────────────

def top_words_per_class(
    df: pd.DataFrame,
    text_column:  str = "message",
    label_column: str = "label",
    top_n:        int = 15,
) -> None:
    """
    Identify and plot the most frequent words for each class.

    Simple whitespace tokenisation is used here (no stemming) so that
    the EDA remains independent of the preprocessing pipeline.

    Args:
        df:           SMS DataFrame.
        text_column:  Column containing the raw SMS text.
        label_column: Column containing the class labels.
        top_n:        Number of top words to display per class (default: 15).
    """
    classes = [c for c in LABEL_ORDER if c in df[label_column].unique()]
    ncols = 3
    nrows = -(-len(classes) // ncols)   # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(f"Top {top_n} Words per Class (raw tokens)", fontsize=15, fontweight="bold")

    print("\n" + "═" * 60)
    print(f"  TOP {top_n} WORDS PER CLASS")
    print("═" * 60)

    for ax, cls in zip(axes_flat, classes):
        subset = df.loc[df[label_column] == cls, text_column].astype(str)
        tokens = " ".join(subset).lower().split()
        counts = Counter(tokens).most_common(top_n)
        words, freqs = zip(*counts) if counts else ([], [])

        print(f"\n  [{cls.upper()}]")
        for w, f in counts:
            print(f"    {w:<20} {f:,}")

        color = CLASS_PALETTE.get(cls, "#999")
        ax.barh(list(words)[::-1], list(freqs)[::-1], color=color, edgecolor="white")
        ax.set_title(f"{cls.upper()}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.tick_params(axis="y", labelsize=9)

    # hide unused axes
    for ax in axes_flat[len(classes):]:
        ax.set_visible(False)

    print("═" * 60 + "\n")

    plt.tight_layout()
    out = get_figure_path("top_words_per_class.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved top-words plot → %s", out)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_eda(
    filepath:     str | Path | None = None,
    text_column:  str = "text",
    label_column: str = "category",
) -> None:
    """
    Execute the full EDA pipeline.

    Args:
        filepath:     Path to the dataset.  Defaults to data/raw/tagalog-sms.csv.
        text_column:  Name of the SMS text column (default: 'text').
        label_column: Name of the label column (default: 'category').
    """
    ensure_dirs()
    if filepath is None:
        filepath = get_data_path("tagalog-sms.csv")

    df = load_data(filepath)
    basic_info(df)
    check_missing_values(df)
    class_distribution(df, label_column=label_column)
    message_length_analysis(df, text_column=text_column, label_column=label_column)
    top_words_per_class(df, text_column=text_column, label_column=label_column)
    logger.info("EDA complete.  Figures saved to outputs/figures/")


if __name__ == "__main__":
    run_eda()