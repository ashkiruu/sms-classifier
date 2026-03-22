"""Exploratory data analysis for the SMS classifier."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_dataset
from utils import setup_logger, get_figure_path

logger = setup_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")
LABEL_ORDER = ["spam", "gov", "notifs", "otp", "ads"]


def basic_info(df: pd.DataFrame) -> None:
    print("\nDataset info")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head().to_string(index=False))


def class_distribution(df: pd.DataFrame, label_column: str = "label") -> None:
    counts = df[label_column].value_counts()
    print("\nClass distribution:\n")
    print(counts.to_string())
    ordered = [c for c in LABEL_ORDER if c in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(ordered, [counts[c] for c in ordered])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, cls in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{counts[cls]:,}", ha="center", va="bottom")
    plt.tight_layout()
    out = get_figure_path("class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure to %s", out)


def message_length_analysis(df: pd.DataFrame, text_column: str = "message", label_column: str = "label") -> None:
    view = df[[label_column, text_column]].copy()
    view["char_len"] = view[text_column].astype(str).str.len()
    view["word_cnt"] = view[text_column].astype(str).str.split().str.len()
    print("\nCharacter length summary:\n")
    print(view.groupby(label_column)["char_len"].describe().round(1).to_string())
    print("\nWord count summary:\n")
    print(view.groupby(label_column)["word_cnt"].describe().round(1).to_string())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=view, x=label_column, y="char_len", order=LABEL_ORDER, ax=ax)
    ax.set_title("Character Length by Class")
    plt.tight_layout()
    out = get_figure_path("message_length_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure to %s", out)


def top_words_per_class(df: pd.DataFrame, text_column: str = "message", label_column: str = "label", top_n: int = 10) -> None:
    print("\nTop words per class:")
    for label in LABEL_ORDER:
        subset = df.loc[df[label_column] == label, text_column].astype(str)
        tokens = []
        for text in subset:
            tokens.extend(text.lower().split())
        common = Counter(tokens).most_common(top_n)
        if common:
            print(f"- {label}: {', '.join([w for w, _ in common])}")


def main() -> None:
    df = load_dataset()
    basic_info(df)
    class_distribution(df)
    message_length_analysis(df)
    top_words_per_class(df)


if __name__ == "__main__":
    main()
