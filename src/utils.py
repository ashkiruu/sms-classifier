"""
utils.py — Shared utility functions for the SMS Classifier project.

Provides path resolution, directory management, logging setup,
and other helpers used across eda.py and preprocessing.py.
"""

import os
import logging
from pathlib import Path


# ── Project root (one level above src/) ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR   = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR    = PROJECT_ROOT / "outputs" / "reports"
MODELS_DIR     = PROJECT_ROOT / "models"


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def get_data_path(filename: str) -> Path:
    """
    Resolve the full path to a raw data file.

    Args:
        filename: Name of the file inside data/raw/ (e.g. 'tagalog-sms.xlsx').

    Returns:
        Absolute Path object to the file.
    """
    return DATA_RAW_DIR / filename


def ensure_dirs() -> None:
    """
    Create all required output directories if they do not already exist.
    Safe to call multiple times (no-op if dirs are present).
    """
    for directory in [FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_figure_path(filename: str) -> Path:
    """
    Resolve the full path for saving a figure.

    Args:
        filename: Desired filename (e.g. 'class_distribution.png').

    Returns:
        Absolute Path object inside outputs/figures/.
    """
    ensure_dirs()
    return FIGURES_DIR / filename


def get_report_path(filename: str) -> Path:
    """
    Resolve the full path for saving a report file.

    Args:
        filename: Desired filename (e.g. 'eda_summary.txt').

    Returns:
        Absolute Path object inside outputs/reports/.
    """
    ensure_dirs()
    return REPORTS_DIR / filename


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a named logger with a consistent format.

    Args:
        name:  Logger name, typically __name__ of the calling module.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Logger ready.")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
