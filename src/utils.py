"""Utility helpers for paths, logging, and label interpretation."""
from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONFIDENCE_DATA_DIR = DATA_DIR / "confidence"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TRAIN_DATA = RAW_DATA_DIR / "main_dataset.xlsx"
DEFAULT_CONFIDENCE_DATA = CONFIDENCE_DATA_DIR / "confidence_set.xlsx"

LABEL_DESCRIPTIONS = {
    "spam": "Suspicious, scam-like, or unsolicited content.",
    "gov": "Government advisory or public-service message.",
    "notifs": "Service or account notification.",
    "notif": "Service or account notification.",
    "otp": "One-time password or verification message.",
    "ads": "Promotional or marketing content.",
}

def ensure_dirs() -> None:
    for directory in [RAW_DATA_DIR, CONFIDENCE_DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_path(filename: str) -> Path:
    ensure_dirs()
    return RAW_DATA_DIR / filename

def get_confidence_path(filename: str) -> Path:
    ensure_dirs()
    return CONFIDENCE_DATA_DIR / filename

def get_report_path(filename: str) -> Path:
    ensure_dirs()
    return REPORTS_DIR / filename

def get_figure_path(filename: str) -> Path:
    ensure_dirs()
    return FIGURES_DIR / filename

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
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

def interpret_label(label: str) -> str:
    return LABEL_DESCRIPTIONS.get(label, "No interpretation available.")
