""" Configuration settings for the project """
# src/config.py
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Project root (assumes pyproject.toml is at the repository root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Data directories
# -----------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# -----------------------------------------------------------------------------
# Reports directories
# -----------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_INTERIM_DIR = REPORTS_DIR / "interim"

# -----------------------------------------------------------------------------
# Filenames
# -----------------------------------------------------------------------------
BRENT_RAW_FILENAME = "brent_prices.csv"

BRENT_CLEAN_FILENAME = "brent_prices_clean.csv"
EDA_SUMMARY_FILENAME = "eda_summary.json"

# UPDATED: match your real events file name
EVENTS_FILENAME = "brent_events.csv"

LOG_PRICE_FILENAME = "brent_log_price.csv"
LOG_RETURNS_FILENAME = "brent_log_returns.csv"
MODEL_INDEX_MAP_FILENAME = "modeling_index_map.csv"

# -----------------------------------------------------------------------------
# Column standards (after cleaning)
# -----------------------------------------------------------------------------
COL_DATE = "date"
COL_PRICE = "price"
COL_LOG_PRICE = "log_price"
COL_LOG_RETURN = "log_return"

# -----------------------------------------------------------------------------
# Defaults for EDA
# -----------------------------------------------------------------------------
ROLLING_VOL_WINDOW_DAYS = 30
STATIONARITY_ALPHA = 0.05

# Optional: default event matching window for later tasks
EVENT_MATCH_WINDOW_DAYS = 14


def ensure_dirs() -> None:
    """Create expected directories if missing."""
    for p in [
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_INTERIM_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
