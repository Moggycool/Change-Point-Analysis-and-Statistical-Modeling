""" Script to clean raw Brent crude oil price data and save the cleaned version """
# scripts/01_clean_prices.py
from __future__ import annotations


import sys
from pathlib import Path
import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.data.cleaning import clean_brent_prices  # noqa: E402
from src.io.load_data import load_brent_prices  # noqa: E402
from src.config import (
    ensure_dirs,
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    BRENT_RAW_FILENAME,
    BRENT_CLEAN_FILENAME,
)  # noqa: E402
from src.io.save_data import save_csv  # noqa: E402
# pylint: enable=wrong-import-position


def main() -> None:
    """Main function to execute the cleaning process."""
    ensure_dirs()

    raw_path = DATA_RAW_DIR / BRENT_RAW_FILENAME
    clean_path = DATA_INTERIM_DIR / BRENT_CLEAN_FILENAME

    raw_df = load_brent_prices(raw_path)
    clean_df = clean_brent_prices(raw_df)

    save_csv(clean_df, clean_path, index=False)

    print(f"[OK] Cleaned data written to: {clean_path}")
    print(clean_df.head())
    print(
        f"Rows: {len(clean_df):,} | Range: {clean_df['date'].min()} → {clean_df['date'].max()}")


if __name__ == "__main__":
    main()
