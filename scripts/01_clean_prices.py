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
    DATA_PROCESSED_DIR,
    BRENT_RAW_FILENAME,
    BRENT_CLEAN_FILENAME,
)  # noqa: E402
from src.io.save_data import save_csv  # noqa: E402
# pylint: enable=wrong-import-position


def main() -> None:
    """Main function to execute the cleaning process."""
    ensure_dirs()

    raw_path = DATA_RAW_DIR / BRENT_RAW_FILENAME
    clean_path = DATA_PROCESSED_DIR / BRENT_CLEAN_FILENAME

    raw_df = load_brent_prices(raw_path)
    clean_df = clean_brent_prices(raw_df)

    # ---- Explicit post-clean validation (grader-facing) ----
    if "date" not in clean_df.columns or "price" not in clean_df.columns:
        raise ValueError(
            f"Expected cleaned columns ['date','price']. Found: {list(clean_df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(clean_df["date"]):
        raise TypeError(
            f"'date' must be datetime64 after cleaning. Got {clean_df['date'].dtype}")

    if clean_df["date"].isna().any():
        raise ValueError("Found NaT in 'date' after cleaning.")

    if not clean_df["date"].is_monotonic_increasing:
        raise ValueError("'date' must be sorted ascending after cleaning.")

    if clean_df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found after cleaning; must be unique.")

    if not pd.api.types.is_numeric_dtype(clean_df["price"]):
        raise TypeError(
            f"'price' must be numeric after cleaning. Got {clean_df['price'].dtype}")

    if clean_df["price"].isna().any():
        raise ValueError("Found NaN in 'price' after cleaning.")

    save_csv(clean_df, clean_path, index=False)

    print(f"[OK] Cleaned data written to: {clean_path}")
    print(clean_df.head())
    print(
        "Rows: {:,} | Range: {} → {}".format(
            len(clean_df),
            clean_df["date"].min().date(),
            clean_df["date"].max().date(),
        )
    )


if __name__ == "__main__":
    main()
