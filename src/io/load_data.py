""" Data loading functions for the project """
# src/io/load_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_brent_prices(path: str | Path) -> pd.DataFrame:
    """
    Load raw Brent prices CSV.

    Expected raw columns:
      - Date (e.g., 20-May-87)
      - Price (float)

    Returns the raw dataframe without heavy cleaning; downstream cleaning will standardize
    column names and parse types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    # Basic validation (lightweight)
    expected = {"Date", "Price"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    return df
