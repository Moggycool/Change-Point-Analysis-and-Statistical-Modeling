# src/io/load_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_brent_prices(
    path: str | Path,
    *,
    parse_dates: bool = False,
) -> pd.DataFrame:
    """
    Load raw Brent prices CSV.

    Expected raw columns (case-sensitive in the file, but we accept common variants):
      - Date / date
      - Price / price

    Parameters
    ----------
    path : str | Path
        Location of raw CSV.
    parse_dates : bool, default False
        If True, parse the date column to datetime (errors="raise") and remove tz info.

    Returns
    -------
    pd.DataFrame
        Raw-ish dataframe. Downstream cleaning should standardize names, sort,
        deduplicate, and enforce numeric types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)

    # Accept common variants but do NOT rename everything here (keep loader lightweight)
    cols = set(df.columns)
    date_col = "Date" if "Date" in cols else (
        "date" if "date" in cols else None)
    price_col = "Price" if "Price" in cols else (
        "price" if "price" in cols else None)

    if date_col is None or price_col is None:
        raise ValueError(
            "Missing required columns. Need Date/date and Price/price. "
            f"Found columns: {list(df.columns)}"
        )

    if parse_dates:
        df[date_col] = pd.to_datetime(
            df[date_col], errors="raise").dt.tz_localize(None)

    return df
