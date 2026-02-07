""" Data cleaning functions for the project """
# src/data/cleaning.py
from __future__ import annotations

import pandas as pd


def clean_brent_prices(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    date_format: str = "%d-%b-%y",
    drop_nonpositive_prices: bool = True,
    duplicate_rule: str = "last",
) -> pd.DataFrame:
    """
    Clean Brent prices dataset.

    Steps:
    - Parse Date strings like '20-May-87' using explicit format '%d-%b-%y'
    - Convert Price to numeric
    - Drop rows with invalid date/price
    - Handle duplicates by date (keep last by default)
    - Sort by date

    Output columns (standardized):
    - date (datetime64[ns])
    - price (float)
    """
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(
            f"Expected columns '{date_col}' and '{price_col}'. Found: {list(df.columns)}"
        )

    out = df[[date_col, price_col]].copy()

    out["date"] = pd.to_datetime(
        out[date_col], format=date_format, errors="coerce")
    out["price"] = pd.to_numeric(out[price_col], errors="coerce")

    # Drop invalid rows
    out = out.dropna(subset=["date", "price"])

    # Optionally drop non-positive prices (log transform requires > 0)
    if drop_nonpositive_prices:
        out = out[out["price"] > 0]

    # Handle duplicates by date
    if out["date"].duplicated().any():
        out = out.sort_values("date")
        if duplicate_rule == "last":
            out = out.drop_duplicates(subset=["date"], keep="last")
        elif duplicate_rule == "first":
            out = out.drop_duplicates(subset=["date"], keep="first")
        elif duplicate_rule == "mean":
            out = out.groupby("date", as_index=False)["price"].mean()
        else:
            raise ValueError(
                "duplicate_rule must be one of: 'last','first','mean'")

    out = out.sort_values("date").reset_index(drop=True)

    return out[["date", "price"]]
