""" Data cleaning functions for the project """
# src/data/cleaning.py
from __future__ import annotations

import pandas as pd


def clean_brent_prices(
    df: pd.DataFrame,
    duplicate_rule: str = "keep_last",
    drop_nonpositive_prices: bool = True,
) -> pd.DataFrame:
    """
    Clean and preprocess the raw Brent price data.

    Parameters
    ----------
    df:
        Raw input DataFrame expected to contain 'Date' and 'Price' columns.
    duplicate_rule:
        How to handle duplicate dates. One of:
        - 'keep_first': keep first occurrence
        - 'keep_last' : keep last occurrence (default)
        - 'mean'      : average prices within same date
        - 'median'    : median price within same date
        - 'drop'      : drop ALL rows whose date is duplicated (remove entire duplicated-date groups)
    drop_nonpositive_prices:
        If True, drop rows with price <= 0 (useful if later computing log transforms).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['date', 'price'] sorted by date.
    """
    # Backwards-compatible aliases
    alias_map = {
        "first": "keep_first",
        "last": "keep_last",
    }
    duplicate_rule = alias_map.get(duplicate_rule, duplicate_rule)

    allowed = {"keep_first", "keep_last", "mean", "median", "drop"}
    if duplicate_rule not in allowed:
        raise ValueError(
            f"Unknown duplicate_rule='{duplicate_rule}'. Expected one of {sorted(allowed)}."
        )

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(
            f"Expected columns 'Date' and 'Price'. Found: {list(df.columns)}")

    out = df[["Date", "Price"]].copy()

    out["date"] = pd.to_datetime(
        out["Date"], format="%d-%b-%y", errors="coerce")
    out["price"] = pd.to_numeric(out["Price"], errors="coerce")

    # Drop invalid rows
    out = out.dropna(subset=["date", "price"])

    # Optionally drop non-positive prices (log transform requires > 0)
    if drop_nonpositive_prices:
        out = out[out["price"] > 0]

    # Handle duplicates by date
    if out["date"].duplicated().any():
        out = out.sort_values(by="date")

        if duplicate_rule == "keep_last":
            out = out.drop_duplicates(subset=["date"], keep="last")

        elif duplicate_rule == "keep_first":
            out = out.drop_duplicates(subset=["date"], keep="first")

        elif duplicate_rule in {"mean", "median"}:
            aggfunc = "mean" if duplicate_rule == "mean" else "median"
            out = (
                out.groupby("date", as_index=False)
                .agg(price=("price", aggfunc))
                .sort_values(by="date")
            )

        elif duplicate_rule == "drop":
            # drop all rows for any date that appears more than once
            dup_mask = out.duplicated(subset=["date"], keep=False)
            out = out.loc[~dup_mask]

    out = out.sort_values(by="date").reset_index(drop=True)
    return out[["date", "price"]]
