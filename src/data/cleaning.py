""" Data cleaning functions for the project """
# src/data/cleaning.py
from __future__ import annotations

import pandas as pd


def clean_brent_prices(
    df: pd.DataFrame,
    duplicate_rule: str = "keep_last",
    drop_nonpositive_prices: bool = True,
    date_format: str | None = "%d-%b-%y",
) -> pd.DataFrame:
    """
    Clean and preprocess the raw Brent price data.

    Notes
    -----
    The raw dataset may contain mixed date string formats (e.g. "22-Apr-20" and
    "Apr 22, 2020"). We keep datetime parsing **strict** (errors='raise') while
    supporting mixed formats using pandas' format='mixed' (pandas >= 2.0).

    Parameters
    ----------
    df:
        Raw input DataFrame expected to contain Date/Price columns (case variants allowed).
    duplicate_rule:
        How to handle duplicate dates. One of:
        - 'keep_first': keep first occurrence
        - 'keep_last' : keep last occurrence (default)
        - 'mean'      : average prices within same date
        - 'median'    : median price within same date
        - 'drop'      : drop ALL rows whose date is duplicated (remove entire duplicated-date groups)
    drop_nonpositive_prices:
        If True, drop rows with price <= 0 (useful if later computing log transforms).
    date_format:
        Optional datetime format string. If None, let pandas infer.
        If provided, we try this format first (strict), then fall back to strict mixed parsing.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['date', 'price'] sorted by date.
    """
    # Backwards-compatible aliases
    alias_map = {"first": "keep_first", "last": "keep_last"}
    duplicate_rule = alias_map.get(duplicate_rule, duplicate_rule)

    allowed = {"keep_first", "keep_last", "mean", "median", "drop"}
    if duplicate_rule not in allowed:
        raise ValueError(
            f"Unknown duplicate_rule='{duplicate_rule}'. Expected one of {sorted(allowed)}."
        )

    # Accept common column variants but enforce presence
    cols = set(df.columns)
    date_col = "Date" if "Date" in cols else (
        "date" if "date" in cols else None)
    price_col = "Price" if "Price" in cols else (
        "price" if "price" in cols else None)

    if date_col is None or price_col is None:
        raise ValueError(
            "Expected columns Date/date and Price/price. "
            f"Found: {list(df.columns)}"
        )

    out = df[[date_col, price_col]].copy()

    # ---- Explicit datetime conversion (strict, mixed-format safe) ----
    # Normalize to stripped strings to avoid whitespace-caused failures.
    s = out[date_col].astype(str).str.strip()

    # Try explicit format first (if provided), then fall back to strict mixed parsing.
    if date_format is not None:
        try:
            out["date"] = pd.to_datetime(
                s, format=date_format, errors="raise").dt.tz_localize(None)
        except ValueError:
            # Mixed formats detected (e.g., "Apr 22, 2020" vs "22-Apr-20")
            out["date"] = pd.to_datetime(
                s, format="mixed", errors="raise").dt.tz_localize(None)
    else:
        # Let pandas infer per-element formats strictly.
        out["date"] = pd.to_datetime(
            s, format="mixed", errors="raise").dt.tz_localize(None)

    # ---- Explicit numeric conversion (tolerant) ----
    out["price"] = pd.to_numeric(out[price_col], errors="coerce")

    # Drop rows where price could not be parsed
    out = out.dropna(subset=["price"])

    # Optionally drop non-positive prices (log transform requires > 0)
    if drop_nonpositive_prices:
        out = out[out["price"] > 0]

    # Sort before duplicate handling for deterministic behavior
    out = out.sort_values(by="date")

    # Handle duplicates by date
    if out["date"].duplicated().any():
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
            dup_mask = out.duplicated(subset=["date"], keep=False)
            out = out.loc[~dup_mask]

    out = out.sort_values(by="date").reset_index(drop=True)

    # Final schema
    out = out[["date", "price"]]

    return out
