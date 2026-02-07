""" Data transformation functions for the project """
# src/data/transforms.py
from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_price(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Adds a 'log_price' column = log(price).
    Requires strictly positive prices.
    """
    if price_col not in df.columns:
        raise ValueError(
            f"Missing column '{price_col}'. Found: {list(df.columns)}")

    if (df[price_col] <= 0).any():
        bad = int((df[price_col] <= 0).sum())
        raise ValueError(f"Found {bad} non-positive prices; cannot take log.")

    df["log_price"] = np.log(df[price_col].astype(float))
    return df


def add_log_returns(df: pd.DataFrame, log_price_col: str = "log_price") -> pd.DataFrame:
    """
    Adds 'log_return' = log_price_t - log_price_{t-1}.
    Assumes df is sorted by date already.
    """
    if log_price_col not in df.columns:
        raise ValueError(
            f"Missing column '{log_price_col}'. Found: {list(df.columns)}")

    df["log_return"] = df[log_price_col].diff()
    return df


def make_modeling_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create mapping from integer index t (0..n-1) to date.
    Useful for interpreting PyMC tau (change point index -> calendar date).
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Missing column '{date_col}'. Found: {list(df.columns)}")

    out = df[[date_col]].copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    out.insert(0, "t", np.arange(len(out), dtype=int))
    return out
