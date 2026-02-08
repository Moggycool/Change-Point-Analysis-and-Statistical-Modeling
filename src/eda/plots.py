"""Module for plotting time series data and related transformations."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _apply_style() -> None:
    """Apply a consistent style for all EDA figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _add_source_footer(fig: plt.Figure, source_note: Optional[str]) -> None:
    """Add a small footer to the figure (optional but professional)."""
    if not source_note:
        return
    fig.text(0.01, 0.01, source_note, ha="left",
             va="bottom", fontsize=9, color="0.35")


def _prep_ts(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Standard prep: select cols, coerce types, sort, drop missing."""
    if date_col not in df.columns:
        raise KeyError(
            f"Missing '{date_col}' column. Found: {list(df.columns)}")
    if value_col not in df.columns:
        raise KeyError(
            f"Missing '{value_col}' column. Found: {list(df.columns)}")

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if out.empty:
        raise ValueError(
            f"After cleaning, no valid rows to plot for '{value_col}'.")
    return out


def plot_price_series(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
) -> plt.Figure:
    _apply_style()
    d = _prep_ts(df, date_col, price_col)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d[date_col], d[price_col], linewidth=1.2, color="tab:blue")
    ax.set_title("Brent Price (USD/bbl)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_log_price_series(
    df: pd.DataFrame,
    date_col: str = "date",
    log_price_col: str = "log_price",
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
) -> plt.Figure:
    _apply_style()
    d = _prep_ts(df, date_col, log_price_col)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d[date_col], d[log_price_col], linewidth=1.2, color="tab:green")
    ax.set_title("Brent Log Price (log USD/bbl)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Price (log USD/bbl)")
    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_price_and_log_price(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    log_price_col: str = "log_price",
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
) -> plt.Figure:
    _apply_style()
    d_price = _prep_ts(df, date_col, price_col)
    d_logp = _prep_ts(df, date_col, log_price_col)

    # align on dates (inner join)
    d = d_price.merge(d_logp, on=date_col, how="inner", suffixes=("", "_log"))
    d = d.rename(columns={log_price_col: "log_price"})

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(d[date_col], d[price_col], linewidth=1.2,
                 color="tab:blue", label="Price")
    axes[0].set_title("Brent Price (USD/bbl)")
    axes[0].set_ylabel("Price (USD/bbl)")
    axes[0].legend(loc="upper left")

    axes[1].plot(d[date_col], d["log_price"], linewidth=1.2,
                 color="tab:green", label="Log Price")
    axes[1].set_title("Brent Log Price (log USD/bbl)")
    axes[1].set_ylabel("Log Price (log USD/bbl)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")

    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_log_returns(
    df: pd.DataFrame,
    date_col: str = "date",
    col: str = "log_return",
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
) -> plt.Figure:
    _apply_style()
    d = _prep_ts(df, date_col, col)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(d[date_col], d[col], linewidth=0.8, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Brent Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_rolling_volatility(
    df: pd.DataFrame,
    date_col: str = "date",
    col: str = "log_return",
    window: int = 30,
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
) -> plt.Figure:
    _apply_style()
    d = _prep_ts(df, date_col, col)

    vol = d[col].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(d[date_col], vol, linewidth=1.2, color="tab:orange")
    ax.set_title(f"Rolling Volatility ({window}-period std of {col})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Dev")
    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
