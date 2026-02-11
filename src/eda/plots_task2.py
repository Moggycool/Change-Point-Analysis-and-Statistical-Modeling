# src/eda/plots_task2.py
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


from src.config import COL_DATE, COL_PRICE, COL_LOG_RETURN
from src.eda.plots import _add_source_footer, _apply_style, _prep_ts


def plot_price(df: pd.DataFrame) -> Figure:
    """Plot the raw price series."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], df[COL_PRICE], lw=1)
    ax.set_title("Brent Price (raw)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return fig


def plot_log_returns(df: pd.DataFrame) -> Figure:
    """Plot the log returns series."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], df[COL_LOG_RETURN], lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.grid(True, alpha=0.3)
    return fig


def plot_rolling_volatility(df: pd.DataFrame, window: int = 30) -> Figure:
    """Plot the rolling volatility (std dev) of log returns."""
    vol = df[COL_LOG_RETURN].rolling(window).std()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], vol, lw=1)
    ax.set_title(f"Rolling Volatility of Log Returns ({window}-day std)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev")
    ax.grid(True, alpha=0.3)
    return fig


def plot_price_series_with_changepoint(
    df: pd.DataFrame,
    cp_date,
    *,
    date_col: str = "date",
    price_col: str = "price",
    hdi_low=None,
    hdi_high=None,
    source_note: Optional[str] = "Source: Brent crude oil price dataset (cleaned).",
    cp_label: str = "Change point (τ mode)",
    hdi_label: str = "τ date 94% HDI",
) -> Figure:
    """
    Rubric-required overlay: raw price + inferred change-point date (+ optional HDI band).

    Parameters
    ----------
    cp_date : any pandas.to_datetime-compatible object
    hdi_low, hdi_high : optional HDI endpoints (dates)
    """
    _apply_style()
    d = _prep_ts(df, date_col, price_col)

    cp_date = pd.to_datetime(cp_date)
    hdi_low = pd.to_datetime(hdi_low) if hdi_low is not None else None
    hdi_high = pd.to_datetime(hdi_high) if hdi_high is not None else None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d[date_col], d[price_col], linewidth=1.2, color="tab:blue")

    # HDI band (optional)
    if hdi_low is not None and hdi_high is not None:
        lo, hi = (hdi_low, hdi_high) if hdi_low <= hdi_high else (
            hdi_high, hdi_low)
        ax.axvspan(lo, hi, color="tab:red", alpha=0.15, label=hdi_label)

    # Change-point line
    ax.axvline(cp_date, color="tab:red", linestyle="--",
               linewidth=1.8, label=cp_label)

    ax.set_title("Brent Price (USD/bbl) with Inferred Change Point")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    ax.legend(loc="best")

    _add_source_footer(fig, source_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
