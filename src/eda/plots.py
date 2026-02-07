""" Module for plotting time series data and related transformations. """
# src/eda/plots.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_price_and_log_price(df: pd.DataFrame, date_col: str = "date") -> plt.Figure:
    """Plot price and log_price on two subplots."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df[date_col], df["price"], linewidth=1)
    axes[0].set_title("Brent Price (Level)")
    axes[0].set_ylabel("Price")

    if "log_price" in df.columns:
        axes[1].plot(df[date_col], df["log_price"], linewidth=1)
        axes[1].set_title("Log(Brent Price)")
        axes[1].set_ylabel("log(price)")
    else:
        axes[1].text(0.01, 0.5, "log_price not found. Run 02_make_returns.py first.",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    return fig


def plot_log_returns(df: pd.DataFrame, date_col: str = "date") -> plt.Figure:
    """Plot log returns over time."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    if "log_return" not in df.columns:
        ax.text(0.01, 0.5, "log_return not found. Run 02_make_returns.py first.",
                transform=ax.transAxes)
        return fig

    ax.plot(df[date_col], df["log_return"], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Log Returns")
    ax.set_ylabel("log_return")
    plt.tight_layout()
    return fig


def plot_rolling_volatility(
    df: pd.DataFrame,
    date_col: str = "date",
    col: str = "log_return",
    window: int = 30,
) -> plt.Figure:
    """Plot rolling volatility (std) of the specified column."""
    fig, ax = plt.subplots(figsize=(12, 3.8))

    if col not in df.columns:
        ax.text(
            0.01, 0.5, f"{col} not found. Run 02_make_returns.py first.", transform=ax.transAxes)
        return fig

    vol = df[col].rolling(window=window).std()
    ax.plot(df[date_col], vol, linewidth=1.2)
    ax.set_title(f"Rolling Volatility ({window}-period std of {col})")
    ax.set_ylabel("std")
    plt.tight_layout()
    return fig
