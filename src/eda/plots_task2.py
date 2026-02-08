# src/eda/plots_task2.py
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import COL_DATE, COL_PRICE, COL_LOG_RETURN


def plot_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], df[COL_PRICE], lw=1)
    ax.set_title("Brent Price (raw)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return fig


def plot_log_returns(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], df[COL_LOG_RETURN], lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("log_return")
    ax.grid(True, alpha=0.3)
    return fig


def plot_rolling_volatility(df: pd.DataFrame, window: int = 30):
    vol = df[COL_LOG_RETURN].rolling(window).std()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[COL_DATE], vol, lw=1)
    ax.set_title(f"Rolling Volatility of Log Returns ({window}-day std)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling std")
    ax.grid(True, alpha=0.3)
    return fig
