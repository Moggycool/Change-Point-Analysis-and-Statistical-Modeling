""" Module for running stationarity tests on time series data. """
# src/eda/time_series_tests.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a numeric series, coercing errors to NaN and dropping them."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    # For returns, first value is often NaN; dropna handles it.
    return s


def adf_test(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: unit root (non-stationary). Small p => reject H0 => stationary.
    """
    s = series.dropna().astype(float)
    res = adfuller(s, autolag=autolag)
    return {
        "test": "ADF",
        "statistic": float(res[0]),
        "p_value": float(res[1]),
        "n_lags": int(res[2]),
        "n_obs": int(res[3]),
    }


def kpss_test(series: pd.Series, regression: str = "c", nlags: str = "auto") -> dict:
    """
    KPSS test.
    H0: stationary (level if regression='c', trend if regression='ct').
    Small p => reject H0 => non-stationary.
    """
    s = series.dropna().astype(float)
    stat, pval, lags, crit = kpss(s, regression=regression, nlags=nlags)
    return {
        "test": f"KPSS({regression})",
        "statistic": float(stat),
        "p_value": float(pval),
        "n_lags": int(lags),
        "n_obs": int(len(s)),
    }


def run_stationarity_suite(
    df: pd.DataFrame,
    date_col: str = "date",
    cols: list[str] = ["price", "log_price", "log_return"],
) -> pd.DataFrame:
    """
    Run ADF + KPSS(level) + KPSS(trend) for each column.
    Returns a tidy table suitable for your Task 1 notebook/report.
    """
    rows = []
    for col in cols:
        if col not in df.columns:
            continue

        s = _safe_series(df, col)
        if len(s) < 20:
            rows.append(
                {
                    "column": col,
                    "test": "SKIP",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_lags": np.nan,
                    "n_obs": int(len(s)),
                    "note": "Too few observations for stationarity testing",
                }
            )
            continue

        try:
            adf = adf_test(s)
            rows.append({"column": col, **adf, "note": ""})
        except Exception as e:
            rows.append(
                {
                    "column": col,
                    "test": "ADF",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_lags": np.nan,
                    "n_obs": int(len(s)),
                    "note": f"ADF failed: {type(e).__name__}: {e}",
                }
            )

        for reg in ("c", "ct"):
            try:
                kp = kpss_test(s, regression=reg)
                rows.append({"column": col, **kp, "note": ""})
            except Exception as e:
                rows.append(
                    {
                        "column": col,
                        "test": f"KPSS({reg})",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "n_lags": np.nan,
                        "n_obs": int(len(s)),
                        "note": f"KPSS failed: {type(e).__name__}: {e}",
                    }
                )

    return pd.DataFrame(rows)
