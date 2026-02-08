# src/data/io_task2.py
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from src.config import COL_DATE, COL_PRICE, COL_LOG_PRICE, COL_LOG_RETURN


def load_log_returns_csv(path: Path) -> pd.DataFrame:
    """
    Load processed log returns CSV and enforce:
    - date parsed to datetime
    - sorted by date
    - required columns exist (date, price, log_price, log_return)
    """
    df = pd.read_csv(path)

    required = {COL_DATE, COL_PRICE, COL_LOG_PRICE, COL_LOG_RETURN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    return df


def ensure_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: ensure log_price/log_return exist (in case upstream changes).
    Uses COL_PRICE as source.
    """
    out = df.copy()
    if (out[COL_PRICE] <= 0).any():
        raise ValueError(
            "Non-positive prices found; cannot compute log features.")

    if COL_LOG_PRICE not in out.columns:
        out[COL_LOG_PRICE] = out[COL_PRICE].map(lambda x: math.log(float(x)))
    if COL_LOG_RETURN not in out.columns:
        out[COL_LOG_RETURN] = out[COL_LOG_PRICE].diff()

    return out
