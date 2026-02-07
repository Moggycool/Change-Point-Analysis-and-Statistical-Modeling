"""Module for saving dataframes to CSV files."""
# src/io/save_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """
    Save dataframe to CSV, ensuring parent directories exist.
    Returns the resolved Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path.resolve()
