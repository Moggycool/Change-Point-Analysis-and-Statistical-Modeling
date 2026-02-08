# src/events/io_task2.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import COL_DATE


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if COL_DATE not in df.columns:
        raise ValueError(f"events file must contain '{COL_DATE}'")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df.sort_values(COL_DATE).reset_index(drop=True)
