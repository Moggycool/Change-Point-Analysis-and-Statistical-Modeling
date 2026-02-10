"""
An example of a more complex event loading function,
with some basic validation and error handling.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import COL_DATE

EVENT_DATE_COL = "event_date"
EVENT_END_DATE_COL = "event_end_date"


def load_events(path: Path) -> pd.DataFrame:
    """Load events from a CSV file, with validation and error handling."""
    df = pd.read_csv(path)

    # 1) Map event_date -> COL_DATE if needed
    if COL_DATE not in df.columns:
        if EVENT_DATE_COL in df.columns:
            df = df.rename(columns={EVENT_DATE_COL: COL_DATE})
        else:
            raise ValueError(
                f"events file must contain '{COL_DATE}' or '{EVENT_DATE_COL}'. "
                f"Found columns: {list(df.columns)}"
            )

    # 2) Parse dates
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    if EVENT_END_DATE_COL in df.columns:
        df[EVENT_END_DATE_COL] = pd.to_datetime(
            df[EVENT_END_DATE_COL], errors="coerce")

    # 3) Basic validation
    if df[COL_DATE].isna().any():
        bad = df.loc[df[COL_DATE].isna()].head(10)
        raise ValueError(
            f"Some events have invalid '{COL_DATE}' values after parsing. "
            f"Examples (first 10 rows):\n{bad.to_string(index=False)}"
        )

    # 4) Sort and return
    return df.sort_values(COL_DATE).reset_index(drop=True)
