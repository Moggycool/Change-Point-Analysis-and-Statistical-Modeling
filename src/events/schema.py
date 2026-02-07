""" Module for validating the events.csv file against the defined schema. """
# src/events/schema.py
from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "event_id",      # unique string or integer
    "event_name",    # short label
    "start_date",    # YYYY-MM-DD recommended
    "end_date",      # optional; can be empty
    "event_type",    # e.g., 'Geopolitical', 'OPEC', 'Sanctions', 'Crisis'
    "description",   # short text
    "region",        # optional but recommended
    "source",        # citation / URL / report name
]


def validate_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate events table for Task 1 (10–15 events).
    Parses dates, checks required columns, and basic consistency rules.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Events CSV missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()

    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")

    if out["start_date"].isna().any():
        bad = out[out["start_date"].isna(
        )][["event_id", "event_name", "start_date"]]
        raise ValueError(f"Some start_date values could not be parsed:\n{bad}")

    # end_date can be NaT, but if present must be >= start_date
    bad_order = out.dropna(subset=["end_date"])
    bad_order = bad_order[bad_order["end_date"] < bad_order["start_date"]]
    if len(bad_order) > 0:
        raise ValueError(
            "Some events have end_date earlier than start_date:\n"
            f"{bad_order[['event_id', 'event_name', 'start_date', 'end_date']]}"
        )

    # Uniqueness of event_id
    if out["event_id"].duplicated().any():
        dups = out[out["event_id"].duplicated(
            keep=False)][["event_id", "event_name"]]
        raise ValueError(f"Duplicate event_id values found:\n{dups}")

    # Minimal completeness checks
    for c in ["event_name", "event_type", "description", "source"]:
        if out[c].isna().any() or (out[c].astype(str).str.strip() == "").any():
            raise ValueError(
                f"Column '{c}' has missing/blank values. Fill them in for Task 1.")

    if len(out) < 10:
        raise ValueError(
            f"Task 1 expects ~10–15 events. Found only {len(out)} rows.")

    return out
