# src/events/schema.py
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "event_id",
    "event_name",
    "start_date",
    "end_date",
    "event_type",
    "description",
    "region",
    "source",
]

OPTIONAL_COLUMNS = [
    "expected_direction",   # up/down/ambiguous
    "expected_channel",     # supply/demand/risk/finance/other
    "confidence",           # high/medium/low
    "source_name",          # e.g., Reuters, EIA, OPEC
    "source_url",           # URL (if available)
]

ALLOWED_DIRECTIONS = {"up", "down", "ambiguous"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _norm(s: pd.Series) -> pd.Series:
    """ Normalize string columns by stripping whitespace and converting to lowercase. """
    return s.astype(str).str.strip().str.lower()


def validate_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Validate the events DataFrame against the expected schema 
         and basic sanity checks.  """
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

    bad_order = out.dropna(subset=["end_date"])
    bad_order = bad_order[bad_order["end_date"] < bad_order["start_date"]]
    if len(bad_order) > 0:
        raise ValueError(
            "Some events have end_date earlier than start_date:\n"
            f"{bad_order[['event_id', 'event_name', 'start_date', 'end_date']]}"
        )

    if out["event_id"].duplicated().any():
        dups = out[out["event_id"].duplicated(
            keep=False)][["event_id", "event_name"]]
        raise ValueError(f"Duplicate event_id values found:\n{dups}")

    for c in ["event_name", "event_type", "description", "source"]:
        if out[c].isna().any() or (_norm(out[c]) == "").any():
            raise ValueError(
                f"Column '{c}' has missing/blank values. Fill them in for Task 1.")

    if len(out) < 10:
        raise ValueError(
            f"Task 1 expects ~10–15 events. Found only {len(out)} rows.")

    # Optional fields validation (only if present)
    if "expected_direction" in out.columns:
        bad = out[~_norm(out["expected_direction"]
                         ).str.lower().isin(ALLOWED_DIRECTIONS)]
        if len(bad) > 0:
            raise ValueError(
                "Invalid expected_direction values. Allowed: up/down/ambiguous.\n"
                f"{bad[['event_id', 'event_name', 'expected_direction']].head(20)}"
            )

    if "confidence" in out.columns:
        bad = out[~_norm(out["confidence"]).str.lower().isin(
            ALLOWED_CONFIDENCE)]
        if len(bad) > 0:
            raise ValueError(
                "Invalid confidence values. Allowed: high/medium/low.\n"
                f"{bad[['event_id', 'event_name', 'confidence']].head(20)}"
            )

    return out
