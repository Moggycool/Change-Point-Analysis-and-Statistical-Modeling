""" A suite of tests for the events schema validation function. """
# tests/test_events_schema.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Get project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.events.schema import validate_events_df  # noqa: E402
# pylint: enable=wrong-import-position


def _valid_events_df(n: int = 10) -> pd.DataFrame:
    """Helper function to generate a valid events DataFrame with n rows."""
    # minimal valid rows
    rows = []
    for i in range(n):
        rows.append(
            {
                "event_id": f"e{i+1}",
                "event_name": f"Event {i+1}",
                "start_date": "1990-01-01",
                "end_date": "1990-01-10",
                "event_type": "Geopolitical",
                "description": "Some description",
                "region": "Global",
                "source": "Some source",
            }
        )
    return pd.DataFrame(rows)


def test_validate_events_df_passes_minimum():
    """Test that validate_events_df accepts a valid DataFrame with 
        the minimum required structure."""
    df = _valid_events_df(10)
    out = validate_events_df(df)
    assert len(out) == 10
    assert pd.api.types.is_datetime64_any_dtype(out["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(out["end_date"])


def test_validate_events_df_requires_columns():
    """Test that validate_events_df raises an error if required 
        columns are missing."""
    df = _valid_events_df(10).drop(columns=["source"])
    with pytest.raises(ValueError):
        validate_events_df(df)


def test_validate_events_df_requires_unique_event_id():
    """Test that validate_events_df raises an error if 
        event_id values are not unique."""
    df = _valid_events_df(10)
    df.loc[1, "event_id"] = df.loc[0, "event_id"]
    with pytest.raises(ValueError):
        validate_events_df(df)


def test_validate_events_df_rejects_end_before_start():
    """Test that validate_events_df raises an error if any event 
       has an end_date before its start_date."""
    df = _valid_events_df(10)
    df.loc[0, "start_date"] = "1990-01-10"
    df.loc[0, "end_date"] = "1990-01-01"
    with pytest.raises(ValueError):
        validate_events_df(df)


def test_validate_events_df_requires_at_least_10():
    """Test that validate_events_df raises an error if 
        there are fewer than 10 events."""
    df = _valid_events_df(9)
    with pytest.raises(ValueError):
        validate_events_df(df)
