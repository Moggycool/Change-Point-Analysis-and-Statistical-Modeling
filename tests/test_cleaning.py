""" A suite of tests for the data cleaning functions. """
# tests/test_cleaning.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.data.cleaning import clean_brent_prices  # noqa: E402
# pylint: enable=wrong-import-position


def test_clean_brent_prices_parses_date_and_price() -> None:
    """Test that clean_brent_prices correctly parses date and price columns."""
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "21-May-87", "22-May-87"],
            "Price": ["18.63", "18.45", "18.55"],
        }
    )
    out = clean_brent_prices(df)

    assert list(out.columns) == ["date", "price"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert pd.api.types.is_numeric_dtype(out["price"])
    assert len(out) == 3


def test_clean_brent_prices_date_is_datetime_monotonic_unique() -> None:
    """
    Explicit rubric checks:
      - dtype of date is datetime64
      - monotonic increasing (sorted ascending)
      - no duplicate dates
    """
    df = pd.DataFrame(
        {
            # intentionally unsorted + duplicated date
            "Date": ["21-May-87", "20-May-87", "20-May-87", "22-May-87"],
            "Price": [19.0, 18.5, 18.7, 19.2],
        }
    )

    out = clean_brent_prices(df, duplicate_rule="keep_last")

    assert pd.api.types.is_datetime64_any_dtype(out["date"]), (
        f"Expected datetime64 dtype for 'date', got {out['date'].dtype}"
    )
    assert out["date"].is_monotonic_increasing, "Expected 'date' to be sorted ascending."
    assert not out["date"].duplicated().any(
    ), "Expected no duplicate dates after cleaning."


def test_clean_brent_prices_raises_on_bad_date() -> None:
    """
    Datetime parsing is strict (errors='raise'); unparseable dates must raise.
    This directly supports the 'explicit datetime conversion' requirement.
    """
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "BADDATE"],
            "Price": ["18.63", "18.45"],
        }
    )
    with pytest.raises(Exception):
        _ = clean_brent_prices(df)


def test_clean_brent_prices_drops_bad_prices_tolerant() -> None:
    """
    Price parsing is tolerant (errors='coerce' + dropna), so bad prices are dropped.
    """
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "21-May-87", "22-May-87"],
            "Price": ["18.63", "NOTNUM", "18.55"],
        }
    )
    out = clean_brent_prices(df)

    # rows with unparseable price should be dropped (date parsing remains strict)
    assert len(out) == 2
    assert out["price"].notna().all()


def test_clean_brent_prices_handles_duplicates_keep_last() -> None:
    """Test that clean_brent_prices handles duplicates correctly when keeping the last occurrence."""
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "20-May-87", "21-May-87"],
            "Price": [10, 11, 12],
        }
    )
    out = clean_brent_prices(df, duplicate_rule="last")
    assert len(out) == 2

    target_date = pd.to_datetime("20-May-87", format="%d-%b-%y")
    v = out.loc[out["date"] == target_date, "price"].iloc[0]
    assert v == 11


def test_clean_brent_prices_rejects_unknown_duplicate_rule() -> None:
    """Test that clean_brent_prices raises an error when given an unknown duplicate handling rule."""
    df = pd.DataFrame({"Date": ["20-May-87"], "Price": [18.63]})
    with pytest.raises(ValueError):
        clean_brent_prices(df, duplicate_rule="unknown")
