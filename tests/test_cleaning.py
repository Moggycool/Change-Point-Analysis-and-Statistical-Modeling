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


def test_clean_brent_prices_parses_date_and_price():
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
    assert pd.api.types.is_float_dtype(out["price"])
    assert len(out) == 3


def test_clean_brent_prices_drops_bad_rows():
    """Test that clean_brent_prices drops rows with invalid dates or prices."""
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "BADDATE", "22-May-87"],
            "Price": ["18.63", "18.45", "NOTNUM"],
        }
    )
    out = clean_brent_prices(df)
    # only first row should survive
    assert len(out) == 1


def test_clean_brent_prices_handles_duplicates_keep_last():
    """Test that clean_brent_prices handles duplicates correctly when keeping the last occurrence."""
    df = pd.DataFrame(
        {
            "Date": ["20-May-87", "20-May-87", "21-May-87"],
            "Price": [10, 11, 12],
        }
    )
    out = clean_brent_prices(df, duplicate_rule="last")
    assert len(out) == 2
    # for 20-May-87, keep last => 11
    v = out.loc[out["date"] == pd.to_datetime(
        "20-May-87", format="%d-%b-%y"), "price"].iloc[0]
    assert v == 11


def test_clean_brent_prices_rejects_unknown_duplicate_rule():
    """Test that clean_brent_prices raises an error when given an unknown duplicate handling rule."""
    df = pd.DataFrame({"Date": ["20-May-87"], "Price": [18.63]})
    with pytest.raises(ValueError):
        clean_brent_prices(df, duplicate_rule="unknown")
