""" Script to validate the events.csv file against the defined schema. """
# scripts/04_validate_events.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.events.schema import validate_events_df  # noqa: E402
from src.config import ensure_dirs, DATA_RAW_DIR, EVENTS_FILENAME  # noqa: E402


def main() -> None:
    """ Main function to validate events dataset. """
    ensure_dirs()

    events_path = DATA_RAW_DIR / EVENTS_FILENAME
    df = pd.read_csv(events_path)

    validated = validate_events_df(df)

    print(f"[OK] Events validated: {events_path}")
    print(f"Rows: {len(validated):,}")
    print(validated.head(10))


if __name__ == "__main__":
    main()
