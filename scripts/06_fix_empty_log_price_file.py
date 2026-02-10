from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    LOG_RETURNS_FILENAME,
    LOG_PRICE_FILENAME,
    COL_DATE,
    COL_PRICE,
    COL_LOG_PRICE,
)

# Ensure project root is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    in_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME
    out_path = DATA_PROCESSED_DIR / LOG_PRICE_FILENAME

    # Load
    df = pd.read_csv(in_path)

    # Parse/validate date column
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", utc=False)
    n_bad_dates = int(df[COL_DATE].isna().sum())
    if n_bad_dates > 0:
        bad_idx = df.index[df[COL_DATE].isna()].tolist()[:10]
        raise ValueError(
            f"[ERROR] {n_bad_dates} rows have invalid {COL_DATE!r} after parsing. "
            f"Example row indices: {bad_idx}. Input file: {in_path}"
        )

    # Validate required columns exist (fail fast with a clear message)
    required = [COL_DATE, COL_PRICE, COL_LOG_PRICE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[ERROR] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}. Input file: {in_path}"
        )

    # Build output
    out = df[required].copy()

    # Ensure output dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    out.to_csv(out_path, index=False)

    # Report
    print(f"[OK] Rebuilt: {out_path}")
    print(f"[INFO] Rows written: {len(out):,}")
    print(f"[INFO] Date range: {out[COL_DATE].min()} -> {out[COL_DATE].max()}")


if __name__ == "__main__":
    main()
