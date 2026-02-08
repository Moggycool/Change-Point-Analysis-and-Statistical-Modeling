from __future__ import annotations
from src.config import DATA_PROCESSED_DIR, LOG_RETURNS_FILENAME, LOG_PRICE_FILENAME, COL_DATE, COL_PRICE, COL_LOG_PRICE
import sys
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    df = pd.read_csv(DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    out = df[[COL_DATE, COL_PRICE, COL_LOG_PRICE]].copy()
    out.to_csv(DATA_PROCESSED_DIR / LOG_PRICE_FILENAME, index=False)
    print("[OK] Rebuilt:", DATA_PROCESSED_DIR / LOG_PRICE_FILENAME)


if __name__ == "__main__":
    main()
