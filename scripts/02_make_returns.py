# scripts/02_make_returns.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.config import (
    ensure_dirs,
    DATA_PROCESSED_DIR,
    BRENT_CLEAN_FILENAME,
    LOG_RETURNS_FILENAME,
)  # noqa: E402
from src.io.save_data import save_csv  # noqa: E402
from src.data.transforms import add_log_price, add_log_returns  # noqa: E402
# pylint: enable=wrong-import-position


def main() -> None:
    """ Main function to process cleaned Brent prices and compute log returns """
    ensure_dirs()

    clean_path = DATA_PROCESSED_DIR / BRENT_CLEAN_FILENAME
    out_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME

    df = pd.read_csv(clean_path, parse_dates=["date"]).sort_values(
        "date").reset_index(drop=True)
    df = add_log_price(df, price_col="price")
    df = add_log_returns(df, log_price_col="log_price")

    save_csv(df, out_path, index=False)

    print(f"[OK] Processed data written to: {out_path}")
    print(df.head(10))
    print(f"Missing log_return: {int(df['log_return'].isna().sum())}")


if __name__ == "__main__":
    main()
