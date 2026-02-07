""" A script to run EDA for Task 1: Stationarity tests and plots. """
# scripts/03_run_task1_eda.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.eda.plots import (
    plot_price_and_log_price,
    plot_log_returns,
    plot_rolling_volatility,
)  # noqa: E402

from src.config import (
    ensure_dirs,
    DATA_PROCESSED_DIR,
    REPORTS_FIGURES_DIR,
    LOG_RETURNS_FILENAME,
    ROLLING_VOL_WINDOW_DAYS,
)  # noqa: E402
from src.eda.time_series_tests import run_stationarity_suite  # noqa: E402

# pylint: enable=wrong-import-position


def main() -> None:
    """ Main function to run EDA for Task 1. """
    ensure_dirs()

    data_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")

    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = plot_price_and_log_price(df)
    fig1.savefig(REPORTS_FIGURES_DIR / "01_price_and_log_price.png",
                 dpi=150, bbox_inches="tight")

    fig2 = plot_log_returns(df)
    fig2.savefig(REPORTS_FIGURES_DIR / "02_log_returns.png",
                 dpi=150, bbox_inches="tight")

    fig3 = plot_rolling_volatility(df, window=ROLLING_VOL_WINDOW_DAYS)
    fig3.savefig(
        REPORTS_FIGURES_DIR /
        f"03_rolling_volatility_{ROLLING_VOL_WINDOW_DAYS}d.png",
        dpi=150,
        bbox_inches="tight",
    )

    results = run_stationarity_suite(
        df, cols=["price", "log_price", "log_return"])
    out_csv = REPORTS_FIGURES_DIR / "stationarity_tests_task1.csv"
    results.to_csv(out_csv, index=False)

    print(f"[OK] Figures saved to: {REPORTS_FIGURES_DIR}")
    print(f"[OK] Stationarity table saved to: {out_csv}")
    print(results)


if __name__ == "__main__":
    main()
