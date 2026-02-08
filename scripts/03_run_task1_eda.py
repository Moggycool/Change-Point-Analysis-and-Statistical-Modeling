"""
A script to run EDA for Task 1: Stationarity tests and plots.

Key guarantees:
- Reads ONLY from data/processed
- Fails loud if inputs are missing or empty
- Writes brent_log_price.csv (canonical)
- Saves figures and then explicitly closes them (prevents 0-byte PNG artifacts on Windows)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.config import (  # noqa: E402
    ensure_dirs,
    DATA_PROCESSED_DIR,
    REPORTS_FIGURES_DIR,
    LOG_RETURNS_FILENAME,
    ROLLING_VOL_WINDOW_DAYS,
)
from src.io.save_data import save_csv  # noqa: E402
from src.eda.plots import (  # noqa: E402
    plot_price_and_log_price,
    plot_log_returns,
    plot_rolling_volatility,
    plot_price_series,
    plot_log_price_series,
)
from src.eda.time_series_tests import run_stationarity_suite  # noqa: E402

# pylint: enable=wrong-import-position


def _save_stationarity_table_png(df: pd.DataFrame, out_path: Path) -> None:
    """Render a DataFrame as a readable PNG table."""
    import matplotlib.pyplot as plt  # local import

    out_path.parent.mkdir(parents=True, exist_ok=True)

    show = df.copy()

    # Light formatting for readability if these columns exist
    for col in ["p_value", "pvalue", "p-val"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").map(
                lambda x: f"{x:.3g}" if pd.notna(x) else ""
            )

    for col in ["statistic", "test_statistic", "adf_statistic", "kpss_statistic"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").map(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            )

    for col in ["n_lags", "nlags", "n_obs", "nobs"]:
        if col in show.columns:
            show[col] = pd.to_numeric(
                show[col], errors="coerce").astype("Int64").astype(str)
            show[col] = show[col].replace("<NA>", "")

    nrows, ncols = show.shape
    fig_w = max(10.0, 1.2 * ncols)
    fig_h = max(2.5, 0.45 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=show.values,
        colLabels=list(show.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.35)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_and_close(fig, out_path: Path, dpi: int = 150) -> None:
    """Save a matplotlib Figure, then close it. Also sanity-check file size."""
    import matplotlib.pyplot as plt  # local import

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Size sanity check (catches the exact 0-byte issue you saw)
    if not out_path.exists():
        raise FileNotFoundError(f"Expected figure not created: {out_path}")
    size = os.path.getsize(out_path)
    if size == 0:
        raise RuntimeError(
            f"Figure saved as 0 bytes: {out_path}. "
            "This usually indicates a backend/file-handle/exception issue."
        )


def main() -> None:
    ensure_dirs()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Canonical inputs (processed only) ---
    clean_prices_path = DATA_PROCESSED_DIR / "brent_prices_clean.csv"
    returns_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME

    if not clean_prices_path.exists():
        raise FileNotFoundError(
            f"Missing {clean_prices_path}. Run scripts/01_clean_prices.py first.")

    prices = pd.read_csv(clean_prices_path, parse_dates=[
                         "date"]).sort_values("date")
    if prices.empty:
        raise ValueError("Loaded brent_prices_clean.csv but it has 0 rows.")

    # Ensure required columns exist
    for c in ["date", "price"]:
        if c not in prices.columns:
            raise KeyError(
                f"Missing column '{c}' in {clean_prices_path}. Found: {list(prices.columns)}")

    # Ensure log_price exists
    if "log_price" not in prices.columns:
        from src.data.transforms import add_log_price  # local import
        prices = add_log_price(prices, price_col="price")

    if prices["log_price"].notna().sum() == 0:
        raise ValueError(
            "log_price is all-NaN after add_log_price; cannot proceed.")

    # Save canonical processed log price output
    log_price_out = DATA_PROCESSED_DIR / "brent_log_price.csv"
    save_csv(prices[["date", "log_price"]].copy(), log_price_out, index=False)

    # --- REQUIRED trend plots ---
    _save_and_close(
        plot_price_and_log_price(prices),
        REPORTS_FIGURES_DIR / "01_price_and_log_price.png",
        dpi=150,
    )

    _save_and_close(
        plot_price_series(prices),
        REPORTS_FIGURES_DIR / "price_series.png",
        dpi=150,
    )

    _save_and_close(
        plot_log_price_series(prices),
        REPORTS_FIGURES_DIR / "log_price_series.png",
        dpi=150,
    )

    # --- Returns + volatility + stationarity outputs ---
    if not returns_path.exists():
        raise FileNotFoundError(
            f"Missing {returns_path}. Run scripts/02_make_returns.py first.")

    df = pd.read_csv(returns_path, parse_dates=["date"]).sort_values("date")
    if df.empty:
        raise ValueError("Loaded brent_log_returns.csv but it has 0 rows.")

    if "log_return" not in df.columns:
        raise KeyError(
            f"Missing 'log_return' in {returns_path}. Found: {list(df.columns)}")

    if df["log_return"].notna().sum() == 0:
        raise ValueError(
            "log_return is all-NaN; cannot plot returns/volatility.")

    _save_and_close(
        plot_log_returns(df),
        REPORTS_FIGURES_DIR / "02_log_returns.png",
        dpi=150,
    )

    # Save both: explicit window name AND a simple rubric-friendly alias if you want it
    vol_path_windowed = REPORTS_FIGURES_DIR / \
        f"03_rolling_volatility_{ROLLING_VOL_WINDOW_DAYS}d.png"
    _save_and_close(
        plot_rolling_volatility(df, window=ROLLING_VOL_WINDOW_DAYS),
        vol_path_windowed,
        dpi=150,
    )

    # Optional alias requested by you
    vol_alias = REPORTS_FIGURES_DIR / "rolling_volatility.png"
    # just copy bytes (avoid recompute)
    vol_alias.write_bytes(vol_path_windowed.read_bytes())

    results = run_stationarity_suite(
        df, cols=["price", "log_price", "log_return"])
    out_csv = REPORTS_FIGURES_DIR / "stationarity_tests_task1.csv"
    results.to_csv(out_csv, index=False)

    out_png = REPORTS_FIGURES_DIR / "stationarity_tests_table.png"
    _save_stationarity_table_png(results, out_png)

    print(f"[OK] Figures saved to: {REPORTS_FIGURES_DIR}")
    print(f"[OK] Wrote processed log price to: {log_price_out}")
    print(f"[OK] Stationarity table saved to: {out_csv}")
    print(f"[OK] Stationarity table PNG saved to: {out_png}")


if __name__ == "__main__":
    main()
