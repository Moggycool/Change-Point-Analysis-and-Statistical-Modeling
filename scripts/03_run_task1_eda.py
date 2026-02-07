"""A script to run EDA for Task 1: Stationarity tests and plots."""
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
from src.eda.plots import (  # noqa: E402
    plot_price_and_log_price,
    plot_log_returns,
    plot_rolling_volatility,
)
from src.config import (  # noqa: E402
    ensure_dirs,
    DATA_PROCESSED_DIR,
    REPORTS_FIGURES_DIR,
    LOG_RETURNS_FILENAME,
    ROLLING_VOL_WINDOW_DAYS,
)
from src.eda.time_series_tests import run_stationarity_suite  # noqa: E402

# pylint: enable=wrong-import-position


def _save_stationarity_table_png(df: pd.DataFrame, out_path: Path) -> None:
    """
    Render a DataFrame as a readable PNG table.

    This avoids the common "blank/empty PNG" issue by:
    - creating an explicit Figure/Axes
    - turning axes off
    - scaling figure size with rows/cols
    - saving via fig.savefig(...) before closing
    """
    # Local import so the script doesn't require matplotlib unless this is called
    import matplotlib.pyplot as plt  # pylint: disable=import-error

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

    # Save BEFORE close (critical)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Main function to run EDA for Task 1."""
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

    # Stationarity suite -> CSV + PNG table
    results = run_stationarity_suite(
        df, cols=["price", "log_price", "log_return"])

    out_csv = REPORTS_FIGURES_DIR / "stationarity_tests_task1.csv"
    results.to_csv(out_csv, index=False)

    out_png = REPORTS_FIGURES_DIR / "stationarity_tests_table.png"
    _save_stationarity_table_png(results, out_png)

    print(f"[OK] Figures saved to: {REPORTS_FIGURES_DIR}")
    print(f"[OK] Stationarity table saved to: {out_csv}")
    print(f"[OK] Stationarity table PNG saved to: {out_png}")
    print(results)


if __name__ == "__main__":
    main()
