"""Task 2: Bayesian change point modeling + insight generation (PyMC)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (  # noqa: E402
    ensure_dirs,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_INTERIM_DIR,
    LOG_RETURNS_FILENAME,
    EVENTS_FILENAME,
    ROLLING_VOL_WINDOW_DAYS,
    EVENT_MATCH_WINDOW_DAYS,
    COL_DATE,
    COL_LOG_RETURN,
)
from src.data.io_task2 import load_log_returns_csv, ensure_log_features  # noqa: E402
from src.events.io_task2 import load_events  # noqa: E402
from src.eda.plots_task2 import plot_price, plot_log_returns, plot_rolling_volatility  # noqa: E402
from src.models.bayes_changepoint_task2 import (  # noqa: E402
    build_switchpoint_mean_model,
    sample_model,
    compute_impact_summary,
)


def main() -> None:
    ensure_dirs()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Data prep
    data_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME  # brent_log_returns.csv
    df = load_log_returns_csv(data_path)
    df = ensure_log_features(df)

    # 1) EDA plots
    fig = plot_price(df)
    fig.savefig(REPORTS_FIGURES_DIR / "task2_01_price_raw.png",
                dpi=150, bbox_inches="tight")

    fig = plot_log_returns(df)
    fig.savefig(REPORTS_FIGURES_DIR / "task2_02_log_returns.png",
                dpi=150, bbox_inches="tight")

    fig = plot_rolling_volatility(df, window=ROLLING_VOL_WINDOW_DAYS)
    fig.savefig(
        REPORTS_FIGURES_DIR /
        f"task2_03_rolling_vol_{ROLLING_VOL_WINDOW_DAYS}d.png",
        dpi=150,
        bbox_inches="tight",
    )

    # 2) Model input: log returns (drop NaN)
    y = df[COL_LOG_RETURN].to_numpy()
    mask = np.isfinite(y)
    y_clean = y[mask]
    dates_clean = df.loc[mask, COL_DATE].reset_index(drop=True)

    # 2) Build + sample model
    model = build_switchpoint_mean_model(y_clean)
    idata = sample_model(model, draws=2000, tune=2000,
                         chains=4, target_accept=0.9)

    # 3) Diagnostics + posterior plots
    import arviz as az
    import matplotlib.pyplot as plt

    summary = az.summary(
        idata, var_names=["tau", "mu_1", "mu_2", "sigma"], round_to=4)
    summary.to_csv(REPORTS_INTERIM_DIR / "task2_pymc_summary.csv")

    az.plot_trace(idata, var_names=["tau", "mu_1", "mu_2", "sigma"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_04_trace.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_posterior(idata, var_names=["tau"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_05_tau_posterior.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_posterior(idata, var_names=["mu_1", "mu_2", "sigma"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_06_params_posterior.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Identify change point + quantify impact
    impact = compute_impact_summary(idata, dates_clean)
    pd.DataFrame([impact.__dict__]).to_csv(
        REPORTS_INTERIM_DIR / "task2_impact_summary.csv", index=False)

    # Save tau samples mapped to dates (useful for reporting uncertainty)
    tau_samples = idata.posterior["tau"].values.reshape(-1).astype(int)
    tau_dates = pd.to_datetime(dates_clean.iloc[tau_samples].values)
    pd.DataFrame({"tau_index": tau_samples, "tau_date": tau_dates}).to_csv(
        REPORTS_INTERIM_DIR / "task2_tau_samples.csv", index=False
    )

    # 4) Associate changes with causes (events near the change point)
    events_path = DATA_RAW_DIR / EVENTS_FILENAME
    if events_path.exists():
        ev = load_events(events_path)
        cp_date = pd.to_datetime(impact.tau_mode_date)
        ev["days_from_cp"] = (ev[COL_DATE] - cp_date).dt.days
        ev_near = ev.loc[ev["days_from_cp"].abs(
        ) <= EVENT_MATCH_WINDOW_DAYS].copy()
        ev_near.to_csv(REPORTS_INTERIM_DIR /
                       "task2_events_near_cp.csv", index=False)
    else:
        ev_near = None

    print("[OK] Task 2 completed.")
    print(" - Figures:", REPORTS_FIGURES_DIR)
    print(" - Tables :", REPORTS_INTERIM_DIR)
    print(" - Change point (mode tau):", impact.tau_mode_date)
    if ev_near is not None:
        print(
            f" - Events within ±{EVENT_MATCH_WINDOW_DAYS} days:", len(ev_near))
    print("\nImpact summary:")
    print(impact)


if __name__ == "__main__":
    main()
