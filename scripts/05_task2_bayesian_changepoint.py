"""
Bayesian changepoint analysis for Task 2.

Models:
- Model 1: switch in mean (mu_1 vs mu_2)
- Model 2: switch in volatility (sigma_1 vs sigma_2)

Updates (Task-2 rubric + robustness):
- Uses EVENTS_FILENAME from config (now brent_events.csv)
- Ensures pointwise log_likelihood is stored during sampling (compute_log_likelihood=True)
- Uses compare_models_safe() to attempt LOO then fallback to WAIC with full traceback logging
- Saves idata to NetCDF for reproducibility/offline debugging
- Writes tau date summary including 94% HDI mapped to calendar dates
- Produces raw-price overlay plots with change-point line + tau-date HDI band
- FIX: avoids KeyError for missing tau_hdi_low_date/high_date in impact_m2 by using tau_summary_m2
       (single source of truth for tau-date HDI) + safe dict .get fallbacks
"""

# scripts/05_task2_bayesian_changepoint.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
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
    COL_PRICE,
)

from src.data.io_task2 import load_log_returns_csv, ensure_log_features  # noqa: E402
from src.events.io_task2 import load_events  # noqa: E402

from src.eda.plots_task2 import (  # noqa: E402
    plot_price,
    plot_log_returns,
    plot_rolling_volatility,
    plot_price_series_with_changepoint,
)

from src.models.bayes_changepoint_task2 import (  # noqa: E402
    build_switchpoint_mean_model,
    build_switchpoint_sigma_model,
    sample_model,
    compute_impact_summary,
    compute_sigma_impact_summary,
    map_tau_samples_to_dates,
    compute_tau_date_summary,
    compute_convergence_report,
    prior_settings_summary,
    standardize,
    compare_models_safe,
    save_idata,
    validate_log_likelihood_finite,
)


def _safe_tau_hdi_dates_from_impact_dict(
    impact: dict[str, Any],
    *,
    fallback_low: Any = None,
    fallback_high: Any = None,
) -> tuple[Any, Any]:
    """
    Extract tau HDI dates from an impact dict safely.

    Some implementations return only tau_mode_date (and other fields) but not
    tau_hdi_low_date / tau_hdi_high_date. This helper prevents KeyError and
    provides optional fallbacks (typically from compute_tau_date_summary()).
    """
    low = impact.get("tau_hdi_low_date", None)
    high = impact.get("tau_hdi_high_date", None)

    # Allow alternate key names if your older code used them
    if low is None:
        low = impact.get("tau_hdi_low", None)
    if high is None:
        high = impact.get("tau_hdi_high", None)

    if low is None:
        low = fallback_low
    if high is None:
        high = fallback_high

    return low, high


def main() -> None:
    """Run Bayesian changepoint analysis for Task 2."""
    ensure_dirs()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    cast(Any, az).style.use("arviz-darkgrid")
    plt.rcParams["figure.dpi"] = 120

    # -------------------------
    # Load data
    # -------------------------
    data_path = DATA_PROCESSED_DIR / LOG_RETURNS_FILENAME
    df = load_log_returns_csv(data_path)
    df = ensure_log_features(df)

    # -------------------------
    # EDA figs
    # -------------------------
    # IMPORTANT: plot_price_series_with_changepoint REQUIRES cp_date -> do NOT use here.
    if (COL_PRICE in df.columns) and (COL_DATE in df.columns):
        fig = plot_price(df)
        fig.savefig(REPORTS_FIGURES_DIR / "task2_01_price_raw.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig = plot_log_returns(df)
    fig.savefig(REPORTS_FIGURES_DIR / "task2_02_log_returns.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_rolling_volatility(df, window=ROLLING_VOL_WINDOW_DAYS)
    fig.savefig(
        REPORTS_FIGURES_DIR /
        f"task2_03_rolling_vol_{ROLLING_VOL_WINDOW_DAYS}d.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------
    # Clean series
    # -------------------------
    y = df[COL_LOG_RETURN].to_numpy(dtype=float)
    mask = np.isfinite(y)
    y_clean = y[mask]
    dates_clean = pd.to_datetime(df.loc[mask, COL_DATE]).reset_index(drop=True)

    # -------------------------
    # Toggle standardization
    # -------------------------
    STANDARDIZE_RETURNS = False
    y_model = y_clean
    y_mean = None
    y_std = None
    if STANDARDIZE_RETURNS:
        y_model, y_mean, y_std = standardize(y_clean)

    priors_used = prior_settings_summary(y_model)

    # -------------------------
    # Model 1: mean switch
    # -------------------------
    model_m1 = build_switchpoint_mean_model(y_model)
    idata_m1 = sample_model(
        model_m1,
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        compute_log_likelihood=True,
    )
    save_idata(idata_m1, str(REPORTS_INTERIM_DIR / "idata_m1_mean_switch.nc"))

    conv_m1 = compute_convergence_report(
        idata_m1, ["tau", "mu_1", "mu_2", "sigma"])

    az.plot_trace(idata_m1, var_names=["tau", "mu_1", "mu_2", "sigma"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m1_trace.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_posterior(idata_m1, var_names=["tau"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m1_tau_posterior.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    with model_m1:
        ppc_m1 = pm.sample_posterior_predictive(idata_m1, random_seed=42)
    idata_m1.extend(ppc_m1)

    az.plot_ppc(idata_m1, data_pairs={"obs": "obs"}, num_pp_samples=200)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m1_ppc.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    impact_m1 = compute_impact_summary(idata_m1, dates_clean)
    pd.DataFrame([impact_m1.__dict__]).to_csv(
        REPORTS_INTERIM_DIR / "task2_m1_impact_summary.csv",
        index=False,
    )

    tau_summary_m1 = compute_tau_date_summary(idata_m1, dates_clean)
    pd.DataFrame([tau_summary_m1.__dict__]).to_csv(
        REPORTS_INTERIM_DIR / "task2_m1_tau_date_summary.csv",
        index=False,
    )

    tau_m1_df = map_tau_samples_to_dates(idata_m1, dates_clean)
    tau_m1_df.to_csv(REPORTS_INTERIM_DIR /
                     "task2_m1_tau_samples.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.hist(pd.to_datetime(tau_m1_df["tau_date"]), bins=60)
    ax.set_title("Posterior mass of change-point date (Model 1: mean switch)")
    ax.set_xlabel("date")
    plt.tight_layout()
    fig.savefig(REPORTS_FIGURES_DIR / "task2_m1_tau_date_mass.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Rubric overlay: CP line + tau HDI band on raw price (only if price exists)
    if (COL_PRICE in df.columns) and (COL_DATE in df.columns):
        fig = plot_price_series_with_changepoint(
            df,
            cp_date=impact_m1.tau_mode_date,
            date_col=COL_DATE,
            price_col=COL_PRICE,
            hdi_low=tau_summary_m1.tau_hdi_low_date,
            hdi_high=tau_summary_m1.tau_hdi_high_date,
            cp_label="Change point (τ mode)",
            hdi_label="τ date 94% HDI",
        )
        fig.savefig(REPORTS_FIGURES_DIR /
                    "task2_m1_price_overlay_cp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    (REPORTS_INTERIM_DIR / "task2_m1_loglik_diagnostics.json").write_text(
        json.dumps(validate_log_likelihood_finite(idata_m1), indent=2)
    )

    # -------------------------
    # Model 2: sigma switch
    # -------------------------
    model_m2 = build_switchpoint_sigma_model(y_model)
    idata_m2 = sample_model(
        model_m2,
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        compute_log_likelihood=True,
    )
    save_idata(idata_m2, str(REPORTS_INTERIM_DIR / "idata_m2_sigma_switch.nc"))

    conv_m2 = compute_convergence_report(
        idata_m2, ["tau", "mu", "sigma_1", "sigma_2"])

    az.plot_trace(idata_m2, var_names=["tau", "mu", "sigma_1", "sigma_2"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m2_trace.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_posterior(idata_m2, var_names=["tau"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m2_tau_posterior.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_posterior(idata_m2, var_names=["sigma_1", "sigma_2"])
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m2_sigma_posterior.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    with model_m2:
        ppc_m2 = pm.sample_posterior_predictive(idata_m2, random_seed=42)
    idata_m2.extend(ppc_m2)

    az.plot_ppc(idata_m2, data_pairs={"obs": "obs"}, num_pp_samples=200)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "task2_m2_ppc.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # NOTE: compute_sigma_impact_summary appears to return a plain dict (not a dataclass)
    impact_m2 = compute_sigma_impact_summary(idata_m2, dates_clean)
    pd.DataFrame([impact_m2]).to_csv(
        REPORTS_INTERIM_DIR / "task2_m2_sigma_impact_summary.csv",
        index=False,
    )

    # This is the canonical source for tau-date HDI for BOTH models
    tau_summary_m2 = compute_tau_date_summary(idata_m2, dates_clean)
    pd.DataFrame([tau_summary_m2.__dict__]).to_csv(
        REPORTS_INTERIM_DIR / "task2_m2_tau_date_summary.csv",
        index=False,
    )

    tau_m2_df = map_tau_samples_to_dates(idata_m2, dates_clean)
    tau_m2_df.to_csv(REPORTS_INTERIM_DIR /
                     "task2_m2_tau_samples.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.hist(pd.to_datetime(tau_m2_df["tau_date"]), bins=60)
    ax.set_title("Posterior mass of change-point date (Model 2: sigma switch)")
    ax.set_xlabel("date")
    plt.tight_layout()
    fig.savefig(REPORTS_FIGURES_DIR / "task2_m2_tau_date_mass.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Rubric overlay for Model 2 (safe: HDI dates from tau_summary_m2)
    if (COL_PRICE in df.columns) and (COL_DATE in df.columns):
        fig = plot_price_series_with_changepoint(
            df,
            cp_date=tau_summary_m2.tau_mode_date,
            date_col=COL_DATE,
            price_col=COL_PRICE,
            hdi_low=tau_summary_m2.tau_hdi_low_date,
            hdi_high=tau_summary_m2.tau_hdi_high_date,
            cp_label="Change point (τ mode)",
            hdi_label="τ date 94% HDI",
        )
        fig.savefig(REPORTS_FIGURES_DIR /
                    "task2_m2_price_overlay_cp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    (REPORTS_INTERIM_DIR / "task2_m2_loglik_diagnostics.json").write_text(
        json.dumps(validate_log_likelihood_finite(idata_m2), indent=2)
    )

    # -------------------------
    # Model comparison (LOO with safe fallback)
    # -------------------------
    cmp = compare_models_safe(
        {"mean_switch": idata_m1, "sigma_switch": idata_m2},
        ic="loo",
        fallback_ic="waic",
        error_path=str(REPORTS_INTERIM_DIR /
                       "task2_model_comparison_loo_error.txt"),
    )
    cmp.to_csv(REPORTS_INTERIM_DIR / "task2_model_comparison.csv")

    # -------------------------
    # Optional: events near CP
    # -------------------------
    events_path = DATA_RAW_DIR / EVENTS_FILENAME
    if events_path.exists():
        ev = load_events(events_path)

        cp1 = pd.to_datetime(impact_m1.tau_mode_date)
        ev1 = ev.copy()
        ev1["days_from_cp"] = (ev1[COL_DATE] - cp1).dt.days
        ev1_near = ev1.loc[ev1["days_from_cp"].abs(
        ) <= EVENT_MATCH_WINDOW_DAYS].copy()
        ev1_near.to_csv(REPORTS_INTERIM_DIR /
                        "task2_events_near_cp_mean_switch.csv", index=False)

        cp2 = pd.to_datetime(tau_summary_m2.tau_mode_date)
        ev2 = ev.copy()
        ev2["days_from_cp"] = (ev2[COL_DATE] - cp2).dt.days
        ev2_near = ev2.loc[ev2["days_from_cp"].abs(
        ) <= EVENT_MATCH_WINDOW_DAYS].copy()
        ev2_near.to_csv(REPORTS_INTERIM_DIR /
                        "task2_events_near_cp_sigma_switch.csv", index=False)

    # Debug prints (kept, but placed before metadata so you can see it even if something fails later)
    print("impact_m2 keys:", sorted(list(impact_m2.keys())))
    print("impact_m2:", impact_m2)

    # -------------------------
    # Metadata (FIXED: no KeyError)
    # -------------------------
    m2_hdi_low, m2_hdi_high = _safe_tau_hdi_dates_from_impact_dict(
        impact_m2,
        fallback_low=tau_summary_m2.tau_hdi_low_date,
        fallback_high=tau_summary_m2.tau_hdi_high_date,
    )

    meta = {
        "standardize_returns": STANDARDIZE_RETURNS,
        "y_mean_if_standardized": y_mean,
        "y_std_if_standardized": y_std,
        "priors_used": priors_used,
        "m1_convergence": conv_m1.__dict__,
        "m2_convergence": conv_m2.__dict__,
        "m1_cp_date_mode": impact_m1.tau_mode_date,
        "m1_cp_date_hdi_94": [tau_summary_m1.tau_hdi_low_date, tau_summary_m1.tau_hdi_high_date],
        "m2_cp_date_mode": tau_summary_m2.tau_mode_date,
        "m2_cp_date_hdi_94": [m2_hdi_low, m2_hdi_high],
        "events_filename_from_config": EVENTS_FILENAME,
        "artifacts": {
            "idata_m1": str(REPORTS_INTERIM_DIR / "idata_m1_mean_switch.nc"),
            "idata_m2": str(REPORTS_INTERIM_DIR / "idata_m2_sigma_switch.nc"),
            "comparison": str(REPORTS_INTERIM_DIR / "task2_model_comparison.csv"),
            "comparison_error_log": str(REPORTS_INTERIM_DIR / "task2_model_comparison_loo_error.txt"),
            "m1_loglik_diag": str(REPORTS_INTERIM_DIR / "task2_m1_loglik_diagnostics.json"),
            "m2_loglik_diag": str(REPORTS_INTERIM_DIR / "task2_m2_loglik_diagnostics.json"),
            "m1_tau_date_summary": str(REPORTS_INTERIM_DIR / "task2_m1_tau_date_summary.csv"),
            "m2_tau_date_summary": str(REPORTS_INTERIM_DIR / "task2_m2_tau_date_summary.csv"),
        },
    }
    (REPORTS_INTERIM_DIR / "task2_run_metadata.json").write_text(json.dumps(meta, indent=2))

    if (not conv_m1.ok) or (not conv_m2.ok):
        warn = {
            "m1_ok": conv_m1.ok,
            "m1_offending": conv_m1.offending,
            "m2_ok": conv_m2.ok,
            "m2_offending": conv_m2.offending,
        }
        (REPORTS_INTERIM_DIR /
         "task2_convergence_warning.json").write_text(json.dumps(warn, indent=2))


if __name__ == "__main__":
    main()
