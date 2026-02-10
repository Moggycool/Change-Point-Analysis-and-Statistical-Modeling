# src/models/bayes_changepoint_task2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple, cast, Literal

import os
import traceback

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az


# -----------------------------
# Convergence gating utilities
# -----------------------------
@dataclass(frozen=True)
class ConvergenceReport:
    """Summary of MCMC convergence diagnostics for a set of variables."""
    ok: bool
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    offending: Dict[str, Dict[str, float]]  # var -> metrics


def compute_convergence_report(
    idata: az.InferenceData,
    var_names: Sequence[str],
    rhat_max: float = 1.01,
    ess_min: float = 400.0,
) -> ConvergenceReport:
    """
    Summarize MCMC health for a set of variables using ArviZ summary.
    Gates: max R-hat <= rhat_max and min ESS (bulk & tail) >= ess_min.
    """
    s = az.summary(
        idata,
        var_names=list(var_names),
        round_to=None,
        kind="all",
    )

    rhat_col = "r_hat" if "r_hat" in s.columns else (
        "rhat" if "rhat" in s.columns else None)
    ess_bulk_col = "ess_bulk" if "ess_bulk" in s.columns else None
    ess_tail_col = "ess_tail" if "ess_tail" in s.columns else None

    if rhat_col is None or ess_bulk_col is None or ess_tail_col is None:
        raise ValueError(
            f"ArviZ summary missing expected columns. Found columns: {list(s.columns)}"
        )

    max_rhat = float(np.nanmax(s[rhat_col].to_numpy()))
    min_ess_bulk = float(np.nanmin(s[ess_bulk_col].to_numpy()))
    min_ess_tail = float(np.nanmin(s[ess_tail_col].to_numpy()))

    offending: Dict[str, Dict[str, float]] = {}
    ok = True
    for idx, row in s.iterrows():
        rhat = float(row[rhat_col])
        eb = float(row[ess_bulk_col])
        et = float(row[ess_tail_col])
        bad = (np.isfinite(rhat) and rhat > rhat_max) or (
            eb < ess_min) or (et < ess_min)
        if bad:
            ok = False
            offending[str(idx)] = {"rhat": rhat,
                                   "ess_bulk": eb, "ess_tail": et}

    return ConvergenceReport(
        ok=ok,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        offending=offending,
    )


# -----------------------------
# Preprocessing helpers
# -----------------------------
def standardize(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standardize data to mean 0 and std 1 for better MCMC performance.
    Returns standardized data, original mean, and original std for later rescaling if needed.
    """
    y = np.asarray(y, dtype=float)
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize: std is not positive.")
    return (y - mu) / sd, mu, sd


def _weakly_informative_prior_scales(y: np.ndarray) -> Dict[str, float]:
    """
    Scale-aware weakly informative prior settings based on data scale.
    For returns, y is often small; this keeps priors sensible.
    """
    y = np.asarray(y, dtype=float)
    sd = float(np.nanstd(y, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    return {
        "mu_prior_sigma": 2.0 * sd,          # wide for mean
        "sigma_prior_sigma": 2.0 * sd,       # half-normal scale
    }


def prior_settings_summary(
    y: np.ndarray,
    mu_prior_sigma: Optional[float] = None,
    sigma_prior_sigma: Optional[float] = None,
) -> Dict[str, float]:
    """Summarize the prior settings used for the model, including any defaults based on data scale."""
    s = _weakly_informative_prior_scales(y)
    if mu_prior_sigma is None:
        mu_prior_sigma = s["mu_prior_sigma"]
    if sigma_prior_sigma is None:
        sigma_prior_sigma = s["sigma_prior_sigma"]
    return {
        "mu_prior_sigma": float(mu_prior_sigma),
        "sigma_prior_sigma": float(sigma_prior_sigma),
        "y_std": float(np.nanstd(np.asarray(y, dtype=float), ddof=1)),
    }


# -----------------------------
# Model builders (Task 2)
# -----------------------------
def build_switchpoint_mean_model(
    y: np.ndarray,
    tau_prior: str = "DiscreteUniform",
    mu_prior_sigma: Optional[float] = None,
    sigma_prior_sigma: Optional[float] = None,
) -> pm.Model:
    """
    Mandatory rubric model:
    - Discrete change point tau ~ DiscreteUniform(0, n-1)
    - Mean shifts: mu_1 before tau, mu_2 after tau
    - Constant sigma
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations for changepoint model.")

    ps = _weakly_informative_prior_scales(y)
    if mu_prior_sigma is None:
        mu_prior_sigma = ps["mu_prior_sigma"]
    if sigma_prior_sigma is None:
        sigma_prior_sigma = ps["sigma_prior_sigma"]

    with pm.Model() as model:
        if tau_prior != "DiscreteUniform":
            raise ValueError(
                "Rubric requires tau ~ DiscreteUniform for the core model.")
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu_1 = pm.Normal("mu_1", mu=0.0, sigma=mu_prior_sigma)
        mu_2 = pm.Normal("mu_2", mu=0.0, sigma=mu_prior_sigma)

        sigma = pm.HalfNormal("sigma", sigma=sigma_prior_sigma)

        t = np.arange(n)
        mu_t = pm.math.switch(pm.math.lt(t, tau), mu_1, mu_2)

        pm.Normal("obs", mu=mu_t, sigma=sigma, observed=y)

    return model


def build_switchpoint_sigma_model(
    y: np.ndarray,
    tau_prior: str = "DiscreteUniform",
    mu_prior_sigma: Optional[float] = None,
    sigma_prior_sigma: Optional[float] = None,
) -> pm.Model:
    """
    Extension model:
    - Discrete change point tau ~ DiscreteUniform(0, n-1)
    - Constant mean mu
    - Volatility shifts: sigma_1 before tau, sigma_2 after tau
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations for changepoint model.")

    ps = _weakly_informative_prior_scales(y)
    if mu_prior_sigma is None:
        mu_prior_sigma = ps["mu_prior_sigma"]
    if sigma_prior_sigma is None:
        sigma_prior_sigma = ps["sigma_prior_sigma"]

    with pm.Model() as model:
        if tau_prior != "DiscreteUniform":
            raise ValueError(
                "Rubric requires tau ~ DiscreteUniform for the core model.")
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu = pm.Normal("mu", mu=0.0, sigma=mu_prior_sigma)

        sigma_1 = pm.HalfNormal("sigma_1", sigma=sigma_prior_sigma)
        sigma_2 = pm.HalfNormal("sigma_2", sigma=sigma_prior_sigma)

        t = np.arange(n)
        sigma_t = pm.math.switch(pm.math.lt(t, tau), sigma_1, sigma_2)

        pm.Normal("obs", mu=mu, sigma=sigma_t, observed=y)

    return model


def build_switchpoint_mean_model_studentt(
    y: np.ndarray,
    tau_prior: str = "DiscreteUniform",
    mu_prior_sigma: Optional[float] = None,
    sigma_prior_sigma: Optional[float] = None,
    nu_prior: Tuple[float, float] = (2.0, 0.1),
) -> pm.Model:
    """
    Robust extension:
    Mean switch with Student-T likelihood (fat tails).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations for changepoint model.")

    ps = _weakly_informative_prior_scales(y)
    if mu_prior_sigma is None:
        mu_prior_sigma = ps["mu_prior_sigma"]
    if sigma_prior_sigma is None:
        sigma_prior_sigma = ps["sigma_prior_sigma"]

    with pm.Model() as model:
        if tau_prior != "DiscreteUniform":
            raise ValueError(
                "Rubric requires tau ~ DiscreteUniform for the core model.")
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu_1 = pm.Normal("mu_1", mu=0.0, sigma=mu_prior_sigma)
        mu_2 = pm.Normal("mu_2", mu=0.0, sigma=mu_prior_sigma)

        sigma = pm.HalfNormal("sigma", sigma=sigma_prior_sigma)
        nu = pm.Exponential("nu", lam=nu_prior[1]) + nu_prior[0]  # nu >= 2

        t = np.arange(n)
        mu_t = pm.math.switch(pm.math.lt(t, tau), mu_1, mu_2)

        pm.StudentT("obs", nu=nu, mu=mu_t, sigma=sigma, observed=y)

    return model


# -----------------------------
# Sampling wrapper
# -----------------------------
def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    compute_log_likelihood: bool = True,
) -> az.InferenceData:
    """
    Sample model with PyMC defaults; discrete tau will be handled by a discrete sampler.

    Key fix for LOO/compare:
    - ensure idata contains pointwise log_likelihood (required for PSIS-LOO).
    """
    idata_kwargs: Dict[str, Any] = {}
    if compute_log_likelihood:
        idata_kwargs["log_likelihood"] = True

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
            idata_kwargs=idata_kwargs if idata_kwargs else None,
        )
    return idata


# -----------------------------
# LOO / Compare helpers (NEW)
# -----------------------------
def _idata_has_log_likelihood(idata: az.InferenceData) -> bool:
    # InferenceData groups are attached dynamically; use getattr for type-checker friendliness.
    return getattr(idata, "log_likelihood", None) is not None


def validate_log_likelihood_finite(idata: az.InferenceData) -> Dict[str, Any]:
    """
    Returns diagnostics about log_likelihood presence and finiteness.
    """
    out: Dict[str, Any] = {
        "has_log_likelihood": _idata_has_log_likelihood(idata)}
    if not out["has_log_likelihood"]:
        return out

    try:
        ll = getattr(idata, "log_likelihood", None)
        if ll is None:
            return out
        ll_arr = cast(Any, ll).to_array().values
        out["finite_fraction"] = float(np.isfinite(ll_arr).mean())
        out["ll_min"] = float(np.nanmin(ll_arr))
        out["ll_max"] = float(np.nanmax(ll_arr))
    except Exception as e:
        out["error"] = f"Could not evaluate finiteness: {e!r}"
    return out


def compare_models_safe(
    models: Dict[str, az.InferenceData],
    ic: Literal["loo", "waic"] = "loo",
    fallback_ic: Literal["loo", "waic"] = "waic",
    error_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Robust model comparison:
    - tries az.compare with ic (default loo)
    - if it fails, writes full traceback to error_path (if provided)
      and falls back to WAIC.

    Returns the compare dataframe.
    """
    try:
        return az.compare(models, ic=ic)

    except Exception:
        tb = traceback.format_exc()
        msg_lines = [
            f"Encountered error in ELPD computation of compare.",
            f"Requested ic={ic!r}. Falling back to ic={fallback_ic!r}.",
            "",
            "Diagnostics per model (log_likelihood checks):",
        ]
        for name, idata in models.items():
            diag = validate_log_likelihood_finite(idata)
            msg_lines.append(f"- {name}: {diag}")

        msg_lines.append("")
        msg_lines.append("Full traceback:")
        msg_lines.append(tb)
        msg = "\n".join(msg_lines)

        if error_path is not None:
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(msg)

        # Fallback
        return az.compare(models, ic=fallback_ic)


def save_idata(idata: az.InferenceData, path: str) -> None:
    """
    Convenience: persist idata so you can debug LOO later without rerunning MCMC.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idata.to_netcdf(path)


# -----------------------------
# Postprocessing / Reporting
# -----------------------------
@dataclass(frozen=True)
class ImpactSummary:
    """Summary of mean-shift impact with calendar-date mapping."""
    # Tau (index + mapped dates)
    tau_mode: int
    tau_mode_date: str

    # NEW: HDI endpoints for tau (index + mapped dates)
    tau_hdi_low: int
    tau_hdi_high: int
    tau_hdi_low_date: str
    tau_hdi_high_date: str

    # Mean-shift effect (for mean-switch model)
    prob_mu2_gt_mu1: float
    delta_mu_mean: float
    delta_mu_hdi_low: float
    delta_mu_hdi_high: float


@dataclass(frozen=True)
class TauDateSummary:
    """Summary of posterior tau with calendar-date mapping."""
    tau_mode: int
    tau_mode_date: str
    tau_hdi_low: int
    tau_hdi_high: int
    tau_hdi_low_date: str
    tau_hdi_high_date: str


def map_tau_samples_to_dates(
    idata: az.InferenceData,
    dates: Sequence[pd.Timestamp] | pd.Series,
    tau_var: str = "tau",
) -> pd.DataFrame:
    """
    Convert posterior samples of tau (index) into calendar dates (same index in `dates`).
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    posterior = cast(Any, idata).posterior
    tau = posterior[tau_var].values.reshape(-1).astype(int)
    tau = np.clip(tau, 0, len(dates) - 1)
    tau_dates = dates.iloc[tau].to_numpy()
    return pd.DataFrame({"tau": tau, "tau_date": pd.to_datetime(tau_dates)})


def _mode_int(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=int)
    vals, counts = np.unique(x, return_counts=True)
    return int(vals[np.argmax(counts)])


def compute_tau_date_summary(
    idata: az.InferenceData,
    dates: Sequence[pd.Timestamp] | pd.Series,
    tau_var: str = "tau",
    hdi_prob: float = 0.94,
) -> TauDateSummary:
    """
    Compute the posterior mode of tau and a discrete HDI interval, then map both
    the mode and HDI endpoints to calendar dates.
    """
    posterior = cast(Any, idata).posterior
    tau = posterior[tau_var].values.reshape(-1).astype(int)

    tau_mode = _mode_int(tau)

    hdi = az.hdi(tau, hdi_prob=hdi_prob)
    # HDI can be float even for integer-valued samples; map to an inclusive index range.
    tau_hdi_low = int(np.floor(float(hdi[0])))
    tau_hdi_high = int(np.ceil(float(hdi[1])))

    dates_s = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    max_idx = len(dates_s) - 1
    tau_mode_i = int(np.clip(tau_mode, 0, max_idx))
    tau_low_i = int(np.clip(tau_hdi_low, 0, max_idx))
    tau_high_i = int(np.clip(tau_hdi_high, 0, max_idx))

    return TauDateSummary(
        tau_mode=tau_mode_i,
        tau_mode_date=str(dates_s.iloc[tau_mode_i].date()),
        tau_hdi_low=tau_low_i,
        tau_hdi_high=tau_high_i,
        tau_hdi_low_date=str(dates_s.iloc[tau_low_i].date()),
        tau_hdi_high_date=str(dates_s.iloc[tau_high_i].date()),
    )


def compute_impact_summary(
    idata: az.InferenceData,
    dates: Sequence[pd.Timestamp] | pd.Series,
    tau_var: str = "tau",
    hdi_prob: float = 0.94,
) -> ImpactSummary:
    """
    For mean-switch model: quantify change in mean and posterior probability of increase.
    Also attaches tau HDI endpoints (index + date) to satisfy rubric/reporting needs.
    """
    posterior = cast(Any, idata).posterior
    mu1 = posterior["mu_1"].values.reshape(-1)
    mu2 = posterior["mu_2"].values.reshape(-1)

    delta = mu2 - mu1
    prob = float(np.mean(delta > 0))

    hdi_delta = az.hdi(delta, hdi_prob=hdi_prob)
    hdi_low = float(hdi_delta[0])
    hdi_high = float(hdi_delta[1])

    # Compute tau mode + tau HDI and map to dates (single source of truth)
    tau_dates = compute_tau_date_summary(
        idata=idata,
        dates=dates,
        tau_var=tau_var,
        hdi_prob=hdi_prob,
    )

    return ImpactSummary(
        tau_mode=int(tau_dates.tau_mode),
        tau_mode_date=str(tau_dates.tau_mode_date),

        tau_hdi_low=int(tau_dates.tau_hdi_low),
        tau_hdi_high=int(tau_dates.tau_hdi_high),
        tau_hdi_low_date=str(tau_dates.tau_hdi_low_date),
        tau_hdi_high_date=str(tau_dates.tau_hdi_high_date),

        prob_mu2_gt_mu1=prob,
        delta_mu_mean=float(np.mean(delta)),
        delta_mu_hdi_low=hdi_low,
        delta_mu_hdi_high=hdi_high,
    )


def compute_sigma_impact_summary(
    idata: az.InferenceData,
    dates: Sequence[pd.Timestamp] | pd.Series,
) -> Dict[str, Any]:
    """
    For sigma-switch model: quantify change in volatility and posterior probability of increase.
    """
    posterior = cast(Any, idata).posterior
    s1 = posterior["sigma_1"].values.reshape(-1)
    s2 = posterior["sigma_2"].values.reshape(-1)
    tau = posterior["tau"].values.reshape(-1).astype(int)

    delta = s2 - s1
    prob = float(np.mean(delta > 0))

    hdi = az.hdi(delta, hdi_prob=0.94)
    tau_mode = _mode_int(tau)

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    tau_mode_date = str(
        dates.iloc[int(np.clip(tau_mode, 0, len(dates) - 1))].date())

    return {
        "tau_mode": int(tau_mode),
        "tau_mode_date": tau_mode_date,
        "prob_sigma2_gt_sigma1": prob,
        "delta_sigma_mean": float(np.mean(delta)),
        "delta_sigma_hdi_low": float(hdi[0]),
        "delta_sigma_hdi_high": float(hdi[1]),
    }
