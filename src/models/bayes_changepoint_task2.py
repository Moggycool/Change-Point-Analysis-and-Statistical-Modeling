# src/models/bayes_changepoint_task2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple, cast

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az


# -----------------------------
# Convergence gating utilities
# -----------------------------
@dataclass(frozen=True)
class ConvergenceReport:
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

    # Handle possible naming differences across arviz versions
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
) -> az.InferenceData:
    """
    Sample model with PyMC defaults; discrete tau will be handled by a discrete sampler.
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


# -----------------------------
# Postprocessing / Reporting
# -----------------------------
@dataclass(frozen=True)
class ImpactSummary:
    tau_mode: int
    tau_mode_date: str
    prob_mu2_gt_mu1: float
    delta_mu_mean: float
    delta_mu_hdi_low: float
    delta_mu_hdi_high: float


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


def compute_impact_summary(
    idata: az.InferenceData,
    dates: Sequence[pd.Timestamp] | pd.Series,
) -> ImpactSummary:
    """
    For mean-switch model: quantify change in mean and posterior probability of increase.
    """
    posterior = cast(Any, idata).posterior
    mu1 = posterior["mu_1"].values.reshape(-1)
    mu2 = posterior["mu_2"].values.reshape(-1)
    tau = posterior["tau"].values.reshape(-1).astype(int)

    delta = mu2 - mu1
    prob = float(np.mean(delta > 0))

    hdi = az.hdi(delta, hdi_prob=0.94)
    hdi_low = float(hdi[0])
    hdi_high = float(hdi[1])

    tau_mode = _mode_int(tau)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    tau_mode_date = str(
        dates.iloc[int(np.clip(tau_mode, 0, len(dates) - 1))].date())

    return ImpactSummary(
        tau_mode=tau_mode,
        tau_mode_date=tau_mode_date,
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
