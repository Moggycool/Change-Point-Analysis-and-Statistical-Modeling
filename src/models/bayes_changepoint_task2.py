# src/models/bayes_changepoint_task2.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ImpactSummary:
    tau_mode_index: int
    tau_mode_date: str
    mu1_mean: float
    mu2_mean: float
    delta_mean: float
    prob_delta_gt_0: float


def build_switchpoint_mean_model(y: np.ndarray):
    """
    Task 2 mandatory model:
      tau ~ DiscreteUniform(0, n-1)
      mu_1, mu_2 ~ Normal(...)
      mu_t = switch(t < tau, mu_1, mu_2)
      y_t ~ Normal(mu_t, sigma)
    """
    import pymc as pm

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y) < 30:
        raise ValueError("Need at least 30 observations")
    if not np.isfinite(y).all():
        raise ValueError("y must be finite; drop NaNs before modeling")

    n = len(y)

    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

        mu_1 = pm.Normal("mu_1", mu=0.0, sigma=1.0)
        mu_2 = pm.Normal("mu_2", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        t = pm.math.arange(n)
        mu = pm.math.switch(pm.math.lt(t, tau), mu_1, mu_2)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    return model


def sample_model(
    model,
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
):
    import pymc as pm

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
    return idata


def compute_impact_summary(idata, dates_clean) -> ImpactSummary:
    import pandas as pd

    tau_samples = idata.posterior["tau"].values.reshape(-1).astype(int)
    mu1 = idata.posterior["mu_1"].values.reshape(-1).astype(float)
    mu2 = idata.posterior["mu_2"].values.reshape(-1).astype(float)

    delta = mu2 - mu1
    tau_mode = int(pd.Series(tau_samples).mode().iloc[0])
    cp_date = str(pd.to_datetime(dates_clean.iloc[tau_mode]).date())

    return ImpactSummary(
        tau_mode_index=tau_mode,
        tau_mode_date=cp_date,
        mu1_mean=float(mu1.mean()),
        mu2_mean=float(mu2.mean()),
        delta_mean=float(delta.mean()),
        prob_delta_gt_0=float((delta > 0).mean()),
    )
