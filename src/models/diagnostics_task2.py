"""" A module for diagnostics of Bayesian models, 
     including convergence checks and summary statistics.
     This is imported by the main Task 2 script, but can also be used independently."""
# src/models/diagnostics_task2.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import arviz as az


@dataclass(frozen=True)
class ConvergenceReport:
    """Summary of convergence diagnostics for a PyMC model."""
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    n_divergences: int
    ok: bool


def check_convergence(idata, var_names, *, rhat_max_ok=1.01, ess_min_ok=400) -> ConvergenceReport:
    """Check convergence diagnostics for a PyMC InferenceData object."""
    summ = az.summary(idata, var_names=var_names, round_to=6)
    rhat_max = float(np.nanmax(summ["r_hat"].to_numpy()))
    ess_bulk_min = float(np.nanmin(summ["ess_bulk"].to_numpy()))
    ess_tail_min = float(np.nanmin(summ["ess_tail"].to_numpy()))
    # divergences
    try:
        n_div = int(idata.sample_stats["diverging"].values.sum())
    except Exception:
        n_div = 0
    ok = (rhat_max <= rhat_max_ok) and (
        ess_bulk_min >= ess_min_ok) and (n_div == 0)
    return ConvergenceReport(rhat_max, ess_bulk_min, ess_tail_min, n_div, ok)
