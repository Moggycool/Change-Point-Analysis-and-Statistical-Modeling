""" A module for posterior predictive checks of Bayesian models,
    including functions to generate PPC figures.
    This is imported by the main Task 2 script, but can also be used independently."""
# src/models/ppc_task2.py
from __future__ import annotations
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def make_ppc_figure(model, idata, *, var_name_obs="obs", n_pp_samples=200, random_seed=42):
    with model:
        ppc = pm.sample_posterior_predictive(idata, random_seed=random_seed)
    idata_ppc = idata.copy()
    idata_ppc.extend(ppc)

    ax = az.plot_ppc(idata_ppc, data_pairs={
                     var_name_obs: var_name_obs}, num_pp_samples=n_pp_samples)
    fig = ax.figure
    plt.tight_layout()
    return fig
