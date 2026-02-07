# Assumptions & Limitations (Task 1)

## Scope

This document records assumptions and limitations for Task 1 (data preparation + EDA) of the Brent oil price change-point analysis project.

## Data assumptions

1. **Source integrity:** Raw Brent price observations are assumed to be correctly reported by the upstream source; this project does not independently audit the original publisher.
2. **Date parsing:** Dates are assumed to follow `%d-%b-%y` format (e.g., `20-May-87`). Unparseable dates are dropped.
3. **Price positivity:** Prices are assumed to be strictly positive for log transforms. Rows with `price <= 0` are removed when `drop_nonpositive_prices=True`.

## Cleaning assumptions

1. **Duplicate dates:** When multiple observations share the same date, the handling rule (`duplicate_rule`) is assumed to be appropriate for the analysis goal. The default is `keep_last`.
2. **Missing values:** Rows with missing `date` or `price` are dropped; no imputation is performed in Task 1.

## EDA / statistical testing assumptions

1. **Stationarity tests:** ADF and KPSS are applied under standard assumptions; p-values from KPSS may be bounded due to lookup-table limits (InterpolationWarning).
2. **Returns modeling:** Log-returns are treated as the main stationary series for subsequent change-point modeling. The first return is undefined due to differencing and is expected to be missing.

## Event annotations: interpretation boundaries

1. **Correlation vs causation:** Event overlays do not prove causal impact; they provide plausible alignment for hypotheses to test in later tasks.
2. **Event timing uncertainty:** Start/end dates for geopolitical/economic events may be approximate and can differ across sources.
3. **Confounding:** Multiple overlapping events and macro factors may affect prices simultaneously; attribution is not identified in Task 1.

## Known limitations

- Structural breaks in prices/returns may arise from market microstructure, policy, technology, or broader macro regimes not captured in the event list.
- Event dataset is curated and not exhaustive.
- Using daily frequency may miss intraday dynamics and announcement effects.
- No backtesting or predictive evaluation is performed in Task 1.

## Implications for Task 2+

Task 2 change-point results should be presented as probabilistic regime shifts in the chosen series (typically log-returns and/or volatility), with event alignment treated as supporting context rather than proof.
