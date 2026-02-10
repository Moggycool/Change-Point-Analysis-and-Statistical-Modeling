# Interim Report — Brent Oil Change Point Analysis (Task 1)

## 1. Objective

Analyze Brent crude oil price dynamics (1987–2020) to identify structural breaks (change points) and generate evidence-based insights aligned with key geopolitical events, OPEC decisions, and macroeconomic shocks.

## 2. Planned Workflow (Data → Insights)

1. **Data ingestion & cleaning**
   - Load raw Brent price data from `data/raw/`.
   - Parse `date` as datetime, sort chronologically, handle missing values/duplicates.
2. **Feature engineering**
   - Compute `log_price = log(price)` and `log_return = diff(log_price)`.
   - Use log returns for modeling because raw prices are non-stationary.
3. **Exploratory data analysis**
   - Plot raw prices to observe long-run trend and regime-like behavior.
   - Plot log returns to inspect volatility clustering and extreme shocks.
   - Rolling volatility (e.g., 30-day rolling std) to highlight volatility regimes.
4. **Stationarity testing**
   - Apply ADF test to raw prices and log returns.
   - Expectation: prices non-stationary; log returns stationary (model-ready).
5. **Change point modeling**
   - Fit Bayesian change point model(s) to log returns to identify structural breaks.
   - Primary output: posterior for change point date(s) and parameters before/after.
6. **Event alignment & interpretation**
   - Match detected change point dates with events in `data/raw/events.csv` using a ±[14/30]-day window.
   - Interpret matches as plausible associations (not proof of causality).
7. **Reporting**
   - Produce figures and tables in `reports/figures/` and `reports/interim/`.
   - Summarize results for stakeholders with clear uncertainty and limitations.

## 3. Event Dataset (Research Output)

A structured list of key events is maintained in:

- `data/raw/events.csv` (≥ 10–15 events; includes approximate dates and descriptions)

Events include (examples): wars/conflicts, sanctions, OPEC announcements, financial crises, demand shocks, and pandemic-related disruptions.

## 4. Assumptions & Limitations (Correlation ≠ Causation)

- **Association not causality:** temporal proximity between a change point and an event suggests correlation; it does **not** prove causal impact.
- **Single-break simplification:** a one-change-point model may miss multiple regime shifts; later iterations may expand to multiple change points.
- **Data limitations:** event dates are approximate; markets may price information before the recorded date.
- **Model limitations:** Gaussian assumptions may understate tail risk; robust alternatives (e.g., Student-t likelihood) can be explored.

## 5. Communication Plan (Stakeholders & Channels)

- **Investors / Investment committee:** 1–2 page memo + key charts (change point date distribution, before/after means, volatility regime plot).
- **Energy company stakeholders:** operational brief focusing on risk regimes, volatility changes, and scenario implications.
- **Policy / academic audience:** technical appendix with stationarity tests, model specification, posterior diagnostics, and reproducibility notes.
- **Formats:** GitHub README + interim report (Markdown/PDF), notebook for reproducibility, and exported PNG figures for quick review.

## 6. Expected Outputs (from Change Point Analysis)

- Estimated change point date(s) with uncertainty (posterior distribution of τ).
- Parameter estimates before/after break (e.g., mean returns μ₁, μ₂; volatility regimes σ).
- Diagnostic plots (trace, posterior) to validate convergence and stability.
- Event alignment table listing events within ±window days of detected change points.
