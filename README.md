# Change Point Analysis and Statistical Modeling

Detecting structural breaks (change points) in **Brent crude oil** time series and associating those changes with **key geopolitical events, OPEC decisions, and macroeconomic shocks**. The project emphasizes a reproducible workflow: data cleaning → feature engineering (log returns) → EDA & stationarity testing → change point modeling → event alignment → insight generation.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Tasks & Deliverables (Interim)](#tasks--deliverables-interim)
- [Setup](#setup)
- [How to Run (Task 1 Pipeline)](#how-to-run-task-1-pipeline)
- [Notebook](#notebook)
- [Interpreting Results (Important)](#interpreting-results-important)
- [Testing](#testing)
- [Configuration](#configuration)
- [Next Steps (Task 2 Preview)](#next-steps-task-2-preview)

## Repository Structure

```
Change-Point-Analysis-and-Statistical-Modeling
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        └─ nodeids
├─ notebooks
│  └─ task1_requirements_notebook.ipynb
├─ README.md
├─ reports
│  ├─ figures
│  │  ├─ 01_price_and_log_price.png
│  │  ├─ 02_log_returns.png
│  │  ├─ 03_rolling_volatility_30d.png
│  │  ├─ log_returns_series.png
│  │  ├─ price_series.png
│  │  ├─ rolling_volatility.png
│  │  ├─ stationarity_tests_table.png
│  │  └─ stationarity_tests_task1.csv
│  └─ interim
│     ├─ assumptions_limitations.md
│     └─ task1_workflow_plan.md
├─ requirements.txt
├─ scripts
│  ├─ 01_clean_prices.py
│  ├─ 02_make_returns.py
│  ├─ 03_run_task1_eda.py
│  └─ 04_validate_events.py
├─ src
│  ├─ config.py
│  ├─ eda
│  │  ├─ plots.py
│  │  ├─ time_series_tests.py
│  │  └─ __init__.py
│  ├─ events
│  │  ├─ schema.py
│  │  └─ __init__.py
│  ├─ io
│  │  ├─ load_data.py
│  │  ├─ save_data.py
│  │  └─ __init__.py
│  ├─ utils
│  │  ├─ dates.py
│  │  ├─ logging.py
│  │  └─ __init__.py
│  └─ __init__.py
└─ tests
   ├─ test_cleaning.py
   └─ test_events_schema.py

```

## Tasks & Deliverables (Interim)

### Task 1 — Workflow & Event Research

- **Workflow documentation:** `reports/interim/task1_workflow_plan.md`
- **Assumptions & limitations (correlation ≠ causation):** `reports/interim/assumptions_limitations.md`
- **Event dataset:** `data/raw/events.csv` (validated by schema checks; see script below)

### Task 1 — Time Series Analysis & Model Understanding

Generated outputs (examples):

- Price trend + transformations: `reports/figures/01_price_and_log_price.png`
- Log returns: `reports/figures/02_log_returns.png`
- Rolling volatility: `reports/figures/03_rolling_volatility_30d.png`
- Stationarity test summary:
  - Figure: `reports/figures/stationarity_tests_table.png`
  - CSV: `reports/figures/stationarity_tests_task1.csv`

## Setup

### 1) Create and activate a virtual environment

**Windows (PowerShell):**

```powershell

- python -m venv .venv
.venv\Scripts\Activate.ps1

macOS/Linux:

bash
python -m venv .venv
source .venv/bin/activate

```

### 2) Install dependencies

bash

- pip install -r requirements.txt
Data
Expected raw input files:

- Brent prices: data/raw/brent_prices.csv
- Event registry: data/raw/events.csv
- Column expectations (standardized in src/config.py):
- date, price, plus engineered columns log_price, log_return after processing.

- Note: If you do not commit raw data to GitHub (recommended for many projects), place the CSVs locally under data/raw/ before running scripts.

### 3) How to Run (Task 1 Pipeline)

Run scripts from the repository root:

- Clean prices
bash
python scripts/01_clean_prices.py
- Create log price + log returns features
bash
python scripts/02_make_returns.py
- Run EDA and stationarity tests (exports figures/tables to reports/)
bash
python scripts/03_run_task1_eda.py
- Validate the event dataset schema
bash
python scripts/04_validate_events.py
- Notebook
notebooks/task1_requirements_notebook.ipynb contains the Task 1 analysis narrative and reproduces key charts and stationarity results.

## Interpreting Results

- Correlation vs causation: Change point detection identifies statistical regime shifts. Matching change points to events (within a ±window) suggests temporal association and should be treated as a hypothesis, not causal proof.

- Modeling choice: Raw prices are typically non-stationary; log returns are commonly used for change point modeling because they are closer to stationary and better reflect shocks/volatility.

## Testing

Run unit tests:

bash
pytest -q

## Configuration

Centralized configuration is in src/config.py (paths, filenames, column standards, defaults like rolling window length).

## Next Steps (Task 2 Preview)

Task 2 will implement Bayesian change point model(s) (PyMC/ArviZ) to estimate:

- change point date(s) with uncertainty (posterior over τ)
- before/after parameters (e.g., mean return shift and/or volatility regime shift)
- event alignment tables to support narrative insight generation

## Project Status

- Current Phase: Task 1 - Exploratory Data Analysis & Stationarity Testing
- Next Phase: Task 2 - Bayesian Change Point Modeling
- Last Updated: February 8 2024
