# Change Point Analysis and Statistical Modeling

Detecting structural breaks (change points) in **Brent crude oil** time series and associating those changes with **key geopolitical events, OPEC decisions, and macroeconomic shocks**. The project emphasizes a reproducible workflow:

**data cleaning → feature engineering (log returns) → EDA & stationarity testing → change point modeling → event alignment → insight generation**

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Tasks & Deliverables (Interim)](#tasks--deliverables-interim)
- [Setup](#setup)
- [Data](#data)
- [How to Run (Task 1 Pipeline)](#how-to-run-task-1-pipeline)
- [Event Registry (Structured Dataset)](#event-registry-structured-dataset)
- [Stakeholder Communication Plan](#stakeholder-communication-plan)
- [Notebook](#notebook)
- [Interpreting Results (Important)](#interpreting-results-important)
- [Testing](#testing)
- [Configuration](#configuration)
- [Next Steps (Task 2 Preview)](#next-steps-task-2-preview)
- [Project Status](#project-status)

---

## Repository Structure

```
Change-Point-Analysis-and-Statistical-Modeling
├─ notebooks/
│  └─ task1_requirements_notebook.ipynb
├─ reports/
│  ├─ figures/
│  │  ├─ 01_price_and_log_price.png
│  │  ├─ 02_log_returns.png
│  │  ├─ 03_rolling_volatility_30d.png
│  │  ├─ stationarity_tests_table.png
│  │  └─ stationarity_tests_task1.csv
│  └─ interim/
│     ├─ assumptions_limitations.md
│     ├─ task1_workflow_plan.md
│     └─ communication_plan.md
├─ requirements.txt
├─ scripts/
│  ├─ 01_clean_prices.py
│  ├─ 02_make_returns.py
│  ├─ 03_run_task1_eda.py
│  └─ 04_validate_events.py
├─ src/
│  ├─ config.py
│  ├─ eda/
│  ├─ events/
│  ├─ io/
│  └─ utils/
└─ tests/
```

---

## Tasks & Deliverables (Interim)

### Task 1 — Workflow & Event Research

- **Workflow documentation:** `reports/interim/task1_workflow_plan.md`
- **Assumptions & limitations (correlation ≠ causation):** `reports/interim/assumptions_limitations.md`
- **Event registry (structured dataset):** `data/raw/events.csv`
- **Event registry validation:** `python scripts/04_validate_events.py`

### Task 1 — Time Series Analysis & Model Understanding

Generated outputs (examples):

- Price trend + transformations: `reports/figures/01_price_and_log_price.png`
- Log returns: `reports/figures/02_log_returns.png`
- Rolling volatility: `reports/figures/03_rolling_volatility_30d.png`
- Stationarity test summary:
  - Figure: `reports/figures/stationarity_tests_table.png`
  - CSV: `reports/figures/stationarity_tests_task1.csv`

---

## Setup

### 1) Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data

Expected raw input files:

- Brent prices: `data/raw/brent_prices.csv`
- Event registry: `data/raw/events.csv`

Column expectations (standardized in `src/config.py`):

- `date`, `price`, plus engineered columns `log_price`, `log_return` after processing.

> **Note:** If raw price data cannot be committed to GitHub (license/size), keep it locally under `data/raw/`.  
> The events registry *is* committed because it is part of the analytical assumptions and is required for reproducibility.

---

### Data standards enforced

All processed Brent price data adheres to these standards:

- **Datetime parsing (strict):** `pd.to_datetime(..., errors="raise")` and timezone removed via `.dt.tz_localize(None)`
- **Sorting:** rows are sorted ascending by `date`
- **Duplicates:** duplicate dates are resolved deterministically via `duplicate_rule` (default: keep last)
- **Numeric conversion:** prices are converted with `pd.to_numeric(..., errors="coerce")` and invalid numeric rows are dropped
- **Output locations (canonical):**
  - `data/processed/brent_prices_clean.csv`
  - `data/processed/brent_log_price.csv`
  - `data/processed/brent_log_returns.csv`

## How to Run (Task 1 Pipeline)

Run scripts from the repository root:

### Clean prices

```bash
python scripts/01_clean_prices.py
```

### Create log price + log returns features

```bash
python scripts/02_make_returns.py
```

### Run EDA and stationarity tests (exports figures/tables to `reports/`)

```bash
python scripts/03_run_task1_eda.py
```

### Validate the event registry (schema + basic consistency)

```bash
python scripts/04_validate_events.py
```

If enabled in `scripts/04_validate_events.py`, this also exports:

- `reports/figures/events_summary.csv`
- `reports/figures/events_summary.png`

---

### Events registry (Task 1 narrative support)

We use a curated historical events registry to interpret structural breaks/change points in Brent crude prices.

**Input (raw):**

- `data/raw/brent_events.csv`

This file is validated against the schema in `src/events/schema.py`.

**Validation script:**

```bash
python scripts/04_validate_events.py
```

## Event Registry & Change-Point Interpretation Dataset

This project includes a **curated historical events registry** used to
interpret detected structural breaks (change points) in Brent crude oil
prices.\
The dataset links **statistical regime shifts** to **real-world
geopolitical, economic, and market drivers**, enabling meaningful
narrative explanations.

### Dataset Artifacts

#### Raw Data

- `data/raw/events.csv` --- primary curated registry of
    oil-market-relevant events (10--15 major items)
- `data/raw/brent_events.csv` --- optional/raw source table (if
    maintained separately)

#### Processed / Validated Outputs

- `data/processed/brent_events_validated.csv` --- schema-validated
    dataset generated during preprocessing

#### Reporting Exports

- `reports/figures/events_summary.csv` --- stakeholder-friendly
    summary table
- `reports/figures/events_summary.png` --- visual preview of the
    summary

### Dataset Purpose

The events registry serves as a **structured reference layer** to:

- Align detected change points with historical events
- Provide causal hypotheses for price shifts
- Support stakeholder interpretation and reporting
- Improve reproducibility of narrative explanations

### Schema (Required Columns)

event_id, event_name, start_date, end_date, event_type, description,
region, source

#### Field Notes

  -----------------------------------------------------------------------
  Column                      Description
  --------------------------- -------------------------------------------
  event_id                    Unique identifier

  event_name                  Short descriptive title

  start_date                  Event start date

  end_date                    Event end date (or same as start for
                              single-day events)

  event_type                  Category (geopolitical, OPEC decision,
                              macro shock, etc.)

  description                 Brief contextual explanation

  region                      Geographic scope

source                      Reference or citation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

### Date Granularity

**Format:** month-start (`YYYY-MM-01`)

Rationale: - Stabilizes joins with daily price data - Reduces noise from
day-level mismatches - Improves interpretability in reports

### How It Integrates With Change-Point Analysis

1. Change-point detection identifies candidate break dates.
2. Break dates are matched to nearby events (nearest or within a time
    window).
3. Matched events provide historical explanations for regime shifts.
4. Results feed into reporting and stakeholder narratives.

### Validation

Schema enforcement and cleaning logic:

src/events/schema.py

Run:

``` bash
python scripts/04_validate_events.py
```

This step: - checks required columns - validates date formats - removes
inconsistencies - outputs the processed validated file

### Summary

The events registry transforms **raw statistical breaks → explainable
business insights**, making the model outputs actionable for analysts,
policymakers, and energy stakeholders.

---

## Stakeholder Communication Plan

To make results directly actionable for decision-makers, the project includes an explicit communication plan:

- `reports/interim/communication_plan.md`

This defines audiences, decisions supported, outputs, cadence, and update triggers.

---

## Notebook

- `notebooks/task1_requirements_notebook.ipynb`

Contains the Task 1 analysis narrative and reproduces key charts and stationarity results.

---

## Key outputs (Task 1)

Figures (generated by `scripts/03_run_task1_eda.py`):

- `reports/figures/price_series.png` — Brent Price (USD/bbl), level trend
- `reports/figures/log_price_series.png` — Brent Log Price (log USD/bbl)
- `reports/figures/01_price_and_log_price.png` — Combined figure containing both series

## Interpreting Results

- **Correlation vs causation:** Change point detection identifies statistical regime shifts. Matching change points to events (within a ±window) suggests temporal association and should be treated as a hypothesis, not causal proof.
- **Modeling choice:** Raw prices are typically non-stationary; log returns are commonly used for change point modeling because they are closer to stationary and better reflect shocks/volatility.

---

## Testing

Run unit tests:

```bash
pytest -q
```

---

## Configuration

Centralized configuration is in:

- `src/config.py`

It defines paths, filenames, column standards, and defaults like rolling volatility window and (later) event matching window.

---

## Next Steps (Task 2 Preview)

Task 2 will implement Bayesian change point model(s) (PyMC/ArviZ) to estimate:

- change point date(s) with uncertainty (posterior over τ)
- before/after parameters (e.g., mean return shift and/or volatility regime shift)
- event alignment tables to support narrative insight generation

---

## Project Status

- **Current Phase:** Task 1 — Exploratory Data Analysis & Stationarity Testing + Event Registry  
- **Next Phase:** Task 2 — Bayesian Change Point Modeling  
- **Last Updated:** 2026-02-08

## Stakeholder Communication Plan (Brent Change-Point Analysis)

found in:

- ...\reports\interim\communication_plan.md
