# Task 1 Workflow Plan (Reproducible Pipeline)

## Goal

Prepare Brent oil price data for change-point analysis and produce EDA outputs that justify the modeling series used in Task 2.

## Inputs

- `data/raw/brent_prices_raw.csv` (or equivalent raw file)
- `data/raw/events.csv`
- Optional: `data/raw/sources.md` for traceable citations

## Steps (scripts)

1. **Clean raw prices**
   - Script: `scripts/01_clean_prices.py`
   - Function: `src/data/cleaning.py::clean_brent_prices`
   - Output: `data/interim/brent_prices_clean.csv`

2. **Create log prices and log-returns**
   - Script: `scripts/02_make_returns.py`
   - Output: `data/processed/brent_log_returns.csv`
   - Note: first `log_return` is expected to be NaN

3. **Run EDA and stationarity tests**
   - Script: `scripts/03_run_task1_eda.py`
   - Outputs:
     - Figures: `reports/figures/*`
     - Table: `reports/figures/stationarity_tests_task1.csv`

4. **Validate event schema**
   - Script: `scripts/04_validate_events.py`
   - Input: `data/raw/events.csv`
   - Output: Console validation summary (and optionally a validated copy in `data/interim/`)

## How to reproduce (Windows / PowerShell)

```powershell
# from repo root in venv
pytest -q
& .\Menv\Scripts\python.exe .\scripts\01_clean_prices.py
& .\Menv\Scripts\python.exe .\scripts\02_make_returns.py
& .\Menv\Scripts\python.exe .\scripts\03_run_task1_eda.py
& .\Menv\Scripts\python.exe .\scripts\04_validate_events.py
