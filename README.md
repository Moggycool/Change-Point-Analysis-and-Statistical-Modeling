# Change-Point-Analysis-and-Statistical-Modeling
Detecting changes and associating causes on time series data

```
Change-Point-Analysis-and-Statistical-Modeling
├─ .env.example
├─ notebooks
│  ├─ 01_task1_data_load_and_clean.ipynb
│  ├─ 02_task1_eda_time_series_properties.ipynb
│  └─ 03_task1_event_research_template.ipynb
├─ pyproject.toml
├─ README.md
├─ reports
│  ├─ figures
│  │  ├─ log_returns_series.png
│  │  ├─ price_series.png
│  │  ├─ rolling_volatility.png
│  │  └─ stationarity_tests_table.png
│  └─ interim
│     ├─ assumptions_limitations.md
│     └─ task1_workflow_plan.md
├─ requirements.txt
├─ scripts
│  ├─ 00_make_folders.py
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