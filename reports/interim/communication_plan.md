# Stakeholder Communication Plan (Brent Change-Point Analysis)

## 1) Purpose

This plan defines how results from the Brent time-series and change-point analysis will be communicated to stakeholders in a way that supports real decisions (risk management, planning, and strategy), not just technical reporting.

The analysis detects statistical regime shifts (change points) in Brent prices/returns and **associates** (not proves) potential drivers using a curated event registry (`data/raw/events.csv`).

---

## 2) Stakeholders, Decisions, and Information Needs

### A) Executive / Investment Committee (IC)

**Decisions supported**

- Adjust commodity exposure (hedging level, portfolio risk limits)
- Approve scenario assumptions for planning (base vs stress)

**Needs**

- Clear headline: “What changed, when, and what it means”
- Uncertainty communicated plainly (credible intervals, not technical jargon)
- Top 3 likely drivers (from event registry) with confidence level

### B) Risk Management (Market Risk / Treasury)

**Decisions supported**

- Update VaR/stress scenarios and risk limits
- Decide whether volatility regime shift warrants model recalibration

**Needs**

- Volatility regime interpretation (sigma shift vs mean shift)
- Timing and persistence of regimes
- Alerts when new breaks are detected

### C) Trading / Commercial Team

**Decisions supported**

- Position sizing and stop-loss rules during high-volatility regimes
- Timing around event risk windows

**Needs**

- Fast updates and visual overlays (events vs returns/volatility)
- “High impact window” list (± event window days)

### D) Analytics / Data Science (Internal)

**Decisions supported**

- Maintain reproducibility, model governance, and auditability

**Needs**

- Full technical appendix: assumptions, diagnostics, sensitivity checks
- Clear runbook to reproduce outputs from raw data to reports

---

## 3) Core Questions the Project Answers

1. **Where are the most likely change points (τ) in the series?**
2. **Are changes driven more by mean shifts or volatility shifts (risk regimes)?**
3. **Which curated events are temporally closest to detected breaks (within a defined window)?**
4. **How uncertain are these estimates (credible intervals, alternative plausible τ values)?**
5. **What is the practical implication for risk and planning (not just statistical significance)?**

---

## 4) Communication Outputs (Deliverables)

### Output 1 — Executive Memo (1 page)

**Audience:** Executive / IC  
**Format:** PDF or Markdown export  
**Contents:**

- Headline findings (1–3 bullets)
- Most likely change-point date(s) + uncertainty band
- “What changed” (mean/volatility; qualitative interpretation)
- Top associated events (from `events.csv`) with confidence level
- Action guidance: monitoring triggers + recommended next steps

### Output 2 — Dashboard / Figure Pack (visual, stakeholder-friendly)

**Audience:** Exec, Risk, Trading  
**Format:** `reports/figures/` + optional slide deck  
**Minimum visuals:**

- Price and log returns series
- Rolling volatility
- Change-point posterior plot (τ distribution)
- Overlay of events on series (vertical markers)
- Summary table: “events near breaks” (± window days)

### Output 3 — Technical Appendix (for Analytics + audit)

**Audience:** Data Science / reviewers  
**Format:** Markdown in `reports/`  
**Contents:**

- Model specifications and priors
- Sampling diagnostics (trace plots, ESS/R-hat)
- Model comparison (e.g., LOO)
- Sensitivity notes (window size, prior robustness)
- Limitations: correlation ≠ causation, confounding, multiple comparisons

---

## 5) Cadence (When We Communicate)

### Regular cadence

- **Weekly** update during active monitoring or volatile periods
- **Monthly** update during stable periods (summary only)

### Trigger-based updates (fast response)

Publish an update within **24–48 hours** when:

- A new extreme return occurs (e.g., top 1% absolute daily return)
- Rolling volatility exceeds a threshold (e.g., 95th percentile of historical)
- Major real-world event occurs (OPEC meeting, sanctions escalation, conflict escalation)
- Model rerun identifies a new high-probability τ region

---

## 6) Channels (How We Communicate)

- **Email/Slack summary**: short bullets + link to latest figures
- **Repo release notes (GitHub)**: what changed, which outputs updated, data/version notes
- **Dashboard / figure pack**: stored under `reports/figures/`
- **Meeting briefing**: 10-minute standing agenda item when triggers occur

---

## 7) Ownership and Responsibilities

- **Data/Model owner:** Data Science (maintains code, runs pipeline, validates outputs)
- **Domain reviewer:** Analyst/Research (reviews event registry updates + plausibility)
- **Primary stakeholder contact:** Risk lead or IC secretary (ensures distribution)

---

## 8) Operational Runbook (Definition of Done)

An “update cycle” is complete when:

1. Data pipeline runs end-to-end:
   - `01_clean_prices.py`
   - `02_make_returns.py`
   - `03_run_task1_eda.py`
   - `04_validate_events.py`
2. Event registry is validated and up-to-date:
   - `data/raw/events.csv` passes schema validation
   - Any new events include a source citation in `source`
3. Outputs refreshed:
   - updated figures in `reports/figures/`
   - updated stationarity summary outputs
   - updated event summary table (CSV/PNG if enabled)
4. Stakeholder-facing summary sent:
   - executive memo updated (if change points materially changed)
   - quick summary message shared to agreed channel

---

## 9) Risk & Limitations (What We Tell Stakeholders)

- Change points indicate **statistical regime shifts**, not definitive causality.
- Multiple events can occur near the same time; attribution is uncertain.
- Event windows and model priors influence alignment; results should be interpreted as evidence-weighted hypotheses.
- Decision-making should combine this analysis with fundamentals, inventories, and forward curves where available.

---
