# Data & Source Register

## Price data

- **Dataset:** Brent crude oil spot price (daily)
- **Provider:** U.S. Energy Information Administration (EIA) / FRED
- **Series name / ID:** DCOILBRENTEU
- **Coverage:** 1987–present (daily)
- **Notes:** Used for aligning event dates with price movements and validating market shocks.

---

## Event sources (institutional)

Below are the primary references used to populate `data/raw/events.csv`.

### 1. U.S. Energy Information Administration (EIA)

- Oil Market Chronology / Energy Disruptions Timeline
- Country Analysis Briefs (Iran, Iraq, Libya, Russia)
- Short-Term Energy Outlook
- Use: Dating of supply disruptions, production losses, historical price responses

### 2. International Energy Agency (IEA)

- Oil Market Report (monthly)
- Global Energy Review
- Oil 2020 / Oil 2021 special reports
- Use: Supply–demand balances and COVID-19 demand collapse analysis

### 3. OPEC / OPEC+

- Monthly Oil Market Report (MOMR)
- Official press releases and ministerial communiqués
- Use: Official production targets, quota agreements, coordinated cuts/increases

### 4. International Monetary Fund (IMF)

- World Economic Outlook (WEO)
- Global Financial Stability Report (GFSR)
- Use: Macro crisis dating and recession context

### 5. World Bank

- Commodity Markets Outlook
- Pink Sheet commodity database
- Use: Long-run commodity price behavior and cross-validation

---

## Mapping of events.csv → sources

| event_id | event_name | Primary source(s) |
|---------|-----------|------------------|
| 1 | 1973 Arab Oil Embargo | EIA, BP Review |
| 2 | 1979 Iranian Revolution Oil Shock | EIA, BP Review |
| 3 | Gulf War Oil Shock | EIA |
| 4 | Asian Financial Crisis | IMF, World Bank |
| 5 | OPEC 1999 Cuts | OPEC |
| 6 | Global Financial Crisis | IMF, EIA |
| 7 | Libya Civil War Supply Loss | IEA, EIA |
| 8 | US Shale Oversupply | EIA |
| 9 | OPEC+ Formation | OPEC |
| 10 | COVID Demand Collapse | IEA, EIA |
| 11 | Russia–Saudi Price War | OPEC, IEA |
| 12 | Pandemic Production Cuts | OPEC |
| 13 | Russia–Ukraine War Energy Shock | EIA, IEA |
| 14 | OPEC+ Voluntary Cuts | OPEC |
| 15 | Red Sea Shipping Disruptions | IEA, EIA |

---

## Citation practice

- Each event row in `events.csv` contains a short source string.
- Full institutional references are documented here for transparency and reproducibility.
