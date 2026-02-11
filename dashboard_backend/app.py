from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -----------------------------
# Paths (match your repo)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_INTERIM_DIR = ROOT / "reports" / "interim"

# Your processed file already has price + log_return in one place
# date,price,log_price,log_return
PRICES_FILE = DATA_PROCESSED_DIR / "brent_log_returns.csv"
# event_date,... metadata
EVENTS_FILE = DATA_RAW_DIR / "brent_events.csv"

TAU_M1_FILE = REPORTS_INTERIM_DIR / "task2_m1_tau_date_summary.csv"
TAU_M2_FILE = REPORTS_INTERIM_DIR / "task2_m2_tau_date_summary.csv"
IMPACT_M1_FILE = REPORTS_INTERIM_DIR / "task2_m1_impact_summary.csv"
IMPACT_M2_FILE = REPORTS_INTERIM_DIR / "task2_m2_sigma_impact_summary.csv"
MODEL_COMPARISON_FILE = REPORTS_INTERIM_DIR / "task2_model_comparison.csv"


def _must_exist(path: Path) -> None:
    """Utility to check if a file exists, and raise an error if not."""
    if not path.exists():
        raise FileNotFoundError(str(path))


def _to_iso_date(x: Any) -> str | None | Any:
    """Converts input to ISO date string (YYYY-MM-DD) if possible, otherwise returns None or original value."""
    if x is None:
        return None

    na = pd.isna(x)
    if isinstance(na, (bool, np.bool_)) and na:
        return None

    ts = pd.to_datetime(x, errors="coerce")
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")

    na_ts = pd.isna(ts)
    if isinstance(na_ts, (bool, np.bool_)) and na_ts:
        return None
    return x


def _read_prices() -> pd.DataFrame:
    """Reads price data from CSV and ensures required columns are present and properly typed."""
    _must_exist(PRICES_FILE)
    df = pd.read_csv(PRICES_FILE)

    required = {"date", "price", "log_price", "log_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{PRICES_FILE} missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["price", "log_price", "log_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df["price"].to_numpy())].copy()
    return df


def _read_events() -> pd.DataFrame:
    """Reads event data from CSV and normalizes columns for frontend use."""
    _must_exist(EVENTS_FILE)
    ev = pd.read_csv(EVENTS_FILE)

    # Your schema
    if "event_date" not in ev.columns:
        raise ValueError(f"{EVENTS_FILE} must contain 'event_date' column")

    ev["event_date"] = pd.to_datetime(ev["event_date"])
    if "event_end_date" in ev.columns:
        ev["event_end_date"] = pd.to_datetime(
            ev["event_end_date"], errors="coerce")

    # normalize for frontend convenience
    if "event_id" not in ev.columns:
        ev["event_id"] = np.arange(len(ev)) + 1

    if "event_title" not in ev.columns:
        ev["event_title"] = "Event"

    ev = ev.sort_values("event_date").reset_index(drop=True)
    return ev


def _read_one_row_csv(path: Path) -> dict[str, Any]:
    """Reads a CSV file expected to have exactly one row, and returns it as a dict."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    d_raw = df.to_dict(orient="records")[0]
    d: dict[str, Any] = {str(k): v for k, v in d_raw.items()}
    # normalize date-ish fields
    for k, v in list(d.items()):
        if "date" in k.lower():
            d[k] = _to_iso_date(v)
    return d


def _rolling_vol(returns: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling volatility (std) with a minimum number of observations."""
    return returns.rolling(window=window, min_periods=max(5, window // 3)).std()

# Root endpoint to list available routes and basic status


@app.get("/")
def index():
    return jsonify({
        "status": "ok",
        "routes": [
            "/api/health",
            "/api/metadata",
            "/api/prices",
            "/api/events",
            "/api/changepoints",
            "/api/event-correlation"
        ]
    })
# -----------------------------
# Endpoints
# -----------------------------


@app.get("/api/health")
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.get("/api/metadata")
def metadata():
    """Returns metadata about available data files and their existence."""
    paths = {
        "prices_file": str(PRICES_FILE),
        "events_file": str(EVENTS_FILE),
        "tau_m1_file": str(TAU_M1_FILE),
        "tau_m2_file": str(TAU_M2_FILE),
        "impact_m1_file": str(IMPACT_M1_FILE),
        "impact_m2_file": str(IMPACT_M2_FILE),
        "model_comparison_file": str(MODEL_COMPARISON_FILE),
    }
    exists = {k: Path(v).exists() for k, v in paths.items()}
    return jsonify({"paths": paths, "exists": exists})


# (i) Historical price data
@app.get("/api/prices")
def get_prices():
    """Returns historical price data with optional volatility.
    GET /api/prices?start=YYYY-MM-DD&end=YYYY-MM-DD&include=volatility
    Returns: [{date, price, log_price, log_return, rolling_vol_30d?}]
    """
    start = request.args.get("start")
    end = request.args.get("end")
    include = (request.args.get("include") or "").lower()

    df = _read_prices()

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    out_cols = [c for c in ["date", "price",
                            "log_price", "log_return"] if c in df.columns]
    out = df[out_cols].copy()

    if "volatility" in include:
        out["rolling_vol_30d"] = _rolling_vol(out["log_return"], window=30)

    out["date"] = out["date"].map(_to_iso_date)
    return jsonify(out.to_dict(orient="records"))


# (ii) Change point results
@app.get("/api/changepoints")
def changepoints():
    """
    GET /api/changepoints
    Returns change point summaries + model comparison if present.
    """
    payload: dict[str, Any] = {
        "mean_switch": {
            "tau_summary": _read_one_row_csv(TAU_M1_FILE),
            "impact_summary": _read_one_row_csv(IMPACT_M1_FILE),
        },
        "sigma_switch": {
            "tau_summary": _read_one_row_csv(TAU_M2_FILE),
            "impact_summary": _read_one_row_csv(IMPACT_M2_FILE),
        },
    }

    if MODEL_COMPARISON_FILE.exists():
        payload["model_comparison"] = pd.read_csv(
            MODEL_COMPARISON_FILE).to_dict(orient="records")

    return jsonify(payload)


# (iii) Event correlation data: raw events
@app.get("/api/events")
def events():
    """
    GET /api/events?start=YYYY-MM-DD&end=YYYY-MM-DD
    Returns events list with your full metadata.
    """
    start = request.args.get("start")
    end = request.args.get("end")

    ev = _read_events()

    if start:
        ev = ev[ev["event_date"] >= pd.to_datetime(start)]
    if end:
        ev = ev[ev["event_date"] <= pd.to_datetime(end)]

    out = ev.copy()
    out["event_date"] = out["event_date"].map(_to_iso_date)
    if "event_end_date" in out.columns:
        out["event_end_date"] = out["event_end_date"].map(_to_iso_date)
    return jsonify(out.to_dict(orient="records"))


@app.get("/api/event-correlation")
def event_correlation():
    """
    GET /api/event-correlation?window=7&metric=mean_abs_return&start=...&end=...
    Computes event-window stats from log_return around event_date.
    """
    window = int(request.args.get("window", 7))
    metric = request.args.get("metric", "mean_abs_return")
    start = request.args.get("start")
    end = request.args.get("end")

    prices_df = _read_prices(
    )[["date", "price", "log_price", "log_return"]].copy()
    ev = _read_events()

    if start:
        ev = ev[ev["event_date"] >= pd.to_datetime(start)]
    if end:
        ev = ev[ev["event_date"] <= pd.to_datetime(end)]

    results: list[dict[str, Any]] = []
    for _, row in ev.iterrows():
        d0 = pd.Timestamp(row["event_date"])

        w = prices_df[
            (prices_df["date"] >= d0 - pd.Timedelta(days=window)) &
            (prices_df["date"] <= d0 + pd.Timedelta(days=window))
        ]
        if w.empty:
            continue

        r = w["log_return"].astype(float)
        rec = {
            "event_id": int(row["event_id"]),
            "event_date": _to_iso_date(d0),
            "event_title": str(row.get("event_title", "Event")),
            "event_type": row.get("event_type", None),
            "region": row.get("region", None),
            "expected_direction": row.get("expected_direction", None),
            "confidence": row.get("confidence", None),
            "window_days": int(window),
            "mean_return": float(r.mean()),
            "mean_abs_return": float(r.abs().mean()),
            "max_abs_return": float(r.abs().max()),
            "n_obs": int(len(w)),
        }
        results.append(rec)

    if metric in {"mean_abs_return", "max_abs_return"}:
        results.sort(key=lambda x: x.get(metric, 0.0), reverse=True)
    elif metric == "mean_return":
        results.sort(key=lambda x: abs(x.get(metric, 0.0)), reverse=True)

    return jsonify(results)


@app.errorhandler(Exception)
def handle_error(e: Exception):
    """Generic error handler to return JSON instead of HTML."""
    return jsonify({"error": str(e), "type": e.__class__.__name__}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
