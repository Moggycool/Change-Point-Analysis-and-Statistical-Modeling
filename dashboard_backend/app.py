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

PRICES_FILE = DATA_PROCESSED_DIR / "brent_log_returns.csv"
EVENTS_FILE = DATA_RAW_DIR / "brent_events.csv"

TAU_M1_FILE = REPORTS_INTERIM_DIR / "task2_m1_tau_date_summary.csv"
TAU_M2_FILE = REPORTS_INTERIM_DIR / "task2_m2_tau_date_summary.csv"
IMPACT_M1_FILE = REPORTS_INTERIM_DIR / "task2_m1_impact_summary.csv"
IMPACT_M2_FILE = REPORTS_INTERIM_DIR / "task2_m2_sigma_impact_summary.csv"
MODEL_COMPARISON_FILE = REPORTS_INTERIM_DIR / "task2_model_comparison.csv"


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def _to_iso_date(x: Any) -> str | None | Any:
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


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert DataFrame to JSON-safe list-of-dicts.
    Critical fix: replace NaN/NaT with None so Flask jsonify emits valid JSON (no NaN tokens).
    """
    safe = df.replace({np.nan: None})
    # Safety: convert datetime64 columns to ISO strings
    for col in safe.columns:
        try:
            if np.issubdtype(safe[col].dtype, np.datetime64):
                safe[col] = safe[col].map(_to_iso_date)
        except Exception:
            pass
    return safe.to_dict(orient="records")


def _read_prices() -> pd.DataFrame:
    _must_exist(PRICES_FILE)
    df = pd.read_csv(PRICES_FILE)

    required = {"date", "price", "log_price", "log_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{PRICES_FILE} missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["price", "log_price", "log_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df["price"].to_numpy())].copy()
    df = df[~pd.isna(df["date"])].copy()
    return df


def _read_events() -> pd.DataFrame:
    _must_exist(EVENTS_FILE)
    ev = pd.read_csv(EVENTS_FILE)

    if "event_date" not in ev.columns:
        raise ValueError(f"{EVENTS_FILE} must contain 'event_date' column")

    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    if "event_end_date" in ev.columns:
        ev["event_end_date"] = pd.to_datetime(
            ev["event_end_date"], errors="coerce")

    if "event_id" not in ev.columns:
        ev["event_id"] = np.arange(len(ev)) + 1

    if "event_title" not in ev.columns:
        ev["event_title"] = "Event"

    ev = ev[~pd.isna(ev["event_date"])].copy()
    ev = ev.sort_values("event_date").reset_index(drop=True)
    return ev


def _read_one_row_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    d_raw = df.replace({np.nan: None}).to_dict(orient="records")[0]
    d: dict[str, Any] = {str(k): v for k, v in d_raw.items()}
    for k, v in list(d.items()):
        if "date" in k.lower():
            d[k] = _to_iso_date(v)
    return d


def _rolling_vol(returns: pd.Series, window: int = 30) -> pd.Series:
    return returns.rolling(window=window, min_periods=max(5, window // 3)).std()


@app.get("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "routes": [
                "/api/health",
                "/api/metadata",
                "/api/prices",
                "/api/events",
                "/api/changepoints",
                "/api/event-correlation",
            ],
        }
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/metadata")
def metadata():
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


@app.get("/api/prices")
def get_prices():
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
    out = out.replace({np.nan: None})
    return jsonify(out.to_dict(orient="records"))


@app.get("/api/changepoints")
def changepoints():
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
        mc = pd.read_csv(MODEL_COMPARISON_FILE).replace({np.nan: None})
        payload["model_comparison"] = mc.to_dict(orient="records")

    return jsonify(payload)


@app.get("/api/events")
def events():
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

    return jsonify(_json_safe_records(out))


@app.get("/api/event-correlation")
def event_correlation():
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
            (prices_df["date"] >= d0 - pd.Timedelta(days=window))
            & (prices_df["date"] <= d0 + pd.Timedelta(days=window))
        ]
        if w.empty:
            continue

        r = w["log_return"].astype(float)

        # Keep event_id as-is (your CSV uses string IDs like 'E01_...').
        event_id = row.get("event_id", None)
        if isinstance(event_id, float) and np.isnan(event_id):
            event_id = None

        rec = {
            "event_id": event_id,
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
        results.sort(key=lambda x: x.get(metric, 0.0) or 0.0, reverse=True)
    elif metric == "mean_return":
        results.sort(key=lambda x: abs(
            x.get(metric, 0.0) or 0.0), reverse=True)

    safe = pd.DataFrame(results).replace({np.nan: None})
    return jsonify(safe.to_dict(orient="records"))


@app.errorhandler(Exception)
def handle_error(e: Exception):
    return jsonify({"error": str(e), "type": e.__class__.__name__}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
