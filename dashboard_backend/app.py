""" Simple Flask backend for dashboard. Adjust paths and data processing as needed. """
from __future__ import annotations
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)  # allow React dev server

# ---- configure paths (adjust to your repo) ----
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_INTERIM = ROOT / "reports" / "interim"

PRICES_FILE = DATA_PROCESSED / "brent_prices_clean.csv"   # <-- set your real file
EVENTS_FILE = DATA_RAW / "brent_events.csv"

TAU_M1_FILE = REPORTS_INTERIM / "task2_m1_tau_date_summary.csv"
TAU_M2_FILE = REPORTS_INTERIM / "task2_m2_tau_date_summary.csv"
IMPACT_M1_FILE = REPORTS_INTERIM / "task2_m1_impact_summary.csv"
IMPACT_M2_FILE = REPORTS_INTERIM / "task2_m2_impact_summary.csv"


def _read_prices() -> pd.DataFrame:
    """ Read and preprocess price data. Adjust as needed for your actual data format. """
    df = pd.read_csv(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def _read_events() -> pd.DataFrame:
    """ Read and preprocess events data. Adjust as needed for your actual data format. """
    ev = pd.read_csv(EVENTS_FILE)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.sort_values("date")
    return ev


@app.get("/api/health")
def health():
    """ Simple health check endpoint. """
    return jsonify({"status": "ok"})


@app.get("/api/prices")
def get_prices():
    """ Return price data, optionally filtered by date range. """
    start = request.args.get("start")
    end = request.args.get("end")
    df = _read_prices()

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    # choose fields to expose
    cols = [c for c in ["date", "price", "log_return",
                        "rolling_vol"] if c in df.columns]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return jsonify(out.to_dict(orient="records"))


@app.get("/api/changepoints")
def changepoints():
    """ Return changepoint summaries from Task 2. Adjust as needed for your actual data format. """
    # read summaries produced by Task 2
    m1_tau = pd.read_csv(TAU_M1_FILE).to_dict(orient="records")[
        0] if TAU_M1_FILE.exists() else {}
    m2_tau = pd.read_csv(TAU_M2_FILE).to_dict(orient="records")[
        0] if TAU_M2_FILE.exists() else {}

    m1_imp = pd.read_csv(IMPACT_M1_FILE).to_dict(orient="records")[
        0] if IMPACT_M1_FILE.exists() else {}
    m2_imp = pd.read_csv(IMPACT_M2_FILE).to_dict(orient="records")[
        0] if IMPACT_M2_FILE.exists() else {}

    return jsonify({
        "mean_switch": {"tau": m1_tau, "impact": m1_imp},
        "sigma_switch": {"tau": m2_tau, "impact": m2_imp},
    })


@app.get("/api/events")
def events():
    """ Return event data. """
    if not EVENTS_FILE.exists():
        return jsonify({"error": f"Missing events file: {EVENTS_FILE}"}), 404

    ev = _read_events()
    cols = [c for c in ev.columns]  # keep all event metadata
    out = ev[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return jsonify(out.to_dict(orient="records"))


@app.get("/api/event-correlation")
def event_correlation():
    """ Compute simple event correlation metrics within a window around each event date. """
    window = int(request.args.get("window", 7))
    metric = request.args.get("metric", "abs_return")

    prices_df = _read_prices()
    if "log_return" not in prices_df.columns:
        return jsonify({"error": "prices file must include 'log_return'"}), 400

    ev = _read_events()

    results = []
    for _, row in ev.iterrows():
        d0 = pd.Timestamp(row["date"])
        w = prices_df[(prices_df["date"] >= d0 - pd.Timedelta(days=window)) &
                      (prices_df["date"] <= d0 + pd.Timedelta(days=window))]

        if w.empty:
            continue

        r = w["log_return"].astype(float)
        rec = {
            "event_id": row.get("event_id", None),
            "date": d0.strftime("%Y-%m-%d"),
            "title": row.get("title", row.get("event", "event")),
            "window_days": window,
            "mean_return": float(r.mean()),
            "mean_abs_return": float(r.abs().mean()),
            "max_abs_return": float(r.abs().max()),
            "n_obs": int(len(w)),
        }
        results.append(rec)

    # optional sorting by requested metric
    if metric == "abs_return":
        results.sort(key=lambda x: x["mean_abs_return"], reverse=True)
    elif metric == "return":
        results.sort(key=lambda x: abs(x["mean_return"]), reverse=True)

    return jsonify(results)


if __name__ == "__main__":
    # flask run is recommended; this is for quick local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
