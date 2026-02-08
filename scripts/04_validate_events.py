""" Script to validate the brent_events.csv file against the defined schema. """
# scripts/04_validate_events.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Get project root (must run before importing from src.* when executing as a script)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# pylint: disable=wrong-import-position
from src.events.schema import validate_events_df  # noqa: E402
from src.config import (  # noqa: E402
    ensure_dirs,
    DATA_RAW_DIR,
    REPORTS_FIGURES_DIR,
)

# ---- concrete events artifact name ----
BRENT_EVENTS_FILENAME = "brent_events.csv"


def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _coerce_expected_direction(x: object) -> str:
    """
    Map common direction labels into the allowed set: up/down/ambiguous.
    Returns '' if x is blank/NA (so you can decide whether to keep/drop column).
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    v = str(x).strip().lower()
    if v in {"up", "increase", "rises", "higher"}:
        return "up"
    if v in {"down", "decrease", "falls", "lower"}:
        return "down"
    if v in {"ambiguous", "unclear", "mixed", "unknown"}:
        return "ambiguous"
    # leave as-is; validate_events_df will raise if invalid
    return v


def _coerce_confidence(x: object) -> str:
    """
    Map common confidence labels into the allowed set: high/medium/low.
    Returns '' if blank/NA.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    v = str(x).strip().lower()
    if v in {"high", "h"}:
        return "high"
    if v in {"medium", "med", "m"}:
        return "medium"
    if v in {"low", "l"}:
        return "low"
    return v


def _load_and_map_brent_events(events_path: Path) -> pd.DataFrame:
    """
    Load the lean Brent events CSV and map it into the canonical schema required
    by src.events.schema.validate_events_df().
    """
    df_raw = pd.read_csv(events_path)

    rename_map = {
        "event_date": "start_date",
        "event_end_date": "end_date",
        "event_title": "event_name",
        "event_description": "description",
        # keep both if present; we'll resolve 'source' below
        "source_name": "source_name",
        "source_url": "source_url",
    }
    df = df_raw.rename(columns=rename_map).copy()

    # --- Ensure required columns exist ---
    required_defaults = {
        "event_id": None,          # must exist; if missing, we will error clearly
        "event_name": "",
        "start_date": None,        # must parse
        "end_date": pd.NaT,
        "event_type": "",
        "description": "",
        "region": "",
        "source": "",              # must be non-blank after filling
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    # If event_id missing in raw, fail early with a helpful message
    if "event_id" not in df_raw.columns:
        raise ValueError(
            f"{events_path} is missing required column 'event_id'. "
            "Add unique IDs like E01, E02, ... and rerun."
        )

    # --- Source handling (schema requires non-blank source) ---
    # Prefer an explicit 'source' column if present, else use source_name, else 'unknown'
    if "source" in df_raw.columns:
        df["source"] = df_raw["source"]
    elif "source_name" in df.columns:
        df["source"] = df["source_name"]
    else:
        df["source"] = "unknown"

    # Guarantee non-blank after stripping (your validator forbids blanks)
    df["source"] = df["source"].astype(str).str.strip()
    df.loc[df["source"] == "", "source"] = "unknown"

    # --- Date parsing ---
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # --- Optional columns: only keep/clean if they exist in input (avoid triggering validation unexpectedly) ---
    if "expected_direction" in df.columns:
        df["expected_direction"] = df["expected_direction"].map(
            _coerce_expected_direction)

    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].map(_coerce_confidence)

    # Trim key string fields (helps satisfy "no blank values" checks)
    for c in ["event_name", "event_type", "description", "region", "source"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Keep a clean column order (optional, but nice for debugging/prints)
    front = [
        "event_id", "event_name", "start_date", "end_date",
        "event_type", "description", "region", "source",
    ]
    tail = [c for c in df.columns if c not in front]
    df = df.loc[:, front + tail]

    return df


def _export_events_summary(validated: pd.DataFrame) -> Path:
    """Export a tidy, stakeholder-friendly events summary (CSV) to reports/figures."""
    ensure_dirs()

    base_cols = [
        "event_id",
        "event_name",
        "start_date",
        "end_date",
        "event_type",
        "region",
        "description",
        "source",
    ]
    optional_cols = [
        "expected_direction",
        "expected_channel",
        "confidence",
        "source_name",
        "source_url",
    ]
    cols = [c for c in base_cols if c in validated.columns] + [
        c for c in optional_cols if c in validated.columns
    ]

    out = validated.loc[:, cols].copy()
    out = out.sort_values(["start_date", "event_id"]).reset_index(drop=True)

    out["start_date"] = pd.to_datetime(
        out["start_date"]).dt.strftime("%Y-%m-%d")
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.strftime("%Y-%m-%d")
    out["end_date"] = out["end_date"].fillna("")

    summary_path = REPORTS_FIGURES_DIR / "events_summary.csv"
    out.to_csv(summary_path, index=False)
    return summary_path


def _export_events_summary_png(validated: pd.DataFrame) -> Path | None:
    """Export a simple PNG table preview; if matplotlib not available, skip."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    ensure_dirs()

    preview_cols = ["event_id", "start_date",
                    "event_name", "event_type", "region"]
    preview_cols = [c for c in preview_cols if c in validated.columns]

    preview = validated.loc[:, preview_cols].copy()
    preview = preview.sort_values(["start_date", "event_id"]).head(15)
    preview["start_date"] = pd.to_datetime(
        preview["start_date"]).dt.strftime("%Y-%m-%d")

    fig_h = 0.45 * (len(preview) + 2)
    fig, ax = plt.subplots(figsize=(14, max(4, fig_h)))
    ax.axis("off")
    ax.set_title("Events Registry (preview)", fontsize=14, pad=12)

    cell_text = preview.fillna("").astype(str).values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=preview.columns.to_list(),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    out_path = REPORTS_FIGURES_DIR / "events_summary.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Main function to validate the brent_events.csv file."""
    ensure_dirs()

    events_path = DATA_RAW_DIR / BRENT_EVENTS_FILENAME
    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing events file: {events_path}\n\n"
            f"Create it at data/raw/{BRENT_EVENTS_FILENAME} and rerun:\n"
            "python scripts/04_validate_events.py"
        )

    df_mapped = _load_and_map_brent_events(events_path)
    validated = validate_events_df(df_mapped)

    summary_csv = _export_events_summary(validated)
    summary_png = _export_events_summary_png(validated)

    print(f"[OK] Events validated: {events_path}")
    print(f"Rows: {len(validated):,}")
    print(f"[OK] Wrote: {summary_csv}")
    if summary_png is not None:
        print(f"[OK] Wrote: {summary_png}")
    print("\nPreview:")
    print(validated.sort_values(["start_date", "event_id"]).head(10))


if __name__ == "__main__":
    main()
