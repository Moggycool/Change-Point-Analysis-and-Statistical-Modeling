""" Script to validate the events.csv file against the defined schema. """
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
    EVENTS_FILENAME,
    REPORTS_FIGURES_DIR,
)


def _export_events_summary(validated: pd.DataFrame) -> Path:
    """
    Export a tidy, stakeholder-friendly events summary (CSV) to reports/figures.
    """
    ensure_dirs()

    # Keep required columns first, then any optional columns if present
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

    # Make dates readable in CSV (YYYY-MM-DD); keep blanks for NaT end_date
    out["start_date"] = pd.to_datetime(
        out["start_date"]).dt.strftime("%Y-%m-%d")
    if "end_date" in out.columns:
        out["end_date"] = pd.to_datetime(
            out["end_date"]).dt.strftime("%Y-%m-%d")
        out["end_date"] = out["end_date"].fillna("")

    summary_path = REPORTS_FIGURES_DIR / "events_summary.csv"
    out.to_csv(summary_path, index=False)
    return summary_path


def _export_events_summary_png(validated: pd.DataFrame) -> Path | None:
    """
    Export a simple PNG table preview for quick review by graders/stakeholders.
    If matplotlib isn't available, skip silently (CSV will still be produced).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    ensure_dirs()

    preview_cols = [
        "event_id",
        "start_date",
        "event_name",
        "event_type",
        "region",
    ]
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
    """ Main function to validate the events.csv file. """
    ensure_dirs()

    events_path = DATA_RAW_DIR / EVENTS_FILENAME
    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing events file: {events_path}\n\n"
            "Create it at data/raw/events.csv (recommended) with columns:\n"
            "event_id,event_name,start_date,end_date,event_type,description,region,source\n"
            "Then rerun: python scripts/04_validate_events.py"
        )

    df = pd.read_csv(events_path)
    validated = validate_events_df(df)

    summary_csv = _export_events_summary(validated)
    summary_png = _export_events_summary_png(validated)

    print(f"[OK] Events validated: {events_path}")
    print(f"Rows: {len(validated):,}")
    print(f"[OK] Wrote: {summary_csv}")
    if summary_png is not None:
        print(f"[OK] Wrote: {summary_png}")
    print("\nPreview:")
    print(validated.sort_values(['start_date', 'event_id']).head(10))


if __name__ == "__main__":
    main()
