"""
Export FULL ArviZ summaries (with diagnostics) from InferenceData (.nc) files.

Diagnostics automatically included:
- mcse_mean
- mcse_sd
- ess_bulk
- ess_tail
- r_hat

Outputs saved to:
D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports
"""

from pathlib import Path
import arviz as az
import pandas as pd


# ======================================================
# INPUT FILES
# ======================================================
FILES = {
    "m1_mean_switch": r"D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports\interim\idata_m1_mean_switch.nc",
    "m2_sigma_switch": r"D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports\interim\idata_m2_sigma_switch.nc",
}


# ======================================================
# OUTPUT DIRECTORY
# ======================================================
OUTPUT_DIR = Path(
    r"D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export_summary(model_name: str, file_path: str) -> pd.DataFrame:
    """Load idata, compute FULL ArviZ summary (with diagnostics), export CSV."""
    path = Path(file_path)

    if not path.exists():
        print(f"❌ File not found: {path}")
        return None

    print(f"\nProcessing → {model_name}")

    idata = az.from_netcdf(path)

    # ✅ FULL summary with diagnostics (reviewer requirement)
    summary = az.summary(
        idata,
        round_to=4
    )

    summary["model"] = model_name
    summary = summary.reset_index().rename(columns={"index": "parameter"})

    out_file = OUTPUT_DIR / f"{model_name}_summary_with_diagnostics.csv"
    summary.to_csv(out_file, index=False)

    print(f"✅ Saved: {out_file}")

    return summary


def main():
    all_summaries = []

    for model, file_path in FILES.items():
        df = export_summary(model, file_path)
        if df is not None:
            all_summaries.append(df)

    # Combined comparison file
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_file = OUTPUT_DIR / "combined_model_summaries_with_diagnostics.csv"
        combined.to_csv(combined_file, index=False)
        print(f"✅ Saved: {combined_file}")


if __name__ == "__main__":
    main()
