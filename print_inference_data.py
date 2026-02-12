"""
Print contents of ArviZ/PyMC NetCDF (.nc) InferenceData files.

Files:
- idata_m1_mean_switch.nc
- idata_m2_sigma_switch.nc
"""

from pathlib import Path
import arviz as az
import xarray as xr


FILES = [
    r"D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports\interim\idata_m1_mean_switch.nc",
    r"D:\Python\Week11\Change-Point-Analysis-and-Statistical-Modeling\reports\interim\idata_m2_sigma_switch.nc",
]


def print_with_arviz(file_path: Path):
    """Pretty Bayesian summary using ArviZ."""
    print("\n" + "=" * 80)
    print(f"ARVIZ SUMMARY → {file_path.name}")
    print("=" * 80)

    idata = az.from_netcdf(file_path)

    print("\nGroups available:")
    print(idata.groups())

    for group in idata.groups():
        print(f"\n--- {group} ---")
        print(idata[group])

    # statistical summary
    print("\nPosterior summary:")
    try:
        print(az.summary(idata, round_to=4))
    except Exception:
        print("No posterior group available.")


def print_raw_xarray(file_path: Path):
    """Raw NetCDF structure using xarray."""
    print("\n" + "=" * 80)
    print(f"RAW XARRAY STRUCTURE → {file_path.name}")
    print("=" * 80)

    ds = xr.open_dataset(file_path)
    print(ds)


def main():
    for file in FILES:
        path = Path(file)

        if not path.exists():
            print(f"\n❌ File not found: {path}")
            continue

        print_raw_xarray(path)
        print_with_arviz(path)


if __name__ == "__main__":
    main()
