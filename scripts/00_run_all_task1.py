""" Run the full Task 1 pipeline end-to-end with one command. """
# scripts/00_run_all_task1.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(script_path: Path) -> None:
    """Run a single script and check for errors."""
    cmd = [sys.executable, str(script_path)]
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run all Task 1 scripts in sequence."""
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"

    _run(scripts / "01_clean_prices.py")
    _run(scripts / "02_make_returns.py")
    _run(scripts / "03_run_task1_eda.py")
    _run(scripts / "04_validate_events.py")

    print("\n[OK] Task 1 pipeline complete.")


if __name__ == "__main__":
    main()
