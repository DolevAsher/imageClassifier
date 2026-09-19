"""
Double-click-friendly launcher for the Streamlit demo app.

Run this file directly (e.g. `python run_app.py`, or just double-click it
in File Explorer if .py files are associated with Python) instead of
typing the `streamlit run ...` command yourself.

It also sets environment variables so Streamlit skips the one-time email
prompt and usage-stats collection, in case that hasn't been configured
via credentials.toml.
"""

import os
import sys
import subprocess
from pathlib import Path

# Suppress the first-run email prompt / telemetry, no config file needed
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

APP_FILE = "streamlit_app.py"


def main():
    app_path = Path(__file__).resolve().parent / APP_FILE

    if not app_path.exists():
        print(f"Could not find {APP_FILE} next to this script ({app_path}).")
        input("Press Enter to exit...")
        sys.exit(1)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )


if __name__ == "__main__":
    main()
