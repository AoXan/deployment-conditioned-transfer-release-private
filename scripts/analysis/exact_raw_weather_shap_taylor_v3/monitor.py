from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    package = Path(args.package).resolve()

    while True:
        subprocess.run(
            [
                args.python,
                str(package / "build_dashboard.py"),
                "--output",
                str(output),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        progress = {}

        if (output / "progress.json").is_file():
            try:
                progress = json.loads(
                    (output / "progress.json").read_text()
                )
            except Exception:
                progress = {}

        os.system("clear")

        print("=" * 84)
        print(" RAW-WEATHER SHAP + TAYLOR MONITOR")
        print("=" * 84)
        print("Status:", progress.get("status", "WAITING"))
        print("Phase:", progress.get("phase", "WAITING"))
        print(
            "Replay:",
            f"{progress.get('replayed_cells', 0)}/30",
        )
        print(
            "Attributed cells:",
            f"{progress.get('attributed_cells', 0)}/30",
        )
        print(
            "Samples:",
            f"{progress.get('completed_samples', 0)}/"
            f"{progress.get('total_samples', 0)}",
        )
        print(
            "Current cell:",
            progress.get("current_cell", "—"),
        )
        print(
            "Current sample:",
            progress.get("current_sample", "—"),
        )
        print(
            "Max replay error:",
            progress.get("max_replay_error", "—"),
        )
        print(
            "Max scaled-array error:",
            progress.get(
                "max_scaled_array_error",
                "—",
            ),
        )
        print(
            "ETA seconds:",
            progress.get("eta_seconds", "—"),
        )
        print()
        print(progress.get("last_message", ""))
        print()
        print("Dashboard:", output / "dashboard.html")
        print("Log:", output / "logs/pipeline.log")
        print("=" * 84)

        if progress.get("status") in {
            "COMPLETED",
            "FAILED",
            "BLOCKED",
        }:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
