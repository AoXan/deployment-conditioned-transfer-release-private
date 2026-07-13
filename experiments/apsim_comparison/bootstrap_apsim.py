#!/usr/bin/env python3
"""Install the pinned APSIM Next Generation source tree without bundling it."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


COMMIT = "2c3639e456746e16ab9ae95a4c492dbfbe6e567f"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("experiments/apsim_comparison/runtime/ApsimX"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    models = args.destination / "bin/Release/net8.0/Models.dll"
    if args.validate_only:
        if not models.is_file():
            raise SystemExit(f"APSIM runtime missing: {models}; rerun without --validate-only")
        return 0
    if not args.destination.exists():
        subprocess.run(["git", "clone", "https://github.com/APSIMInitiative/ApsimX.git", str(args.destination)], check=True)
    subprocess.run(["git", "-C", str(args.destination), "checkout", COMMIT], check=True)
    subprocess.run(["dotnet", "build", str(args.destination / "Models/Models.csproj"), "-c", "Release", "-f", "net8.0"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
