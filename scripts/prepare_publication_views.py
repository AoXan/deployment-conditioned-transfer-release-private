#!/usr/bin/env python3
"""Build the harmonised CY-Bench views consumed by the formal Stage 8 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = (
    ("PRIMARY_CYBENCH_MAIZE_US", "maize", "US"),
    ("CY-Bench_wheat_AU", "wheat", "AU"),
)
REQUIRED_PREFIXES = ("yield", "meteo", "crop_calendar", "soil", "location", "crop_mask")


def locate_dataset_sources(raw_root: Path, crop: str, country: str) -> tuple[Path, list[Path]]:
    names = [f"{prefix}_{crop}_{country}.csv" for prefix in REQUIRED_PREFIXES]
    direct = [raw_root / name for name in names]
    if all(path.is_file() for path in direct):
        return raw_root, direct
    located: list[Path] = []
    for name in names:
        matches = sorted(raw_root.rglob(name)) if raw_root.is_dir() else []
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one {name} under {raw_root}, found {len(matches)}"
            )
        located.append(matches[0])
    parents = {path.parent for path in located}
    if len(parents) != 1:
        raise ValueError(f"CY-Bench files for {crop}_{country} do not share one directory: {sorted(map(str, parents))}")
    return parents.pop(), located


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cybench-root",
        type=Path,
        required=True,
        help="Directory tree containing CY-Bench source CSV files.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for harmonised publication views.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    layouts = {
        dataset_id: locate_dataset_sources(args.cybench_root, crop, country)
        for dataset_id, crop, country in DATASETS
    }
    required = [path for _, files in layouts.values() for path in files]
    outputs = [args.output_root / f"{dataset_id}.csv.gz" for dataset_id, _, _ in DATASETS]
    if args.dry_run or args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "executed": False,
                    "inputs": len(required),
                    "source_directories": {
                        dataset_id: str(directory) for dataset_id, (directory, _) in layouts.items()
                    },
                    "outputs": [str(path) for path in outputs],
                },
                indent=2,
            )
        )
        return 0
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.distillation_v4.universal_weather_v2.adapters import build_cybench_view

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for (dataset_id, crop, country), output in zip(DATASETS, outputs, strict=True):
        results.append(
            build_cybench_view(
                dataset_id=dataset_id,
                directory=layouts[dataset_id][0],
                crop=crop,
                country=country,
                output_path=output,
            )
        )
    print(json.dumps({"status": "PASS", "views": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
