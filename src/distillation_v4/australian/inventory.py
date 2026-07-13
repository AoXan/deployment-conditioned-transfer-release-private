from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["view"])
    if not path.is_file():
        return {**spec, "contract_status": "ENGINEERING_BLOCKED", "reason": "VIEW_NOT_FOUND"}
    frame = pd.read_csv(path, nrows=5000)
    target = spec["target_column"]
    required = [spec["id_column"], spec["year_column"], target]
    missing = [name for name in required if name not in frame]
    if missing:
        return {**spec, "contract_status": "ENGINEERING_BLOCKED", "reason": "MISSING_REQUIRED_COLUMNS", "missing_columns": missing}
    return {
        **spec,
        "contract_status": "CONTRACT_VALID",
        "view_sha256": sha256_file(path),
        "sample_rows_profiled": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "target_non_null_profiled": int(frame[target].notna().sum()),
    }


def inventory_datasets(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [profile_dataset({"dataset_id": key, **value}) for key, value in config["datasets"].items()]
