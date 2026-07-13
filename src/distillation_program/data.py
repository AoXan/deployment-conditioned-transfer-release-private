from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_declared_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit_datasets(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset_id, contract in config["datasets"].items():
        if contract["status"] == "EXTERNAL_BLOCKED":
            rows.append({
                "dataset_id": dataset_id,
                "status": "EXTERNAL_BLOCKED",
                "path_accessed": False,
                "reason": contract["reason"],
            })
            continue
        declared = contract.get("view") or contract.get("provenance")
        if declared is None and contract.get("views"):
            declared_paths = contract["views"]
        else:
            declared_paths = [declared]
        for declared_path in declared_paths:
            path = resolve_declared_path(declared_path, root)
            if not path.is_file():
                rows.append({"dataset_id": dataset_id, "status": "BLOCKED_DATA_CONTRACT", "path": str(path), "path_accessed": False, "reason": "declared_safe_input_missing"})
                continue
            if path.suffix == ".json":
                payload = json.loads(path.read_text())
                columns = sorted(payload)
                row_count = None
            else:
                sample = pd.read_csv(path, nrows=5)
                columns = list(sample.columns)
                row_count = None
            rows.append({
                "dataset_id": dataset_id,
                "status": "DECLARED_INPUT_VERIFIED",
                "path": str(path),
                "path_accessed": True,
                "column_count": len(columns),
                "modality_counts": {
                    "weather": sum(column.startswith(("weather_", "silo_", "tmin_", "tmax_", "prec_")) for column in columns),
                    "soil": sum(column.startswith(("soil_", "slga_", "openmeteo_")) for column in columns),
                    "ec": sum(column.startswith("ec_") for column in columns),
                    "management": sum("treatment" in column.lower() or "management" in column.lower() or "irrig" in column.lower() for column in columns),
                },
                "row_count": row_count,
            })
    return rows


def load_declared_frame_and_fold(config: dict[str, Any], dataset_id: str, root: Path, fold: str = "test_2023"):
    contract = config["datasets"][dataset_id]
    if contract["status"].startswith("BLOCKED") or contract["status"] == "EXTERNAL_BLOCKED":
        raise PermissionError(contract["status"])
    frame = pd.read_csv(resolve_declared_path(contract["view"], root))
    fold_path = resolve_declared_path(contract["folds"].format(fold=fold), root)
    folds = pd.read_csv(fold_path)
    return frame, folds, fold_path
