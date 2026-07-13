from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(raw: str, repository_root: Path) -> Path:
    normalised = "/" + str(raw).replace("\\", "/").casefold().strip("/") + "/"
    forbidden = ("/nvt/", "g2f_2024", "g2f-2024", "2024_observed", "observed_target")
    if any(token in normalised for token in forbidden):
        raise PermissionError("FORBIDDEN_DATA_PATH_POLICY")
    path = Path(raw)
    return path if path.is_absolute() else repository_root / path


def load_frame(contract: dict, repository_root: Path) -> tuple[pd.DataFrame, Path]:
    path = resolve_dataset_path(contract["path"], repository_root)
    if not path.is_file():
        raise FileNotFoundError(f"DATASET_PATH_MISSING:{path}")
    frame = pd.read_csv(path, low_memory=False)
    sample_column = contract["sample_id_column"]
    if sample_column not in frame:
        template = contract.get("sample_id_template")
        if not template:
            raise ValueError(f"SAMPLE_ID_COLUMN_MISSING:{sample_column}:{path}")
        needed = [field for field in ("crop", "year") if "{" + field + "}" in template]
        missing = [field for field in needed if field not in frame]
        if missing:
            raise ValueError(f"SAMPLE_ID_TEMPLATE_FIELDS_MISSING:{missing}")
        frame = frame.copy()
        frame[sample_column] = [template.format(**row) for row in frame.to_dict("records")]
    required = [sample_column, contract["target_column"], contract["year_column"]]
    required.extend(contract.get("weather_columns", []))
    required.extend(contract.get("soil_columns", []))
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"DATASET_REQUIRED_COLUMNS_MISSING:{missing}:{path}")
    if frame[sample_column].astype(str).duplicated().any():
        raise ValueError(f"DUPLICATE_SAMPLE_IDS:{path}")
    target = pd.to_numeric(frame[contract["target_column"]], errors="coerce")
    valid = np.isfinite(target.to_numpy(dtype=float))
    if not valid.any():
        raise ValueError(f"NO_FINITE_TARGET:{path}")
    frame = frame.loc[valid].copy()
    frame[contract["target_column"]] = target.loc[valid].astype(float)
    return frame, path


def _weather_repeated(frame: pd.DataFrame, contract: dict) -> bool:
    weather = [column for column in contract.get("weather_columns", []) if column in frame]
    year = contract["year_column"]
    if not weather:
        return False
    return bool((frame.groupby(year, dropna=False)[weather].nunique(dropna=False).max() <= 1).all())


def audit_datasets(config: dict, repository_root: Path) -> dict:
    results = []
    for dataset_id, contract in config["datasets"].items():
        frame, path = load_frame(contract, repository_root)
        crop_column = contract.get("crop_column")
        results.append(
            {
                "dataset_id": dataset_id,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(len(frame)),
                "unique_sample_ids": int(frame[contract["sample_id_column"]].astype(str).nunique()),
                "sample_unit": contract["sample_unit"],
                "target_column": contract["target_column"],
                "target_unit": contract["target_unit"],
                "years": int(frame[contract["year_column"]].nunique()),
                "modalities": list(contract["modalities"]),
                "weather_repeated_within_year": _weather_repeated(frame, contract),
                "crop_counts": (
                    {str(key): int(value) for key, value in frame[crop_column].value_counts(dropna=False).items()}
                    if crop_column and crop_column in frame
                    else {}
                ),
                "splits": list(contract["splits"]),
            }
        )
    return {"schema_version": "universal_weather_v2_dataset_audit_v1", "datasets": results}
