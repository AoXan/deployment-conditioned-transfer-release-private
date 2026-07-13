from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REQUIRED_FP = {
    "job",
    "resolved_config_hash",
    "code_tree_hash",
    "runtime_hash",
    "data_hash",
    "view_hash",
    "split_hash",
    "feature_hash",
    "model_hash",
    "lineage_hash",
    "schema_version",
    "namespace",
}


def execution_fingerprint(payload: dict) -> str:
    missing = REQUIRED_FP - set(payload)
    if missing:
        raise ValueError(f"MISSING_FINGERPRINT_INPUTS:{sorted(missing)}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def code_tree_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "universal_weather_v2_remediation").glob("*.py"))
    files += sorted((root / "universal_weather_v2").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def build_fingerprint(job: dict, *, namespace: str, extra: dict, resolved_config: dict | None = None) -> str:
    """Build the complete versioned fingerprint required by remediation jobs."""
    def hashed(value: object) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    payload = {
        "job": job,
        "resolved_config_hash": hashed(resolved_config or job),
        "code_tree_hash": code_tree_hash(),
        "runtime_hash": hashed({"python": os.sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}),
        "data_hash": sha256_file(Path(extra["dataset_path"])),
        "view_hash": sha256_file(Path(extra["dataset_path"])),
        "split_hash": str(extra.get("split")),
        "feature_hash": hashed((resolved_config or {}).get("datasets", {}).get(job.get("dataset_id"), job.get("model_contract"))),
        "model_hash": hashed({"route": job.get("source_route"), "method": job.get("missing_modality_method")}),
        "lineage_hash": hashed({"source": job.get("source_dataset"), "strategy": job.get("transfer_strategy"), "checkpoint": extra.get("checkpoint_hash")}),
        "schema_version": "universal_weather_v2_remediation_artifact_v1",
        "namespace": namespace,
    }
    return execution_fingerprint(payload)


def accept_artifacts(attempt: Path, *, expected_fingerprint: str) -> dict:
    return accept_attempt(attempt, fingerprint=expected_fingerprint)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def accept_attempt(attempt: Path, *, fingerprint: str) -> dict:
    predictions_path, folds_path = attempt / "predictions.csv", attempt / "fold_ids.json"
    if not predictions_path.is_file():
        raise ValueError("PREDICTIONS_REQUIRED")
    if not folds_path.is_file():
        raise ValueError("FOLD_IDS_REQUIRED")
    predictions = pd.read_csv(predictions_path)
    required = {"sample_id", "y_true", "y_pred"}
    if not required <= set(predictions):
        raise ValueError("PREDICTION_SCHEMA_INVALID")
    folds = json.loads(folds_path.read_text())
    if set(predictions.sample_id.astype(str)) != set(map(str, folds.get("test_ids", []))):
        raise ValueError("PREDICTION_FOLD_ID_MISMATCH")
    y, p = predictions.y_true.to_numpy(float), predictions.y_pred.to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("NONFINITE_OR_EMPTY_PREDICTIONS")
    metrics = {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "n": int(len(y)),
    }
    atomic_json(attempt / "metrics.json", metrics)
    acceptance = {
        "status": "ARTIFACT_ACCEPTED",
        "fingerprint": fingerprint,
        "artifact_sha256": {"predictions": sha256_file(predictions_path), "fold_ids": sha256_file(folds_path), "metrics": sha256_file(attempt / "metrics.json")},
    }
    atomic_json(attempt / "acceptance.json", acceptance)
    atomic_json(attempt / "COMPLETED.json", {"status": "COMPLETED", "fingerprint": fingerprint, "acceptance_sha256": sha256_file(attempt / "acceptance.json")})
    return acceptance
