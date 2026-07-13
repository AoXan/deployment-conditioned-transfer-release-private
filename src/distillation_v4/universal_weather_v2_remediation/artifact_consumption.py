from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SyntheticCheckpoint:
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def write_synthetic_checkpoint(path: Path, *, weights: Sequence[float], bias: float) -> SyntheticCheckpoint:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"weights": list(map(float, weights)), "bias": float(bias)}, sort_keys=True) + "\n")
    temporary.replace(path)
    return SyntheticCheckpoint(path=path, sha256=sha256_file(path))


def assert_dependency_hash_matches_loaded_artifact(path: Path, expected_hash: str) -> str:
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"DEPENDENCY_HASH_MISMATCH:expected={expected_hash}:actual={actual}:path={path}")
    return actual


def load_dependency_and_predict(
    *,
    expected_dependency_job_id: str,
    checkpoint_path: SyntheticCheckpoint | Path,
    expected_checkpoint_hash: str,
    arrays: np.ndarray,
    split_ids: Sequence[str],
    feature_order: Sequence[str],
) -> dict:
    path = checkpoint_path.path if isinstance(checkpoint_path, SyntheticCheckpoint) else Path(checkpoint_path)
    actual_hash = assert_dependency_hash_matches_loaded_artifact(path, expected_checkpoint_hash)
    payload = json.loads(path.read_text())
    weights = np.asarray(payload["weights"], dtype=float)
    inputs = np.asarray(arrays, dtype=float)
    if inputs.ndim != 2:
        raise ValueError("INPUT_ARRAYS_MUST_BE_2D")
    if inputs.shape[1] != len(weights):
        raise ValueError("FEATURE_COUNT_CHECKPOINT_WEIGHT_MISMATCH")
    if len(split_ids) != inputs.shape[0]:
        raise ValueError("SPLIT_ID_ARRAY_ROW_MISMATCH")
    predictions = inputs @ weights + float(payload["bias"])
    return {
        "expected_dependency_job_id": expected_dependency_job_id,
        "expected_checkpoint_hash": expected_checkpoint_hash,
        "actual_file_path_opened": str(path),
        "actual_checkpoint_hash_loaded": actual_hash,
        "input_array_hash": stable_hash(inputs.tolist()),
        "input_sample_id_hash": stable_hash(list(map(str, split_ids))),
        "feature_order_hash": stable_hash(list(feature_order)),
        "prediction_hash": stable_hash(predictions.tolist()),
        "predictions": predictions.tolist(),
    }
