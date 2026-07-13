from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import atomic_json, sha256_file, stable_hash


REQUIRED_CHILD_FILES = {
    "predictions.csv",
    "fold_ids.json",
    "metrics.json",
    "training_ledger.json",
    "manifest.json",
    "acceptance.json",
    "COMPLETED.json",
}

SUPPORTED_M_METHODS = {
    "imputation",
    "missing_indicators",
    "late_fusion",
    "teacher_student_m2",
    "missing_aware",
}

SUPPORTED_DEPLOYMENT_CONDITIONS = {
    "complete",
    "natural",
    "synthetic_no_soil",
    "synthetic_no_weather",
    "synthetic_random_0_15",
}


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _target_job_directory(
    output_root: Path,
    candidate_id: str,
    *,
    smoke: bool,
) -> Path:
    return (
        output_root
        / ("smoke" if smoke else "jobs")
        / "targets"
        / candidate_id
    )


def validate_m_child_artifact(
    output_root: Path,
    candidate_id: str,
    *,
    smoke: bool,
) -> dict[str, Any]:
    directory = _target_job_directory(
        output_root,
        candidate_id,
        smoke=smoke,
    )

    if not directory.is_dir():
        raise FileNotFoundError(
            f"M_EVALUATION_CHILD_DIRECTORY_MISSING:{candidate_id}"
        )

    missing = sorted(
        name
        for name in REQUIRED_CHILD_FILES
        if not (directory / name).is_file()
    )
    if missing:
        raise ValueError(
            f"M_EVALUATION_CHILD_ARTIFACTS_MISSING:"
            f"{candidate_id}:{','.join(missing)}"
        )

    completed = _load_json(directory / "COMPLETED.json")
    acceptance = _load_json(directory / "acceptance.json")
    manifest = _load_json(directory / "manifest.json")
    folds = _load_json(directory / "fold_ids.json")
    metrics = _load_json(directory / "metrics.json")
    ledger = _load_json(directory / "training_ledger.json")

    if completed.get("status") != "COMPLETED":
        raise ValueError(
            f"M_EVALUATION_CHILD_NOT_COMPLETED:{candidate_id}"
        )
    if acceptance.get("status") != "ARTIFACT_ACCEPTED":
        raise ValueError(
            f"M_EVALUATION_CHILD_NOT_ACCEPTED:{candidate_id}"
        )

    completed_fingerprint = completed.get("fingerprint")
    accepted_fingerprint = acceptance.get("fingerprint")
    manifest_fingerprint = manifest.get("fingerprint")

    if not (
        completed_fingerprint
        and completed_fingerprint == accepted_fingerprint
        and accepted_fingerprint == manifest_fingerprint
    ):
        raise ValueError(
            f"M_EVALUATION_CHILD_FINGERPRINT_MISMATCH:{candidate_id}"
        )

    job = manifest.get("job")
    if not isinstance(job, dict):
        raise ValueError(
            f"M_EVALUATION_CHILD_JOB_MISSING:{candidate_id}"
        )

    method = job.get("missing_modality_method")
    condition = job.get("deployment_condition")

    if method not in SUPPORTED_M_METHODS:
        raise ValueError(
            f"M_EVALUATION_UNSUPPORTED_METHOD:{candidate_id}:{method}"
        )
    if condition not in SUPPORTED_DEPLOYMENT_CONDITIONS:
        raise ValueError(
            f"M_EVALUATION_UNSUPPORTED_CONDITION:"
            f"{candidate_id}:{condition}"
        )

    predictions = pd.read_csv(directory / "predictions.csv")
    required_columns = {"sample_id", "y_true", "y_pred"}
    if not required_columns <= set(predictions):
        raise ValueError(
            f"M_EVALUATION_PREDICTION_SCHEMA_INVALID:{candidate_id}"
        )

    if predictions.empty:
        raise ValueError(
            f"M_EVALUATION_EMPTY_PREDICTIONS:{candidate_id}"
        )

    y_true = predictions["y_true"].to_numpy(float)
    y_pred = predictions["y_pred"].to_numpy(float)

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(
            f"M_EVALUATION_NONFINITE_PREDICTIONS:{candidate_id}"
        )

    expected_test_ids = set(map(str, folds.get("test_ids", [])))
    observed_test_ids = set(
        predictions["sample_id"].astype(str)
    )
    if observed_test_ids != expected_test_ids:
        raise ValueError(
            f"M_EVALUATION_CHILD_TEST_IDS_MISMATCH:{candidate_id}"
        )

    file_hashes = {
        name: sha256_file(directory / name)
        for name in sorted(REQUIRED_CHILD_FILES)
    }

    return {
        "candidate_id": candidate_id,
        "directory": str(directory),
        "job": job,
        "method": method,
        "condition": condition,
        "dataset_id": job.get("dataset_id"),
        "split_id": job.get("split_id"),
        "seed": job.get("seed"),
        "source_dataset": job.get("source_dataset"),
        "source_route": job.get("source_route"),
        "transfer_strategy": job.get("transfer_strategy"),
        "adaptation_mode": job.get("adaptation_mode"),
        "adaptation_fraction": job.get("adaptation_fraction"),
        "fingerprint": manifest_fingerprint,
        "metrics": metrics,
        "training_ledger_hash": file_hashes["training_ledger.json"],
        "file_hashes": file_hashes,
        "prediction_rows": int(len(predictions)),
        "test_ids_hash": stable_hash(
            sorted(observed_test_ids)
        ),
        "predictions": predictions,
        "ledger": ledger,
    }


def _bundle_identity(child: dict[str, Any]) -> tuple[Any, ...]:
    return (
        child["dataset_id"],
        child["split_id"],
        child["method"],
        child["seed"],
        child["source_dataset"],
        child["source_route"],
        child["transfer_strategy"],
        child["adaptation_mode"],
        child["adaptation_fraction"],
    )


def execute_m_evaluation_bundle(
    *,
    output_root: Path,
    bundle_id: str,
    child_candidate_ids: Iterable[str],
    smoke: bool,
) -> dict[str, Any]:
    candidate_ids = list(dict.fromkeys(map(str, child_candidate_ids)))
    if not candidate_ids:
        raise ValueError("M_EVALUATION_REQUIRES_CHILD_JOBS")

    children = [
        validate_m_child_artifact(
            output_root,
            candidate_id,
            smoke=smoke,
        )
        for candidate_id in candidate_ids
    ]

    identities = {_bundle_identity(child) for child in children}
    if len(identities) != 1:
        raise ValueError(
            "M_EVALUATION_BUNDLE_IDENTITY_MISMATCH"
        )

    conditions = [child["condition"] for child in children]
    if len(conditions) != len(set(conditions)):
        raise ValueError(
            "M_EVALUATION_DUPLICATE_DEPLOYMENT_CONDITION"
        )

    test_hashes = {child["test_ids_hash"] for child in children}
    if len(test_hashes) != 1:
        raise ValueError(
            "M_EVALUATION_TEST_COHORT_MISMATCH"
        )

    condition_order = sorted(conditions)
    by_condition = {
        child["condition"]: child
        for child in children
    }

    merged_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for condition in condition_order:
        child = by_condition[condition]
        part = child["predictions"].copy()
        part.insert(0, "deployment_condition", condition)
        part.insert(0, "child_candidate_id", child["candidate_id"])
        merged_parts.append(part)

        metric_rows.append(
            {
                "deployment_condition": condition,
                "child_candidate_id": child["candidate_id"],
                "mae": child["metrics"].get("mae"),
                "rmse": child["metrics"].get("rmse"),
                "r2": child["metrics"].get("r2"),
                "n": child["metrics"].get("n"),
                "child_fingerprint": child["fingerprint"],
                "predictions_sha256": child["file_hashes"][
                    "predictions.csv"
                ],
            }
        )

    merged = pd.concat(merged_parts, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows)

    root = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "m_evaluation"
        / bundle_id
    )
    root.mkdir(parents=True, exist_ok=True)

    merged.to_csv(root / "predictions_by_condition.csv", index=False)
    metrics_frame.to_csv(root / "metrics_by_condition.csv", index=False)

    first = children[0]
    child_records = [
        {
            key: value
            for key, value in child.items()
            if key not in {"predictions", "ledger"}
        }
        for child in children
    ]

    manifest = {
        "schema_version": "m_evaluation_bundle_manifest_v1",
        "bundle_id": bundle_id,
        "bundle_identity": {
            "dataset_id": first["dataset_id"],
            "split_id": first["split_id"],
            "missing_modality_method": first["method"],
            "seed": first["seed"],
            "source_dataset": first["source_dataset"],
            "source_route": first["source_route"],
            "transfer_strategy": first["transfer_strategy"],
            "adaptation_mode": first["adaptation_mode"],
            "adaptation_fraction": first["adaptation_fraction"],
        },
        "deployment_conditions": condition_order,
        "test_ids_hash": first["test_ids_hash"],
        "child_candidate_ids": candidate_ids,
        "child_fingerprints": {
            child["candidate_id"]: child["fingerprint"]
            for child in children
        },
        "child_file_hashes": {
            child["candidate_id"]: child["file_hashes"]
            for child in children
        },
        "evaluation_only": True,
        "training_performed": False,
        "smoke_only": smoke,
    }
    manifest["bundle_fingerprint"] = _sha256_json(manifest)
    atomic_json(root / "manifest.json", manifest)

    acceptance = {
        "schema_version": "m_evaluation_bundle_acceptance_v1",
        "status": "ARTIFACT_ACCEPTED",
        "bundle_id": bundle_id,
        "bundle_fingerprint": manifest["bundle_fingerprint"],
        "child_count": len(children),
        "deployment_condition_count": len(condition_order),
        "test_ids_hash": first["test_ids_hash"],
        "artifact_sha256": {
            "manifest": sha256_file(root / "manifest.json"),
            "predictions_by_condition": sha256_file(
                root / "predictions_by_condition.csv"
            ),
            "metrics_by_condition": sha256_file(
                root / "metrics_by_condition.csv"
            ),
        },
        "training_performed": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    atomic_json(
        root / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "bundle_id": bundle_id,
            "bundle_fingerprint": manifest["bundle_fingerprint"],
            "acceptance_sha256": sha256_file(
                root / "acceptance.json"
            ),
            "training_performed": False,
        },
    )

    return {
        "status": "COMPLETED",
        "bundle_id": bundle_id,
        "bundle_directory": str(root),
        "bundle_fingerprint": manifest["bundle_fingerprint"],
        "child_count": len(children),
        "deployment_conditions": condition_order,
        "training_performed": False,
        "smoke_only": smoke,
        "child_records": child_records,
    }


M_EVALUATION_HANDLER_REGISTRY = {
    "m_evaluation_bundles": execute_m_evaluation_bundle,
}
