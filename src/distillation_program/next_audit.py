from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _prediction_digest(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values("sample_id")[["sample_id", "y_pred"]]
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def audit_round1_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"sample_id", "y_true", "y_pred", "variant", "seed"}
    if not required <= set(predictions):
        raise ValueError(f"round1_prediction_columns_missing:{sorted(required - set(predictions))}")
    if predictions.empty or not np.isfinite(predictions[["y_true", "y_pred"]]).all().all():
        raise ValueError("round1_predictions_invalid")
    variants = {}
    for variant, rows in predictions.groupby("variant", observed=True):
        variants[str(variant)] = {
            "n": int(len(rows)),
            "unique_sample_ids": int(rows.sample_id.nunique()),
            "unique_predictions": int(rows.y_pred.nunique()),
            "mae": float(mean_absolute_error(rows.y_true, rows.y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(rows.y_true, rows.y_pred))),
            "r2": float(r2_score(rows.y_true, rows.y_pred)) if rows.y_true.nunique() > 1 else None,
            "prediction_sha256": _prediction_digest(rows),
        }
    controls: dict[str, Any] = {}
    student = variants.get("deployable_supervised_student")
    capacity = variants.get("deployable_capacity_control")
    if capacity:
        same = bool(student and student["prediction_sha256"] == capacity["prediction_sha256"])
        controls["deployable_capacity_control"] = {
            "status": "CONTROL_INVALID_OR_NON_DISTINCT" if same else "DISTINCT_CONTROL_REQUIRES_CAPACITY_AUDIT",
            "prediction_sha_equal": same,
        }
    if "current_m2_fixed_blend" in variants:
        controls["current_m2_fixed_blend"] = {"status": "HISTORICAL_CONTROL_ONLY", "mechanism": "target_blending_not_separate_kd_losses"}
    return {"variants": variants, "controls": controls, "scientific_promotion_performed": False}


def grouped_regression_metrics(predictions: pd.DataFrame, group_column: str) -> dict[str, Any]:
    if group_column not in predictions:
        raise ValueError(f"group_column_missing:{group_column}")
    groups = predictions.groupby(group_column, observed=True).apply(lambda rows: float(np.mean(np.abs(rows.y_true - rows.y_pred))), include_groups=False)
    return {
        "group_column": group_column,
        "group_count": int(len(groups)),
        "balanced_mae": float(groups.mean()),
        "worst_group_mae": float(groups.max()),
    }


def audit_round1_run(run_directory: Path) -> dict[str, Any]:
    required_paths = {
        "predictions": run_directory / "predictions.csv",
        "metrics": run_directory / "metrics.csv",
        "folds": run_directory / "fold_assignments.csv",
        "manifest": run_directory / "manifest.json",
        "marker": run_directory / "completion_marker.json",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        return {"run": str(run_directory), "engineering_status": "COMPLETE_UNVERIFIED", "failures": [f"missing_{name}" for name in missing]}
    predictions = pd.read_csv(required_paths["predictions"])
    folds = pd.read_csv(required_paths["folds"])
    metrics = pd.read_csv(required_paths["metrics"])
    manifest = json.loads(required_paths["manifest"].read_text())
    marker = json.loads(required_paths["marker"].read_text())
    failures = []
    expected_keys = {"code", "config", "data", "view", "split", "feature", "model", "runtime"}
    if set(manifest.get("fingerprint_inputs", {})) != expected_keys:
        failures.append("incomplete_fingerprints")
    prediction_ids = set(predictions.sample_id.astype(str))
    test_ids = set(folds.loc[folds.split.eq("test"), "sample_id"].astype(str))
    if prediction_ids != test_ids:
        failures.append("prediction_fold_id_mismatch")
    independent = audit_round1_predictions(predictions)
    reported = {str(row.variant): row for row in metrics.itertuples()}
    for variant, recomputed in independent["variants"].items():
        row = reported.get(variant)
        if row is None or not np.isclose(float(row.mae), recomputed["mae"], rtol=1e-8, atol=1e-10) or not np.isclose(float(row.rmse), recomputed["rmse"], rtol=1e-8, atol=1e-10):
            failures.append("metric_drift")
            break
    manifest_hash = marker.get("manifest_sha256")
    if manifest_hash and manifest_hash != hashlib.sha256(required_paths["manifest"].read_bytes()).hexdigest():
        failures.append("marker_manifest_hash_mismatch")
    return {
        "run": str(run_directory),
        "engineering_status": "FORMAL_ACCEPTED" if not failures and marker.get("status") == "ARTIFACT_ACCEPTED_NOT_SCIENTIFIC_VERDICT" else "COMPLETE_UNVERIFIED",
        "scientific_status": "INSUFFICIENT_EVIDENCE",
        "marker_status": marker.get("status"),
        "failures": sorted(set(failures)),
        "independent_metrics_and_controls": independent,
    }
