from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .artifacts import sha256_file


def independently_accept(output_root: Path) -> dict:
    accepted, rejected = [], []
    for directory in sorted((output_root / "jobs" / "targets").glob("*")):
        try:
            predictions_path = directory / "predictions.csv"
            fold_path = directory / "fold_ids.json"
            manifest_path = directory / "manifest.json"
            marker_path = directory / "COMPLETED.json"
            if not all(path.is_file() for path in (predictions_path, fold_path, manifest_path, marker_path)):
                raise ValueError("ARTIFACT_SET_INCOMPLETE")
            predictions = pd.read_csv(predictions_path)
            folds = json.loads(fold_path.read_text())
            if set(predictions.sample_id.astype(str)) != set(map(str, folds["test_ids"])):
                raise ValueError("PREDICTION_FOLD_ID_MISMATCH")
            y = predictions.y_true.to_numpy(float)
            p = predictions.y_pred.to_numpy(float)
            if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or np.std(p) == 0:
                raise ValueError("INVALID_PREDICTIONS")
            metrics = {
                "mae": float(mean_absolute_error(y, p)),
                "rmse": float(mean_squared_error(y, p) ** 0.5),
                "r2": float(r2_score(y, p)) if len(y) > 1 else None,
                "n": len(y),
            }
            accepted.append(
                {
                    "job_id": directory.name,
                    "metrics": metrics,
                    "prediction_sha256": sha256_file(predictions_path),
                    "fold_sha256": sha256_file(fold_path),
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
        except Exception as error:
            rejected.append({"job_id": directory.name, "reason": type(error).__name__, "detail": str(error)})
    return {
        "schema_version": "universal_weather_v2_remediation_acceptance_v1",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "scientific_status": "NOT_EVALUATED" if not accepted else "ENGINEERING_ARTIFACT_ACCEPTED_ONLY",
    }
