from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_predictions(
    *,
    formal_predictions: Path,
    replay_predictions: Path,
    prediction_tolerance: float = 1e-5,
    metric_tolerance: float = 1e-6,
) -> dict:
    formal = pd.read_csv(formal_predictions)
    replay = pd.read_csv(replay_predictions)
    required = {"sample_id", "y_true", "y_pred"}
    if not required.issubset(formal.columns) or not required.issubset(replay.columns):
        return {"reproduction_status": "FAIL", "reason": "PREDICTION_COLUMNS_MISSING"}
    merged = formal.merge(replay, on="sample_id", suffixes=("_formal", "_replay"), how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        return {
            "reproduction_status": "FAIL",
            "reason": "SAMPLE_ID_ALIGNMENT_MISMATCH",
            "left_only": int((merged["_merge"] == "left_only").sum()),
            "right_only": int((merged["_merge"] == "right_only").sum()),
        }
    prediction_diff = np.abs(merged["y_pred_formal"].astype(float) - merged["y_pred_replay"].astype(float))
    target_diff = np.abs(merged["y_true_formal"].astype(float) - merged["y_true_replay"].astype(float))
    replay_metrics = metric_bundle(replay)
    formal_metrics = metric_bundle(formal)
    metric_diff = {metric: abs(formal_metrics[metric] - replay_metrics[metric]) for metric in formal_metrics}
    status = (
        "PASS"
        if float(prediction_diff.max()) <= prediction_tolerance
        and float(target_diff.max()) <= prediction_tolerance
        and all(value <= metric_tolerance for value in metric_diff.values())
        else "FAIL"
    )
    return {
        "reproduction_status": status,
        "prediction_max_abs_diff": float(prediction_diff.max()),
        "target_max_abs_diff": float(target_diff.max()),
        "formal_metrics": formal_metrics,
        "replay_metrics": replay_metrics,
        "metric_diff": metric_diff,
        "row_count": int(len(merged)),
    }


def write_reproduction_record(output_root: Path, candidate_id: str, record: dict) -> Path:
    path = Path(output_root) / "reproduction" / f"{candidate_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return path

