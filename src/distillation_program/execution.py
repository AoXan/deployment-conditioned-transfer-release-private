from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _hash_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _pipeline(columns: list[str], seed: int, min_samples_leaf: int) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), columns)
    ])
    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _select_and_fit(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], seed: int):
    candidates = {}
    for leaf in (2, 5):
        model = _pipeline(features, seed, leaf)
        model.fit(train[features], train.target_yield)
        candidates[leaf] = float(mean_absolute_error(validation.target_yield, model.predict(validation[features])))
    selected = min(candidates, key=lambda leaf: (candidates[leaf], leaf))
    final = _pipeline(features, seed, selected)
    final.fit(pd.concat([train, validation])[features], pd.concat([train, validation]).target_yield)
    return final, {"candidate_validation_mae": candidates, "selected_min_samples_leaf": selected}


def _modality_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "weather": [column for column in frame if column.startswith("weather_")],
        "soil": [column for column in frame if column.startswith("soil_")],
        "ec": [column for column in frame if column.startswith("ec_")],
    }
    return {name: columns for name, columns in groups.items() if columns}


def run_core_experiment(
    *,
    job_id: str,
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    output: Path,
    seed: int,
    job_status: str = "DESIGN_IMPLEMENTED_NOT_RUN",
    config_fingerprint: str | None = None,
) -> dict:
    if job_status.startswith("BLOCKED") or job_status == "EXTERNAL_BLOCKED":
        row = {"job_id": job_id, "status": job_status, "reason": "preflight_gate_not_satisfied"}
        _atomic_json(output / "status.json", row)
        return row

    required = {"sample_id", "target_yield", "Year"}
    if not required <= set(frame) or not {"sample_id", "split"} <= set(folds):
        raise ValueError("frame_or_fold_contract_missing")
    merged = frame.merge(folds, on="sample_id", validate="one_to_one")
    outer_train = merged[merged.split.eq("train")].copy()
    test = merged[merged.split.eq("test")].copy()
    if outer_train.empty or test.empty:
        raise ValueError("empty_outer_train_or_test")
    validation_year = int(outer_train.Year.max())
    validation = outer_train[outer_train.Year.eq(validation_year)].copy()
    train = outer_train[outer_train.Year.lt(validation_year)].copy()
    if train.empty or validation.empty:
        raise ValueError("inner_validation_unavailable")

    groups = _modality_columns(frame)
    deployable = groups.get("weather", [])
    full = [column for columns in groups.values() for column in columns]
    if not deployable or len(full) <= len(deployable):
        raise ValueError("teacher_student_modality_contract_unavailable")

    variants: dict[str, tuple[list[str], Pipeline]] = {}
    selection = {}
    if job_id == "core_g2f_teacher_advantage":
        for variant, features in {
            "deployable_supervised_student": deployable,
            "deployable_capacity_control": deployable,
            "full_modal_teacher": full,
        }.items():
            model, ledger = _select_and_fit(train, validation, features, seed)
            variants[variant] = (features, model)
            selection[variant] = ledger
    elif job_id == "core_g2f_m1_fusion":
        modality_predictions = []
        modality_ledgers = {}
        for name, features in groups.items():
            model, ledger = _select_and_fit(train, validation, features, seed)
            validation_model = _pipeline(features, seed, ledger["selected_min_samples_leaf"])
            validation_model.fit(train[features], train.target_yield)
            validation_prediction = validation_model.predict(validation[features])
            modality_predictions.append((name, features, model, validation_prediction))
            modality_ledgers[name] = ledger
        rows = []
        validation_matrix = np.column_stack([prediction for _, _, _, prediction in modality_predictions])
        test_matrix = np.column_stack([model.predict(test[features]) for _, features, model, _ in modality_predictions])
        validation_availability = np.column_stack([validation[features].notna().any(axis=1).to_numpy(float) for _, features, _, _ in modality_predictions])
        test_availability = np.column_stack([test[features].notna().any(axis=1).to_numpy(float) for _, features, _, _ in modality_predictions])
        equal_weights = np.where(test_availability.sum(1, keepdims=True) > 0, test_availability, 1.0)
        equal_prediction = (test_matrix * equal_weights).sum(1) / equal_weights.sum(1)
        validation_errors = np.mean(np.abs(validation_matrix - validation.target_yield.to_numpy()[:, None]), axis=0)
        global_weights = 1 / np.clip(validation_errors, 1e-8, None)
        weighted = test_availability * global_weights[None, :]
        weighted = np.where(weighted.sum(1, keepdims=True) > 0, weighted, global_weights[None, :])
        weighted_prediction = (test_matrix * weighted).sum(1) / weighted.sum(1)
        gate = Ridge(alpha=1.0)
        gate.fit(np.column_stack([validation_matrix, validation_availability]), validation.target_yield)
        gated_prediction = gate.predict(np.column_stack([test_matrix, test_availability]))
        for variant, prediction in {
            "modality_specific_equal_late_fusion": equal_prediction,
            "modality_specific_validation_weighted_late_fusion": weighted_prediction,
            "modality_specific_missing_aware_gated_fusion": gated_prediction,
        }.items():
            for index, sample in enumerate(test.itertuples()):
                rows.append({"sample_id": sample.sample_id, "y_true": sample.target_yield, "y_pred": prediction[index], "variant": variant, "seed": seed})
        predictions = pd.DataFrame(rows)
        selection = {"modality_models": modality_ledgers, "validation_inverse_mae_weights": global_weights.tolist(), "gate": "ridge_on_validation_predictions_and_availability"}
    elif job_id == "core_current_m2_control":
        teacher, teacher_ledger = _select_and_fit(train, validation, full, seed)
        pseudo = teacher.predict(train[full])
        blended = train.copy()
        blended["target_yield"] = 0.5 * train.target_yield.to_numpy() + 0.5 * pseudo
        student, student_ledger = _select_and_fit(blended, validation, deployable, seed)
        variants["current_m2_fixed_blend"] = (deployable, student)
        selection = {"teacher": teacher_ledger, "student": student_ledger, "blend": {"supervised": 0.5, "pseudo": 0.5}}
    else:
        raise ValueError(f"NO_EXECUTABLE_HANDLER:{job_id}")

    if job_id != "core_g2f_m1_fusion":
        rows = []
        for variant, (features, model) in variants.items():
            prediction = model.predict(test[features])
            for index, sample in enumerate(test.itertuples()):
                rows.append({"sample_id": sample.sample_id, "y_true": sample.target_yield, "y_pred": prediction[index], "variant": variant, "seed": seed})
        predictions = pd.DataFrame(rows)
    if not np.isfinite(predictions[["y_true", "y_pred"]]).all().all() or predictions.y_pred.nunique() <= 1:
        raise RuntimeError("invalid_predictions")

    metrics = []
    for variant, rows in predictions.groupby("variant"):
        metrics.append({
            "variant": variant,
            "mae": float(mean_absolute_error(rows.y_true, rows.y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(rows.y_true, rows.y_pred))),
            "n": len(rows),
        })
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "predictions.csv", predictions)
    _atomic_csv(output / "fold_assignments.csv", folds)
    _atomic_csv(output / "metrics.csv", pd.DataFrame(metrics))
    fingerprint_inputs = {
        "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config_fingerprint or hashlib.sha256(f"direct_test|{job_id}".encode()).hexdigest(),
        "data": _hash_frame(frame),
        "view": hashlib.sha256(json.dumps({"columns": list(frame), "sample_unit": "declared_by_adapter"}, sort_keys=True).encode()).hexdigest(),
        "split": _hash_frame(folds),
        "feature": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest(),
        "model": hashlib.sha256(json.dumps({"job_id": job_id, "seed": seed, "selection": selection}, sort_keys=True, default=str).encode()).hexdigest(),
        "runtime": hashlib.sha256(json.dumps({"python": platform.python_version(), "numpy": np.__version__}, sort_keys=True).encode()).hexdigest(),
    }
    manifest = {
        "job_id": job_id,
        "status": "EXECUTION_COMPLETE_UNVERIFIED",
        "scientific_status": "INSUFFICIENT_EVIDENCE",
        "seed": seed,
        "selection_ledger": {
            "validation_year": validation_year,
            "validation_ids": validation.sample_id.astype(str).tolist(),
            "variants": selection,
        },
        "test_labels_used_for_selection": False,
        "test_ids": test.sample_id.astype(str).tolist(),
        "data_fingerprint": _hash_frame(frame),
        "split_fingerprint": _hash_frame(folds),
        "fingerprint_inputs": fingerprint_inputs,
        "fingerprint": hashlib.sha256(json.dumps(fingerprint_inputs, sort_keys=True).encode()).hexdigest(),
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "completion_marker.json", {
        "status": "EXECUTION_COMPLETE_UNVERIFIED",
        "scientific_status": "INSUFFICIENT_EVIDENCE",
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
    })
    return {"status": "EXECUTION_COMPLETE_UNVERIFIED", "output": str(output), "variants": sorted(predictions.variant.unique())}
