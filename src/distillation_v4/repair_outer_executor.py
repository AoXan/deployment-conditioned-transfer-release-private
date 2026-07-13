from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .campaign import load_primary_frame
from .contracts import CampaignConfig
from .phase4_pretraining import pretrain_weather_encoder
from .phase4_student import (
    DeployableStudent,
    fit_phase4_student,
)
from .phase4_teacher_adapter import (
    TeacherSignals,
    infer_frozen_teacher_signals,
)
from .phase4_training_contract import (
    Phase4LossWeights,
)
from .repair_outer_artifacts import sha256_file
from .teacher_registry import (
    validate_frozen_teacher_registry,
)


TeacherSignalProvider = Callable[..., TeacherSignals]


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    temporary.replace(path)


def _relative(
    *,
    output: Path,
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        relative = resolved.relative_to(
            output.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "REPAIR_OUTER_ARTIFACT_OUTSIDE_OUTPUT"
        ) from exc

    return str(relative)


def _frame_fingerprint(
    frame: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()

    digest.update(
        json.dumps(
            list(frame.columns),
            separators=(",", ":"),
        ).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(
            frame,
            index=True,
        ).values.tobytes()
    )

    return digest.hexdigest()


def _partitions(
    *,
    frame: pd.DataFrame,
    outer_test_year: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    calibration_year = outer_test_year - 1

    train = frame.loc[
        frame.year < calibration_year
    ].copy()
    calibration = frame.loc[
        frame.year.eq(calibration_year)
    ].copy()
    test = frame.loc[
        frame.year.eq(outer_test_year)
    ].copy()

    if train.empty:
        raise RuntimeError(
            "REPAIR_OUTER_TRAIN_EMPTY"
        )
    if calibration.empty:
        raise RuntimeError(
            "REPAIR_OUTER_CALIBRATION_EMPTY"
        )
    if test.empty:
        raise RuntimeError(
            "REPAIR_OUTER_TEST_EMPTY"
        )

    train_ids = set(
        train.sample_id.astype(str)
    )
    calibration_ids = set(
        calibration.sample_id.astype(str)
    )
    test_ids = set(
        test.sample_id.astype(str)
    )

    if (
        train_ids & calibration_ids
        or train_ids & test_ids
        or calibration_ids & test_ids
    ):
        raise RuntimeError(
            "REPAIR_OUTER_SPLIT_SAMPLE_OVERLAP"
        )

    return (
        train.reset_index(drop=True),
        calibration.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _features(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    weather = sorted(
        column
        for column in frame.columns
        if column.startswith("weather_")
    )
    soil = sorted(
        column
        for column in frame.columns
        if (
            column.startswith("soil_")
            and not column.endswith("__missing")
        )
    )

    if not weather:
        raise RuntimeError(
            "REPAIR_OUTER_WEATHER_FEATURES_EMPTY"
        )

    return weather, soil


def _preprocess(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Pipeline,
]:
    pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    train_values = pipeline.fit_transform(
        train[features]
    )
    calibration_values = pipeline.transform(
        calibration[features]
    )
    test_values = pipeline.transform(
        test[features]
    )

    return (
        torch.tensor(
            train_values,
            dtype=torch.float32,
        ),
        torch.tensor(
            calibration_values,
            dtype=torch.float32,
        ),
        torch.tensor(
            test_values,
            dtype=torch.float32,
        ),
        pipeline,
    )


def _phase4_manifest(
    output: Path,
) -> dict[str, Any]:
    path = (
        output
        / "control/frozen_phase4_route_manifest.json"
    )

    if not path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_MANIFEST_MISSING"
        )

    payload: dict[str, Any] = json.loads(
        path.read_text()
    )

    if not isinstance(
        payload.get("routes"),
        list,
    ):
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_MANIFEST_INVALID"
        )

    return payload


def _phase4_route_record(
    *,
    output: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    manifest = _phase4_manifest(output)

    matching = [
        record
        for record in manifest["routes"]
        if (
            str(record.get("dataset"))
            == str(job["dataset"])
            and str(record.get("fold"))
            == str(job["fold"])
            and int(record.get("seed"))
            == int(job["seed"])
            and str(record.get("route"))
            == str(job["route"])
        )
    ]

    if len(matching) != 1:
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_ROUTE_RESOLUTION_FAILED:"
            f"{job['dataset']}:{job['fold']}:"
            f"{job['seed']}:{job['route']}:"
            f"{len(matching)}"
        )

    return matching[0]


def _teacher_lookup(
    *,
    output: Path,
) -> dict[
    tuple[str, str, str],
    dict[str, Any],
]:
    registry_path = (
        output
        / "control/frozen_teacher_registry.json"
    )

    registry = validate_frozen_teacher_registry(
        registry_path,
        output_root=output,
        require_formal_release=True,
    )

    if registry.get(
        "outer_refit_allowed"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_TEACHER_REFIT_NOT_BLOCKED"
        )

    lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for record in registry["teachers"]:
        key = (
            str(record["dataset"]),
            str(record["fold"]),
            str(record["role"]),
        )

        if key in lookup:
            raise RuntimeError(
                "REPAIR_OUTER_TEACHER_KEY_DUPLICATE:"
                + ":".join(key)
            )

        lookup[key] = record

    return lookup


def _resolve_teacher(
    *,
    lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    key: Any,
    required: bool,
    label: str,
) -> dict[str, Any] | None:
    if key is None:
        if required:
            raise RuntimeError(
                f"REPAIR_OUTER_{label}_TEACHER_KEY_MISSING"
            )
        return None

    if not isinstance(key, (list, tuple)):
        raise RuntimeError(
            f"REPAIR_OUTER_{label}_TEACHER_KEY_INVALID"
        )

    normalized = tuple(
        str(value)
        for value in key
    )

    record = lookup.get(normalized)

    if record is None:
        raise RuntimeError(
            f"REPAIR_OUTER_{label}_TEACHER_NOT_FOUND:"
            + ":".join(normalized)
        )

    return record


def _loss_weights(
    *,
    route: str,
    phase4_record: dict[str, Any],
) -> Phase4LossWeights:
    declared = phase4_record.get(
        "loss_weights"
    )

    if declared is not None:
        if not isinstance(declared, dict):
            raise RuntimeError(
                "REPAIR_OUTER_PHASE4_LOSS_WEIGHTS_INVALID"
            )

        return Phase4LossWeights(
            supervised=float(
                declared["supervised"]
            ),
            prediction=float(
                declared["prediction"]
            ),
            representation=float(
                declared["representation"]
            ),
        )

    if route == "combined_kd":
        raise RuntimeError(
            "REPAIR_OUTER_CKD_FROZEN_WEIGHTS_MISSING"
        )

    return Phase4LossWeights(
        supervised=1.0,
        prediction=(
            0.5
            if route == "prediction_kd"
            else 0.0
        ),
        representation=(
            0.5
            if route == "representation_kd"
            else 0.0
        ),
    )


def _metrics(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(
        target,
        dtype=float,
    )
    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(
        np.sqrt(np.mean(error ** 2))
    )

    denominator = float(
        np.sum(
            (target - np.mean(target)) ** 2
        )
    )

    r2 = (
        None
        if denominator <= 1e-12
        else float(
            1.0
            - np.sum(error ** 2)
            / denominator
        )
    )

    if not (
        math.isfinite(mae)
        and math.isfinite(rmse)
    ):
        raise RuntimeError(
            "REPAIR_OUTER_METRIC_NONFINITE"
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rows": int(len(target)),
    }


def _predict_checkpoint(
    *,
    checkpoint: Path,
    values: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = DeployableStudent(
        int(payload["input_dim"]),
        hidden=int(payload["hidden"]),
        representation_dim=int(
            payload["representation_dim"]
        ),
    )
    model.load_state_dict(
        payload["state_dict"]
    )
    model.eval()

    with torch.no_grad():
        scaled, representation = model(values)
        prediction = (
            scaled
            * float(payload["target_scale"])
            + float(payload["target_mean"])
        )

    return (
        prediction.cpu().numpy(),
        representation.cpu().numpy(),
    )


def execute_repair_outer_job(
    config: CampaignConfig,
    output: Path,
    job: dict[str, Any],
    root: Path,
    *,
    teacher_signal_provider: (
        TeacherSignalProvider
    ) = infer_frozen_teacher_signals,
) -> dict[str, Any]:
    route = str(job["route"])
    dataset = str(job["dataset"])
    fold = str(job["fold"])
    seed = int(job["seed"])
    outer_test_year = int(
        job["outer_test_year"]
    )

    allowed = {
        "supervised",
        "fine_tune",
        "prediction_kd",
        "representation_kd",
        "combined_kd",
    }

    if route not in allowed:
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_UNSUPPORTED:"
            + route
        )

    if fold != f"test_{outer_test_year}":
        raise RuntimeError(
            "REPAIR_OUTER_FOLD_YEAR_MISMATCH"
        )

    frame = load_primary_frame(
        config,
        output,
        dataset,
    )

    (
        train,
        calibration,
        test,
    ) = _partitions(
        frame=frame,
        outer_test_year=outer_test_year,
    )

    weather_features, soil_features = (
        _features(frame)
    )

    (
        train_weather,
        calibration_weather,
        test_weather,
        weather_preprocessor,
    ) = _preprocess(
        train=train,
        calibration=calibration,
        test=test,
        features=weather_features,
    )

    train_target = torch.tensor(
        train.target_yield.to_numpy(float),
        dtype=torch.float32,
    )
    calibration_target = torch.tensor(
        calibration.target_yield.to_numpy(float),
        dtype=torch.float32,
    )

    phase4_record = _phase4_route_record(
        output=output,
        job=job,
    )

    lookup = _teacher_lookup(output=output)

    prediction_record = _resolve_teacher(
        lookup=lookup,
        key=phase4_record.get(
            "prediction_teacher_key"
        ),
        required=route in {
            "prediction_kd",
            "combined_kd",
        },
        label="PREDICTION",
    )

    representation_record = _resolve_teacher(
        lookup=lookup,
        key=phase4_record.get(
            "representation_teacher_key"
        ),
        required=route in {
            "fine_tune",
            "representation_kd",
            "combined_kd",
        },
        label="REPRESENTATION",
    )

    prediction_signals = None
    if prediction_record is not None:
        prediction_signals = (
            teacher_signal_provider(
                output_root=output,
                teacher_record=prediction_record,
                train=train,
                validation=calibration,
            )
        )

    representation_signals = None
    if route in {
        "representation_kd",
        "combined_kd",
    }:
        if (
            prediction_record
            == representation_record
            and prediction_signals is not None
        ):
            representation_signals = (
                prediction_signals
            )
        elif representation_record is not None:
            representation_signals = (
                teacher_signal_provider(
                    output_root=output,
                    teacher_record=(
                        representation_record
                    ),
                    train=train,
                    validation=calibration,
                )
            )

    pretrained_encoder_state = None
    soil_preprocessor = None

    if route == "fine_tune":
        if not soil_features:
            raise RuntimeError(
                "REPAIR_OUTER_FT_SOIL_FEATURES_EMPTY"
            )

        (
            train_soil,
            calibration_soil,
            _test_soil,
            soil_preprocessor,
        ) = _preprocess(
            train=train,
            calibration=calibration,
            test=test,
            features=soil_features,
        )

        pretrained_encoder_state = (
            pretrain_weather_encoder(
                train_weather=train_weather,
                train_soil=train_soil,
                train_target=train_target,
                validation_weather=(
                    calibration_weather
                ),
                validation_soil=(
                    calibration_soil
                ),
                validation_target=(
                    calibration_target
                ),
                contract=config.neural_training,
                seed=seed,
            )
        )

    loss_weights = _loss_weights(
        route=route,
        phase4_record=phase4_record,
    )

    result = fit_phase4_student(
        route=route,
        train_values=train_weather,
        train_target=train_target,
        validation_values=(
            calibration_weather
        ),
        validation_target=(
            calibration_target
        ),
        contract=config.neural_training,
        seed=seed,
        loss_weights=loss_weights,
        teacher_prediction_train=(
            None
            if prediction_signals is None
            else torch.tensor(
                prediction_signals
                .train_prediction,
                dtype=torch.float32,
            )
        ),
        teacher_prediction_validation=(
            None
            if prediction_signals is None
            else torch.tensor(
                prediction_signals
                .validation_prediction,
                dtype=torch.float32,
            )
        ),
        teacher_representation_train=(
            None
            if representation_signals is None
            else torch.tensor(
                representation_signals
                .train_representation,
                dtype=torch.float32,
            )
        ),
        teacher_representation_validation=(
            None
            if representation_signals is None
            else torch.tensor(
                representation_signals
                .validation_representation,
                dtype=torch.float32,
            )
        ),
        pretrained_encoder_state=(
            pretrained_encoder_state
        ),
    )

    root.mkdir(parents=True, exist_ok=True)

    checkpoint = root / "model.pt"
    preprocessor = root / "preprocessor.joblib"
    predictions_path = root / "predictions.csv"
    calibration_path = (
        root / "calibration_predictions.csv"
    )
    metrics_path = root / "metrics.json"
    folds_path = root / "fold_assignments.csv"
    manifest_path = root / "manifest.json"
    loss_path = root / "loss_ledger.json"
    test_representation_path = (
        root / "test_representation.npy"
    )

    representation_dim = int(
        result.representation.shape[1]
    )

    torch.save(
        {
            "state_dict": result.best_state_dict,
            "route": route,
            "input_dim": int(
                train_weather.shape[1]
            ),
            "hidden": 32,
            "representation_dim": (
                representation_dim
            ),
            "target_mean": result.target_mean,
            "target_scale": result.target_scale,
            "selected_epoch": (
                result.selected_epoch
            ),
            "features": weather_features,
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "teacher_refit_performed": False,
        },
        checkpoint,
    )

    joblib.dump(
        weather_preprocessor,
        preprocessor,
    )

    if soil_preprocessor is not None:
        joblib.dump(
            soil_preprocessor,
            root / "soil_preprocessor.joblib",
        )

    (
        test_prediction,
        test_representation,
    ) = _predict_checkpoint(
        checkpoint=checkpoint,
        values=test_weather,
    )

    replay_prediction, _ = _predict_checkpoint(
        checkpoint=checkpoint,
        values=test_weather,
    )

    replay_error = float(
        np.max(
            np.abs(
                replay_prediction
                - test_prediction
            )
        )
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "REPAIR_OUTER_CHECKPOINT_REPLAY_MISMATCH"
        )

    calibration_frame = pd.DataFrame(
        {
            "sample_id": (
                calibration.sample_id
                .astype(str)
                .to_numpy()
            ),
            "y_true": (
                calibration.target_yield
                .to_numpy(float)
            ),
            "y_pred": result.prediction,
        }
    )
    calibration_frame.to_csv(
        calibration_path,
        index=False,
    )

    predictions = pd.DataFrame(
        {
            "sample_id": (
                test.sample_id.astype(str)
                .to_numpy()
            ),
            "y_true": (
                test.target_yield.to_numpy(float)
            ),
            "y_pred": test_prediction,
        }
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics = _metrics(
        target=predictions.y_true.to_numpy(),
        prediction=predictions.y_pred.to_numpy(),
    )
    _atomic_json(metrics_path, metrics)

    fold_assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": (
                        train.sample_id.astype(str)
                    ),
                    "split": "train",
                    "year": train.year,
                }
            ),
            pd.DataFrame(
                {
                    "sample_id": (
                        calibration.sample_id
                        .astype(str)
                    ),
                    "split": "calibration",
                    "year": calibration.year,
                }
            ),
            pd.DataFrame(
                {
                    "sample_id": (
                        test.sample_id.astype(str)
                    ),
                    "split": "test",
                    "year": test.year,
                }
            ),
        ],
        ignore_index=True,
    )
    fold_assignments.to_csv(
        folds_path,
        index=False,
    )

    np.save(
        test_representation_path,
        test_representation,
    )

    _atomic_json(
        loss_path,
        {
            "route": route,
            "loss_weights": (
                loss_weights
                .normalized()
                .to_mapping()
            ),
            "epochs": result.loss_ledger,
        },
    )

    teacher_lineage: dict[str, Any] = {}

    if prediction_signals is not None:
        teacher_lineage["prediction"] = {
            "candidate": (
                prediction_signals
                .teacher_candidate
            ),
            "checkpoint": (
                prediction_signals
                .teacher_checkpoint
            ),
            "checkpoint_sha256": (
                prediction_signals
                .teacher_checkpoint_sha256
            ),
        }

    if representation_signals is not None:
        teacher_lineage["representation"] = {
            "candidate": (
                representation_signals
                .teacher_candidate
            ),
            "checkpoint": (
                representation_signals
                .teacher_checkpoint
            ),
            "checkpoint_sha256": (
                representation_signals
                .teacher_checkpoint_sha256
            ),
        }

    manifest = {
        "job_id": job["job_id"],
        "dataset": dataset,
        "route": route,
        "fold": fold,
        "seed": seed,
        "outer_test_year": outer_test_year,
        "calibration_year": (
            outer_test_year - 1
        ),
        "features": weather_features,
        "feature_order_frozen": True,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_max_year": int(
            train.year.max()
        ),
        "calibration_year_observed": int(
            calibration.year.unique()[0]
        ),
        "test_year_observed": int(
            test.year.unique()[0]
        ),
        "target_scaling": (
            "TRAIN_ONLY_STANDARDISATION"
        ),
        "target_mean": result.target_mean,
        "target_scale": result.target_scale,
        "loss_weights": (
            loss_weights.normalized().to_mapping()
        ),
        "phase4_route_id": phase4_record.get(
            "route_id"
        ),
        "phase4_route_manifest_fingerprint": (
            _phase4_manifest(output).get(
                "phase4_route_manifest_fingerprint"
            )
        ),
        "teacher_lineage": teacher_lineage,
        "teacher_refit_performed": False,
        "outer_test_role": (
            "FINAL_ESTIMATION_ONLY"
        ),
        "outer_test_may_enable_downstream_jobs": (
            False
        ),
        "route_fingerprint_before_outer_test": (
            job[
                "route_fingerprint_before_outer_test"
            ]
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "dataframe_fingerprint": (
            _frame_fingerprint(frame)
        ),
    }
    _atomic_json(manifest_path, manifest)

    marker = {
        "job_id": job["job_id"],
        "status": "ENGINEERING_ACCEPTED",
        "outer_test_used": True,
        "teacher_refit_performed": False,
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
    }
    _atomic_json(
        root / "completion_marker.json",
        marker,
    )

    return {
        "status": "ENGINEERING_ACCEPTED",
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "optimizer_steps": (
            result.optimizer_steps
        ),
        "parameter_delta_l2": (
            result.parameter_delta_l2
        ),
        "selected_epoch": (
            result.selected_epoch
        ),
        "fit_time_seconds": (
            result.fit_time_seconds
        ),
        "inference_time_seconds": (
            result.inference_time_seconds
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "teacher_refit_performed": False,
        "predictions": _relative(
            output=output,
            path=predictions_path,
        ),
        "predictions_sha256": sha256_file(
            predictions_path
        ),
        "metrics": _relative(
            output=output,
            path=metrics_path,
        ),
        "metrics_sha256": sha256_file(
            metrics_path
        ),
        "fold_assignments": _relative(
            output=output,
            path=folds_path,
        ),
        "fold_assignments_sha256": (
            sha256_file(folds_path)
        ),
        "checkpoint": _relative(
            output=output,
            path=checkpoint,
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint
        ),
        "preprocessor": _relative(
            output=output,
            path=preprocessor,
        ),
        "preprocessor_sha256": sha256_file(
            preprocessor
        ),
        "manifest": _relative(
            output=output,
            path=manifest_path,
        ),
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "loss_ledger": _relative(
            output=output,
            path=loss_path,
        ),
        "loss_ledger_sha256": sha256_file(
            loss_path
        ),
        "calibration_predictions": _relative(
            output=output,
            path=calibration_path,
        ),
        "calibration_predictions_sha256": (
            sha256_file(calibration_path)
        ),
        "test_representation": _relative(
            output=output,
            path=test_representation_path,
        ),
        "test_representation_sha256": (
            sha256_file(
                test_representation_path
            )
        ),
    }


# STAGE8_V4_SOIL_OUTER_DISPATCH
_execute_repair_outer_job_without_soil = (
    execute_repair_outer_job
)


def execute_repair_outer_job(
    config: CampaignConfig,
    output: Path,
    job: dict[str, Any],
    root: Path,
    *,
    teacher_signal_provider: (
        TeacherSignalProvider
    ) = infer_frozen_teacher_signals,
) -> dict[str, Any]:
    route = str(job["route"])

    if route in {
        "soil_direct",
        "missing_aware",
    }:
        from .repair_outer_soil_execution import (
            execute_repair_outer_soil_job,
        )

        return execute_repair_outer_soil_job(
            config,
            output,
            job,
            root,
        )

    return (
        _execute_repair_outer_job_without_soil(
            config,
            output,
            job,
            root,
            teacher_signal_provider=(
                teacher_signal_provider
            ),
        )
    )
