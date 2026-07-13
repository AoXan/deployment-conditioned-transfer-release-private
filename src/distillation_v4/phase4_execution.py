from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .phase4_pretraining import (
    pretrain_weather_encoder,
)
from .phase4_reference import (
    materialize_frozen_reference,
)
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
from .training_contract import NeuralTrainingContract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
        )
        + "\n"
    )
    temporary.replace(path)


def _split_development(
    frame: pd.DataFrame,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_year = int(fold.split("_")[-1])
    validation_year = outer_year - 1

    train = frame.loc[
        frame.year < validation_year
    ].copy()
    validation = frame.loc[
        frame.year.eq(validation_year)
    ].copy()

    if train.empty:
        raise RuntimeError(
            f"PHASE4_TRAIN_EMPTY:{fold}"
        )
    if validation.empty:
        raise RuntimeError(
            f"PHASE4_VALIDATION_EMPTY:{fold}"
        )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    return train, validation


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
            "PHASE4_WEATHER_FEATURES_EMPTY"
        )

    return weather, soil


def _preprocess(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[
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
    validation_values = pipeline.transform(
        validation[features]
    )

    return (
        torch.tensor(
            train_values,
            dtype=torch.float32,
        ),
        torch.tensor(
            validation_values,
            dtype=torch.float32,
        ),
        pipeline,
    )


def _teacher_record(
    *,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    key: list[str] | tuple[str, str, str] | None,
) -> dict[str, Any] | None:
    if key is None:
        return None

    normalized = tuple(str(value) for value in key)

    if normalized not in teacher_lookup:
        raise RuntimeError(
            "PHASE4_FROZEN_TEACHER_KEY_MISSING:"
            + ":".join(normalized)
        )

    return teacher_lookup[normalized]


def _replay_student(
    *,
    checkpoint: Path,
    validation_values: torch.Tensor,
    reference_prediction: np.ndarray,
) -> float:
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
        scaled, _ = model(validation_values)
        prediction = (
            scaled * float(payload["target_scale"])
            + float(payload["target_mean"])
        ).cpu().numpy()

    return float(
        np.max(
            np.abs(
                prediction
                - np.asarray(reference_prediction)
            )
        )
    )


def execute_phase4_route(
    *,
    route_record: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    contract: NeuralTrainingContract,
    ckd_weights: dict[str, float] | None,
    teacher_signal_provider: Callable[..., TeacherSignals] = (
        infer_frozen_teacher_signals
    ),
) -> dict[str, Any]:
    route = str(route_record["route"])
    dataset = str(route_record["dataset"])
    fold = str(route_record["fold"])
    seed = int(route_record["seed"])

    if route == "reference":
        train, validation = _split_development(
            frame,
            fold,
        )

        prediction_record = _teacher_record(
            teacher_lookup=teacher_lookup,
            key=route_record.get(
                "prediction_teacher_key"
            ),
        )
        representation_record = _teacher_record(
            teacher_lookup=teacher_lookup,
            key=route_record.get(
                "representation_teacher_key"
            ),
        )
        selected = (
            prediction_record
            if prediction_record is not None
            else representation_record
        )

        if selected is None:
            raise RuntimeError(
                "PHASE4_REFERENCE_TEACHER_MISSING"
            )

        signals = teacher_signal_provider(
            output_root=output_root,
            teacher_record=selected,
            train=train,
            validation=validation,
        )

        if signals.validation_prediction is None:
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_MISSING"
            )

        prediction = np.asarray(
            signals.validation_prediction,
            dtype=float,
        )

        if len(prediction) != len(validation):
            raise RuntimeError(
                "PHASE4_REFERENCE_PREDICTION_MISALIGNED"
            )

        validation_mae = float(
            np.mean(
                np.abs(
                    prediction
                    - validation.target_yield.to_numpy(
                        float
                    )
                )
            )
        )

        record = materialize_frozen_reference(
            output_root=output_root,
            teacher_record=selected,
            output=output,
            route_id=str(
                route_record["route_id"]
            ),
        )

        prediction_path = (
            output / "validation_prediction.npy"
        )
        np.save(prediction_path, prediction)

        record.update(
            {
                "validation_mae": validation_mae,
                "prediction_std": float(
                    np.std(prediction)
                ),
                "prediction_artifact": str(
                    prediction_path.relative_to(
                        output_root
                    )
                ),
                "prediction_artifact_sha256": (
                    _sha256(prediction_path)
                ),
                "teacher_candidate": (
                    signals.teacher_candidate
                ),
                "selection_scope": (
                    "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                ),
            }
        )

        _atomic_json(
            output / "reference_record.json",
            record,
        )
        _atomic_json(
            output / "job_record.json",
            record,
        )
        _atomic_json(
            output / "completion_marker.json",
            {
                "route_id": route_record["route_id"],
                "status": "ENGINEERING_ACCEPTED",
                "outer_test_used": False,
                "teacher_refit_performed": False,
                "student_training_performed": False,
            },
        )

        return record

    train, validation = _split_development(
        frame,
        fold,
    )
    weather_features, soil_features = _features(
        frame
    )

    (
        train_weather,
        validation_weather,
        weather_preprocessor,
    ) = _preprocess(
        train=train,
        validation=validation,
        features=weather_features,
    )

    train_target = torch.tensor(
        train.target_yield.to_numpy(float),
        dtype=torch.float32,
    )
    validation_target = torch.tensor(
        validation.target_yield.to_numpy(float),
        dtype=torch.float32,
    )

    prediction_teacher_record = _teacher_record(
        teacher_lookup=teacher_lookup,
        key=route_record.get(
            "prediction_teacher_key"
        ),
    )
    representation_teacher_record = _teacher_record(
        teacher_lookup=teacher_lookup,
        key=route_record.get(
            "representation_teacher_key"
        ),
    )

    prediction_signals = None
    if prediction_teacher_record is not None:
        prediction_signals = teacher_signal_provider(
            output_root=output_root,
            teacher_record=(
                prediction_teacher_record
            ),
            train=train,
            validation=validation,
        )

    representation_signals = None
    if (
        route in {
            "representation_kd",
            "combined_kd",
        }
        and representation_teacher_record
        is not None
    ):
        if (
            prediction_teacher_record
            == representation_teacher_record
            and prediction_signals is not None
        ):
            representation_signals = (
                prediction_signals
            )
        else:
            representation_signals = (
                teacher_signal_provider(
                    output_root=output_root,
                    teacher_record=(
                        representation_teacher_record
                    ),
                    train=train,
                    validation=validation,
                )
            )

    pretrained_encoder_state = None

    if route == "fine_tune":
        if not soil_features:
            raise RuntimeError(
                "PHASE4_FT_SOIL_FEATURES_EMPTY"
            )

        (
            train_soil,
            validation_soil,
            _soil_preprocessor,
        ) = _preprocess(
            train=train,
            validation=validation,
            features=soil_features,
        )

        pretrained_encoder_state = (
            pretrain_weather_encoder(
                train_weather=train_weather,
                train_soil=train_soil,
                train_target=train_target,
                validation_weather=(
                    validation_weather
                ),
                validation_soil=validation_soil,
                validation_target=(
                    validation_target
                ),
                contract=contract,
                seed=seed,
            )
        )

    if route == "combined_kd":
        if ckd_weights is None:
            raise RuntimeError(
                "PHASE4_CKD_WEIGHTS_MISSING"
            )
        loss_weights = Phase4LossWeights(
            supervised=float(
                ckd_weights["supervised"]
            ),
            prediction=float(
                ckd_weights["prediction"]
            ),
            representation=float(
                ckd_weights["representation"]
            ),
        )
    else:
        loss_weights = Phase4LossWeights(
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

    result = fit_phase4_student(
        route=route,
        train_values=train_weather,
        train_target=train_target,
        validation_values=validation_weather,
        validation_target=validation_target,
        contract=contract,
        seed=seed,
        loss_weights=loss_weights,
        teacher_prediction_train=(
            None
            if prediction_signals is None
            else torch.tensor(
                prediction_signals.train_prediction,
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

    output.mkdir(parents=True, exist_ok=True)

    checkpoint = output / "model.pt"
    preprocessor = output / "preprocessor.joblib"
    prediction_path = (
        output / "validation_prediction.npy"
    )
    representation_path = (
        output / "validation_representation.npy"
    )
    loss_path = output / "loss_ledger.json"

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
        },
        checkpoint,
    )
    joblib.dump(
        weather_preprocessor,
        preprocessor,
    )
    np.save(
        prediction_path,
        result.prediction,
    )
    np.save(
        representation_path,
        result.representation,
    )
    _atomic_json(
        loss_path,
        {
            "route": route,
            "loss_weights": (
                loss_weights.normalized().to_mapping()
            ),
            "epochs": result.loss_ledger,
        },
    )

    replay_error = _replay_student(
        checkpoint=checkpoint,
        validation_values=validation_weather,
        reference_prediction=result.prediction,
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "PHASE4_STUDENT_CHECKPOINT_REPLAY_MISMATCH"
        )

    validation_mae = float(
        np.mean(
            np.abs(
                result.prediction
                - validation.target_yield.to_numpy(
                    float
                )
            )
        )
    )

    record = {
        "route_id": route_record["route_id"],
        "route": route,
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "status": "ENGINEERING_ACCEPTED",
        "validation_mae": validation_mae,
        "prediction_std": float(
            np.std(result.prediction)
        ),
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
        "checkpoint": str(
            checkpoint.relative_to(output_root)
        ),
        "checkpoint_sha256": (
            _sha256(checkpoint)
        ),
        "preprocessor": str(
            preprocessor.relative_to(output_root)
        ),
        "preprocessor_sha256": (
            _sha256(preprocessor)
        ),
        "prediction_artifact": str(
            prediction_path.relative_to(
                output_root
            )
        ),
        "prediction_artifact_sha256": (
            _sha256(prediction_path)
        ),
        "representation_artifact": str(
            representation_path.relative_to(
                output_root
            )
        ),
        "representation_artifact_sha256": (
            _sha256(representation_path)
        ),
        "loss_ledger": str(
            loss_path.relative_to(output_root)
        ),
        "loss_ledger_sha256": (
            _sha256(loss_path)
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "teacher_refit_performed": False,
        "ft_qualification_teacher_key": (
            route_record.get(
                "representation_teacher_key"
            )
            if route == "fine_tune"
            else None
        ),
        "outer_test_used": False,
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
    }

    if prediction_signals is not None:
        record["prediction_teacher"] = {
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
        record["representation_teacher"] = {
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

    _atomic_json(
        output / "job_record.json",
        record,
    )
    _atomic_json(
        output / "completion_marker.json",
        {
            "route_id": route_record["route_id"],
            "status": "ENGINEERING_ACCEPTED",
            "outer_test_used": False,
        },
    )

    return record

# STAGE8_V4_SOIL_PHASE4_DISPATCH
_execute_phase4_route_without_soil = (
    execute_phase4_route
)


def execute_phase4_route(
    *,
    route_record: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    contract: NeuralTrainingContract,
    ckd_weights: (
        dict[str, float]
        | None
    ),
    teacher_signal_provider: (
        Callable[..., TeacherSignals]
    ) = infer_frozen_teacher_signals,
) -> dict[str, Any]:
    route = str(
        route_record["route"]
    )

    if route in {
        "soil_direct",
        "missing_aware",
    }:
        from .phase4_soil_execution import (
            execute_phase4_soil_route,
        )

        return execute_phase4_soil_route(
            route_record=route_record,
            frame=frame,
            output_root=output_root,
            output=output,
            contract=contract,
        )

    return (
        _execute_phase4_route_without_soil(
            route_record=route_record,
            frame=frame,
            output_root=output_root,
            output=output,
            teacher_lookup=(
                teacher_lookup
            ),
            contract=contract,
            ckd_weights=ckd_weights,
            teacher_signal_provider=(
                teacher_signal_provider
            ),
        )
    )
