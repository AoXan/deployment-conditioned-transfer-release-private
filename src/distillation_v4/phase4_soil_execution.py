from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .phase4_execution import (
    _atomic_json,
    _features,
    _preprocess,
    _sha256,
    _split_development,
)
from .soil_missing_aware import (
    SoilRepresentationTransformer,
    build_soil_checkpoint,
    fit_flexible_soil_model,
    predict_soil_checkpoint,
    soil_availability_mask,
)
from .training_contract import (
    NeuralTrainingContract,
)


SOIL_PHASE4_ROUTES = {
    "soil_direct",
    "missing_aware",
}


def execute_phase4_soil_route(
    *,
    route_record: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    contract: NeuralTrainingContract,
) -> dict[str, Any]:
    route = str(route_record["route"])

    if route not in SOIL_PHASE4_ROUTES:
        raise ValueError(
            "PHASE4_SOIL_ROUTE_INVALID:"
            + route
        )

    dataset = str(
        route_record["dataset"]
    )
    fold = str(route_record["fold"])
    seed = int(route_record["seed"])

    train, validation = (
        _split_development(
            frame,
            fold,
        )
    )

    weather_features, soil_features = (
        _features(frame)
    )

    if not soil_features:
        raise RuntimeError(
            "PHASE4_SOIL_FEATURES_EMPTY"
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

    train_target_array = (
        train.target_yield.to_numpy(
            float
        )
    )
    validation_target_array = (
        validation.target_yield.to_numpy(
            float
        )
    )

    train_soil_raw = (
        train[soil_features]
        .to_numpy(float)
    )
    validation_soil_raw = (
        validation[soil_features]
        .to_numpy(float)
    )

    representation_method = str(
        route_record[
            "soil_representation"
        ]
    )
    component_cap = int(
        route_record[
            "soil_compact_components"
        ]
    )
    hidden = int(
        route_record["soil_hidden"]
    )
    representation_dim = int(
        route_record[
            "soil_representation_dim"
        ]
    )
    dropout_probability = float(
        route_record[
            "soil_dropout_probability"
        ]
    )
    explicit_mask = bool(
        route_record[
            "explicit_availability_mask"
        ]
    )

    transformer = (
        SoilRepresentationTransformer(
            method=representation_method,
            max_components=component_cap,
        )
    )

    train_soil_array = (
        transformer.fit_transform(
            train_soil_raw,
            train_target_array,
        )
    )
    validation_soil_array = (
        transformer.transform(
            validation_soil_raw
        )
    )

    train_available_array = (
        soil_availability_mask(
            train_soil_raw
        )
    )
    validation_available_array = (
        soil_availability_mask(
            validation_soil_raw
        )
    )

    train_soil = torch.tensor(
        train_soil_array,
        dtype=torch.float32,
    )
    validation_soil = torch.tensor(
        validation_soil_array,
        dtype=torch.float32,
    )
    train_available = torch.tensor(
        train_available_array,
        dtype=torch.float32,
    )
    validation_available = (
        torch.tensor(
            validation_available_array,
            dtype=torch.float32,
        )
    )
    train_target = torch.tensor(
        train_target_array,
        dtype=torch.float32,
    )
    validation_target = torch.tensor(
        validation_target_array,
        dtype=torch.float32,
    )

    result = fit_flexible_soil_model(
        route=route,
        train_weather=train_weather,
        train_soil=train_soil,
        train_soil_available=(
            train_available
        ),
        train_target=train_target,
        validation_weather=(
            validation_weather
        ),
        validation_soil=(
            validation_soil
        ),
        validation_soil_available=(
            validation_available
        ),
        validation_target=(
            validation_target
        ),
        contract=contract,
        seed=seed,
        hidden=hidden,
        representation_dim=(
            representation_dim
        ),
        soil_dropout_probability=(
            dropout_probability
        ),
        explicit_availability_mask=(
            explicit_mask
        ),
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = output / "model.pt"
    preprocessor_path = (
        output / "preprocessor.joblib"
    )
    prediction_path = (
        output
        / "validation_prediction.npy"
    )
    representation_path = (
        output
        / "validation_representation.npy"
    )
    loss_path = (
        output / "loss_ledger.json"
    )

    checkpoint_payload = (
        build_soil_checkpoint(
            result=result,
            route=route,
            weather_dim=int(
                train_weather.shape[1]
            ),
            soil_dim=int(
                train_soil.shape[1]
            ),
            hidden=hidden,
            representation_dim=(
                representation_dim
            ),
            soil_representation=(
                representation_method
            ),
            soil_dropout_probability=(
                dropout_probability
            ),
            explicit_availability_mask=(
                explicit_mask
            ),
            weather_features=(
                weather_features
            ),
            soil_features=soil_features,
        )
    )

    torch.save(
        checkpoint_payload,
        checkpoint,
    )

    joblib.dump(
        {
            "weather_preprocessor": (
                weather_preprocessor
            ),
            "soil_transformer": (
                transformer
            ),
            "weather_features": (
                weather_features
            ),
            "soil_features": (
                soil_features
            ),
            "soil_availability_rule": (
                "AT_LEAST_ONE_NATIVE_"
                "SOIL_VALUE_PRESENT"
            ),
        },
        preprocessor_path,
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
            "soil_representation": (
                representation_method
            ),
            "soil_dropout_probability": (
                dropout_probability
            ),
            "explicit_availability_mask": (
                explicit_mask
            ),
            "epochs": (
                result.loss_ledger
            ),
        },
    )

    loaded_checkpoint = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    replay_prediction, _ = (
        predict_soil_checkpoint(
            payload=loaded_checkpoint,
            weather=validation_weather,
            soil=validation_soil,
            soil_available=(
                validation_available
            ),
        )
    )

    replay_error = float(
        np.max(
            np.abs(
                replay_prediction
                - result.prediction
            )
        )
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "PHASE4_SOIL_CHECKPOINT_"
            "REPLAY_MISMATCH"
        )

    validation_mae = float(
        np.mean(
            np.abs(
                result.prediction
                - validation_target_array
            )
        )
    )

    record = {
        "route_id": (
            route_record["route_id"]
        ),
        "route": route,
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "status": (
            "ENGINEERING_ACCEPTED"
        ),
        "validation_mae": (
            validation_mae
        ),
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
        "parameter_count": (
            result.parameter_count
        ),
        "parameter_matching_group": (
            "SOIL_DIRECT_MISSING_AWARE_V1"
        ),
        "checkpoint": str(
            checkpoint.relative_to(
                output_root
            )
        ),
        "checkpoint_sha256": (
            _sha256(checkpoint)
        ),
        "preprocessor": str(
            preprocessor_path.relative_to(
                output_root
            )
        ),
        "preprocessor_sha256": (
            _sha256(preprocessor_path)
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
            _sha256(
                representation_path
            )
        ),
        "loss_ledger": str(
            loss_path.relative_to(
                output_root
            )
        ),
        "loss_ledger_sha256": (
            _sha256(loss_path)
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "soil_contract": {
            "representation": (
                representation_method
            ),
            "compact_components_cap": (
                component_cap
            ),
            "dropout_probability": (
                dropout_probability
            ),
            "explicit_availability_mask": (
                explicit_mask
            ),
            "mask_granularity": (
                "WHOLE_SOIL_MODALITY"
            ),
            "weather_modality_dropped": (
                False
            ),
            "observed_soil_fraction_train": (
                result
                .observed_soil_fraction_train
            ),
            "observed_soil_fraction_validation": (
                result
                .observed_soil_fraction_validation
            ),
        },
        "teacher_refit_performed": (
            False
        ),
        "outer_test_used": False,
        "selection_scope": (
            "OUTER_TRAIN_INNER_"
            "VALIDATION_ONLY"
        ),
    }

    _atomic_json(
        output / "job_record.json",
        record,
    )

    _atomic_json(
        output / "completion_marker.json",
        {
            "route_id": (
                route_record["route_id"]
            ),
            "status": (
                "ENGINEERING_ACCEPTED"
            ),
            "outer_test_used": False,
            "teacher_refit_performed": (
                False
            ),
        },
    )

    return record
