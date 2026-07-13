from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .phase4_execution import (
    _features,
    _preprocess,
    _split_development,
)
from .phase5_sources import (
    load_validated_phase4_source,
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


SOIL_PHASE5_CONTROLS = {
    "soil_representation_sensitivity_control",
    "missing_aware_ablation_suite_control",
}


def _variant_contracts(
    control: str,
) -> tuple[
    str,
    tuple[dict[str, Any], ...],
]:
    if control == (
        "soil_representation_sensitivity_control"
    ):
        return (
            "pls",
            (
                {
                    "variant_id": "pca",
                    "soil_representation": "pca",
                    "soil_dropout_probability": 0.0,
                    "explicit_availability_mask": True,
                    "scientific_role": (
                        "COMPACT_UNSUPERVISED_"
                        "REPRESENTATION_CONTROL"
                    ),
                },
                {
                    "variant_id": "pls",
                    "soil_representation": "pls",
                    "soil_dropout_probability": 0.0,
                    "explicit_availability_mask": True,
                    "scientific_role": (
                        "COMPACT_SUPERVISED_"
                        "REPRESENTATION_CONTROL"
                    ),
                },
            ),
        )

    if control == (
        "missing_aware_ablation_suite_control"
    ):
        return (
            "no_mask",
            (
                {
                    "variant_id": "no_mask",
                    "soil_representation": "raw",
                    "soil_dropout_probability": 0.5,
                    "explicit_availability_mask": False,
                    "scientific_role": (
                        "EXPLICIT_AVAILABILITY_"
                        "MASK_ABLATION"
                    ),
                },
                {
                    "variant_id": "dropout_025",
                    "soil_representation": "raw",
                    "soil_dropout_probability": 0.25,
                    "explicit_availability_mask": True,
                    "scientific_role": (
                        "LOW_DROPOUT_"
                        "SENSITIVITY_CONTROL"
                    ),
                },
                {
                    "variant_id": "dropout_075",
                    "soil_representation": "raw",
                    "soil_dropout_probability": 0.75,
                    "explicit_availability_mask": True,
                    "scientific_role": (
                        "HIGH_DROPOUT_"
                        "SENSITIVITY_CONTROL"
                    ),
                },
            ),
        )

    raise RuntimeError(
        "PHASE5_SOIL_CONTROL_UNKNOWN:"
        + control
    )


def execute_phase5_soil_control(
    *,
    job: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    contract: NeuralTrainingContract,
) -> dict[str, Any]:
    dataset = str(job["dataset"])
    fold = str(job["fold"])
    seed = int(job["seed"])
    control = str(job["control"])

    if control not in SOIL_PHASE5_CONTROLS:
        raise RuntimeError(
            "PHASE5_SOIL_CONTROL_INVALID:"
            + control
        )

    source = load_validated_phase4_source(
        output=output_root,
        route_id=str(
            job["source_route_id"]
        ),
        expected_artifact_fingerprint=str(
            job[
                "source_route_artifact_fingerprint"
            ]
        ),
    )

    baseline = load_validated_phase4_source(
        output=output_root,
        route_id=str(
            job["matched_s0_route_id"]
        ),
        expected_artifact_fingerprint=str(
            job[
                "matched_s0_artifact_fingerprint"
            ]
        ),
    )

    if str(source["dataset"]) != dataset:
        raise RuntimeError(
            "PHASE5_SOIL_SOURCE_DATASET_MISMATCH"
        )

    if str(source["fold"]) != fold:
        raise RuntimeError(
            "PHASE5_SOIL_SOURCE_FOLD_MISMATCH"
        )

    if int(source["seed"]) != seed:
        raise RuntimeError(
            "PHASE5_SOIL_SOURCE_SEED_MISMATCH"
        )

    if baseline.get("route") != "supervised":
        raise RuntimeError(
            "PHASE5_SOIL_BASELINE_NOT_SUPERVISED"
        )

    required_source_route = {
        "soil_representation_sensitivity_control": (
            "soil_direct"
        ),
        "missing_aware_ablation_suite_control": (
            "missing_aware"
        ),
    }[control]

    if source.get("route") != required_source_route:
        raise RuntimeError(
            "PHASE5_SOIL_SOURCE_ROUTE_MISMATCH:"
            f"{control}:{source.get('route')}"
        )

    train, validation = _split_development(
        frame,
        fold,
    )

    weather_features, soil_features = (
        _features(frame)
    )

    if not soil_features:
        raise RuntimeError(
            "PHASE5_SOIL_FEATURES_EMPTY"
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
        train.target_yield.to_numpy(float)
    )
    validation_target_array = (
        validation.target_yield.to_numpy(
            float
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

    train_soil_raw = (
        train[soil_features].to_numpy(float)
    )
    validation_soil_raw = (
        validation[soil_features]
        .to_numpy(float)
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

    train_available = torch.tensor(
        train_available_array,
        dtype=torch.float32,
    )
    validation_available = torch.tensor(
        validation_available_array,
        dtype=torch.float32,
    )

    extension = dict(
        job["soil_extension_contract"]
    )

    hidden = int(
        extension.get("hidden", 32)
    )
    representation_dim = int(
        extension.get(
            "representation_dim",
            32,
        )
    )
    component_cap = int(
        extension.get(
            "compact_components",
            8,
        )
    )

    primary_variant, variants = (
        _variant_contracts(control)
    )

    results: dict[str, Any] = {}
    transformers: dict[
        str,
        SoilRepresentationTransformer,
    ] = {}
    checkpoint_payloads: dict[
        str,
        dict[str, Any],
    ] = {}
    variant_records: list[
        dict[str, Any]
    ] = []

    for variant_index, variant in enumerate(
        variants
    ):
        variant_id = str(
            variant["variant_id"]
        )
        method = str(
            variant[
                "soil_representation"
            ]
        )
        dropout = float(
            variant[
                "soil_dropout_probability"
            ]
        )
        explicit_mask = bool(
            variant[
                "explicit_availability_mask"
            ]
        )

        transformer = (
            SoilRepresentationTransformer(
                method=method,
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

        train_soil = torch.tensor(
            train_soil_array,
            dtype=torch.float32,
        )
        validation_soil = torch.tensor(
            validation_soil_array,
            dtype=torch.float32,
        )

        variant_seed = (
            seed + variant_index
        )

        result = fit_flexible_soil_model(
            route="soil_control",
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
            seed=variant_seed,
            hidden=hidden,
            representation_dim=(
                representation_dim
            ),
            soil_dropout_probability=(
                dropout
            ),
            explicit_availability_mask=(
                explicit_mask
            ),
        )

        payload = build_soil_checkpoint(
            result=result,
            route="soil_control",
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
            soil_representation=method,
            soil_dropout_probability=(
                dropout
            ),
            explicit_availability_mask=(
                explicit_mask
            ),
            weather_features=(
                weather_features
            ),
            soil_features=soil_features,
        )

        payload["control"] = control
        payload["variant_id"] = variant_id
        payload["scientific_role"] = (
            variant["scientific_role"]
        )
        payload["variant_seed"] = (
            variant_seed
        )

        validation_mae = float(
            np.mean(
                np.abs(
                    result.prediction
                    - validation_target_array
                )
            )
        )

        results[variant_id] = result
        transformers[variant_id] = (
            transformer
        )
        checkpoint_payloads[
            variant_id
        ] = payload

        variant_records.append(
            {
                "variant_id": variant_id,
                "primary": (
                    variant_id
                    == primary_variant
                ),
                "scientific_role": (
                    variant[
                        "scientific_role"
                    ]
                ),
                "soil_representation": (
                    method
                ),
                "soil_dropout_probability": (
                    dropout
                ),
                "explicit_availability_mask": (
                    explicit_mask
                ),
                "variant_seed": variant_seed,
                "validation_mae": (
                    validation_mae
                ),
                "prediction_std": float(
                    np.std(
                        result.prediction
                    )
                ),
                "optimizer_steps": (
                    result.optimizer_steps
                ),
                "parameter_delta_l2": (
                    result
                    .parameter_delta_l2
                ),
                "fit_time_seconds": (
                    result.fit_time_seconds
                ),
                "inference_time_seconds": (
                    result
                    .inference_time_seconds
                ),
                "parameter_count": (
                    result.parameter_count
                ),
                "selected_epoch": (
                    result.selected_epoch
                ),
                "observed_soil_fraction_train": (
                    result
                    .observed_soil_fraction_train
                ),
                "observed_soil_fraction_validation": (
                    result
                    .observed_soil_fraction_validation
                ),
            }
        )

    if primary_variant not in results:
        raise RuntimeError(
            "PHASE5_SOIL_PRIMARY_VARIANT_MISSING"
        )

    primary_result = results[
        primary_variant
    ]
    primary_payload = (
        checkpoint_payloads[
            primary_variant
        ]
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

    torch.save(
        {
            "model_family": (
                "phase5_soil_control_bundle"
            ),
            "control": control,
            "primary_variant": (
                primary_variant
            ),
            "primary_payload": (
                primary_payload
            ),
            "variant_payloads": (
                checkpoint_payloads
            ),
            "variant_selection_policy": (
                "PREDECLARED_NOT_"
                "PERFORMANCE_SELECTED"
            ),
        },
        checkpoint,
    )

    joblib.dump(
        {
            "weather_preprocessor": (
                weather_preprocessor
            ),
            "soil_transformers": (
                transformers
            ),
            "weather_features": (
                weather_features
            ),
            "soil_features": (
                soil_features
            ),
            "primary_variant": (
                primary_variant
            ),
            "variant_selection_policy": (
                "PREDECLARED_NOT_"
                "PERFORMANCE_SELECTED"
            ),
        },
        preprocessor_path,
    )

    np.save(
        prediction_path,
        primary_result.prediction,
    )
    np.save(
        representation_path,
        primary_result.representation,
    )

    from .phase5_execution import (
        _atomic_json,
        _sha256,
    )

    _atomic_json(
        loss_path,
        {
            "control": control,
            "primary_variant": (
                primary_variant
            ),
            "variant_selection_policy": (
                "PREDECLARED_NOT_"
                "PERFORMANCE_SELECTED"
            ),
            "variants": {
                variant_id: (
                    results[
                        variant_id
                    ].loss_ledger
                )
                for variant_id
                in results
            },
        },
    )

    loaded_bundle = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    primary_transformer = (
        transformers[primary_variant]
    )
    replay_soil_array = (
        primary_transformer.transform(
            validation_soil_raw
        )
    )
    replay_soil = torch.tensor(
        replay_soil_array,
        dtype=torch.float32,
    )

    replay_prediction, _ = (
        predict_soil_checkpoint(
            payload=loaded_bundle[
                "primary_payload"
            ],
            weather=validation_weather,
            soil=replay_soil,
            soil_available=(
                validation_available
            ),
        )
    )

    replay_error = float(
        np.max(
            np.abs(
                replay_prediction
                - primary_result.prediction
            )
        )
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "PHASE5_SOIL_CHECKPOINT_"
            "REPLAY_MISMATCH"
        )

    primary_record = next(
        item
        for item in variant_records
        if item["variant_id"]
        == primary_variant
    )

    if control == (
        "missing_aware_ablation_suite_control"
    ):
        parameter_counts = {
            int(item["parameter_count"])
            for item in variant_records
        }

        if len(parameter_counts) != 1:
            raise RuntimeError(
                "PHASE5_MISSING_AWARE_"
                "PARAMETER_MATCHING_FAILED"
            )

    record = {
        "job_id": job["job_id"],
        "phase": "phase5",
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "control": control,
        "required_by_route": (
            job["required_by_route"]
        ),
        "status": (
            "ENGINEERING_ACCEPTED"
        ),
        "source_route_id": (
            source["route_id"]
        ),
        "matched_s0_route_id": (
            baseline["route_id"]
        ),
        "validation_mae": (
            primary_record[
                "validation_mae"
            ]
        ),
        "optimizer_steps": (
            primary_result.optimizer_steps
        ),
        "parameter_delta_l2": (
            primary_result
            .parameter_delta_l2
        ),
        "fit_time_seconds": float(
            sum(
                result.fit_time_seconds
                for result in results.values()
            )
        ),
        "inference_time_seconds": float(
            sum(
                result
                .inference_time_seconds
                for result in results.values()
            )
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "control_metadata": {
            "primary_variant": (
                primary_variant
            ),
            "variant_selection_policy": (
                "PREDECLARED_NOT_"
                "PERFORMANCE_SELECTED"
            ),
            "variants": (
                variant_records
            ),
            "source_validation_mae": (
                source.get(
                    "validation_mae"
                )
            ),
            "matched_s0_validation_mae": (
                baseline.get(
                    "validation_mae"
                )
            ),
            "selection_scope": (
                "OUTER_TRAIN_INNER_"
                "VALIDATION_ONLY"
            ),
        },
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
        "selection_scope": (
            "OUTER_TRAIN_INNER_"
            "VALIDATION_ONLY"
        ),
        "outer_test_used": False,
        "seed_selected_by_performance": (
            False
        ),
        "scientific_status": (
            "CONTROL_EXECUTED_NOT_INTERPRETED"
        ),
    }

    _atomic_json(
        output / "job_record.json",
        record,
    )

    _atomic_json(
        output / "completion_marker.json",
        {
            "job_id": job["job_id"],
            "status": (
                "ENGINEERING_ACCEPTED"
            ),
            "outer_test_used": False,
        },
    )

    return record
