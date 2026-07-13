from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import joblib
import numpy as np
import pandas as pd
import torch

from .soil_missing_aware import (
    SoilRepresentationTransformer,
    build_soil_checkpoint,
    fit_flexible_soil_model,
    predict_soil_checkpoint,
    soil_availability_mask,
)


SOIL_OUTER_ROUTES = {
    "soil_direct",
    "missing_aware",
}


def _scenario_metrics(
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

    if len(target) == 0:
        return {
            "status": "NOT_ESTIMABLE",
            "rows": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    error = prediction - target

    mae = float(
        np.mean(np.abs(error))
    )
    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    denominator = float(
        np.sum(
            (
                target
                - np.mean(target)
            )
            ** 2
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
            "REPAIR_OUTER_SOIL_METRIC_NONFINITE"
        )

    return {
        "status": "ESTIMATED",
        "rows": int(len(target)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def execute_repair_outer_soil_job(
    config,
    output: Path,
    job: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    from . import repair_outer_executor as outer

    route = str(job["route"])
    dataset = str(job["dataset"])
    fold = str(job["fold"])
    seed = int(job["seed"])
    outer_test_year = int(
        job["outer_test_year"]
    )

    if route not in SOIL_OUTER_ROUTES:
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_ROUTE_INVALID:"
            + route
        )

    if fold != f"test_{outer_test_year}":
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_FOLD_YEAR_MISMATCH"
        )

    frame = outer.load_primary_frame(
        config,
        output,
        dataset,
    )

    (
        train,
        calibration,
        test,
    ) = outer._partitions(
        frame=frame,
        outer_test_year=outer_test_year,
    )

    weather_features, soil_features = (
        outer._features(frame)
    )

    if not soil_features:
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_FEATURES_EMPTY"
        )

    (
        train_weather,
        calibration_weather,
        test_weather,
        weather_preprocessor,
    ) = outer._preprocess(
        train=train,
        calibration=calibration,
        test=test,
        features=weather_features,
    )

    train_target_array = (
        train.target_yield.to_numpy(float)
    )
    calibration_target_array = (
        calibration.target_yield.to_numpy(
            float
        )
    )
    test_target_array = (
        test.target_yield.to_numpy(float)
    )

    train_target = torch.tensor(
        train_target_array,
        dtype=torch.float32,
    )
    calibration_target = torch.tensor(
        calibration_target_array,
        dtype=torch.float32,
    )

    train_soil_raw = (
        train[soil_features]
        .to_numpy(float)
    )
    calibration_soil_raw = (
        calibration[soil_features]
        .to_numpy(float)
    )
    test_soil_raw = (
        test[soil_features]
        .to_numpy(float)
    )

    phase4_record = (
        outer._phase4_route_record(
            output=output,
            job=job,
        )
    )

    soil_contract = dict(
        config.raw.get(
            "soil_missing_aware",
            {},
        )
    )

    method = str(
        phase4_record.get(
            "soil_representation",
            soil_contract.get(
                "primary_soil_representation",
                "raw",
            ),
        )
    )
    component_cap = int(
        phase4_record.get(
            "soil_compact_components",
            soil_contract.get(
                "compact_components",
                8,
            ),
        )
    )
    hidden = int(
        phase4_record.get(
            "soil_hidden",
            soil_contract.get(
                "hidden",
                32,
            ),
        )
    )
    representation_dim = int(
        phase4_record.get(
            "soil_representation_dim",
            soil_contract.get(
                "representation_dim",
                32,
            ),
        )
    )

    dropout_probability = (
        0.0
        if route == "soil_direct"
        else float(
            phase4_record.get(
                "soil_dropout_probability",
                soil_contract.get(
                    "primary_dropout_probability",
                    0.5,
                ),
            )
        )
    )

    explicit_mask = bool(
        phase4_record.get(
            "explicit_availability_mask",
            soil_contract.get(
                "explicit_availability_mask",
                True,
            ),
        )
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
    calibration_soil_array = (
        transformer.transform(
            calibration_soil_raw
        )
    )
    test_soil_array = (
        transformer.transform(
            test_soil_raw
        )
    )

    train_available_array = (
        soil_availability_mask(
            train_soil_raw
        )
    )
    calibration_available_array = (
        soil_availability_mask(
            calibration_soil_raw
        )
    )
    test_available_array = (
        soil_availability_mask(
            test_soil_raw
        )
    )

    train_soil = torch.tensor(
        train_soil_array,
        dtype=torch.float32,
    )
    calibration_soil = torch.tensor(
        calibration_soil_array,
        dtype=torch.float32,
    )
    test_soil = torch.tensor(
        test_soil_array,
        dtype=torch.float32,
    )

    train_available = torch.tensor(
        train_available_array,
        dtype=torch.float32,
    )
    calibration_available = (
        torch.tensor(
            calibration_available_array,
            dtype=torch.float32,
        )
    )
    test_available = torch.tensor(
        test_available_array,
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
            calibration_weather
        ),
        validation_soil=(
            calibration_soil
        ),
        validation_soil_available=(
            calibration_available
        ),
        validation_target=(
            calibration_target
        ),
        contract=config.neural_training,
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

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = root / "model.pt"
    preprocessor_path = (
        root / "preprocessor.joblib"
    )
    predictions_path = (
        root / "predictions.csv"
    )
    calibration_path = (
        root / "calibration_predictions.csv"
    )
    metrics_path = root / "metrics.json"
    folds_path = (
        root / "fold_assignments.csv"
    )
    manifest_path = (
        root / "manifest.json"
    )
    loss_path = (
        root / "loss_ledger.json"
    )
    test_representation_path = (
        root / "test_representation.npy"
    )
    scenario_predictions_path = (
        root / "scenario_predictions.csv"
    )
    scenario_metrics_path = (
        root / "scenario_metrics.json"
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
            soil_representation=method,
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

    checkpoint_payload.update(
        {
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "teacher_refit_performed": (
                False
            ),
            "outer_scenarios": [
                "native",
                "weather_only",
                "complete_case",
            ],
        }
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
            "complete_case_rule": (
                "ALL_NATIVE_SOIL_"
                "FEATURES_PRESENT"
            ),
        },
        preprocessor_path,
    )

    native_prediction, native_representation = (
        predict_soil_checkpoint(
            payload=checkpoint_payload,
            weather=test_weather,
            soil=test_soil,
            soil_available=(
                test_available
            ),
        )
    )

    weather_only_prediction, _ = (
        predict_soil_checkpoint(
            payload=checkpoint_payload,
            weather=test_weather,
            soil=torch.zeros_like(
                test_soil
            ),
            soil_available=torch.zeros_like(
                test_available
            ),
        )
    )

    complete_case_mask = (
        np.isfinite(
            test_soil_raw
        ).all(axis=1)
    )

    complete_indices = np.flatnonzero(
        complete_case_mask
    )

    if len(complete_indices) > 0:
        complete_index_tensor = (
            torch.tensor(
                complete_indices,
                dtype=torch.long,
            )
        )

        (
            complete_case_prediction,
            _,
        ) = predict_soil_checkpoint(
            payload=checkpoint_payload,
            weather=test_weather[
                complete_index_tensor
            ],
            soil=test_soil[
                complete_index_tensor
            ],
            soil_available=test_available[
                complete_index_tensor
            ],
        )
    else:
        complete_case_prediction = (
            np.asarray([], dtype=float)
        )

    replay_prediction, _ = (
        predict_soil_checkpoint(
            payload=torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            ),
            weather=test_weather,
            soil=test_soil,
            soil_available=(
                test_available
            ),
        )
    )

    replay_error = float(
        np.max(
            np.abs(
                replay_prediction
                - native_prediction
            )
        )
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "REPAIR_OUTER_SOIL_"
            "CHECKPOINT_REPLAY_MISMATCH"
        )

    calibration_frame = pd.DataFrame(
        {
            "sample_id": (
                calibration.sample_id
                .astype(str)
                .to_numpy()
            ),
            "y_true": (
                calibration_target_array
            ),
            "y_pred": (
                result.prediction
            ),
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
            "y_true": test_target_array,
            "y_pred": native_prediction,
        }
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    native_metrics = _scenario_metrics(
        test_target_array,
        native_prediction,
    )
    weather_only_metrics = (
        _scenario_metrics(
            test_target_array,
            weather_only_prediction,
        )
    )
    complete_case_metrics = (
        _scenario_metrics(
            test_target_array[
                complete_case_mask
            ],
            complete_case_prediction,
        )
    )

    outer._atomic_json(
        metrics_path,
        {
            "mae": native_metrics["mae"],
            "rmse": (
                native_metrics["rmse"]
            ),
            "r2": native_metrics["r2"],
            "rows": native_metrics["rows"],
            "scenario": "native",
        },
    )

    scenario_frames = [
        pd.DataFrame(
            {
                "sample_id": (
                    test.sample_id
                    .astype(str)
                    .to_numpy()
                ),
                "scenario": "native",
                "y_true": test_target_array,
                "y_pred": (
                    native_prediction
                ),
                "soil_available": (
                    test_available_array
                ),
            }
        ),
        pd.DataFrame(
            {
                "sample_id": (
                    test.sample_id
                    .astype(str)
                    .to_numpy()
                ),
                "scenario": (
                    "weather_only"
                ),
                "y_true": test_target_array,
                "y_pred": (
                    weather_only_prediction
                ),
                "soil_available": 0.0,
            }
        ),
    ]

    if len(complete_indices) > 0:
        scenario_frames.append(
            pd.DataFrame(
                {
                    "sample_id": (
                        test.iloc[
                            complete_indices
                        ].sample_id
                        .astype(str)
                        .to_numpy()
                    ),
                    "scenario": (
                        "complete_case"
                    ),
                    "y_true": (
                        test_target_array[
                            complete_case_mask
                        ]
                    ),
                    "y_pred": (
                        complete_case_prediction
                    ),
                    "soil_available": 1.0,
                }
            )
        )

    pd.concat(
        scenario_frames,
        ignore_index=True,
    ).to_csv(
        scenario_predictions_path,
        index=False,
    )

    outer._atomic_json(
        scenario_metrics_path,
        {
            "primary_scenario": "native",
            "scenario_selection_policy": (
                "PREDECLARED_NOT_"
                "PERFORMANCE_SELECTED"
            ),
            "native": native_metrics,
            "weather_only": (
                weather_only_metrics
            ),
            "complete_case": (
                complete_case_metrics
            ),
        },
    )

    fold_assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": (
                        train.sample_id
                        .astype(str)
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
                    "year": (
                        calibration.year
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "sample_id": (
                        test.sample_id
                        .astype(str)
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
        native_representation,
    )

    outer._atomic_json(
        loss_path,
        {
            "route": route,
            "soil_representation": method,
            "soil_dropout_probability": (
                dropout_probability
            ),
            "explicit_availability_mask": (
                explicit_mask
            ),
            "epochs": result.loss_ledger,
        },
    )

    manifest = {
        "job_id": job["job_id"],
        "dataset": dataset,
        "route": route,
        "fold": fold,
        "seed": seed,
        "outer_test_year": (
            outer_test_year
        ),
        "calibration_year": (
            outer_test_year - 1
        ),
        "weather_features": (
            weather_features
        ),
        "soil_features": soil_features,
        "feature_order_frozen": True,
        "train_rows": int(len(train)),
        "calibration_rows": int(
            len(calibration)
        ),
        "test_rows": int(len(test)),
        "target_scaling": (
            "TRAIN_ONLY_STANDARDISATION"
        ),
        "target_mean": (
            result.target_mean
        ),
        "target_scale": (
            result.target_scale
        ),
        "soil_representation": method,
        "soil_dropout_probability": (
            dropout_probability
        ),
        "explicit_availability_mask": (
            explicit_mask
        ),
        "outer_scenarios": {
            "native": {
                "rows": int(len(test)),
            },
            "weather_only": {
                "rows": int(len(test)),
            },
            "complete_case": {
                "rows": int(
                    complete_case_mask.sum()
                ),
            },
        },
        "primary_outer_scenario": (
            "native"
        ),
        "scenario_selection_policy": (
            "PREDECLARED_NOT_"
            "PERFORMANCE_SELECTED"
        ),
        "phase4_route_id": (
            phase4_record.get(
                "route_id"
            )
        ),
        "phase4_route_manifest_fingerprint": (
            outer._phase4_manifest(
                output
            ).get(
                "phase4_route_manifest_fingerprint"
            )
        ),
        "teacher_lineage": {},
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
            outer._frame_fingerprint(
                frame
            )
        ),
    }
    outer._atomic_json(
        manifest_path,
        manifest,
    )

    outer._atomic_json(
        root / "completion_marker.json",
        {
            "job_id": job["job_id"],
            "status": (
                "ENGINEERING_ACCEPTED"
            ),
            "outer_test_used": True,
            "teacher_refit_performed": (
                False
            ),
            "checkpoint_replay_max_abs_error": (
                replay_error
            ),
            "outer_scenarios_complete": (
                True
            ),
        },
    )

    return {
        "status": (
            "ENGINEERING_ACCEPTED"
        ),
        "mae": native_metrics["mae"],
        "rmse": native_metrics["rmse"],
        "r2": native_metrics["r2"],
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
        "predictions": outer._relative(
            output=output,
            path=predictions_path,
        ),
        "predictions_sha256": (
            outer.sha256_file(
                predictions_path
            )
        ),
        "metrics": outer._relative(
            output=output,
            path=metrics_path,
        ),
        "metrics_sha256": (
            outer.sha256_file(
                metrics_path
            )
        ),
        "fold_assignments": (
            outer._relative(
                output=output,
                path=folds_path,
            )
        ),
        "fold_assignments_sha256": (
            outer.sha256_file(
                folds_path
            )
        ),
        "checkpoint": outer._relative(
            output=output,
            path=checkpoint,
        ),
        "checkpoint_sha256": (
            outer.sha256_file(
                checkpoint
            )
        ),
        "preprocessor": outer._relative(
            output=output,
            path=preprocessor_path,
        ),
        "preprocessor_sha256": (
            outer.sha256_file(
                preprocessor_path
            )
        ),
        "manifest": outer._relative(
            output=output,
            path=manifest_path,
        ),
        "manifest_sha256": (
            outer.sha256_file(
                manifest_path
            )
        ),
        "loss_ledger": outer._relative(
            output=output,
            path=loss_path,
        ),
        "loss_ledger_sha256": (
            outer.sha256_file(
                loss_path
            )
        ),
        "calibration_predictions": (
            outer._relative(
                output=output,
                path=calibration_path,
            )
        ),
        "calibration_predictions_sha256": (
            outer.sha256_file(
                calibration_path
            )
        ),
        "test_representation": (
            outer._relative(
                output=output,
                path=(
                    test_representation_path
                ),
            )
        ),
        "test_representation_sha256": (
            outer.sha256_file(
                test_representation_path
            )
        ),
        "scenario_predictions": (
            outer._relative(
                output=output,
                path=(
                    scenario_predictions_path
                ),
            )
        ),
        "scenario_predictions_sha256": (
            outer.sha256_file(
                scenario_predictions_path
            )
        ),
        "scenario_metrics": (
            outer._relative(
                output=output,
                path=scenario_metrics_path,
            )
        ),
        "scenario_metrics_sha256": (
            outer.sha256_file(
                scenario_metrics_path
            )
        ),
        "outer_scenarios": {
            "native": native_metrics,
            "weather_only": (
                weather_only_metrics
            ),
            "complete_case": (
                complete_case_metrics
            ),
        },
    }
