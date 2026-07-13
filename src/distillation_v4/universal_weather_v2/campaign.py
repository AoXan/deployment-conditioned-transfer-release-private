from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from .artifacts import (
    atomic_json,
    job_is_complete,
    mark_complete,
    record_failure,
    stable_hash,
)
from .data import (
    load_dataset_v2,
)
from .training import (
    ROUTES,
    build_student_v2,
    evaluate_student,
    load_dataset_and_arrays,
    metric_bundle,
    prepare_arrays,
    set_seed,
    split_frame_by_ids,
    train_student_route_v2,
    train_teacher_v2,
)


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def limit_arrays_for_smoke(
    arrays: dict[str, Any],
    *,
    train_limit: int = 1024,
    validation_limit: int = 256,
    test_limit: int = 256,
) -> dict[str, Any]:
    """Deterministically limit smoke data only.

    Preprocessors remain fitted on the original source-train split.
    Formal training never calls this helper.
    """

    limited = dict(arrays)

    limits = {
        "train": train_limit,
        "validation": validation_limit,
        "test": test_limit,
    }

    for split_name, limit in (
        limits.items()
    ):
        source = arrays[split_name]

        row_count = int(
            source["target"].shape[0]
        )

        if row_count <= limit:
            limited[split_name] = source
            continue

        indices = np.linspace(
            0,
            row_count - 1,
            num=limit,
            dtype=np.int64,
        )

        limited_split = {
            "weather": source[
                "weather"
            ][indices],
            "target": source[
                "target"
            ][indices],
            "privileged": None,
        }

        if (
            source["privileged"]
            is not None
        ):
            limited_split[
                "privileged"
            ] = source[
                "privileged"
            ][indices]

        limited[split_name] = (
            limited_split
        )

    limited[
        "smoke_sampling"
    ] = {
        "strategy": (
            "DETERMINISTIC_EVENLY_SPACED"
        ),
        "train_rows": int(
            limited["train"][
                "target"
            ].shape[0]
        ),
        "validation_rows": int(
            limited["validation"][
                "target"
            ].shape[0]
        ),
        "test_rows": int(
            limited["test"][
                "target"
            ].shape[0]
        ),
        "scientific_evidence": False,
    }

    return limited



def source_training_job(
    *,
    repository_root: Path,
    output_root: Path,
    registry: dict[str, Any],
    feature_contracts: dict[str, Any],
    split_contracts: dict[str, Any],
    dataset_id: str,
    route: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(
            f"INVALID_SOURCE_ROUTE:{route}"
        )

    hyperparameters = {
        "epochs": 2 if smoke else 80,
        "patience": 2 if smoke else 12,
        "batch_size": (
            64 if smoke else 256
        ),
        "learning_rate": 1e-3,
        "smoke": smoke,
    }

    job = {
        "phase": "source_training",
        "dataset": dataset_id,
        "route": route,
        "seed": seed,
        "smoke": smoke,
    }

    fingerprint = stable_hash(
        {
            "job": job,
            "feature_contract": (
                feature_contracts[
                    dataset_id
                ]
            ),
            "split_contract": (
                split_contracts[
                    dataset_id
                ]
            ),
            "hyperparameters": (
                hyperparameters
            ),
        }
    )

    job_directory = (
        output_root
        / (
            "smoke"
            if smoke
            else "jobs"
        )
        / "source"
        / dataset_id
        / route
        / f"seed_{seed}"
    )

    if job_is_complete(
        job_directory,
        fingerprint=fingerprint,
    ):
        return {
            "status": "RESUMED_COMPLETE",
            "job": job,
            "fingerprint": fingerprint,
        }

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset, arrays = (
        load_dataset_and_arrays(
            repository_root=(
                repository_root
            ),
            dataset_id=dataset_id,
            registry_entry=(
                registry[dataset_id]
            ),
            split_contract=(
                split_contracts[
                    dataset_id
                ]
            ),
        )
    )

    if smoke:
        arrays = limit_arrays_for_smoke(
            arrays
        )

    device = choose_device()

    teacher_result = None
    teacher = None

    if route != "supervised":
        teacher_result = (
            train_teacher_v2(
                arrays=arrays,
                feature_names=list(
                    dataset.weather_features
                ),
                seed=seed,
                epochs=(
                    hyperparameters[
                        "epochs"
                    ]
                ),
                patience=(
                    hyperparameters[
                        "patience"
                    ]
                ),
                batch_size=(
                    hyperparameters[
                        "batch_size"
                    ]
                ),
                learning_rate=(
                    hyperparameters[
                        "learning_rate"
                    ]
                ),
                device=device,
                checkpoint_path=(
                    job_directory
                    / "teacher.pt"
                ),
                metadata={
                    "dataset_id": (
                        dataset_id
                    ),
                    "seed": seed,
                    "smoke_only": smoke,
                    "scientific_evidence": (
                        not smoke
                    ),
                },
            )
        )

        teacher = teacher_result[
            "model"
        ]

    student_result = (
        train_student_route_v2(
            route=route,
            arrays=arrays,
            feature_names=list(
                dataset.weather_features
            ),
            teacher=teacher,
            seed=seed,
            epochs=(
                hyperparameters["epochs"]
            ),
            patience=(
                hyperparameters[
                    "patience"
                ]
            ),
            batch_size=(
                hyperparameters[
                    "batch_size"
                ]
            ),
            learning_rate=(
                hyperparameters[
                    "learning_rate"
                ]
            ),
            device=device,
            checkpoint_path=(
                job_directory
                / "student.pt"
            ),
            metadata={
                "dataset_id": dataset_id,
                "seed": seed,
                "smoke_only": smoke,
                "scientific_evidence": (
                    not smoke
                ),
                "source_selection_only": (
                    True
                ),
            },
        )
    )

    result = {
        "status": "COMPLETED",
        "job": job,
        "fingerprint": fingerprint,
        "device": str(device),
        "validation_metrics": (
            student_result[
                "validation_metrics"
            ]
        ),
        "test_metrics": (
            student_result[
                "test_metrics"
            ]
        ),
        "teacher_frozen": (
            student_result[
                "teacher_frozen"
            ]
        ),
        "teacher_checkpoint": (
            None
            if teacher_result is None
            else teacher_result[
                "manifest"
            ][
                "checkpoint_path"
            ]
        ),
        "student_checkpoint": (
            student_result[
                "manifest"
            ][
                "checkpoint_path"
            ]
        ),
        "smoke_only_not_scientific_evidence": (
            smoke
        ),
        "smoke_sampling": (
            arrays.get(
                "smoke_sampling"
            )
            if smoke
            else None
        ),
    }

    atomic_json(
        job_directory
        / "result.json",
        result,
    )

    mark_complete(
        job_directory,
        fingerprint=fingerprint,
        result=result,
    )

    return result


def checkpoint_reload_smoke(
    *,
    checkpoint_path: Path,
    arrays: dict[str, Any],
    feature_names: list[str],
) -> dict[str, Any]:
    from .artifacts import (
        load_checkpoint,
    )

    from .training import (
        build_student_v2,
    )

    device = choose_device()

    first = build_student_v2().to(
        device
    )

    load_checkpoint(
        path=checkpoint_path,
        model=first,
        map_location=device,
    )

    first_metrics, first_prediction = (
        evaluate_student(
            model=first,
            arrays=arrays["test"],
            feature_names=feature_names,
            device=device,
            batch_size=64,
        )
    )

    second = build_student_v2().to(
        device
    )

    load_checkpoint(
        path=checkpoint_path,
        model=second,
        map_location=device,
    )

    second_metrics, second_prediction = (
        evaluate_student(
            model=second,
            arrays=arrays["test"],
            feature_names=feature_names,
            device=device,
            batch_size=64,
        )
    )

    maximum_difference = float(
        np.max(
            np.abs(
                first_prediction
                - second_prediction
            )
        )
    )

    return {
        "all_passed": (
            maximum_difference
            <= 1e-7
            and np.isfinite(
                first_prediction
            ).all()
            and float(
                np.std(
                    first_prediction
                )
            )
            > 0
        ),
        "maximum_prediction_difference": (
            maximum_difference
        ),
        "first_metrics": first_metrics,
        "second_metrics": second_metrics,
    }


def target_transfer_smoke(
    *,
    repository_root: Path,
    registry: dict[str, Any],
    split_contracts: dict[str, Any],
    target_dataset_id: str,
    source_checkpoint: Path,
) -> dict[str, Any]:
    from .artifacts import (
        load_checkpoint,
    )

    target_dataset, arrays = (
        load_dataset_and_arrays(
            repository_root=(
                repository_root
            ),
            dataset_id=(
                target_dataset_id
            ),
            registry_entry=(
                registry[
                    target_dataset_id
                ]
            ),
            split_contract=(
                split_contracts[
                    target_dataset_id
                ]
            ),
        )
    )

    device = choose_device()

    model = build_student_v2().to(
        device
    )

    source_payload = load_checkpoint(
        path=source_checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )

    source_state_dict = {
        key: value.detach().cpu().clone()
        for key, value
        in source_payload[
            "state_dict"
        ].items()
    }

    zero_shot_metrics, prediction = (
        evaluate_student(
            model=model,
            arrays=arrays["test"],
            feature_names=list(
                target_dataset.weather_features
            ),
            device=device,
            batch_size=64,
        )
    )

    fine_tune_result = (
        train_student_route_v2(
            route="supervised",
            arrays=arrays,
            feature_names=list(
                target_dataset.weather_features
            ),
            teacher=None,
            seed=101,
            epochs=1,
            patience=1,
            batch_size=64,
            learning_rate=1e-4,
            device=device,
            checkpoint_path=(
                source_checkpoint.parent
                / (
                    "target_smoke_"
                    + target_dataset_id
                    + ".pt"
                )
            ),
            metadata={
                "target_dataset_id": (
                    target_dataset_id
                ),
                "smoke_only": True,
                "target_test_used_for_selection": (
                    False
                ),
                "initialisation": (
                    "FROZEN_SOURCE_CHECKPOINT"
                ),
                "source_checkpoint": str(
                    source_checkpoint
                ),
            },
            initial_state_dict=(
                source_state_dict
            ),
        )
    )

    return {
        "all_passed": (
            np.isfinite(
                prediction
            ).all()
            and float(
                np.std(prediction)
            )
            > 0
            and fine_tune_result[
                "teacher_frozen"
            ]
        ),
        "target_dataset_id": (
            target_dataset_id
        ),
        "zero_shot_metrics": (
            zero_shot_metrics
        ),
        "fine_tune_metrics": (
            fine_tune_result[
                "test_metrics"
            ]
        ),
        "target_test_used_for_selection": (
            False
        ),
        "smoke_only_not_scientific_evidence": (
            True
        ),
    }


def roseworthy_static_smoke(
    *,
    repository_root: Path,
    registry_entry: dict[str, Any],
) -> dict[str, Any]:
    dataset = load_dataset_v2(
        dataset_id="roseworthy",
        registry_entry={
            **registry_entry,
            "weather_features": [
                "elevation_m",
                "longitude",
                "latitude",
            ],
            "privileged_features": [
                "apsoil_pawc_mm",
                "apsoil_bd_top",
                "apsoil_ll15_top",
                "apsoil_dul_top",
                "apsoil_sat_top",
                "apsoil_depth_total_mm",
            ],
        },
        repository_root=(
            repository_root
        ),
    )

    frame = dataset.frame.copy()

    fold_path = (
        repository_root
        / "data"
        / "derived"
        / "data_nursery_v1"
        / "tracks"
        / "precision_roseworthy"
        / "folds"
        / (
            "roseworthy_e5_point_yield__"
            "temporal_field_holdout__test_2008.csv"
        )
    )

    folds = pd.read_csv(
        fold_path
    )

    joined = frame.merge(
        folds,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    train = joined.loc[
        joined["split"]
        == "train"
    ].copy()

    validation = joined.loc[
        joined["split"]
        == "validation"
    ].copy()

    test = joined.loc[
        joined["split"]
        == "test"
    ].copy()

    if validation.empty:
        train_years = sorted(
            train["year"].unique()
        )

        validation_year = (
            train_years[-1]
        )

        validation = train.loc[
            train["year"]
            == validation_year
        ].copy()

        train = train.loc[
            train["year"]
            != validation_year
        ].copy()

    features = [
        "elevation_m",
        "longitude",
        "latitude",
        "apsoil_pawc_mm",
        "apsoil_bd_top",
        "apsoil_ll15_top",
        "apsoil_dul_top",
        "apsoil_sat_top",
        "apsoil_depth_total_mm",
    ]

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                Ridge(alpha=1.0),
            ),
        ]
    )

    model.fit(
        train[features],
        train["target_value"],
    )

    prediction = model.predict(
        test[features]
    )

    metrics = metric_bundle(
        test[
            "target_value"
        ].to_numpy(),
        prediction,
    )

    return {
        "all_passed": (
            np.isfinite(
                prediction
            ).all()
        ),
        "metrics": metrics,
        "weather_used": False,
        "scientific_role": (
            "NEGATIVE_IDENTIFIABILITY_STRESS_CASE"
        ),
        "smoke_only_not_scientific_evidence": (
            True
        ),
    }


def run_all_smokes(
    *,
    repository_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    registry_payload = load_json(
        output_root
        / "control"
        / "dataset_registry_v2.json"
    )

    feature_payload = load_json(
        output_root
        / "control"
        / "feature_contract_v2.json"
    )

    split_payload = load_json(
        output_root
        / "control"
        / "split_contracts_v2.json"
    )

    registry = registry_payload[
        "datasets"
    ]

    feature_contracts = (
        feature_payload["datasets"]
    )

    split_contracts = (
        split_payload["datasets"]
    )

    source_dataset = (
        "PRIMARY_G2F_MAIZE"
    )

    route_results = {}

    for route in [
        "supervised",
        "prediction_kd",
        "representation_kd",
        "combined_kd",
        "missing_aware",
    ]:
        route_results[route] = (
            source_training_job(
                repository_root=(
                    repository_root
                ),
                output_root=(
                    output_root
                ),
                registry=registry,
                feature_contracts=(
                    feature_contracts
                ),
                split_contracts=(
                    split_contracts
                ),
                dataset_id=(
                    source_dataset
                ),
                route=route,
                seed=101,
                smoke=True,
            )
        )

    route_smoke = {
        "all_passed": all(
            item.get("status")
            in {
                "COMPLETED",
                "RESUMED_COMPLETE",
            }
            and item.get(
                "teacher_frozen",
                True,
            )
            for item in (
                route_results.values()
            )
        ),
        "routes": route_results,
        "smoke_only_not_scientific_evidence": (
            True
        ),
    }

    atomic_json(
        output_root
        / "status"
        / "route_training_smoke_v2.json",
        route_smoke,
    )

    dataset, arrays = (
        load_dataset_and_arrays(
            repository_root=(
                repository_root
            ),
            dataset_id=(
                source_dataset
            ),
            registry_entry=(
                registry[
                    source_dataset
                ]
            ),
            split_contract=(
                split_contracts[
                    source_dataset
                ]
            ),
        )
    )

    supervised_checkpoint = Path(
        route_results[
            "supervised"
        ]["student_checkpoint"]
    )

    reload_smoke = (
        checkpoint_reload_smoke(
            checkpoint_path=(
                supervised_checkpoint
            ),
            arrays=arrays,
            feature_names=list(
                dataset.weather_features
            ),
        )
    )

    atomic_json(
        output_root
        / "status"
        / "checkpoint_reload_smoke_v2.json",
        reload_smoke,
    )

    selection_smoke = {
        "all_passed": True,
        "selection_scope": (
            "SOURCE_VALIDATION_ONLY"
        ),
        "target_metrics_used": False,
        "selected_route": min(
            route_results,
            key=lambda route: (
                route_results[route]
                .get(
                    "validation_metrics",
                    {},
                )
                .get(
                    "rmse",
                    float("inf"),
                )
            ),
        ),
        "selection_metric": (
            "SOURCE_VALIDATION_RMSE"
        ),
        "note": (
            "Smoke selection validates mechanics only; "
            "formal Phase 5 must select using source validation artifacts, "
            "not smoke test metrics."
        ),
        "smoke_only_not_scientific_evidence": (
            True
        ),
    }

    atomic_json(
        output_root
        / "status"
        / "source_selection_smoke_v2.json",
        selection_smoke,
    )

    transfer_results = {}

    for target in [
        "CY-Bench_wheat_AU",
        "waite",
    ]:
        transfer_results[target] = (
            target_transfer_smoke(
                repository_root=(
                    repository_root
                ),
                registry=registry,
                split_contracts=(
                    split_contracts
                ),
                target_dataset_id=(
                    target
                ),
                source_checkpoint=(
                    supervised_checkpoint
                ),
            )
        )

    transfer_smoke = {
        "all_passed": all(
            item["all_passed"]
            for item in (
                transfer_results.values()
            )
        ),
        "targets": transfer_results,
        "target_test_used_for_selection": (
            False
        ),
        "smoke_only_not_scientific_evidence": (
            True
        ),
    }

    atomic_json(
        output_root
        / "status"
        / "target_transfer_smoke_v2.json",
        transfer_smoke,
    )

    return {
        "route_training": (
            route_smoke
        ),
        "checkpoint_reload": (
            reload_smoke
        ),
        "source_selection": (
            selection_smoke
        ),
        "target_transfer": (
            transfer_smoke
        ),
        "all_passed": all(
            [
                route_smoke[
                    "all_passed"
                ],
                reload_smoke[
                    "all_passed"
                ],
                selection_smoke[
                    "all_passed"
                ],
                transfer_smoke[
                    "all_passed"
                ],
            ]
        ),
    }
