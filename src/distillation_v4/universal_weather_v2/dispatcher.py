from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .artifacts import (
    atomic_json,
    atomic_torch_save,
    job_is_complete,
    load_checkpoint,
    mark_complete,
    record_failure,
    sha256_file,
    stable_hash,
)
from .campaign import (
    choose_device,
    limit_arrays_for_smoke,
    load_json,
    source_training_job,
)
from .training import (
    build_student_v2,
    evaluate_student,
    load_dataset_and_arrays,
    metric_bundle,
    train_student_route_v2,
)


SOURCE_PHASES = {
    "phase1",
    "phase2",
    "phase3",
}

CONTROL_PHASES = {
    "phase4",
    "phase5",
}

TARGET_MODES = {
    "frozen_zero_shot",
    "fine_tune",
    "local_scratch_matched",
    "traditional_baseline",
}

SOURCE_ROUTES = {
    "supervised",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
    "missing_aware",
}


def classify_job(
    job: dict[str, Any],
) -> str:
    phase = str(
        job.get("phase", "")
    )

    mode = job.get("mode")

    if phase in SOURCE_PHASES:
        return "source_training"

    if phase == "phase4":
        return "source_selection_gate"

    if phase == "phase5":
        return "source_checkpoint_freeze"

    if phase == "outer":
        return "source_outer_evaluation"

    if phase == "targets":
        if mode == "frozen_zero_shot":
            return "target_zero_shot"

        if mode == "fine_tune":
            return "target_fine_tune"

        if mode == "local_scratch_matched":
            return "target_local_scratch"

        if mode == "traditional_baseline":
            return "target_traditional_baseline"

        raise ValueError(
            "UNSUPPORTED_TARGET_MODE:"
            + repr(mode)
        )

    if phase == "report":
        return "aggregate_report"

    raise ValueError(
        "UNSUPPORTED_JOB_PHASE:"
        + repr(phase)
    )


def normalise_source_route(
    job: dict[str, Any],
) -> str:
    route = str(
        job.get("route", "")
    )

    if route in SOURCE_ROUTES:
        return route

    phase = str(
        job.get("phase", "")
    )

    phase_defaults = {
        "phase1": "supervised",
        "phase2": "prediction_kd",
        "phase3": "combined_kd",
    }

    if phase in phase_defaults:
        return phase_defaults[phase]

    raise ValueError(
        "SOURCE_ROUTE_NOT_RESOLVED:"
        + json.dumps(
            job,
            sort_keys=True,
        )
    )


def source_dataset_id(
    job: dict[str, Any],
) -> str:
    for key in [
        "dataset",
        "source_dataset",
        "source",
    ]:
        value = job.get(key)

        if value:
            return str(value)

    raise ValueError(
        "SOURCE_DATASET_NOT_RESOLVED:"
        + json.dumps(
            job,
            sort_keys=True,
        )
    )


def target_dataset_id(
    job: dict[str, Any],
) -> str:
    value = job.get("dataset")

    if not value:
        raise ValueError(
            "TARGET_DATASET_NOT_RESOLVED"
        )

    return str(value)


def seed_value(
    job: dict[str, Any],
) -> int:
    value = job.get("seed")

    if value is None:
        return 101

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return 101

    return int(value)


def load_control_assets(
    output_root: Path,
) -> dict[str, Any]:
    registry = load_json(
        output_root
        / "control"
        / "dataset_registry_v2.json"
    )["datasets"]

    feature_contracts = load_json(
        output_root
        / "control"
        / "feature_contract_v2.json"
    )["datasets"]

    split_contracts = load_json(
        output_root
        / "control"
        / "split_contracts_v2.json"
    )["datasets"]

    return {
        "registry": registry,
        "feature_contracts": (
            feature_contracts
        ),
        "split_contracts": (
            split_contracts
        ),
    }




def source_smoke_results(
    output_root: Path,
    dataset_id: str,
) -> dict[str, Any]:
    source_root = (
        output_root
        / "smoke"
        / "source"
        / dataset_id
    )

    if not source_root.is_dir():
        raise FileNotFoundError(
            "SOURCE_SPECIFIC_SMOKE_DIRECTORY_MISSING:"
            + str(source_root)
        )

    routes: dict[str, dict[str, Any]] = {}

    for json_path in sorted(
        source_root.rglob("*.json")
    ):
        try:
            payload = load_json(json_path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        job = payload.get("job", {})

        route = (
            job.get("route")
            or payload.get("route")
        )

        checkpoint = payload.get(
            "student_checkpoint"
        )

        validation = payload.get(
            "validation_metrics"
        )

        status = payload.get("status")

        if (
            not route
            or not checkpoint
            or not validation
            or status
            not in {
                "COMPLETED",
                "RESUMED_COMPLETE",
            }
        ):
            continue

        checkpoint_path = Path(
            checkpoint
        )

        if not checkpoint_path.is_file():
            continue

        resolved_checkpoint = str(
            checkpoint_path.resolve()
        )

        required_fragment = (
            "/smoke/source/"
            + dataset_id
            + "/"
        )

        if required_fragment not in (
            resolved_checkpoint
        ):
            raise RuntimeError(
                "SOURCE_SMOKE_LINEAGE_VIOLATION:"
                + dataset_id
                + ":"
                + resolved_checkpoint
            )

        candidate = {
            **payload,
            "student_checkpoint": (
                resolved_checkpoint
            ),
            "result_path": str(
                json_path.resolve()
            ),
        }

        previous = routes.get(
            str(route)
        )

        if previous is None:
            routes[str(route)] = candidate
            continue

        candidate_rmse = float(
            candidate[
                "validation_metrics"
            ]["rmse"]
        )

        previous_rmse = float(
            previous[
                "validation_metrics"
            ]["rmse"]
        )

        if candidate_rmse < previous_rmse:
            routes[str(route)] = candidate

    if not routes:
        raise RuntimeError(
            "NO_SOURCE_SPECIFIC_SMOKE_RESULTS:"
            + dataset_id
        )

    return routes

def formal_source_results(
    *,
    output_root: Path,
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    root = (
        output_root
        / "jobs"
        / "source"
        / dataset_id
    )

    if not root.is_dir():
        raise FileNotFoundError(
            "FORMAL_SOURCE_DIRECTORY_MISSING:"
            + str(root)
        )

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for path in sorted(
        root.rglob("result.json")
    ):
        payload = load_json(path)

        if payload.get("status") != "COMPLETED":
            continue

        job = payload.get("job", {})

        route = job.get("route")

        validation = payload.get(
            "validation_metrics"
        )

        checkpoint = payload.get(
            "student_checkpoint"
        )

        if (
            not route
            or not validation
            or not checkpoint
        ):
            continue

        checkpoint_path = Path(
            checkpoint
        )

        if not checkpoint_path.is_file():
            continue

        candidate = {
            **payload,
            "result_path": str(path),
        }

        existing = results.get(
            str(route)
        )

        if existing is None:
            results[str(route)] = candidate
            continue

        current_rmse = float(
            candidate[
                "validation_metrics"
            ]["rmse"]
        )

        existing_rmse = float(
            existing[
                "validation_metrics"
            ]["rmse"]
        )

        if current_rmse < existing_rmse:
            results[str(route)] = candidate

    if not results:
        raise RuntimeError(
            "NO_COMPLETED_FORMAL_SOURCE_RESULTS:"
            + dataset_id
        )

    return results



def select_source_checkpoint(
    *,
    output_root: Path,
    dataset_id: str,
    canary: bool,
) -> dict[str, Any]:
    if canary:
        routes = source_smoke_results(
            output_root,
            dataset_id,
        )

        artifact_namespace = (
            "SMOKE_SOURCE_RESULTS"
        )
    else:
        routes = formal_source_results(
            output_root=output_root,
            dataset_id=dataset_id,
        )

        artifact_namespace = (
            "FORMAL_SOURCE_RESULTS"
        )

    eligible = {
        route: result
        for route, result in routes.items()
        if (
            result.get("status")
            in {
                "COMPLETED",
                "RESUMED_COMPLETE",
            }
            and result.get(
                "validation_metrics"
            )
            and result.get(
                "student_checkpoint"
            )
        )
    }

    if not eligible:
        raise RuntimeError(
            "NO_SOURCE_VALIDATION_RESULTS:"
            + dataset_id
        )

    selected_route = min(
        eligible,
        key=lambda route: float(
            eligible[route][
                "validation_metrics"
            ]["rmse"]
        ),
    )

    selected = eligible[
        selected_route
    ]

    checkpoint = Path(
        selected["student_checkpoint"]
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    checkpoint_text = str(
        checkpoint.resolve()
    )

    if canary:
        if "/smoke/" not in checkpoint_text:
            raise RuntimeError(
                "CANARY_SELECTION_USED_NON_SMOKE_CHECKPOINT:"
                + checkpoint_text
            )
    else:
        required_fragment = (
            "/jobs/source/"
            + dataset_id
            + "/"
        )

        if required_fragment not in (
            checkpoint_text
        ):
            raise RuntimeError(
                "FORMAL_SELECTION_USED_NON_FORMAL_CHECKPOINT:"
                + checkpoint_text
            )

    return {
        "dataset_id": dataset_id,
        "selected_route": (
            selected_route
        ),
        "selection_metric": (
            "SOURCE_VALIDATION_RMSE"
        ),
        "validation_metrics": (
            selected[
                "validation_metrics"
            ]
        ),
        "source_checkpoint": str(
            checkpoint.resolve()
        ),
        "artifact_namespace": (
            artifact_namespace
        ),
        "target_metrics_used": False,
        "outer_metrics_used": False,
    }


def run_source_selection_gate(
    *,
    output_root: Path,
    job: dict[str, Any],
    canary: bool,
    canary_directory: Path,
) -> dict[str, Any]:
    dataset_id = source_dataset_id(
        job
    )

    selection = (
        select_source_checkpoint(
            output_root=output_root,
            dataset_id=dataset_id,
            canary=canary,
        )
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "source_selection_gate"
        ),
        **selection,
        "gate_passed": True,
        "canary": canary,
        "scientific_evidence": (
            not canary
        ),
    }

    atomic_json(
        canary_directory
        / "selection.json",
        result,
    )

    return result


def run_source_checkpoint_freeze(
    *,
    output_root: Path,
    job: dict[str, Any],
    canary: bool,
    canary_directory: Path,
) -> dict[str, Any]:
    dataset_id = source_dataset_id(
        job
    )

    selection = (
        select_source_checkpoint(
            output_root=output_root,
            dataset_id=dataset_id,
            canary=canary,
        )
    )

    source_path = Path(
        selection[
            "source_checkpoint"
        ]
    )

    payload = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    if canary:
        canonical_directory = (
            output_root
            / "canary_dispatch"
            / "source_checkpoint_freeze"
            / dataset_id
        )
    else:
        canonical_directory = (
            output_root
            / "formal_artifacts"
            / "source_checkpoint_freeze"
            / dataset_id
        )

    canonical_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_path = (
        canonical_directory
        / "frozen_source_checkpoint.pt"
    )

    frozen_metadata = dict(
        payload.get(
            "metadata",
            {},
        )
    )

    frozen_metadata.update(
        {
            "checkpoint_role": (
                "FROZEN_SOURCE_SELECTED"
            ),
            "dataset_id": dataset_id,
            "selected_route": (
                selection[
                    "selected_route"
                ]
            ),
            "selection_metric": (
                "SOURCE_VALIDATION_RMSE"
            ),
            "artifact_namespace": (
                selection[
                    "artifact_namespace"
                ]
            ),
            "target_metrics_used": False,
            "outer_metrics_used": False,
            "canary_only": canary,
            "scientific_evidence": (
                not canary
            ),
        }
    )

    atomic_torch_save(
        frozen_path,
        {
            "state_dict": payload[
                "state_dict"
            ],
            "metadata": frozen_metadata,
        },
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "source_checkpoint_freeze"
        ),
        **selection,
        "frozen_checkpoint": str(
            frozen_path.resolve()
        ),
        "frozen_checkpoint_sha256": (
            sha256_file(
                frozen_path
            )
        ),
        "refit_performed": False,
        "canary": canary,
        "scientific_evidence": (
            not canary
        ),
    }

    atomic_json(
        canary_directory
        / "freeze.json",
        result,
    )

    return result


def resolve_frozen_checkpoint(
    *,
    output_root: Path,
    dataset_id: str,
    canary: bool,
) -> Path:
    if canary:
        path = (
            output_root
            / "canary_dispatch"
            / "source_checkpoint_freeze"
            / dataset_id
            / "frozen_source_checkpoint.pt"
        )
    else:
        path = (
            output_root
            / "formal_artifacts"
            / "source_checkpoint_freeze"
            / dataset_id
            / "frozen_source_checkpoint.pt"
        )

    if not path.is_file():
        raise FileNotFoundError(
            "FROZEN_CHECKPOINT_MISSING:"
            + str(path)
        )

    resolved = str(path.resolve())

    if canary:
        if "/canary_dispatch/" not in resolved:
            raise RuntimeError(
                "CANARY_FROZEN_NAMESPACE_VIOLATION:"
                + resolved
            )
    else:
        if "/formal_artifacts/" not in resolved:
            raise RuntimeError(
                "FORMAL_FROZEN_NAMESPACE_VIOLATION:"
                + resolved
            )

    return path

def run_source_outer_evaluation(
    *,
    repository_root: Path,
    output_root: Path,
    assets: dict[str, Any],
    job: dict[str, Any],
    canary_directory: Path,
    canary: bool,
) -> dict[str, Any]:
    dataset_id = source_dataset_id(
        job
    )

    if dataset_id not in assets[
        "registry"
    ]:
        dataset_id = (
            "PRIMARY_G2F_MAIZE"
        )

    dataset, arrays = (
        load_dataset_and_arrays(
            repository_root=(
                repository_root
            ),
            dataset_id=dataset_id,
            registry_entry=assets[
                "registry"
            ][dataset_id],
            split_contract=assets[
                "split_contracts"
            ][dataset_id],
        )
    )

    arrays = limit_arrays_for_smoke(
        arrays,
        train_limit=512,
        validation_limit=128,
        test_limit=256,
    )

    device = choose_device()

    model = build_student_v2().to(
        device
    )

    checkpoint = (
        resolve_frozen_checkpoint(
            output_root=output_root,
            dataset_id=dataset_id,
            canary=canary,
        )
    )

    load_checkpoint(
        path=checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )

    metrics, prediction = (
        evaluate_student(
            model=model,
            arrays=arrays["test"],
            feature_names=list(
                dataset.weather_features
            ),
            device=device,
            batch_size=64,
        )
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "source_outer_evaluation"
        ),
        "dataset_id": dataset_id,
        "checkpoint": str(
            checkpoint
        ),
        "refit_performed": False,
        "outer_metrics": metrics,
        "prediction_finite": bool(
            np.isfinite(
                prediction
            ).all()
        ),
        "prediction_nonconstant": bool(
            float(
                np.std(prediction)
            )
            > 0
        ),
        "outer_metrics_used_for_selection": (
            False
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / "outer_result.json",
        result,
    )

    return result


def load_target_arrays(
    *,
    repository_root: Path,
    assets: dict[str, Any],
    dataset_id: str,
) -> tuple[Any, dict[str, Any]]:
    dataset, arrays = (
        load_dataset_and_arrays(
            repository_root=(
                repository_root
            ),
            dataset_id=dataset_id,
            registry_entry=assets[
                "registry"
            ][dataset_id],
            split_contract=assets[
                "split_contracts"
            ][dataset_id],
        )
    )

    arrays = limit_arrays_for_smoke(
        arrays,
        train_limit=512,
        validation_limit=128,
        test_limit=128,
    )

    return dataset, arrays



def load_frozen_model(
    *,
    output_root: Path,
    dataset_id: str,
    canary: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = (
        resolve_frozen_checkpoint(
            output_root=output_root,
            dataset_id=dataset_id,
            canary=canary,
        )
    )

    model = build_student_v2().to(
        device
    )

    payload = load_checkpoint(
        path=checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )

    return model, payload

def run_target_zero_shot(
    *,
    repository_root: Path,
    output_root: Path,
    assets: dict[str, Any],
    job: dict[str, Any],
    canary_directory: Path,
    canary: bool,
) -> dict[str, Any]:
    dataset_id = target_dataset_id(
        job
    )

    dataset, arrays = load_target_arrays(
        repository_root=(
            repository_root
        ),
        assets=assets,
        dataset_id=dataset_id,
    )

    device = choose_device()

    source_id = str(
        job.get(
            "source_dataset"
        )
        or job.get("source")
        or "PRIMARY_G2F_MAIZE"
    )

    model, _payload = load_frozen_model(
        output_root=output_root,
        dataset_id=source_id,
        canary=canary,
        device=device,
    )

    metrics, prediction = (
        evaluate_student(
            model=model,
            arrays=arrays["test"],
            feature_names=list(
                dataset.weather_features
            ),
            device=device,
            batch_size=64,
        )
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "target_zero_shot"
        ),
        "dataset_id": dataset_id,
        "metrics": metrics,
        "refit_performed": False,
        "target_train_used": False,
        "target_validation_used": False,
        "target_test_used_for_selection": (
            False
        ),
        "prediction_finite": bool(
            np.isfinite(
                prediction
            ).all()
        ),
        "prediction_nonconstant": bool(
            float(
                np.std(prediction)
            )
            > 0
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / "zero_shot.json",
        result,
    )

    return result


def run_target_fine_tune(
    *,
    repository_root: Path,
    output_root: Path,
    assets: dict[str, Any],
    job: dict[str, Any],
    canary_directory: Path,
    canary: bool,
) -> dict[str, Any]:
    dataset_id = target_dataset_id(
        job
    )

    dataset, arrays = load_target_arrays(
        repository_root=(
            repository_root
        ),
        assets=assets,
        dataset_id=dataset_id,
    )

    device = choose_device()

    source_id = str(
        job.get(
            "source_dataset"
        )
        or job.get("source")
        or "PRIMARY_G2F_MAIZE"
    )

    _model, payload = load_frozen_model(
        output_root=output_root,
        dataset_id=source_id,
        canary=canary,
        device=device,
    )

    initial_state = {
        key: value.detach().cpu().clone()
        for key, value
        in payload[
            "state_dict"
        ].items()
    }

    result_payload = (
        train_student_route_v2(
            route="supervised",
            arrays=arrays,
            feature_names=list(
                dataset.weather_features
            ),
            teacher=None,
            seed=seed_value(job),
            epochs=1,
            patience=1,
            batch_size=64,
            learning_rate=1e-4,
            device=device,
            checkpoint_path=(
                canary_directory
                / "fine_tuned.pt"
            ),
            metadata={
                "dataset_id": dataset_id,
                "initialisation": (
                    "FROZEN_SOURCE_CHECKPOINT"
                ),
                "target_test_used_for_selection": (
                    False
                ),
                "canary_only": True,
            },
            initial_state_dict=(
                initial_state
            ),
        )
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "target_fine_tune"
        ),
        "dataset_id": dataset_id,
        "validation_metrics": (
            result_payload[
                "validation_metrics"
            ]
        ),
        "test_metrics": (
            result_payload[
                "test_metrics"
            ]
        ),
        "initialisation": (
            "FROZEN_SOURCE_CHECKPOINT"
        ),
        "target_test_used_for_selection": (
            False
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / "fine_tune.json",
        result,
    )

    return result


def run_target_local_scratch(
    *,
    repository_root: Path,
    assets: dict[str, Any],
    job: dict[str, Any],
    canary_directory: Path,
) -> dict[str, Any]:
    dataset_id = target_dataset_id(
        job
    )

    dataset, arrays = load_target_arrays(
        repository_root=(
            repository_root
        ),
        assets=assets,
        dataset_id=dataset_id,
    )

    device = choose_device()

    result_payload = (
        train_student_route_v2(
            route="supervised",
            arrays=arrays,
            feature_names=list(
                dataset.weather_features
            ),
            teacher=None,
            seed=seed_value(job),
            epochs=1,
            patience=1,
            batch_size=64,
            learning_rate=1e-3,
            device=device,
            checkpoint_path=(
                canary_directory
                / "local_scratch.pt"
            ),
            metadata={
                "dataset_id": dataset_id,
                "initialisation": (
                    "RANDOM_MATCHED_ARCHITECTURE"
                ),
                "target_test_used_for_selection": (
                    False
                ),
                "canary_only": True,
            },
            initial_state_dict=None,
        )
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "target_local_scratch"
        ),
        "dataset_id": dataset_id,
        "validation_metrics": (
            result_payload[
                "validation_metrics"
            ]
        ),
        "test_metrics": (
            result_payload[
                "test_metrics"
            ]
        ),
        "architecture": (
            "UNIVERSAL_WEATHER_STUDENT_V2"
        ),
        "initialisation": (
            "RANDOM_MATCHED_ARCHITECTURE"
        ),
        "target_test_used_for_selection": (
            False
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / "local_scratch.json",
        result,
    )

    return result



def run_target_traditional_baseline(
    *,
    repository_root: Path,
    assets: dict[str, Any],
    job: dict[str, Any],
    canary_directory: Path,
) -> dict[str, Any]:
    dataset_id = target_dataset_id(
        job
    )

    _dataset, arrays = load_target_arrays(
        repository_root=(
            repository_root
        ),
        assets=assets,
        dataset_id=dataset_id,
    )

    requested = str(
        job.get("route", "ridge")
    ).strip().lower()

    aliases = {
        "linear": "linear_regression",
        "ols": "linear_regression",
        "linear_regression": "linear_regression",
        "ridge": "ridge",
        "lasso": "lasso",
        "elasticnet": "elastic_net",
        "elastic_net": "elastic_net",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "extratrees": "extra_trees",
        "extra_trees": "extra_trees",
        "gbdt": "gradient_boosting",
        "gradient_boosting": "gradient_boosting",
        "histgbdt": "hist_gradient_boosting",
        "histgradientboosting": (
            "hist_gradient_boosting"
        ),
        "hist_gradient_boosting": (
            "hist_gradient_boosting"
        ),
        "xgb": "xgboost",
        "xgboost": "xgboost",
        "svr": "svr",
        "knn": "knn",
    }

    canonical = aliases.get(
        requested
    )

    if canonical is None:
        raise ValueError(
            "UNSUPPORTED_TRADITIONAL_BASELINE:"
            + requested
        )

    estimators = {
        "linear_regression": (
            LinearRegression()
        ),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(
            alpha=0.001,
            max_iter=10000,
        ),
        "elastic_net": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=100,
                random_state=seed_value(
                    job
                ),
                n_jobs=-1,
            )
        ),
        "extra_trees": (
            ExtraTreesRegressor(
                n_estimators=100,
                random_state=seed_value(
                    job
                ),
                n_jobs=-1,
            )
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(
                random_state=seed_value(
                    job
                )
            )
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                random_state=seed_value(
                    job
                )
            )
        ),
        "xgboost": XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed_value(
                job
            ),
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ),
        "svr": SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
        ),
        "knn": KNeighborsRegressor(
            n_neighbors=5,
        ),
    }

    estimator = estimators[
        canonical
    ]

    scale_required = canonical in {
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "svr",
        "knn",
    }

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]

    if scale_required:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    steps.append(
        (
            "regressor",
            estimator,
        )
    )

    model = Pipeline(steps)

    model.fit(
        arrays["train"]["weather"],
        arrays["train"]["target"],
    )

    validation_prediction = (
        model.predict(
            arrays[
                "validation"
            ]["weather"]
        )
    )

    test_prediction = model.predict(
        arrays["test"]["weather"]
    )

    result = {
        "status": "COMPLETED",
        "handler": (
            "target_traditional_baseline"
        ),
        "dataset_id": dataset_id,
        "requested_route": requested,
        "canonical_route": canonical,
        "executed_model": (
            estimator.__class__.__name__
        ),
        "validation_metrics": (
            metric_bundle(
                arrays[
                    "validation"
                ]["target"],
                validation_prediction,
            )
        ),
        "test_metrics": metric_bundle(
            arrays["test"]["target"],
            test_prediction,
        ),
        "target_test_used_for_selection": (
            False
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / (
            "traditional_baseline_"
            + canonical
            + ".json"
        ),
        result,
    )

    return result

def run_aggregate_report(
    *,
    output_root: Path,
    canary_directory: Path,
) -> dict[str, Any]:
    canary_root = (
        output_root
        / "canary_dispatch"
    )

    records = []

    for path in sorted(
        canary_root.rglob("*.json")
    ):
        if (
            path.name
            in {
                "COMPLETED.json",
                "dispatcher_canary_status.json",
            }
        ):
            continue

        try:
            payload = load_json(path)
        except Exception:
            continue

        if isinstance(payload, dict):
            records.append(
                {
                    "path": str(path),
                    "handler": payload.get(
                        "handler"
                    ),
                    "status": payload.get(
                        "status"
                    ),
                }
            )

    result = {
        "status": "COMPLETED",
        "handler": (
            "aggregate_report"
        ),
        "record_count": len(records),
        "records": records,
        "formal_metrics_mixed_with_canary": (
            False
        ),
        "scientific_evidence": False,
    }

    atomic_json(
        canary_directory
        / "aggregate_report.json",
        result,
    )

    return result


def execute_job(
    *,
    repository_root: Path,
    output_root: Path,
    job: dict[str, Any],
    canary: bool,
) -> dict[str, Any]:
    kind = classify_job(job)

    assets = load_control_assets(
        output_root
    )

    fingerprint = stable_hash(
        {
            "schema": (
                "universal_weather_dispatch_v2"
            ),
            "kind": kind,
            "job": job,
            "canary": canary,
            "feature_contracts": (
                assets[
                    "feature_contracts"
                ]
            ),
            "split_hashes": {
                dataset_id: contract.get(
                    "split_sha256"
                )
                for dataset_id, contract
                in assets[
                    "split_contracts"
                ].items()
            },
        }
    )

    job_directory = (
        output_root
        / (
            "canary_dispatch"
            if canary
            else "jobs"
        )
        / kind
        / fingerprint[:16]
    )

    if job_is_complete(
        job_directory,
        fingerprint=fingerprint,
    ):
        result_path = (
            job_directory
            / "dispatcher_result.json"
        )

        if not result_path.is_file():
            raise RuntimeError(
                "COMPLETION_MARKER_WITHOUT_DISPATCHER_RESULT:"
                + str(job_directory)
            )

        resumed_result = load_json(
            result_path
        )

        if (
            resumed_result.get(
                "dispatcher_fingerprint"
            )
            != fingerprint
        ):
            raise RuntimeError(
                "RESUME_FINGERPRINT_MISMATCH:"
                + str(job_directory)
            )

        return {
            **resumed_result,
            "status": "RESUMED_COMPLETE",
            "resumed": True,
            "original_status": (
                resumed_result.get(
                    "status"
                )
            ),
        }

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        if kind == "source_training":
            result = source_training_job(
                repository_root=(
                    repository_root
                ),
                output_root=output_root,
                registry=assets[
                    "registry"
                ],
                feature_contracts=assets[
                    "feature_contracts"
                ],
                split_contracts=assets[
                    "split_contracts"
                ],
                dataset_id=(
                    source_dataset_id(job)
                ),
                route=(
                    normalise_source_route(
                        job
                    )
                ),
                seed=seed_value(job),
                smoke=canary,
            )

        elif kind == (
            "source_selection_gate"
        ):
            result = (
                run_source_selection_gate(
                    output_root=output_root,
                    job=job,
                    canary=canary,
                    canary_directory=(
                        job_directory
                    ),
                )
            )

        elif kind == (
            "source_checkpoint_freeze"
        ):
            result = (
                run_source_checkpoint_freeze(
                    output_root=output_root,
                    job=job,
                    canary=canary,
                    canary_directory=(
                        job_directory
                    ),
                )
            )

        elif kind == (
            "source_outer_evaluation"
        ):
            result = (
                run_source_outer_evaluation(
                    repository_root=(
                        repository_root
                    ),
                    output_root=(
                        output_root
                    ),
                    assets=assets,
                    job=job,
                    canary_directory=(
                        job_directory
                    ),
                    canary=canary,
)
            )

        elif kind == "target_zero_shot":
            result = run_target_zero_shot(
                repository_root=(
                    repository_root
                ),
                output_root=output_root,
                assets=assets,
                job=job,
                canary_directory=(
                    job_directory
                ),
                canary=canary,
            )

        elif kind == "target_fine_tune":
            result = run_target_fine_tune(
                repository_root=(
                    repository_root
                ),
                output_root=output_root,
                assets=assets,
                job=job,
                canary_directory=(
                    job_directory
                ),
                canary=canary,
            )

        elif kind == (
            "target_local_scratch"
        ):
            result = (
                run_target_local_scratch(
                    repository_root=(
                        repository_root
                    ),
                    assets=assets,
                    job=job,
                    canary_directory=(
                        job_directory
                    ),
                )
            )

        elif kind == (
            "target_traditional_baseline"
        ):
            result = (
                run_target_traditional_baseline(
                    repository_root=(
                        repository_root
                    ),
                    assets=assets,
                    job=job,
                    canary_directory=(
                        job_directory
                    ),
                )
            )

        elif kind == "aggregate_report":
            result = run_aggregate_report(
                output_root=output_root,
                canary_directory=(
                    job_directory
                ),
            )

        else:
            raise RuntimeError(
                "HANDLER_NOT_BOUND:"
                + kind
            )

        result = {
            **result,
            "dispatcher_kind": kind,
            "dispatcher_fingerprint": (
                fingerprint
            ),
            "canary": canary,
        }

        atomic_json(
            job_directory
            / "dispatcher_result.json",
            result,
        )

        mark_complete(
            job_directory,
            fingerprint=fingerprint,
            result=result,
        )

        return result

    except Exception as error:
        record_failure(
            failure_ledger=(
                output_root
                / "failures"
                / (
                    "canary_dispatch_failures.json"
                    if canary
                    else "formal_dispatch_failures.json"
                )
            ),
            job=job,
            error=error,
        )

        raise



def coverage_report(
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures = []

    supported_baselines = {
        "linear",
        "ols",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "elastic_net",
        "rf",
        "randomforest",
        "random_forest",
        "extratrees",
        "extra_trees",
        "gbdt",
        "gradient_boosting",
        "histgbdt",
        "histgradientboosting",
        "hist_gradient_boosting",
        "xgb",
        "xgboost",
        "svr",
        "knn",
    }

    requested_baselines = set()
    unsupported_baselines = set()

    for index, job in enumerate(jobs):
        try:
            kind = classify_job(job)
        except Exception as error:
            failures.append(
                {
                    "index": index,
                    "job": job,
                    "reason": str(error),
                }
            )
            continue

        counts[kind] = (
            counts.get(kind, 0)
            + 1
        )

        if kind == (
            "target_traditional_baseline"
        ):
            route = str(
                job.get("route", "")
            ).strip().lower()

            requested_baselines.add(
                route
            )

            if route not in (
                supported_baselines
            ):
                unsupported_baselines.add(
                    route
                )

    expected_handlers = {
        "source_training",
        "source_selection_gate",
        "source_checkpoint_freeze",
        "source_outer_evaluation",
        "target_zero_shot",
        "target_fine_tune",
        "target_local_scratch",
        "target_traditional_baseline",
        "aggregate_report",
    }

    observed_handlers = set(
        counts
    )

    all_jobs_mapped = (
        not failures
        and expected_handlers
        <= observed_handlers
        and not unsupported_baselines
    )

    return {
        "planned_jobs": len(jobs),
        "covered_jobs": (
            len(jobs) - len(failures)
        ),
        "uncovered_jobs": len(
            failures
        ),
        "handler_counts": counts,
        "expected_handlers": sorted(
            expected_handlers
        ),
        "observed_handlers": sorted(
            observed_handlers
        ),
        "missing_handler_kinds": sorted(
            expected_handlers
            - observed_handlers
        ),
        "requested_baseline_routes": (
            sorted(
                requested_baselines
            )
        ),
        "unsupported_baseline_routes": (
            sorted(
                unsupported_baselines
            )
        ),
        "baseline_implementation_coverage": (
            not unsupported_baselines
        ),
        "failures": failures,
        "all_jobs_mapped": (
            all_jobs_mapped
        ),
    }

def representative_jobs(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    representatives: dict[
        str,
        dict[str, Any],
    ] = {}

    for job in jobs:
        kind = classify_job(job)

        if kind not in representatives:
            representatives[kind] = (
                job
            )

    ordered = [
        "source_training",
        "source_selection_gate",
        "source_checkpoint_freeze",
        "source_outer_evaluation",
        "target_zero_shot",
        "target_fine_tune",
        "target_local_scratch",
        "target_traditional_baseline",
        "aggregate_report",
    ]

    return [
        representatives[kind]
        for kind in ordered
        if kind in representatives
    ]



def run_dispatch_canaries(
    *,
    repository_root: Path,
    output_root: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = coverage_report(
        jobs
    )

    atomic_json(
        output_root
        / "status"
        / "dispatcher_coverage_status.json",
        {
            "schema_version": (
                "universal_weather_dispatch_coverage_v3"
            ),
            **coverage,
            "formal_training_started": False,
        },
    )

    if not coverage[
        "all_jobs_mapped"
    ]:
        raise RuntimeError(
            "DISPATCH_COVERAGE_FAILED:"
            + json.dumps(
                coverage,
                sort_keys=True,
            )
        )

    results = {}

    for job in representative_jobs(
        jobs
    ):
        kind = classify_job(job)

        results[kind] = execute_job(
            repository_root=(
                repository_root
            ),
            output_root=output_root,
            job=job,
            canary=True,
        )

    baseline_canaries = {}

    for route in coverage[
        "requested_baseline_routes"
    ]:
        candidates = [
            job
            for job in jobs
            if (
                classify_job(job)
                == "target_traditional_baseline"
                and str(
                    job.get("route", "")
                ).strip().lower()
                == route
            )
        ]

        if not candidates:
            raise RuntimeError(
                "BASELINE_CANARY_JOB_MISSING:"
                + route
            )

        result = execute_job(
            repository_root=(
                repository_root
            ),
            output_root=output_root,
            job=candidates[0],
            canary=True,
        )

        baseline_canaries[route] = (
            result
        )

    handler_passed = all(
        result.get("status")
        in {
            "COMPLETED",
            "RESUMED_COMPLETE",
        }
        for result in results.values()
    )

    baseline_passed = all(
        result.get("status")
        in {
            "COMPLETED",
            "RESUMED_COMPLETE",
        }
        and result.get(
            "executed_model"
        )
        for result in (
            baseline_canaries.values()
        )
    )

    status = {
        "schema_version": (
            "universal_weather_dispatch_canary_v3"
        ),
        "coverage": coverage,
        "handler_canaries": results,
        "baseline_canaries": (
            baseline_canaries
        ),
        "all_baseline_canaries_passed": (
            baseline_passed
        ),
        "all_handler_canaries_passed": (
            handler_passed
            and baseline_passed
        ),
        "all_jobs_mapped": coverage[
            "all_jobs_mapped"
        ],
        "formal_training_started": False,
        "scientific_evidence": False,
    }

    atomic_json(
        output_root
        / "status"
        / "dispatcher_canary_status.json",
        status,
    )

    return status

