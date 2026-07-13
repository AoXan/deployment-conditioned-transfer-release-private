from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.distillation_v4.universal_weather_v2.artifacts import (
    load_checkpoint,
)
from src.distillation_v4.universal_weather_v2.campaign import (
    source_training_job,
)
from src.distillation_v4.universal_weather_v2.models import (
    UniversalWeatherEncoderV2,
    UniversalWeatherRegressorV2,
)
from src.distillation_v4.universal_weather_v2.training import (
    evaluate_student,
    prepare_arrays,
)

from .artifacts import (
    accept_artifacts,
    atomic_json,
    build_fingerprint,
    code_tree_hash,
    sha256_file,
    stable_hash,
)
from .datasets import load_frame
from .execution import _dataset, _source_assets
from .splits import build_contract_splits


PRIMARY_SOURCE_ROUTES = {
    "supervised",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
    "missing_aware",
}


def source_route_job_id(
    *,
    source_dataset: str,
    source_route: str,
    seed: int,
) -> str:
    return f"source::{source_dataset}::{source_route}::seed_{seed}"


def source_route_directory(
    *,
    output_root: Path,
    source_dataset: str,
    source_route: str,
    seed: int,
    smoke: bool,
) -> Path:
    return (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source"
        / source_dataset
        / source_route
        / f"seed_{seed}"
    )


def execute_primary_source_route(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    source_dataset = str(job["source_dataset"])
    source_route = str(job["source_route"])
    seed = int(job["seed"])

    if source_route not in PRIMARY_SOURCE_ROUTES:
        raise ValueError(
            f"UNSUPPORTED_PRIMARY_SOURCE_ROUTE:{source_route}"
        )

    registry, feature_contracts, split_contracts = _source_assets(
        config,
        repository_root,
        source_dataset,
    )

    result = source_training_job(
        repository_root=repository_root,
        output_root=output_root,
        registry=registry,
        feature_contracts=feature_contracts,
        split_contracts=split_contracts,
        dataset_id=source_dataset,
        route=source_route,
        seed=seed,
        smoke=smoke,
    )

    directory = source_route_directory(
        output_root=output_root,
        source_dataset=source_dataset,
        source_route=source_route,
        seed=seed,
        smoke=smoke,
    )
    checkpoint = directory / "student.pt"

    if result.get("student_checkpoint"):
        checkpoint = Path(result["student_checkpoint"])

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"ROUTE_SPECIFIC_CHECKPOINT_MISSING:{checkpoint}"
        )

    payload = {
        "schema_version": "source_route_handler_v1",
        "job_id": source_route_job_id(
            source_dataset=source_dataset,
            source_route=source_route,
            seed=seed,
        ),
        "source_dataset": source_dataset,
        "source_route": source_route,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "handler": "execute_primary_source_route",
        "legacy_source_training_result": result,
        "smoke_only": smoke,
    }
    atomic_json(directory / "remediation_source_handler.json", payload)
    return payload


def resolve_source_checkpoint(
    *,
    output_root: Path,
    source_dataset: str,
    source_route: str,
    seed: int,
    smoke: bool,
) -> Path:
    directory = source_route_directory(
        output_root=output_root,
        source_dataset=source_dataset,
        source_route=source_route,
        seed=seed,
        smoke=smoke,
    )
    checkpoint = directory / "student.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "SOURCE_DEPENDENCY_NOT_EXECUTED:"
            f"{source_dataset}:{source_route}:{seed}"
        )

    return checkpoint


def execute_source_outer(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    source_dataset = str(job["source_dataset"])
    source_route = str(job["source_route"])
    seed = int(job["seed"])

    if source_route not in PRIMARY_SOURCE_ROUTES:
        raise ValueError(
            f"UNSUPPORTED_SOURCE_OUTER_ROUTE:{source_route}"
        )

    contract = config["source_datasets"][source_dataset]
    frame, dataset_path = load_frame(contract, repository_root)
    splits = build_contract_splits(
        frame,
        contract,
        seed=seed,
    )
    split = splits[0]

    sample_column = contract["sample_id_column"]
    indexed = (
        frame.assign(
            _sample_id=frame[sample_column].astype(str)
        )
        .set_index("_sample_id", drop=False)
    )

    train = indexed.loc[list(split.train_ids)].reset_index(drop=True)
    validation = indexed.loc[
        list(split.validation_ids)
    ].reset_index(drop=True)
    test = indexed.loc[list(split.test_ids)].reset_index(drop=True)

    if smoke:
        train = train.sort_values(sample_column).head(256).copy()
        validation = (
            validation.sort_values(sample_column).head(64).copy()
        )
        test = test.sort_values(sample_column).head(64).copy()

    dataset = _dataset(
        frame,
        dataset_path,
        source_dataset,
        contract,
    )
    arrays = prepare_arrays(
        dataset=dataset,
        train_frame=train,
        validation_frame=validation,
        test_frame=test,
    )

    checkpoint = resolve_source_checkpoint(
        output_root=output_root,
        source_dataset=source_dataset,
        source_route=source_route,
        seed=seed,
        smoke=smoke,
    )

    model = UniversalWeatherRegressorV2(
        encoder=UniversalWeatherEncoderV2()
    ).to(torch.device("cpu"))
    loaded = load_checkpoint(
        path=checkpoint,
        model=model,
        map_location=torch.device("cpu"),
        strict=True,
    )

    metrics, predictions = evaluate_student(
        model=model,
        arrays=arrays["test"],
        feature_names=list(dataset.weather_features),
        device=torch.device("cpu"),
        batch_size=64,
    )

    prediction_values = np.asarray(predictions, dtype=float)
    if (
        prediction_values.size != len(test)
        or not np.isfinite(prediction_values).all()
    ):
        raise RuntimeError("INVALID_SOURCE_OUTER_PREDICTIONS")

    candidate_id = str(
        job.get(
            "candidate_id",
            (
                f"source_outer__{source_dataset}__"
                f"{source_route}__seed_{seed}"
            ),
        )
    )
    job_directory = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source_outer"
        / candidate_id
    )
    job_directory.mkdir(parents=True, exist_ok=True)

    prediction_frame = pd.DataFrame(
        {
            "sample_id": test[sample_column].astype(str),
            "y_true": test[contract["target_column"]].astype(float),
            "y_pred": prediction_values,
        }
    )
    prediction_frame.to_csv(
        job_directory / "predictions.csv",
        index=False,
    )

    fold_payload = {
        "split_id": split.split_id,
        "split_fingerprint": split.id_hash,
        "train_ids": train[sample_column].astype(str).tolist(),
        "validation_ids": (
            validation[sample_column].astype(str).tolist()
        ),
        "test_ids": test[sample_column].astype(str).tolist(),
        "algorithm": split.algorithm,
        "random_state": split.random_state,
        "grouping_field": split.grouping_field,
    }
    atomic_json(job_directory / "fold_ids.json", fold_payload)

    feature_order = {
        "weather": list(dataset.weather_features),
        "privileged": list(dataset.privileged_features),
    }
    atomic_json(
        job_directory / "feature_order.json",
        feature_order,
    )

    checkpoint_hash = sha256_file(checkpoint)
    checkpoint_metadata = loaded.get("metadata", {})

    lineage = {
        "dependency_job_id": source_route_job_id(
            source_dataset=source_dataset,
            source_route=source_route,
            seed=seed,
        ),
        "source_dataset": source_dataset,
        "source_route": source_route,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata,
    }
    atomic_json(
        job_directory / "checkpoint_lineage.json",
        lineage,
    )

    execution_job = {
        **job,
        "candidate_id": candidate_id,
        "job_family": "source_outer",
    }
    fingerprint = build_fingerprint(
        execution_job,
        namespace=str(output_root),
        extra={
            "dataset_path": str(dataset_path),
            "split": split.id_hash,
            "checkpoint_hash": checkpoint_hash,
        },
        resolved_config=config,
    )

    manifest = {
        "schema_version": "source_outer_manifest_v1",
        "job": execution_job,
        "fingerprint": fingerprint,
        "code_tree_hash": code_tree_hash(),
        "resolved_config_hash": stable_hash(config),
        "data_hash": sha256_file(dataset_path),
        "dataset_path": str(dataset_path),
        "split_fingerprint": split.id_hash,
        "namespace": str(output_root),
        "source_checkpoint_hash": checkpoint_hash,
        "source_dependency_job_id": lineage["dependency_job_id"],
        "feature_order_hash": stable_hash(feature_order),
        "train_ids_hash": stable_hash(fold_payload["train_ids"]),
        "validation_ids_hash": stable_hash(
            fold_payload["validation_ids"]
        ),
        "test_ids_hash": stable_hash(fold_payload["test_ids"]),
        "reported_evaluation_metrics": metrics,
        "smoke_only": smoke,
    }
    atomic_json(job_directory / "manifest.json", manifest)

    atomic_json(
        job_directory / "training_ledger.json",
        {
            "handler": "execute_source_outer",
            "training_performed": False,
            "outer_forward_only": True,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_hash,
            "test_labels_used_for_training": False,
            "smoke_only": smoke,
        },
    )

    acceptance = accept_artifacts(
        job_directory,
        expected_fingerprint=fingerprint,
    )

    return {
        "status": "COMPLETED",
        "job_id": candidate_id,
        "handler": "execute_source_outer",
        "acceptance": acceptance,
        "checkpoint_sha256": checkpoint_hash,
        "smoke_only": smoke,
    }


SOURCE_HANDLER_REGISTRY = {
    "primary_source_routes": execute_primary_source_route,
    "source_outer": execute_source_outer,
}
