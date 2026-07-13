from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.distillation_v4.outer import fit_outer_route
from src.distillation_v4.universal_weather_v2.campaign import (
    checkpoint_reload_smoke,
)
from src.distillation_v4.universal_weather_v2.training import (
    prepare_arrays,
)

from .artifacts import atomic_json, sha256_file, stable_hash
from .datasets import load_frame
from .execution import _dataset
from .source_handlers import (
    execute_primary_source_route,
    source_route_directory,
)
from .splits import build_contract_splits


TEACHER_REQUIRED_ROUTES = {
    "prediction_kd",
    "representation_kd",
    "combined_kd",
    "missing_aware",
}

PRIMARY_ROUTES = {
    "supervised",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
    "missing_aware",
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _find_validation_metric(payload: Any) -> tuple[str, float] | None:
    """Find a validation-only loss without accepting test metrics."""

    candidates: list[tuple[str, float]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                lowered = child_path.lower()

                metric = _safe_float(child)
                if (
                    metric is not None
                    and "validation" in lowered
                    and any(
                        token in lowered
                        for token in ("mae", "rmse", "loss")
                    )
                    and "test" not in lowered
                    and "outer" not in lowered
                ):
                    candidates.append((child_path, metric))

                visit(child, child_path)

        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[1])


def _source_frames(
    config: dict[str, Any],
    repository_root: Path,
    source_dataset: str,
    seed: int,
) -> tuple[Any, Any, Any, Any, Path, dict[str, Any]]:
    contract = config["source_datasets"][source_dataset]
    frame, path = load_frame(contract, repository_root)

    split = build_contract_splits(
        frame,
        contract,
        seed=seed,
    )[0]

    sample = contract["sample_id_column"]
    indexed = (
        frame.assign(_sample_id=frame[sample].astype(str))
        .set_index("_sample_id", drop=False)
    )

    train = indexed.loc[list(split.train_ids)].reset_index(drop=True)
    validation = indexed.loc[
        list(split.validation_ids)
    ].reset_index(drop=True)
    test = indexed.loc[list(split.test_ids)].reset_index(drop=True)

    return frame, train, validation, test, path, contract


def execute_source_screening(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Train a declared source route and mechanically reload its checkpoint."""

    source_dataset = str(job["source_dataset"])
    source_route = str(job["source_route"])
    seed = int(job["seed"])

    if source_route not in PRIMARY_ROUTES:
        raise ValueError(
            f"UNSUPPORTED_SOURCE_SCREENING_ROUTE:{source_route}"
        )

    route_result = execute_primary_source_route(
        config,
        job,
        repository_root=repository_root,
        output_root=output_root,
        smoke=smoke,
    )

    checkpoint = Path(route_result["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SCREENING_CHECKPOINT_MISSING:{checkpoint}"
        )

    (
        frame,
        train,
        validation,
        test,
        dataset_path,
        contract,
    ) = _source_frames(
        config,
        repository_root,
        source_dataset,
        seed,
    )

    if smoke:
        sample = contract["sample_id_column"]
        train = train.sort_values(sample).head(256).copy()
        validation = validation.sort_values(sample).head(64).copy()
        test = test.sort_values(sample).head(64).copy()

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

    reload_result = checkpoint_reload_smoke(
        checkpoint_path=checkpoint,
        arrays=arrays,
        feature_names=list(dataset.weather_features),
    )

    reload_status = str(
        reload_result.get("status", "")
    ).upper()

    reload_passed = bool(
        reload_result.get("all_passed") is True
        or reload_status in {
            "PASS",
            "PASSED",
            "COMPLETED",
            "VERIFIED",
        }
    )

    maximum_difference = reload_result.get(
        "maximum_prediction_difference"
    )

    if maximum_difference is not None:
        try:
            maximum_difference = float(maximum_difference)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "SOURCE_CHECKPOINT_RELOAD_DIFFERENCE_INVALID:"
                f"{source_dataset}:{source_route}:{seed}:"
                f"{reload_result}"
            ) from error

        if not np.isfinite(maximum_difference):
            reload_passed = False

    if not reload_passed:
        raise RuntimeError(
            "SOURCE_CHECKPOINT_RELOAD_SCREENING_FAILED:"
            f"{source_dataset}:{source_route}:{seed}:"
            f"{reload_result}"
        )

    directory = source_route_directory(
        output_root=output_root,
        source_dataset=source_dataset,
        source_route=source_route,
        seed=seed,
        smoke=smoke,
    )

    payload = {
        "schema_version": "source_screening_handler_v1",
        "status": "SCREENING_ACCEPTED",
        "source_dataset": source_dataset,
        "source_route": source_route,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_reload": reload_result,
        "validation_only_selection_required": True,
        "outer_test_used_for_selection": False,
        "smoke_only": smoke,
    }
    payload["screening_fingerprint"] = stable_hash(payload)

    atomic_json(directory / "screening.json", payload)
    return payload


def execute_teacher_qualification(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Qualify the teacher produced by a screened source route."""

    source_dataset = str(job["source_dataset"])
    source_route = str(job["source_route"])
    seed = int(job["seed"])

    directory = source_route_directory(
        output_root=output_root,
        source_dataset=source_dataset,
        source_route=source_route,
        seed=seed,
        smoke=smoke,
    )

    screening_path = directory / "screening.json"
    handler_path = directory / "remediation_source_handler.json"

    if not screening_path.is_file():
        raise FileNotFoundError(
            "TEACHER_QUALIFICATION_REQUIRES_SCREENING:"
            f"{source_dataset}:{source_route}:{seed}"
        )
    if not handler_path.is_file():
        raise FileNotFoundError(
            "SOURCE_HANDLER_ARTIFACT_REQUIRED_FOR_QUALIFICATION:"
            f"{source_dataset}:{source_route}:{seed}"
        )

    screening = _json(screening_path)
    handler = _json(handler_path)

    if screening.get("status") != "SCREENING_ACCEPTED":
        raise ValueError("SOURCE_SCREENING_NOT_ACCEPTED")

    result = handler.get("legacy_source_training_result", {})
    if not isinstance(result, dict):
        raise ValueError("SOURCE_TRAINING_RESULT_REQUIRED")

    teacher_path_value = result.get("teacher_checkpoint")
    teacher_required = source_route in TEACHER_REQUIRED_ROUTES

    if teacher_required and not teacher_path_value:
        raise ValueError(
            f"ROUTE_REQUIRES_TEACHER_CHECKPOINT:{source_route}"
        )

    teacher_path = (
        Path(teacher_path_value)
        if teacher_path_value
        else None
    )

    if teacher_path is not None and not teacher_path.is_file():
        raise FileNotFoundError(
            f"TEACHER_CHECKPOINT_MISSING:{teacher_path}"
        )

    teacher_frozen = result.get("teacher_frozen")

    if teacher_required and teacher_frozen is not True:
        raise ValueError(
            f"TEACHER_NOT_VERIFIED_FROZEN:{source_route}"
        )

    payload = {
        "schema_version": "teacher_qualification_handler_v1",
        "status": (
            "TEACHER_QUALIFIED"
            if teacher_required
            else "TEACHER_NOT_REQUIRED"
        ),
        "source_dataset": source_dataset,
        "source_route": source_route,
        "seed": seed,
        "teacher_required": teacher_required,
        "teacher_frozen": (
            bool(teacher_frozen)
            if teacher_required
            else None
        ),
        "teacher_checkpoint": (
            str(teacher_path)
            if teacher_path is not None
            else None
        ),
        "teacher_checkpoint_sha256": (
            sha256_file(teacher_path)
            if teacher_path is not None
            else None
        ),
        "student_checkpoint": handler["checkpoint"],
        "student_checkpoint_sha256": handler[
            "checkpoint_sha256"
        ],
        "screening_fingerprint": screening[
            "screening_fingerprint"
        ],
        "outer_test_used_for_qualification": False,
        "smoke_only": smoke,
    }
    payload["qualification_fingerprint"] = stable_hash(payload)

    atomic_json(
        directory / "teacher_qualification.json",
        payload,
    )
    return payload


def execute_multimodal_pretrain_finetune(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Run the recovered multimodal pretrain/fine-tune implementation."""

    source_dataset = str(job["source_dataset"])
    seed = int(job["seed"])

    contract = config["source_datasets"][source_dataset]
    frame, dataset_path = load_frame(contract, repository_root)

    weather = list(contract.get("weather_columns", []))
    soil = list(contract.get("soil_columns", []))

    if not weather or not soil:
        raise ValueError(
            "MULTIMODAL_PRETRAIN_FINETUNE_REQUIRES_WEATHER_AND_SOIL"
        )

    target = contract["target_column"]
    year_column = contract["year_column"]
    sample_column = contract["sample_id_column"]

    renamed = frame.copy()

    rename_map: dict[str, str] = {}
    if target != "target_yield":
        rename_map[target] = "target_yield"
    if year_column != "year":
        rename_map[year_column] = "year"
    if sample_column != "sample_id":
        rename_map[sample_column] = "sample_id"

    renamed = renamed.rename(columns=rename_map)

    years = sorted(
        renamed["year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if len(years) < 3:
        raise ValueError(
            "MULTIMODAL_PRETRAIN_FINETUNE_INSUFFICIENT_YEARS"
        )

    outer_test_year = years[-1]
    destination = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source_multimodal_pretrain_finetune"
        / source_dataset
        / f"seed_{seed}"
    )

    route_fingerprint = stable_hash(
        {
            "source_dataset": source_dataset,
            "route": "multimodal_pretrain_finetune",
            "legacy_route": "fine_tune",
            "seed": seed,
            "data_hash": sha256_file(dataset_path),
            "outer_test_year": outer_test_year,
            "smoke": smoke,
        }
    )

    result = fit_outer_route(
        route="fine_tune",
        frame=renamed,
        outer_test_year=outer_test_year,
        seed=seed,
        output=destination,
        route_fingerprint=route_fingerprint,
        max_epochs=2 if smoke else 80,
        execution_namespace=str(output_root),
    )

    pretraining = destination / "multimodal_pretraining.pt"
    if not pretraining.is_file():
        raise FileNotFoundError(
            "MULTIMODAL_PRETRAINING_ARTIFACT_MISSING"
        )

    payload = {
        "schema_version": (
            "multimodal_pretrain_finetune_handler_v1"
        ),
        "status": "COMPLETED",
        "source_dataset": source_dataset,
        "seed": seed,
        "route": "multimodal_pretrain_finetune",
        "legacy_execution_route": "fine_tune",
        "outer_test_year": outer_test_year,
        "outer_test_used_for_selection": False,
        "multimodal_pretraining_checkpoint": str(pretraining),
        "multimodal_pretraining_sha256": sha256_file(
            pretraining
        ),
        "route_fingerprint": route_fingerprint,
        "result": result,
        "smoke_only": smoke,
    }
    payload["handler_fingerprint"] = stable_hash(payload)

    atomic_json(destination / "handler.json", payload)
    return payload


def execute_source_controls(
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Select by validation-only metric and freeze one immutable checkpoint."""

    source_dataset = str(job["source_dataset"])

    source_root = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source"
        / source_dataset
    )

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"SOURCE_CONTROL_INPUT_ROOT_MISSING:{source_root}"
        )

    candidates: list[dict[str, Any]] = []

    for handler_path in sorted(
        source_root.glob("*/seed_*/remediation_source_handler.json")
    ):
        directory = handler_path.parent

        screening_path = directory / "screening.json"
        qualification_path = (
            directory / "teacher_qualification.json"
        )

        if not screening_path.is_file():
            continue
        if not qualification_path.is_file():
            continue

        handler = _json(handler_path)
        screening = _json(screening_path)
        qualification = _json(qualification_path)

        if screening.get("status") != "SCREENING_ACCEPTED":
            continue
        if qualification.get("status") not in {
            "TEACHER_QUALIFIED",
            "TEACHER_NOT_REQUIRED",
        }:
            continue

        legacy = handler.get(
            "legacy_source_training_result",
            {},
        )
        metric = _find_validation_metric(legacy)

        if metric is None:
            continue

        checkpoint = Path(handler["checkpoint"])
        if not checkpoint.is_file():
            continue

        candidates.append(
            {
                "source_route": handler["source_route"],
                "seed": int(handler["seed"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_metric_path": metric[0],
                "validation_metric_value": metric[1],
                "screening_fingerprint": screening[
                    "screening_fingerprint"
                ],
                "qualification_fingerprint": qualification[
                    "qualification_fingerprint"
                ],
            }
        )

    if not candidates:
        raise ValueError(
            f"NO_VALIDATION_QUALIFIED_SOURCE_CANDIDATES:"
            f"{source_dataset}"
        )

    selected = min(
        candidates,
        key=lambda item: (
            item["validation_metric_value"],
            item["source_route"],
            item["seed"],
        ),
    )

    control_root = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source_controls"
        / source_dataset
    )
    control_root.mkdir(parents=True, exist_ok=True)

    frozen_checkpoint = (
        control_root / "frozen_source_checkpoint.pt"
    )
    temporary = frozen_checkpoint.with_suffix(".pt.tmp")
    shutil.copy2(selected["checkpoint"], temporary)
    temporary.replace(frozen_checkpoint)

    frozen_hash = sha256_file(frozen_checkpoint)
    if frozen_hash != selected["checkpoint_sha256"]:
        raise ValueError("FROZEN_CHECKPOINT_HASH_MISMATCH")

    selection = {
        "schema_version": "source_selection_control_v1",
        "status": "COMPLETED",
        "selection_scope": "VALIDATION_ONLY",
        "outer_test_used_for_selection": False,
        "source_dataset": source_dataset,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "smoke_only": smoke,
    }
    selection["selection_fingerprint"] = stable_hash(selection)
    atomic_json(control_root / "selection.json", selection)

    freeze = {
        "schema_version": "source_checkpoint_freeze_v1",
        "status": "COMPLETED",
        "source_dataset": source_dataset,
        "selected_source_route": selected["source_route"],
        "selected_seed": selected["seed"],
        "selection_metric_path": selected[
            "validation_metric_path"
        ],
        "selection_metric_value": selected[
            "validation_metric_value"
        ],
        "source_checkpoint": selected["checkpoint"],
        "source_checkpoint_sha256": selected[
            "checkpoint_sha256"
        ],
        "frozen_checkpoint": str(frozen_checkpoint),
        "frozen_checkpoint_sha256": frozen_hash,
        "selection_fingerprint": selection[
            "selection_fingerprint"
        ],
        "immutable_downstream_dependency": True,
        "smoke_only": smoke,
    }
    freeze["freeze_fingerprint"] = stable_hash(freeze)
    atomic_json(control_root / "freeze.json", freeze)

    return {
        "status": "COMPLETED",
        "source_dataset": source_dataset,
        "selection": selection,
        "freeze": freeze,
        "smoke_only": smoke,
    }


REMAINING_HANDLER_REGISTRY = {
    "source_screening": execute_source_screening,
    "teacher_qualification": execute_teacher_qualification,
    "multimodal_pretrain_finetune": (
        execute_multimodal_pretrain_finetune
    ),
    "source_controls": execute_source_controls,
}
