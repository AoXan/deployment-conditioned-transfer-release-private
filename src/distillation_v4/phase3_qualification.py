from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
from typing import Any, Iterable

import numpy as np

from .teacher_registry import sha256_file


PREDICTION_CANDIDATES = {
    "hgb_privileged",
    "shared_early_fusion",
}

REPRESENTATION_CANDIDATES = {
    "shared_early_fusion",
    "modality_specific_late_fusion",
}


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _finite(record: dict[str, Any], field: str) -> float:
    value = float(record[field])
    if not math.isfinite(value):
        raise RuntimeError(
            f"NONFINITE_PHASE3_QUALIFICATION_FIELD:"
            f"{record.get('job_id')}:{field}"
        )
    return value


def _mean(
    records: Iterable[dict[str, Any]],
    field: str,
) -> float:
    values = [
        _finite(record, field)
        for record in records
    ]
    if not values:
        raise RuntimeError(
            f"EMPTY_PHASE3_QUALIFICATION_VALUES:{field}"
        )
    return float(np.mean(values))


def load_phase3_records(
    output: Path,
) -> list[dict[str, Any]]:
    root = (
        output
        / "development/phase3_repair_matrix"
    )

    records = [
        json.loads(path.read_text())
        for path in sorted(
            root.glob("**/job_record.json")
        )
    ]

    identifiers = [
        str(record["job_id"])
        for record in records
    ]

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "DUPLICATE_PHASE3_JOB_RECORD"
        )

    for record in records:
        if record.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "UNACCEPTED_PHASE3_JOB_RECORD"
            )
        if record.get("outer_test_used") is not False:
            raise RuntimeError(
                "OUTER_TEST_PHASE3_QUALIFICATION_FORBIDDEN"
            )

    return records


def _group(
    records: list[dict[str, Any]],
) -> dict[
    tuple[str, str, str],
    list[dict[str, Any]],
]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped[
            (
                str(record["dataset"]),
                str(record["fold"]),
                str(record["candidate"]),
            )
        ].append(record)

    return dict(grouped)


def _checkpoint_record(
    records: list[dict[str, Any]],
    *,
    score_field: str,
) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record.get("checkpoint")
    ]
    if not eligible:
        raise RuntimeError(
            "QUALIFIED_TEACHER_CHECKPOINT_UNAVAILABLE"
        )

    return min(
        eligible,
        key=lambda record: (
            _finite(record, score_field),
            int(record["seed"]),
        ),
    )


def qualify_phase3_teachers(
    *,
    output: Path,
    datasets: Iterable[str],
    folds_by_dataset: dict[str, tuple[str, ...]],
    expected_jobs: int,
    replay_tolerance: float = 1e-6,
) -> dict[str, Any]:
    records = load_phase3_records(output)

    if len(records) != expected_jobs:
        raise RuntimeError(
            "PHASE3_QUALIFICATION_JOB_COUNT_MISMATCH:"
            f"{len(records)}:{expected_jobs}"
        )

    grouped = _group(records)
    selected: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}

    for dataset in datasets:
        dataset_gate: dict[str, Any] = {
            "dataset": dataset,
            "selection_scope": (
                "OUTER_TRAIN_INNER_VALIDATION_ONLY"
            ),
            "outer_test_used": False,
            "folds": {},
        }

        for fold in folds_by_dataset[dataset]:
            hgb_teacher = grouped.get(
                (dataset, fold, "hgb_privileged"),
                [],
            )
            hgb_control = grouped.get(
                (dataset, fold, "hgb_deployable"),
                [],
            )
            shared_teacher = grouped.get(
                (dataset, fold, "shared_early_fusion"),
                [],
            )
            modality_teacher = grouped.get(
                (
                    dataset,
                    fold,
                    "modality_specific_late_fusion",
                ),
                [],
            )
            shared_control = grouped.get(
                (
                    dataset,
                    fold,
                    "shared_neural_receiver",
                ),
                [],
            )
            random_control = grouped.get(
                (
                    dataset,
                    fold,
                    "random_representation",
                ),
                [],
            )
            shuffled_control = grouped.get(
                (
                    dataset,
                    fold,
                    "shuffled_soil_neural",
                ),
                [],
            )

            required_counts = {
                "hgb_privileged": (
                    len(hgb_teacher),
                    1,
                ),
                "hgb_deployable": (
                    len(hgb_control),
                    1,
                ),
                "shared_early_fusion": (
                    len(shared_teacher),
                    3,
                ),
                "modality_specific_late_fusion": (
                    len(modality_teacher),
                    3,
                ),
                "shared_neural_receiver": (
                    len(shared_control),
                    3,
                ),
                "random_representation": (
                    len(random_control),
                    3,
                ),
                "shuffled_soil_neural": (
                    len(shuffled_control),
                    3,
                ),
            }

            invalid_counts = {
                candidate: {
                    "observed": observed,
                    "expected": expected,
                }
                for candidate, (
                    observed,
                    expected,
                ) in required_counts.items()
                if observed != expected
            }

            if invalid_counts:
                raise RuntimeError(
                    "PHASE3_FOLD_SLOT_COUNT_MISMATCH:"
                    f"{dataset}:{fold}:"
                    f"{json.dumps(invalid_counts, sort_keys=True)}"
                )

            hgb_teacher_mae = _mean(
                hgb_teacher,
                "validation_mae",
            )
            hgb_control_mae = _mean(
                hgb_control,
                "validation_mae",
            )

            shared_teacher_mae = _mean(
                shared_teacher,
                "validation_mae",
            )
            shared_control_mae = _mean(
                shared_control,
                "validation_mae",
            )

            prediction_candidates = {
                "hgb_privileged": {
                    "qualified": (
                        hgb_teacher_mae
                        < hgb_control_mae
                    ),
                    "teacher_mae": hgb_teacher_mae,
                    "matched_control_mae": (
                        hgb_control_mae
                    ),
                    "improvement": (
                        hgb_control_mae
                        - hgb_teacher_mae
                    ),
                },
                "shared_early_fusion": {
                    "qualified": (
                        shared_teacher_mae
                        < shared_control_mae
                    ),
                    "teacher_mae": shared_teacher_mae,
                    "matched_control_mae": (
                        shared_control_mae
                    ),
                    "improvement": (
                        shared_control_mae
                        - shared_teacher_mae
                    ),
                },
            }

            random_probe_mae = _mean(
                random_control,
                "representation_probe_mae",
            )
            shuffled_probe_mae = _mean(
                shuffled_control,
                "representation_probe_mae",
            )

            representation_candidates = {}

            for candidate, candidate_records in (
                (
                    "shared_early_fusion",
                    shared_teacher,
                ),
                (
                    "modality_specific_late_fusion",
                    modality_teacher,
                ),
            ):
                candidate_probe = _mean(
                    candidate_records,
                    "representation_probe_mae",
                )
                replay_valid = all(
                    _finite(
                        record,
                        "checkpoint_replay_max_abs_error",
                    )
                    <= replay_tolerance
                    for record in candidate_records
                )
                representation_valid = all(
                    _finite(
                        record,
                        "representation_std",
                    )
                    > 1e-12
                    for record in candidate_records
                )

                representation_candidates[candidate] = {
                    "qualified": bool(
                        candidate_probe
                        < random_probe_mae
                        and candidate_probe
                        < shuffled_probe_mae
                        and replay_valid
                        and representation_valid
                    ),
                    "probe_mae": candidate_probe,
                    "random_control_probe_mae": (
                        random_probe_mae
                    ),
                    "shuffled_soil_probe_mae": (
                        shuffled_probe_mae
                    ),
                    "random_control_improvement": (
                        random_probe_mae
                        - candidate_probe
                    ),
                    "shuffled_control_improvement": (
                        shuffled_probe_mae
                        - candidate_probe
                    ),
                    "checkpoint_replay_valid": (
                        replay_valid
                    ),
                    "representation_non_degenerate": (
                        representation_valid
                    ),
                }

            qualified_prediction = [
                candidate
                for candidate, evidence
                in prediction_candidates.items()
                if evidence["qualified"]
            ]
            qualified_representation = [
                candidate
                for candidate, evidence
                in representation_candidates.items()
                if evidence["qualified"]
            ]

            selected_prediction = None
            if qualified_prediction:
                selected_prediction = min(
                    qualified_prediction,
                    key=lambda candidate: (
                        prediction_candidates[
                            candidate
                        ]["teacher_mae"],
                        candidate,
                    ),
                )

                source_records = grouped[
                    (
                        dataset,
                        fold,
                        selected_prediction,
                    )
                ]
                checkpoint = _checkpoint_record(
                    source_records,
                    score_field="validation_mae",
                )

                selected.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "role": "prediction",
                        "candidate": (
                            selected_prediction
                        ),
                        "decision": "DEVELOPMENT_OPEN",
                        "seed": checkpoint["seed"],
                        "checkpoint": (
                            checkpoint["checkpoint"]
                        ),
                        "checkpoint_sha256": (
                            checkpoint[
                                "checkpoint_sha256"
                            ]
                        ),
                        "preprocessor": (
                            checkpoint.get("preprocessor")
                        ),
                        "preprocessor_sha256": (
                            checkpoint.get(
                                "preprocessor_sha256"
                            )
                        ),
                        "source_job_id": (
                            checkpoint["job_id"]
                        ),
                        "source_component": (
                            checkpoint.get("component")
                        ),
                        "model_family": (
                            checkpoint.get("model_family")
                        ),
                        "selection_scope": (
                            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                        ),
                        "outer_refit_allowed": False,
                    }
                )

            selected_representation = None
            if qualified_representation:
                selected_representation = min(
                    qualified_representation,
                    key=lambda candidate: (
                        representation_candidates[
                            candidate
                        ]["probe_mae"],
                        candidate,
                    ),
                )

                source_records = grouped[
                    (
                        dataset,
                        fold,
                        selected_representation,
                    )
                ]
                checkpoint = _checkpoint_record(
                    source_records,
                    score_field=(
                        "representation_probe_mae"
                    ),
                )

                selected.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "role": "representation",
                        "candidate": (
                            selected_representation
                        ),
                        "decision": "DEVELOPMENT_OPEN",
                        "seed": checkpoint["seed"],
                        "checkpoint": (
                            checkpoint["checkpoint"]
                        ),
                        "checkpoint_sha256": (
                            checkpoint[
                                "checkpoint_sha256"
                            ]
                        ),
                        "preprocessor": (
                            checkpoint.get("preprocessor")
                        ),
                        "preprocessor_sha256": (
                            checkpoint.get(
                                "preprocessor_sha256"
                            )
                        ),
                        "source_job_id": (
                            checkpoint["job_id"]
                        ),
                        "source_component": (
                            checkpoint.get("component")
                        ),
                        "model_family": (
                            checkpoint.get("model_family")
                        ),
                        "selection_scope": (
                            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                        ),
                        "outer_refit_allowed": False,
                    }
                )

            dataset_gate["folds"][fold] = {
                "prediction_candidates": (
                    prediction_candidates
                ),
                "representation_candidates": (
                    representation_candidates
                ),
                "selected_prediction_teacher": (
                    selected_prediction
                ),
                "selected_representation_teacher": (
                    selected_representation
                ),
            }

        fold_values = list(
            dataset_gate["folds"].values()
        )
        prediction_complete = all(
            fold[
                "selected_prediction_teacher"
            ]
            is not None
            for fold in fold_values
        )
        representation_complete = all(
            fold[
                "selected_representation_teacher"
            ]
            is not None
            for fold in fold_values
        )

        dataset_gate["prediction_teacher"] = (
            "DEVELOPMENT_OPEN"
            if prediction_complete
            else "DEVELOPMENT_BLOCKED"
        )
        dataset_gate[
            "representation_teacher"
        ] = (
            "DEVELOPMENT_OPEN"
            if representation_complete
            else "DEVELOPMENT_BLOCKED"
        )
        dataset_gate[
            "prediction_teacher_complete_across_folds"
        ] = prediction_complete
        dataset_gate[
            "representation_teacher_complete_across_folds"
        ] = representation_complete

        gate_path = (
            output
            / "control/gates"
            / f"{dataset}__teachers_repair.json"
        )
        _atomic_json(gate_path, dataset_gate)
        gates[dataset] = dataset_gate

    registry = {
        "status": "FROZEN_TEACHER_REGISTRY",
        "release_status": "FORMALLY_RELEASED",
        "outer_refit_allowed": False,
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "qualification_job_count": len(records),
        "teachers": selected,
    }

    registry_path = (
        output
        / "control/frozen_teacher_registry.json"
    )
    _atomic_json(registry_path, registry)

    return {
        "gates": gates,
        "registry": registry,
        "registry_path": str(
            registry_path.relative_to(output)
        ),
        "registry_sha256": sha256_file(
            registry_path
        ),
        "qualification_job_count": len(records),
    }
