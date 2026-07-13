from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from .phase5_fingerprint import (
    validate_frozen_phase5_matrix,
)
from .phase5_sources import (
    load_validated_phase4_source,
)


CONTROL_INTERPRETATION = {
    "soil_representation_sensitivity_control": (
        "SOIL_REPRESENTATION_SENSITIVITY"
    ),
    "missing_aware_ablation_suite_control": (
        "MISSING_MODALITY_MASKING_AND_RATE_SENSITIVITY"
    ),
    "active_compute_steps_control": (
        "COMPUTE_FAIRNESS"
    ),
    "shuffled_teacher_prediction_control": (
        "PREDICTION_TEACHER_INFORMATION"
    ),
    "random_representation_transfer_control": (
        "REPRESENTATION_INFORMATION"
    ),
    "ckd_balancing_sensitivity_control": (
        "LOSS_BALANCING_SENSITIVITY"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"PHASE5_REQUIRED_FILE_MISSING:{path.name}"
        )

    return json.loads(path.read_text())


def _finite_float(
    value: Any,
    *,
    field: str,
) -> float:
    converted = float(value)

    if not math.isfinite(converted):
        raise RuntimeError(
            f"PHASE5_NONFINITE_FIELD:{field}"
        )

    return converted


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError(
            "PHASE5_EMPTY_SUMMARY_VALUES"
        )

    return float(np.mean(values))


def _load_phase5_records(
    output: Path,
) -> list[dict[str, Any]]:
    root = (
        output
        / "development/phase5_repair_matrix"
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
            "DUPLICATE_PHASE5_JOB_RECORD"
        )

    for record in records:
        if record.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "UNACCEPTED_PHASE5_JOB_RECORD"
            )

        if record.get("outer_test_used") is not False:
            raise RuntimeError(
                "OUTER_TEST_PHASE5_AGGREGATION_FORBIDDEN"
            )

        _finite_float(
            record.get("validation_mae"),
            field="validation_mae",
        )

    return records


def _mechanism_interpretation(
    *,
    control: str,
    source_gain_over_control: list[float],
) -> dict[str, Any]:
    mean_gain = _mean(
        source_gain_over_control
    )
    positive_fraction = float(
        np.mean(
            [
                value > 0.0
                for value in source_gain_over_control
            ]
        )
    )

    if mean_gain > 0.0:
        status = "CONTROL_SEPARATION_OBSERVED"
    elif mean_gain < 0.0:
        status = "CONTROL_NOT_SEPARATED"
    else:
        status = "CONTROL_TIED"

    return {
        "mechanism_dimension": (
            CONTROL_INTERPRETATION[control]
        ),
        "development_control_status": status,
        "mean_source_gain_over_control": (
            mean_gain
        ),
        "median_source_gain_over_control": float(
            np.median(source_gain_over_control)
        ),
        "positive_separation_fraction": (
            positive_fraction
        ),
        "interpretation_scope": (
            "OUTER_TRAIN_DEVELOPMENT_DIAGNOSTIC_ONLY"
        ),
        "may_delete_phase4_route": False,
        "may_enable_outer_test": False,
    }


def aggregate_phase5_controls(
    *,
    output: Path,
) -> dict[str, Any]:
    manifest = validate_frozen_phase5_matrix(
        output
        / "control/frozen_phase5_job_matrix.json"
    )

    execution = _load_json(
        output
        / "status/phase5_repair_matrix_execution.json"
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE5_ENGINEERING_MATRIX_INCOMPLETE"
        )

    if int(execution.get("failed_jobs", -1)) != 0:
        raise RuntimeError(
            "PHASE5_ENGINEERING_FAILURES_PRESENT"
        )

    records = _load_phase5_records(output)

    planned_jobs = manifest.get("jobs")

    if not isinstance(planned_jobs, list):
        raise RuntimeError(
            "PHASE5_MANIFEST_JOBS_INVALID"
        )

    planned_ids = {
        str(job["job_id"])
        for job in planned_jobs
    }
    observed_ids = {
        str(record["job_id"])
        for record in records
    }

    missing = sorted(
        planned_ids.difference(observed_ids)
    )
    unexpected = sorted(
        observed_ids.difference(planned_ids)
    )

    if missing or unexpected:
        raise RuntimeError(
            "PHASE5_RECORD_MANIFEST_MISMATCH:"
            f"missing={missing}:"
            f"unexpected={unexpected}"
        )

    manifest_by_id = {
        str(job["job_id"]): job
        for job in planned_jobs
    }

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    matched_records: list[dict[str, Any]] = []

    for record in records:
        job_id = str(record["job_id"])
        job = manifest_by_id[job_id]

        for field in (
            "dataset",
            "fold",
            "seed",
            "control",
            "required_by_route",
            "source_route_id",
            "matched_s0_route_id",
        ):
            if record.get(field) != job.get(field):
                raise RuntimeError(
                    "PHASE5_RECORD_JOB_LINEAGE_MISMATCH:"
                    f"{job_id}:{field}"
                )

        source = load_validated_phase4_source(
            output=output,
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
            output=output,
            route_id=str(
                job["matched_s0_route_id"]
            ),
            expected_artifact_fingerprint=str(
                job[
                    "matched_s0_artifact_fingerprint"
                ]
            ),
        )

        if source.get("route") != job.get(
            "required_by_route"
        ):
            raise RuntimeError(
                "PHASE5_SOURCE_ROUTE_ROLE_MISMATCH:"
                + job_id
            )

        if baseline.get("route") != "supervised":
            raise RuntimeError(
                "PHASE5_MATCHED_BASELINE_NOT_SUPERVISED:"
                + job_id
            )

        for field in ("dataset", "fold", "seed"):
            expected = job[field]

            if source.get(field) != expected:
                raise RuntimeError(
                    "PHASE5_SOURCE_PAIRING_MISMATCH:"
                    f"{job_id}:{field}"
                )

            if baseline.get(field) != expected:
                raise RuntimeError(
                    "PHASE5_BASELINE_PAIRING_MISMATCH:"
                    f"{job_id}:{field}"
                )

        control_mae = _finite_float(
            record["validation_mae"],
            field="control_validation_mae",
        )
        source_mae = _finite_float(
            source["validation_mae"],
            field="source_validation_mae",
        )
        baseline_mae = _finite_float(
            baseline["validation_mae"],
            field="baseline_validation_mae",
        )

        source_gain_over_control = (
            control_mae - source_mae
        )
        source_gain_over_s0 = (
            baseline_mae - source_mae
        )
        control_gain_over_s0 = (
            baseline_mae - control_mae
        )

        matched = {
            "job_id": job_id,
            "dataset": str(job["dataset"]),
            "fold": str(job["fold"]),
            "seed": int(job["seed"]),
            "control": str(job["control"]),
            "required_by_route": str(
                job["required_by_route"]
            ),
            "source_route_id": str(
                source["route_id"]
            ),
            "matched_s0_route_id": str(
                baseline["route_id"]
            ),
            "source_validation_mae": source_mae,
            "control_validation_mae": control_mae,
            "s0_validation_mae": baseline_mae,
            "source_gain_over_control": (
                source_gain_over_control
            ),
            "source_gain_over_s0": (
                source_gain_over_s0
            ),
            "control_gain_over_s0": (
                control_gain_over_s0
            ),
            "source_separates_from_control": (
                source_gain_over_control > 0.0
            ),
            "control_outperforms_source": (
                source_gain_over_control < 0.0
            ),
            "outer_test_used": False,
        }

        if job["control"] == (
            "active_compute_steps_control"
        ):
            metadata = record.get(
                "control_metadata"
            )

            if not isinstance(metadata, dict):
                raise RuntimeError(
                    "PHASE5_COMPUTE_METADATA_MISSING:"
                    + job_id
                )

            target_steps = int(
                metadata["target_optimizer_steps"]
            )
            actual_steps = int(
                metadata["actual_optimizer_steps"]
            )

            if target_steps != int(
                source["optimizer_steps"]
            ):
                raise RuntimeError(
                    "PHASE5_COMPUTE_TARGET_SOURCE_MISMATCH:"
                    + job_id
                )

            if actual_steps != target_steps:
                raise RuntimeError(
                    "PHASE5_COMPUTE_STEP_MATCH_FAILED:"
                    + job_id
                )

            source_fit_time = _finite_float(
                source["fit_time_seconds"],
                field="source_fit_time_seconds",
            )
            control_fit_time = _finite_float(
                record["fit_time_seconds"],
                field="control_fit_time_seconds",
            )

            matched[
                "optimizer_steps_exact_match"
            ] = True
            matched["fit_time_difference_seconds"] = (
                control_fit_time - source_fit_time
            )
            matched["fit_time_ratio"] = (
                control_fit_time / source_fit_time
                if source_fit_time > 0.0
                else None
            )

        grouped[
            (
                str(job["dataset"]),
                str(job["control"]),
            )
        ].append(matched)

        matched_records.append(matched)

    dataset_summaries: dict[str, Any] = {}

    for dataset in sorted(
        {
            str(job["dataset"])
            for job in planned_jobs
        }
    ):
        control_summaries: dict[str, Any] = {}

        controls = sorted(
            {
                str(job["control"])
                for job in planned_jobs
                if str(job["dataset"]) == dataset
            }
        )

        for control in controls:
            pairs = grouped[(dataset, control)]

            source_gains = [
                float(
                    pair[
                        "source_gain_over_control"
                    ]
                )
                for pair in pairs
            ]

            summary = {
                "pair_count": len(pairs),
                **_mechanism_interpretation(
                    control=control,
                    source_gain_over_control=(
                        source_gains
                    ),
                ),
                "mean_source_gain_over_s0": (
                    _mean(
                        [
                            float(
                                pair[
                                    "source_gain_over_s0"
                                ]
                            )
                            for pair in pairs
                        ]
                    )
                ),
                "mean_control_gain_over_s0": (
                    _mean(
                        [
                            float(
                                pair[
                                    "control_gain_over_s0"
                                ]
                            )
                            for pair in pairs
                        ]
                    )
                ),
                "pairs": pairs,
            }

            if control == (
                "active_compute_steps_control"
            ):
                summary[
                    "optimizer_steps_exact_match"
                ] = all(
                    bool(
                        pair[
                            "optimizer_steps_exact_match"
                        ]
                    )
                    for pair in pairs
                )
                summary[
                    "mean_fit_time_difference_seconds"
                ] = _mean(
                    [
                        float(
                            pair[
                                "fit_time_difference_seconds"
                            ]
                        )
                        for pair in pairs
                    ]
                )

            control_summaries[control] = summary

        dataset_summaries[dataset] = {
            "controls": control_summaries,
            "outer_test_used": False,
            "claim_boundary": (
                "DEVELOPMENT_MECHANISM_DIAGNOSTICS_ONLY"
            ),
        }

    return {
        "phase": "phase5",
        "status": (
            "CONTROL_AGGREGATION_COMPLETE"
        ),
        "phase5_job_matrix_fingerprint": (
            manifest[
                "phase5_job_matrix_fingerprint"
            ]
        ),
        "planned_jobs": len(planned_ids),
        "observed_jobs": len(observed_ids),
        "datasets": dataset_summaries,
        "matched_records": matched_records,
        "outer_test_used": False,
        "control_results_may_delete_phase4_routes": (
            False
        ),
        "control_results_may_enable_outer_test": (
            False
        ),
        "outer_test_may_enable_downstream_jobs": False,
    }
