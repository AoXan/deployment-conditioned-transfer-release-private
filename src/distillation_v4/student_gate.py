from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class StudentCredibilityDecision(Enum):
    OPEN = "STUDENT_CREDIBILITY_OPEN"
    BLOCKED = "STUDENT_CREDIBILITY_BLOCKED"
    INSUFFICIENT = "STUDENT_CREDIBILITY_INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class StudentCredibilityResult:
    decision: StudentCredibilityDecision
    evidence: dict[str, Any]


def evaluate_student_credibility(
    records: list[dict[str, Any]],
    *,
    data_scope: str,
    minimum_folds: int = 3,
    minimum_improved_folds: int = 2,
    maximum_mean_classical_gap: float = 0.25,
) -> StudentCredibilityResult:
    if data_scope != "OUTER_TRAIN_INNER_VALIDATION_ONLY":
        raise ValueError("OUTER_TEST_CREDIBILITY_INPUT_FORBIDDEN")

    if len(records) < minimum_folds:
        return StudentCredibilityResult(
            StudentCredibilityDecision.INSUFFICIENT,
            {
                "reason": "INSUFFICIENT_DEVELOPMENT_FOLDS",
                "records": len(records),
                "minimum_folds": minimum_folds,
            },
        )

    required = {
        "fold",
        "student_mae",
        "naive_mae",
        "classical_mae",
        "optimizer_steps",
        "parameter_delta_l2",
        "checkpoint_replay_max_abs_error",
        "prediction_std",
        "converged",
    }

    missing = sorted(
        {
            field
            for record in records
            for field in required.difference(record)
        }
    )
    if missing:
        return StudentCredibilityResult(
            StudentCredibilityDecision.INSUFFICIENT,
            {
                "reason": "CREDIBILITY_FIELDS_MISSING",
                "missing_fields": missing,
                "records": len(records),
            },
        )

    numeric_fields = (
        "student_mae",
        "naive_mae",
        "classical_mae",
        "optimizer_steps",
        "parameter_delta_l2",
        "checkpoint_replay_max_abs_error",
        "prediction_std",
    )

    for record in records:
        for field in numeric_fields:
            if not math.isfinite(float(record[field])):
                return StudentCredibilityResult(
                    StudentCredibilityDecision.BLOCKED,
                    {
                        "reason": "NONFINITE_CREDIBILITY_EVIDENCE",
                        "fold": record["fold"],
                        "field": field,
                    },
                )

    improved_over_naive = [
        float(record["student_mae"]) < float(record["naive_mae"])
        for record in records
    ]
    classical_gaps = [
        (
            float(record["student_mae"])
            - float(record["classical_mae"])
        )
        / max(float(record["classical_mae"]), 1e-12)
        for record in records
    ]

    training_valid = [
        int(record["optimizer_steps"]) > 0
        and float(record["parameter_delta_l2"]) > 0
        and float(record["checkpoint_replay_max_abs_error"]) <= 1e-6
        and float(record["prediction_std"]) > 1e-12
        and bool(record["converged"])
        for record in records
    ]

    improved_count = sum(improved_over_naive)
    mean_classical_gap = sum(classical_gaps) / len(classical_gaps)

    evidence = {
        "records": len(records),
        "improved_over_naive_folds": improved_count,
        "minimum_improved_folds": minimum_improved_folds,
        "mean_relative_gap_to_classical": mean_classical_gap,
        "maximum_mean_classical_gap": maximum_mean_classical_gap,
        "all_training_audits_valid": all(training_valid),
        "all_checkpoint_replays_valid": all(
            float(record["checkpoint_replay_max_abs_error"]) <= 1e-6
            for record in records
        ),
        "all_predictions_non_degenerate": all(
            float(record["prediction_std"]) > 1e-12
            for record in records
        ),
    }

    if not all(training_valid):
        return StudentCredibilityResult(
            StudentCredibilityDecision.BLOCKED,
            {
                **evidence,
                "reason": "TRAINING_OR_REPLAY_AUDIT_FAILED",
            },
        )

    if improved_count < minimum_improved_folds:
        return StudentCredibilityResult(
            StudentCredibilityDecision.BLOCKED,
            {
                **evidence,
                "reason": "INSUFFICIENT_NAIVE_BASELINE_IMPROVEMENT",
            },
        )

    if mean_classical_gap > maximum_mean_classical_gap:
        return StudentCredibilityResult(
            StudentCredibilityDecision.BLOCKED,
            {
                **evidence,
                "reason": "EXCESSIVE_GAP_TO_CLASSICAL_BASELINE",
            },
        )

    return StudentCredibilityResult(
        StudentCredibilityDecision.OPEN,
        {
            **evidence,
            "reason": "STUDENT_CREDIBILITY_CRITERIA_MET",
        },
    )
