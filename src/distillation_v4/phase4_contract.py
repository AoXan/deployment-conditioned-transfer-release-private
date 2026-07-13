from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from .phase_contract import read_phase_completion
from .teacher_registry import (
    validate_frozen_teacher_registry,
)


@dataclass(frozen=True)
class Phase4Prerequisites:
    phase3_status: dict[str, Any]
    teacher_registry: dict[str, Any]
    teachers: dict[
        tuple[str, str, str],
        dict[str, Any],
    ]


def validate_phase4_prerequisites(
    output: Path,
) -> Phase4Prerequisites:
    phase3_path = (
        output / "status/phase3_repair.json"
    )

    completion = read_phase_completion(
        phase3_path,
        expected_phase="phase3",
    )
    phase3_status = json.loads(
        phase3_path.read_text()
    )

    if phase3_status.get(
        "teacher_registry_formally_released"
    ) is not True:
        raise RuntimeError(
            "PHASE3_TEACHER_REGISTRY_NOT_RELEASED"
        )

    if phase3_status.get(
        "phase4_release_allowed"
    ) is not True:
        raise RuntimeError(
            "PHASE4_RELEASE_NOT_ALLOWED"
        )

    if phase3_status.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE3_RELEASE_FORBIDDEN"
        )

    registry_path = (
        output
        / "control/frozen_teacher_registry.json"
    )
    registry = validate_frozen_teacher_registry(
        registry_path,
        output_root=output,
        require_formal_release=True,
    )

    lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for teacher in registry["teachers"]:
        key = (
            str(teacher["dataset"]),
            str(teacher["fold"]),
            str(teacher["role"]),
        )

        if key in lookup:
            raise RuntimeError(
                "DUPLICATE_PHASE4_TEACHER_LINEAGE:"
                + ":".join(key)
            )

        if teacher.get(
            "selection_scope"
        ) != "OUTER_TRAIN_INNER_VALIDATION_ONLY":
            raise RuntimeError(
                "INVALID_PHASE4_TEACHER_SELECTION_SCOPE"
            )

        if teacher.get("outer_refit_allowed") is not False:
            raise RuntimeError(
                "PHASE4_TEACHER_OUTER_REFIT_NOT_FORBIDDEN"
            )

        lookup[key] = teacher

    gate_termination = bool(
        phase3_status.get(
            "scientific_gate_termination",
            False,
        )
    )
    if completion.accepted_jobs <= 0 and not gate_termination:
        raise RuntimeError(
            "PHASE3_ACCEPTED_JOB_COUNT_INVALID"
        )

    return Phase4Prerequisites(
        phase3_status=phase3_status,
        teacher_registry=registry,
        teachers=lookup,
    )
