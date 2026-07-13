from __future__ import annotations

from pathlib import Path
from typing import Any

from .phase4_artifacts import (
    load_and_validate_phase4_record,
)


def find_phase4_record_path(
    *,
    output: Path,
    route_id: str,
) -> Path:
    matches = [
        path
        for path in (
            output
            / "development/phase4_repair_matrix"
        ).glob("**/job_record.json")
        if path.parent.name == route_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "PHASE5_PHASE4_SOURCE_RECORD_RESOLUTION_FAILED:"
            f"{route_id}:{len(matches)}"
        )

    return matches[0]


def load_validated_phase4_source(
    *,
    output: Path,
    route_id: str,
    expected_artifact_fingerprint: str,
) -> dict[str, Any]:
    record_path = find_phase4_record_path(
        output=output,
        route_id=route_id,
    )

    raw = __import__("json").loads(
        record_path.read_text()
    )

    execution_fingerprint = raw.get(
        "execution_fingerprint"
    )

    if not execution_fingerprint:
        raise RuntimeError(
            "PHASE5_SOURCE_EXECUTION_FINGERPRINT_MISSING"
        )

    record = load_and_validate_phase4_record(
        record_path=record_path,
        output_root=output,
        expected_execution_fingerprint=(
            str(execution_fingerprint)
        ),
    )

    from .phase5_matrix import (
        _record_artifact_fingerprint,
    )

    actual = _record_artifact_fingerprint(record)

    if actual != expected_artifact_fingerprint:
        raise RuntimeError(
            "PHASE5_SOURCE_ARTIFACT_FINGERPRINT_MISMATCH"
        )

    return record


def resolve_teacher_record(
    *,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    dataset: str,
    fold: str,
    role: str,
    expected_summary: dict[str, Any],
) -> dict[str, Any]:
    key = (dataset, fold, role)

    if key not in teacher_lookup:
        raise RuntimeError(
            "PHASE5_FROZEN_TEACHER_KEY_MISSING:"
            + ":".join(key)
        )

    record = teacher_lookup[key]

    for field in (
        "candidate",
        "checkpoint",
        "checkpoint_sha256",
    ):
        if (
            expected_summary.get(field)
            != record.get(field)
        ):
            raise RuntimeError(
                "PHASE5_TEACHER_LINEAGE_MISMATCH:"
                f"{role}:{field}"
            )

    return record
