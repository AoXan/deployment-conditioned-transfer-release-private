from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class PhaseCompletion:
    phase: str
    status: str
    planned_jobs: int
    accepted_jobs: int
    failed_jobs: int
    evidence_files: tuple[str, ...]


def read_phase_completion(
    path: Path,
    *,
    expected_phase: str,
) -> PhaseCompletion:
    if not path.is_file():
        raise RuntimeError(
            f"REPAIR_PHASE_STATUS_MISSING:{expected_phase}"
        )

    payload: dict[str, Any] = json.loads(path.read_text())

    if payload.get("phase") != expected_phase:
        raise RuntimeError(
            f"REPAIR_PHASE_NAME_MISMATCH:{expected_phase}"
        )

    if payload.get("status") != "SCIENTIFIC_PHASE_COMPLETE":
        raise RuntimeError(
            f"REPAIR_PHASE_NOT_COMPLETE:{expected_phase}"
        )

    planned = int(payload.get("planned_jobs", -1))
    accepted = int(payload.get("accepted_jobs", -1))
    failed = int(payload.get("failed_jobs", -1))

    gate_termination = bool(
        payload.get("scientific_gate_termination", False)
    )

    if planned < 0 or (planned == 0 and not gate_termination):
        raise RuntimeError(
            f"REPAIR_PHASE_INVALID_PLAN:{expected_phase}"
        )

    if accepted != planned:
        raise RuntimeError(
            f"REPAIR_PHASE_ACCEPTED_COUNT_MISMATCH:{expected_phase}"
        )

    if failed != 0:
        raise RuntimeError(
            f"REPAIR_PHASE_FAILURES_PRESENT:{expected_phase}"
        )

    evidence = payload.get("evidence_files")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError(
            f"REPAIR_PHASE_EVIDENCE_MISSING:{expected_phase}"
        )

    missing = [
        item
        for item in evidence
        if not (path.parents[1] / item).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"REPAIR_PHASE_EVIDENCE_FILE_MISSING:"
            f"{expected_phase}:{','.join(missing)}"
        )

    return PhaseCompletion(
        phase=expected_phase,
        status="SCIENTIFIC_PHASE_COMPLETE",
        planned_jobs=planned,
        accepted_jobs=accepted,
        failed_jobs=failed,
        evidence_files=tuple(str(item) for item in evidence),
    )
