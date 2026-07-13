from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_frozen_phase5_matrix(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "FROZEN_PHASE5_JOB_MATRIX_MISSING"
        )

    payload: dict[str, Any] = json.loads(
        path.read_text()
    )

    if payload.get("status") != (
        "JOB_MATRIX_FROZEN"
    ):
        raise RuntimeError(
            "PHASE5_JOB_MATRIX_NOT_FROZEN"
        )

    stored = payload.get(
        "phase5_job_matrix_fingerprint"
    )

    unsigned = dict(payload)
    unsigned.pop(
        "phase5_job_matrix_fingerprint",
        None,
    )

    if stored != _fingerprint(unsigned):
        raise RuntimeError(
            "PHASE5_JOB_MATRIX_FINGERPRINT_MISMATCH"
        )

    if payload.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE5_MATRIX_FORBIDDEN"
        )

    if payload.get("execution_started") is not False:
        raise RuntimeError(
            "PHASE5_MATRIX_ALREADY_EXECUTING"
        )

    jobs = payload.get("jobs")

    if not isinstance(jobs, list):
        raise RuntimeError(
            "PHASE5_JOB_MATRIX_JOBS_INVALID"
        )

    identifiers = [
        str(job["job_id"])
        for job in jobs
    ]

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "PHASE5_JOB_MATRIX_DUPLICATE_ID"
        )

    ceiling = int(
        payload["per_dataset_job_ceiling"]
    )
    counts = payload[
        "materialized_jobs_by_dataset"
    ]

    for dataset, count in counts.items():
        if int(count) > ceiling:
            raise RuntimeError(
                "PHASE5_JOB_MATRIX_CEILING_EXCEEDED:"
                f"{dataset}:{count}:{ceiling}"
            )

    return payload
