from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

from .contracts import CampaignConfig
from .repair_outer_artifacts import (
    validate_repair_outer_record,
)
from .repair_outer_matrix import (
    validate_frozen_repair_outer_job_matrix,
)


OuterJobExecutor = Callable[
    [
        CampaignConfig,
        Path,
        dict[str, Any],
        Path,
    ],
    dict[str, Any],
]


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    temporary.replace(path)


def run_repair_outer_matrix(
    *,
    config: CampaignConfig,
    output: Path,
    job_executor: OuterJobExecutor,
) -> dict[str, Any]:
    matrix = (
        validate_frozen_repair_outer_job_matrix(
            output=output,
            path=(
                output
                / "control/"
                "frozen_repair_outer_job_matrix.json"
            ),
        )
    )

    registry_path = (
        output
        / "control/frozen_route_registry_repair.json"
    )
    registry_before = hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fingerprints: set[str] = set()

    for job in matrix["jobs"]:
        execution_fingerprint = _fingerprint(
            {
                "job": job,
                "matrix_fingerprint": matrix[
                    "repair_outer_job_matrix_fingerprint"
                ],
                "training_contract": (
                    config.raw.get("training", {})
                ),
                "code_contract": (
                    "stage8_v4_repair_outer_v1"
                ),
            }
        )

        if execution_fingerprint in fingerprints:
            raise RuntimeError(
                "REPAIR_OUTER_EXECUTION_FINGERPRINT_DUPLICATE"
            )

        fingerprints.add(execution_fingerprint)

        root = (
            output
            / "outer_test_repair"
            / str(job["dataset"])
            / str(job["job_id"])
        )
        record_path = (
            root / "job_record.json"
        )

        if record_path.is_file():
            try:
                record = validate_repair_outer_record(
                    output=output,
                    record_path=record_path,
                    expected_job=job,
                    expected_execution_fingerprint=(
                        execution_fingerprint
                    ),
                )
                record["resume_status"] = (
                    "REUSED_VALIDATED_ARTIFACT"
                )
                accepted.append(record)
                continue
            except RuntimeError as exc:
                failures.append(
                    {
                        "job_id": job["job_id"],
                        "status": (
                            "BLOCKED_EXISTING_ARTIFACT_INVALID"
                        ),
                        "failure_type": (
                            type(exc).__name__
                        ),
                        "reason": str(exc),
                    }
                )
                continue

        try:
            record = job_executor(
                config,
                output,
                job,
                root,
            )

            record = {
                **record,
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "route": job["route"],
                "fold": job["fold"],
                "seed": job["seed"],
                "route_fingerprint_before_outer_test": (
                    job[
                        "route_fingerprint_before_outer_test"
                    ]
                ),
                "execution_fingerprint": (
                    execution_fingerprint
                ),
            }

            _atomic_json(record_path, record)

            validated = validate_repair_outer_record(
                output=output,
                record_path=record_path,
                expected_job=job,
                expected_execution_fingerprint=(
                    execution_fingerprint
                ),
            )
            validated["resume_status"] = (
                "EXECUTED_NEW"
            )
            accepted.append(validated)

        except Exception as exc:
            failure = {
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "route": job["route"],
                "fold": job["fold"],
                "seed": job["seed"],
                "status": "FAILED_RETRYABLE",
                "failure_type": type(exc).__name__,
                "reason": str(exc),
            }
            failures.append(failure)

            ledger = (
                output
                / "failures/"
                "repair_outer_failure_ledger.jsonl"
            )
            ledger.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with ledger.open("a") as handle:
                handle.write(
                    json.dumps(
                        failure,
                        sort_keys=True,
                    )
                    + "\n"
                )

    registry_after = hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()

    if registry_after != registry_before:
        raise RuntimeError(
            "REPAIR_OUTER_MUTATED_ROUTE_REGISTRY"
        )

    planned = len(matrix["jobs"])
    completed = len(accepted)
    failed = len(failures)

    complete = (
        completed == planned
        and failed == 0
    )

    status = {
        "phase": "repair_outer",
        "status": (
            "ENGINEERING_MATRIX_COMPLETE"
            if complete
            else "ENGINEERING_MATRIX_INCOMPLETE"
        ),
        "planned_jobs": planned,
        "accepted_jobs": completed,
        "failed_jobs": failed,
        "route_registry_unchanged": True,
        "repair_outer_job_matrix_fingerprint": (
            matrix[
                "repair_outer_job_matrix_fingerprint"
            ]
        ),
        "outer_test_used": completed > 0,
        "scientific_status": (
            "OUTER_RESULTS_NOT_FINALIZED"
        ),
    }

    _atomic_json(
        output
        / "status/"
        "repair_outer_matrix_execution.json",
        status,
    )

    return {
        "execution_status": status,
        "accepted_records": accepted,
        "failures": failures,
    }


def unavailable_repair_outer_executor(
    config: CampaignConfig,
    output: Path,
    job: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    raise RuntimeError(
        "REPAIR_OUTER_MODEL_EXECUTOR_NOT_CONNECTED"
    )


def connected_repair_outer_executor(
    config: CampaignConfig,
    output: Path,
    job: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    from .repair_outer_executor import (
        execute_repair_outer_job,
    )

    return execute_repair_outer_job(
        config,
        output,
        job,
        root,
    )
