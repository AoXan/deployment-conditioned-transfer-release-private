from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

import pandas as pd

from .contracts import CampaignConfig
from .phase4_contract import (
    validate_phase4_prerequisites,
)
from .phase4_fingerprint import (
    dataframe_fingerprint,
)
from .phase5_artifacts import (
    validate_phase5_record,
)
from .phase5_execution import execute_phase5_job
from .phase5_fingerprint import (
    validate_frozen_phase5_matrix,
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
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
        )
        + "\n"
    )
    temporary.replace(path)


def run_phase5_repair_matrix(
    config: CampaignConfig,
    output: Path,
    *,
    frame_loader: Callable[
        [CampaignConfig, Path, str],
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    manifest = validate_frozen_phase5_matrix(
        output
        / "control/frozen_phase5_job_matrix.json"
    )

    prerequisites = (
        validate_phase4_prerequisites(output)
    )

    jobs = manifest["jobs"]

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fingerprints: set[str] = set()

    frames: dict[str, pd.DataFrame] = {}
    frame_hashes: dict[str, str] = {}

    for job in jobs:
        dataset = str(job["dataset"])

        if dataset not in frames:
            frames[dataset] = frame_loader(
                config,
                output,
                dataset,
            )
            frame_hashes[dataset] = (
                dataframe_fingerprint(
                    frames[dataset]
                )
            )

        execution_fingerprint = _fingerprint(
            {
                "job": job,
                "frame_fingerprint": (
                    frame_hashes[dataset]
                ),
                "training_contract": (
                    config.raw.get(
                        "training",
                        {},
                    )
                ),
                "phase5_matrix_fingerprint": (
                    manifest[
                        "phase5_job_matrix_fingerprint"
                    ]
                ),
                "code_contract": (
                    "stage8_v4_phase5_execution_v1"
                ),
            }
        )

        if execution_fingerprint in fingerprints:
            raise RuntimeError(
                "PHASE5_EXECUTION_FINGERPRINT_DUPLICATE"
            )

        fingerprints.add(execution_fingerprint)

        job_output = (
            output
            / "development/phase5_repair_matrix"
            / dataset
            / str(job["job_id"])
        )
        record_path = (
            job_output / "job_record.json"
        )

        if record_path.is_file():
            try:
                record = json.loads(
                    record_path.read_text()
                )

                validate_phase5_record(
                    record=record,
                    output=output,
                    expected_execution_fingerprint=(
                        execution_fingerprint
                    ),
                    record_path=record_path,
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
                        "dataset": dataset,
                        "fold": job["fold"],
                        "seed": job["seed"],
                        "control": job["control"],
                        "status": (
                            "BLOCKED_EXISTING_ARTIFACT_INVALID"
                        ),
                        "failure_type": (
                            type(exc).__name__
                        ),
                        "reason": str(exc),
                        "outer_test_used": False,
                    }
                )
                continue

        try:
            record = execute_phase5_job(
                job=job,
                frame=frames[dataset],
                output_root=output,
                output=job_output,
                teacher_lookup=(
                    prerequisites.teachers
                ),
                contract=config.neural_training,
            )

            record["execution_fingerprint"] = (
                execution_fingerprint
            )
            record["resume_status"] = (
                "EXECUTED_NEW"
            )

            _atomic_json(record_path, record)
            accepted.append(record)

        except Exception as exc:
            failure = {
                "job_id": job["job_id"],
                "dataset": dataset,
                "fold": job["fold"],
                "seed": job["seed"],
                "control": job["control"],
                "status": "FAILED_RETRYABLE",
                "failure_type": type(exc).__name__,
                "reason": str(exc),
                "outer_test_used": False,
            }
            failures.append(failure)

            ledger = (
                output
                / "failures/phase5_repair_matrix.jsonl"
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

    planned = len(jobs)
    completed = len(accepted)
    failed = len(failures)

    engineering_complete = (
        completed == planned
        and failed == 0
    )

    status = {
        "phase": "phase5",
        "status": (
            "ENGINEERING_MATRIX_COMPLETE"
            if engineering_complete
            else "ENGINEERING_MATRIX_INCOMPLETE"
        ),
        "planned_jobs": planned,
        "accepted_jobs": completed,
        "failed_jobs": failed,
        "unique_execution_fingerprints": len(
            fingerprints
        ),
        "phase5_job_matrix_fingerprint": (
            manifest[
                "phase5_job_matrix_fingerprint"
            ]
        ),
        "outer_test_used": False,
        "scientific_phase_complete": False,
        "outer_release_allowed": False,
    }

    _atomic_json(
        output
        / "status/phase5_repair_matrix_execution.json",
        status,
    )

    _atomic_json(
        output / "status/phase5_repair.json",
        {
            **status,
            "status": (
                "SCIENTIFIC_PHASE_INCOMPLETE"
            ),
            "reason": (
                "CONTROL_RESULTS_REQUIRE_"
                "MATCHED_AGGREGATION_AND_FINALIZATION"
                if engineering_complete
                else "ENGINEERING_MATRIX_INCOMPLETE"
            ),
        },
    )

    return {
        "execution_status": status,
        "accepted_records": accepted,
        "failures": failures,
    }
