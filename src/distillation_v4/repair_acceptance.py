from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .repair_outer_artifacts import (
    validate_repair_outer_record,
)
from .repair_outer_matrix import (
    validate_frozen_repair_outer_job_matrix,
)


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


def accept_repair_outer_artifacts(
    *,
    output: Path,
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

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for job in matrix["jobs"]:
        record_path = (
            output
            / "outer_test_repair"
            / str(job["dataset"])
            / str(job["job_id"])
            / "job_record.json"
        )

        if not record_path.is_file():
            rejected.append(
                {
                    "job_id": job["job_id"],
                    "reason": (
                        "REPAIR_OUTER_RECORD_MISSING"
                    ),
                }
            )
            continue

        record = json.loads(
            record_path.read_text()
        )

        try:
            validated = (
                validate_repair_outer_record(
                    output=output,
                    record_path=record_path,
                    expected_job=job,
                    expected_execution_fingerprint=str(
                        record.get(
                            "execution_fingerprint",
                            "",
                        )
                    ),
                )
            )
            accepted.append(
                {
                    "job_id": job["job_id"],
                    "dataset": job["dataset"],
                    "route": job["route"],
                    "fold": job["fold"],
                    "seed": job["seed"],
                    "mae": validated["mae"],
                    "predictions_sha256": (
                        validated[
                            "predictions_sha256"
                        ]
                    ),
                    "metrics_sha256": validated[
                        "metrics_sha256"
                    ],
                }
            )
        except RuntimeError as exc:
            rejected.append(
                {
                    "job_id": job["job_id"],
                    "reason": str(exc),
                }
            )

    status = {
        "engineering_status": (
            "ENGINEERING_ACCEPTED"
            if (
                len(accepted) == len(matrix["jobs"])
                and not rejected
            )
            else "ENGINEERING_INCOMPLETE"
        ),
        "planned_count": len(matrix["jobs"]),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }

    _atomic_json(
        output
        / "acceptance/"
        "repair_outer_accepted_artifact_index.json",
        status,
    )

    return status


def write_repair_final_report(
    *,
    output: Path,
) -> dict[str, Any]:
    acceptance_path = (
        output
        / "acceptance/"
        "repair_outer_accepted_artifact_index.json"
    )
    outer_path = (
        output
        / "status/repair_outer_final.json"
    )

    if not acceptance_path.is_file():
        raise RuntimeError(
            "REPAIR_ACCEPTANCE_STATUS_MISSING"
        )

    if not outer_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_FINAL_STATUS_MISSING"
        )

    acceptance = json.loads(
        acceptance_path.read_text()
    )
    outer = json.loads(
        outer_path.read_text()
    )

    payload = {
        "ENGINEERING_STATUS": acceptance[
            "engineering_status"
        ],
        "SCIENTIFIC_STATUS": outer[
            "scientific_verdict"
        ],
        "OUTER_TEST_STATUS": (
            "OUTER_TEST_COMPLETE"
            if outer.get("outer_test_complete")
            is True
            else "OUTER_TEST_INCOMPLETE"
        ),
        "ROUTE_MATRIX_MUTATED": False,
        "CONTROL_MATRIX_MUTATED": False,
        "HISTORICAL_V4_REUSED": False,
        "V3_MUTATED_OR_RESUMED": False,
        "CLAIM_BOUNDARY": outer[
            "claim_boundary"
        ],
    }

    _atomic_json(
        output
        / "reports/"
        "repair_final_evidence_registry.json",
        payload,
    )
    _atomic_json(
        output
        / "status/repair_final_status.json",
        payload,
    )

    report = (
        "# Stage 8 V4 Repair Final Report\n\n"
        + "\n".join(
            f"- **{key}**: `{value}`"
            for key, value in payload.items()
        )
        + "\n\n"
        + "The route and control matrices remained frozen "
        + "throughout outer evaluation.\n"
    )

    report_path = (
        output
        / "reports/"
        "repair_final_scientific_report.md"
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(report)

    return payload
