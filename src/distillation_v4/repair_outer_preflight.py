from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .contracts import CampaignConfig
from .repair_outer_adapter_contract import (
    validate_repair_outer_job_dependencies,
)
from .repair_outer_matrix import (
    validate_frozen_repair_outer_job_matrix,
)
from .repair_outer_release import (
    validate_repair_outer_release_token,
)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
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


def run_repair_outer_preflight(
    *,
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    if config.raw.get("schema_version") != (
        "stage8_v4_repair_v1"
    ):
        raise RuntimeError(
            "REPAIR_PREFLIGHT_CONFIG_SCHEMA_INVALID"
        )

    token_path = (
        output
        / "control/repair_outer_release_token.json"
    )
    token = validate_repair_outer_release_token(
        output=output,
        token_path=token_path,
    )

    matrix_path = (
        output
        / "control/"
        "frozen_repair_outer_job_matrix.json"
    )
    matrix = (
        validate_frozen_repair_outer_job_matrix(
            output=output,
            path=matrix_path,
        )
    )

    if matrix.get(
        "repair_outer_release_fingerprint"
    ) != token.get(
        "repair_outer_release_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_PREFLIGHT_RELEASE_MATRIX_MISMATCH"
        )

    records: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for job in matrix["jobs"]:
        try:
            dependency = (
                validate_repair_outer_job_dependencies(
                    output=output,
                    job=job,
                )
            )

            implementation_status = dependency[
                "capability"
            ]["implementation_status"]

            if implementation_status != "CONNECTED":
                blocked.append(
                    {
                        "job_id": job["job_id"],
                        "dataset": job["dataset"],
                        "route": job["route"],
                        "fold": job["fold"],
                        "seed": job["seed"],
                        "status": (
                            "BLOCKED_MODEL_ADAPTER_NOT_CONNECTED"
                        ),
                        "reason": (
                            "REPAIR_OUTER_MODEL_EXECUTOR_"
                            "NOT_CONNECTED"
                        ),
                    }
                )
            else:
                records.append(dependency)

        except RuntimeError as exc:
            blocked.append(
                {
                    "job_id": job["job_id"],
                    "dataset": job["dataset"],
                    "route": job["route"],
                    "fold": job["fold"],
                    "seed": job["seed"],
                    "status": (
                        "BLOCKED_DEPENDENCY_VALIDATION_FAILED"
                    ),
                    "reason": str(exc),
                }
            )

    all_dependencies_valid = all(
        record.get("dependency_status")
        == "DEPENDENCIES_VALIDATED"
        for record in records
    )

    adapters_connected = (
        len(records) == len(matrix["jobs"])
        and not blocked
    )

    payload = {
        "phase": "repair_outer_preflight",
        "status": (
            "REPAIR_OUTER_PREFLIGHT_READY"
            if adapters_connected
            else "REPAIR_OUTER_PREFLIGHT_BLOCKED"
        ),
        "planned_jobs": len(matrix["jobs"]),
        "ready_jobs": len(records),
        "blocked_jobs": len(blocked),
        "dependency_records": records,
        "blocked_records": blocked,
        "all_dependencies_valid": (
            all_dependencies_valid
        ),
        "all_model_adapters_connected": (
            adapters_connected
        ),
        "training_started": False,
        "outer_test_used": False,
        "artifacts_created": False,
        "formal_release_token_validated": True,
        "repair_outer_job_matrix_fingerprint": (
            matrix[
                "repair_outer_job_matrix_fingerprint"
            ]
        ),
        "repair_outer_release_fingerprint": (
            token[
                "repair_outer_release_fingerprint"
            ]
        ),
        "execution_allowed": adapters_connected,
        "execution_started": False,
        "claim_boundary": (
            "PREFLIGHT_ONLY_NO_MODEL_FITTING"
        ),
    }

    payload["preflight_fingerprint"] = (
        _fingerprint(payload)
    )

    _atomic_json(
        output
        / "status/repair_outer_preflight.json",
        payload,
    )

    return payload
