from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .contracts import CampaignConfig
from .repair_acceptance import (
    accept_repair_outer_artifacts,
    write_repair_final_report,
)
from .repair_outer_finalize import (
    finalize_repair_outer,
)
from .repair_outer_matrix import (
    validate_frozen_repair_outer_job_matrix,
)
from .repair_outer_orchestrator import (
    connected_repair_outer_executor,
    run_repair_outer_matrix,
)
from .repair_outer_preflight import (
    run_repair_outer_preflight,
)
from .repair_outer_release import (
    validate_repair_outer_release_token,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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


def _formal_flags(
    config: CampaignConfig,
) -> dict[str, bool]:
    if config.raw.get("schema_version") != (
        "stage8_v4_repair_v1"
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_CONFIG_SCHEMA_INVALID"
        )

    repair = config.raw.get(
        "repair_contract",
        {},
    )

    approved = bool(
        repair.get(
            "formal_run_approved",
            False,
        )
    )
    outer_enabled = bool(
        repair.get(
            "outer_release_enabled",
            False,
        )
    )
    refit_forbidden = bool(
        repair.get(
            "teacher_refit_inside_outer_forbidden",
            False,
        )
    )
    historical_read_only = bool(
        repair.get(
            "historical_namespace_read_only",
            False,
        )
    )

    if not approved:
        raise RuntimeError(
            "FORMAL_RUN_NOT_APPROVED"
        )

    if not outer_enabled:
        raise RuntimeError(
            "REPAIR_OUTER_RELEASE_NOT_ENABLED"
        )

    if not refit_forbidden:
        raise RuntimeError(
            "REPAIR_OUTER_TEACHER_REFIT_GUARD_DISABLED"
        )

    if not historical_read_only:
        raise RuntimeError(
            "REPAIR_HISTORICAL_READ_ONLY_GUARD_DISABLED"
        )

    return {
        "formal_run_approved": approved,
        "outer_release_enabled": outer_enabled,
        "teacher_refit_inside_outer_forbidden": (
            refit_forbidden
        ),
        "historical_namespace_read_only": (
            historical_read_only
        ),
    }


def run_repair_formal_outer_pipeline(
    *,
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    flags = _formal_flags(config)

    token_path = (
        output
        / "control/"
        "repair_outer_release_token.json"
    )
    matrix_path = (
        output
        / "control/"
        "frozen_repair_outer_job_matrix.json"
    )
    registry_path = (
        output
        / "control/"
        "frozen_route_registry_repair.json"
    )

    token = validate_repair_outer_release_token(
        output=output,
        token_path=token_path,
    )
    matrix = (
        validate_frozen_repair_outer_job_matrix(
            output=output,
            path=matrix_path,
        )
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "REPAIR_FROZEN_ROUTE_REGISTRY_MISSING"
        )

    registry_hash_before = _sha256(
        registry_path
    )
    matrix_hash_before = _sha256(
        matrix_path
    )
    token_hash_before = _sha256(
        token_path
    )

    preflight = run_repair_outer_preflight(
        config=config,
        output=output,
    )

    if preflight.get("status") != (
        "REPAIR_OUTER_PREFLIGHT_READY"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_PREFLIGHT_NOT_READY"
        )

    if preflight.get(
        "execution_allowed"
    ) is not True:
        raise RuntimeError(
            "REPAIR_OUTER_PREFLIGHT_EXECUTION_BLOCKED"
        )

    if preflight.get(
        "training_started"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_PREFLIGHT_STARTED_TRAINING"
        )

    if preflight.get(
        "outer_test_used"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_PREFLIGHT_USED_OUTER_TEST"
        )

    if preflight.get(
        "repair_outer_job_matrix_fingerprint"
    ) != matrix.get(
        "repair_outer_job_matrix_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_PREFLIGHT_MATRIX_MISMATCH"
        )

    if preflight.get(
        "repair_outer_release_fingerprint"
    ) != token.get(
        "repair_outer_release_fingerprint"
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_PREFLIGHT_RELEASE_MISMATCH"
        )

    started = {
        "phase": "repair_outer_formal",
        "status": "FORMAL_EXECUTION_STARTED",
        "formal_flags": flags,
        "planned_jobs": matrix["job_count"],
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
        "registry_sha256_before": (
            registry_hash_before
        ),
        "matrix_sha256_before": (
            matrix_hash_before
        ),
        "token_sha256_before": (
            token_hash_before
        ),
        "preflight_fingerprint": preflight[
            "preflight_fingerprint"
        ],
        "outer_test_role": (
            "FINAL_ESTIMATION_ONLY"
        ),
        "outer_test_may_enable_downstream_jobs": (
            False
        ),
    }
    _atomic_json(
        output
        / "status/"
        "repair_outer_formal_started.json",
        started,
    )

    execution = run_repair_outer_matrix(
        config=config,
        output=output,
        job_executor=(
            connected_repair_outer_executor
        ),
    )

    execution_status = execution[
        "execution_status"
    ]

    if execution_status.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        blocked = {
            **started,
            "status": (
                "FORMAL_EXECUTION_INCOMPLETE"
            ),
            "execution_status": execution_status,
            "failure_count": len(
                execution.get("failures", [])
            ),
            "scientific_finalization_started": (
                False
            ),
            "acceptance_started": False,
            "report_written": False,
        }
        _atomic_json(
            output
            / "status/"
            "repair_outer_formal_incomplete.json",
            blocked,
        )
        return blocked

    if execution_status.get(
        "accepted_jobs"
    ) != matrix["job_count"]:
        raise RuntimeError(
            "REPAIR_FORMAL_ACCEPTED_JOB_COUNT_MISMATCH"
        )

    if execution_status.get(
        "failed_jobs"
    ) != 0:
        raise RuntimeError(
            "REPAIR_FORMAL_FAILURE_COUNT_NONZERO"
        )

    if _sha256(registry_path) != (
        registry_hash_before
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_ROUTE_REGISTRY_MUTATED"
        )

    if _sha256(matrix_path) != (
        matrix_hash_before
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_OUTER_MATRIX_MUTATED"
        )

    if _sha256(token_path) != (
        token_hash_before
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_RELEASE_TOKEN_MUTATED"
        )

    finalized = finalize_repair_outer(
        output=output,
    )
    final_status = finalized["status"]

    if final_status.get("status") != (
        "SCIENTIFIC_PHASE_COMPLETE"
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_SCIENTIFIC_FINALIZATION_INCOMPLETE"
        )

    if final_status.get(
        "route_matrix_mutated"
    ) is not False:
        raise RuntimeError(
            "REPAIR_FORMAL_FINALIZER_MUTATED_ROUTE_MATRIX"
        )

    if final_status.get(
        "control_matrix_mutated"
    ) is not False:
        raise RuntimeError(
            "REPAIR_FORMAL_FINALIZER_MUTATED_CONTROL_MATRIX"
        )

    if final_status.get(
        "downstream_jobs_created"
    ) is not False:
        raise RuntimeError(
            "REPAIR_FORMAL_FINALIZER_CREATED_JOBS"
        )

    acceptance = (
        accept_repair_outer_artifacts(
            output=output,
        )
    )

    if acceptance.get(
        "engineering_status"
    ) != "ENGINEERING_ACCEPTED":
        raise RuntimeError(
            "REPAIR_FORMAL_ACCEPTANCE_INCOMPLETE"
        )

    if acceptance.get(
        "accepted_count"
    ) != matrix["job_count"]:
        raise RuntimeError(
            "REPAIR_FORMAL_ACCEPTANCE_COUNT_MISMATCH"
        )

    if acceptance.get(
        "rejected_count"
    ) != 0:
        raise RuntimeError(
            "REPAIR_FORMAL_ACCEPTANCE_REJECTIONS_PRESENT"
        )

    report = write_repair_final_report(
        output=output,
    )

    if report.get(
        "ENGINEERING_STATUS"
    ) != "ENGINEERING_ACCEPTED":
        raise RuntimeError(
            "REPAIR_FORMAL_REPORT_ENGINEERING_INCOMPLETE"
        )

    completed = {
        "phase": "repair_outer_formal",
        "status": "FORMAL_EXECUTION_COMPLETE",
        "planned_jobs": matrix["job_count"],
        "accepted_jobs": acceptance[
            "accepted_count"
        ],
        "rejected_jobs": acceptance[
            "rejected_count"
        ],
        "scientific_status": report[
            "SCIENTIFIC_STATUS"
        ],
        "engineering_status": report[
            "ENGINEERING_STATUS"
        ],
        "outer_test_status": report[
            "OUTER_TEST_STATUS"
        ],
        "route_registry_unchanged": (
            _sha256(registry_path)
            == registry_hash_before
        ),
        "outer_matrix_unchanged": (
            _sha256(matrix_path)
            == matrix_hash_before
        ),
        "release_token_unchanged": (
            _sha256(token_path)
            == token_hash_before
        ),
        "teacher_refit_performed": False,
        "downstream_jobs_created": False,
        "outer_test_may_enable_downstream_jobs": (
            False
        ),
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
    }

    if not all(
        (
            completed[
                "route_registry_unchanged"
            ],
            completed[
                "outer_matrix_unchanged"
            ],
            completed[
                "release_token_unchanged"
            ],
        )
    ):
        raise RuntimeError(
            "REPAIR_FORMAL_FROZEN_CONTROL_MUTATED"
        )

    _atomic_json(
        output
        / "status/"
        "repair_outer_formal_complete.json",
        completed,
    )

    return completed
