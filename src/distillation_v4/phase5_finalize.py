from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .phase5_aggregation import (
    aggregate_phase5_controls,
)
from .outer_readiness import (
    freeze_outer_readiness_contract,
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


def finalize_phase5_repair(
    *,
    output: Path,
) -> dict[str, Any]:
    execution_path = (
        output
        / "status/phase5_repair_matrix_execution.json"
    )

    if not execution_path.is_file():
        raise RuntimeError(
            "PHASE5_EXECUTION_STATUS_MISSING"
        )

    execution = json.loads(
        execution_path.read_text()
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE5_ENGINEERING_MATRIX_INCOMPLETE"
        )

    if int(execution.get("failed_jobs", -1)) != 0:
        raise RuntimeError(
            "PHASE5_ENGINEERING_FAILURES_PRESENT"
        )

    aggregation = aggregate_phase5_controls(
        output=output,
    )

    aggregation_path = (
        output
        / "development/phase5_aggregation.json"
    )
    _atomic_json(
        aggregation_path,
        aggregation,
    )

    status = {
        "phase": "phase5",
        "status": "SCIENTIFIC_PHASE_COMPLETE",
        "scientific_completion_scope": (
            "OUTER_TRAIN_DEVELOPMENT_CONTROLS_ONLY"
        ),
        "planned_jobs": execution[
            "planned_jobs"
        ],
        "accepted_jobs": execution[
            "accepted_jobs"
        ],
        "failed_jobs": 0,
        "engineering_matrix_complete": True,
        "matched_control_aggregation_complete": True,
        "control_results_may_delete_phase4_routes": (
            False
        ),
        "control_results_may_enable_outer_test": (
            False
        ),
        "outer_readiness_contract_frozen": False,
        "outer_readiness_planning_allowed": True,
        "outer_test_execution_allowed": False,
        "outer_release_allowed": False,
        "outer_test_used": False,
        "development_aggregation": str(
            aggregation_path.relative_to(output)
        ),
        "phase5_job_matrix_fingerprint": (
            aggregation[
                "phase5_job_matrix_fingerprint"
            ]
        ),
        "claim_boundary": (
            "PHASE5_DEVELOPMENT_CONTROLS_ARE_"
            "MECHANISM_DIAGNOSTICS_NOT_OUTER_VERDICTS"
        ),
        "evidence_files": [
            str(
                execution_path.relative_to(output)
            ),
            str(
                aggregation_path.relative_to(output)
            ),
            (
                "control/"
                "frozen_phase5_job_matrix.json"
            ),
        ],
    }

    _atomic_json(
        output / "status/phase5_repair.json",
        status,
    )

    readiness = freeze_outer_readiness_contract(
        output=output,
    )

    status[
        "outer_readiness_contract_frozen"
    ] = True
    status["outer_readiness_contract"] = (
        "control/"
        "frozen_outer_readiness_contract.json"
    )
    status["outer_readiness_fingerprint"] = (
        readiness["outer_readiness_fingerprint"]
    )

    _atomic_json(
        output / "status/phase5_repair.json",
        status,
    )

    return {
        "status": status,
        "aggregation": aggregation,
        "outer_readiness": readiness,
    }
