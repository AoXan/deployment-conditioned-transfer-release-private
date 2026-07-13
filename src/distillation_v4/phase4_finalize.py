from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .contracts import CampaignConfig
from .phase4_aggregation import (
    aggregate_phase4_development,
)
from .phase5_requirements import (
    freeze_phase5_control_requirements,
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


def finalize_phase4_repair(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    execution_path = (
        output
        / "status/phase4_repair_matrix_execution.json"
    )

    if not execution_path.is_file():
        raise RuntimeError(
            "PHASE4_EXECUTION_STATUS_MISSING"
        )

    execution = json.loads(
        execution_path.read_text()
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE4_ENGINEERING_MATRIX_INCOMPLETE"
        )

    if int(execution.get("failed_jobs", -1)) != 0:
        raise RuntimeError(
            "PHASE4_ENGINEERING_FAILURES_PRESENT"
        )

    aggregation = aggregate_phase4_development(
        output=output,
    )

    aggregation_path = (
        output
        / "development/phase4_aggregation.json"
    )
    _atomic_json(
        aggregation_path,
        aggregation,
    )

    phase5_requirements = (
        freeze_phase5_control_requirements(
            config=config,
            output=output,
            aggregation=aggregation,
        )
    )

    status = {
        "phase": "phase4",
        "status": "SCIENTIFIC_PHASE_COMPLETE",
        "scientific_completion_scope": (
            "OUTER_TRAIN_DEVELOPMENT_PHASE_ONLY"
        ),
        "planned_jobs": execution[
            "planned_jobs"
        ],
        "accepted_jobs": execution[
            "accepted_jobs"
        ],
        "failed_jobs": 0,
        "engineering_matrix_complete": True,
        "paired_aggregation_complete": True,
        "route_survival_gate_complete": True,
        "phase5_control_requirements_frozen": True,
        "phase5_release_allowed": True,
        "phase5_execution_allowed": False,
        "outer_release_allowed": False,
        "outer_test_used": False,
        "development_effects_may_change_route_matrix": (
            False
        ),
        "development_aggregation": str(
            aggregation_path.relative_to(output)
        ),
        "phase5_control_requirements": (
            "control/"
            "frozen_phase5_control_requirements.json"
        ),
        "route_manifest_fingerprint": (
            aggregation[
                "route_manifest_fingerprint"
            ]
        ),
        "claim_boundary": (
            "PHASE4_DEVELOPMENT_COMPLETION_IS_NOT_"
            "AN_OUTER_TEST_GO_VERDICT"
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
                "frozen_phase5_control_requirements.json"
            ),
        ],
    }

    _atomic_json(
        output / "status/phase4_repair.json",
        status,
    )

    return {
        "status": status,
        "aggregation": aggregation,
        "phase5_requirements": (
            phase5_requirements
        ),
    }
