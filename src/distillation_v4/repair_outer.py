from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .contracts import CampaignConfig
from .repair_outer_release import (
    validate_repair_outer_release_token,
)


def preflight_repair_outer_test(
    *,
    config: CampaignConfig,
    output: Path,
    token_path: Path,
) -> dict[str, Any]:
    schema_version = str(
        config.raw.get("schema_version", "")
    )

    if schema_version != "stage8_v4_repair_v1":
        raise RuntimeError(
            "REPAIR_OUTER_CONFIG_SCHEMA_REQUIRED"
        )

    token = validate_repair_outer_release_token(
        output=output,
        token_path=token_path,
    )

    registry_path = (
        output
        / "control/frozen_route_registry_repair.json"
    )
    registry = json.loads(
        registry_path.read_text()
    )

    jobs = registry.get("jobs")

    if not isinstance(jobs, list):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_JOBS_INVALID"
        )

    route_ids = [
        (
            str(job["dataset"]),
            str(job["route"]),
        )
        for job in jobs
    ]

    if len(set(route_ids)) != len(route_ids):
        raise RuntimeError(
            "REPAIR_OUTER_DUPLICATE_ROUTE"
        )

    return {
        "status": "REPAIR_OUTER_AUTHORIZED_NOT_EXECUTED",
        "authorized_route_count": len(jobs),
        "route_fingerprint_before_outer_test": (
            token[
                "route_fingerprint_before_outer_test"
            ]
        ),
        "outer_test_role": "FINAL_ESTIMATION_ONLY",
        "route_matrix_mutated": False,
        "control_matrix_mutated": False,
        "training_started": False,
        "outer_test_used": False,
        "reason": (
            "REPAIR_OUTER_EXECUTION_LAYER_NOT_YET_CONNECTED"
        ),
    }
