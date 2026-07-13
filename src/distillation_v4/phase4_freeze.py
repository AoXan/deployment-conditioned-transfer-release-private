from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .contracts import CampaignConfig
from .phase4_planner import build_phase4_routes


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def freeze_phase4_routes(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    manifest = build_phase4_routes(
        config,
        output,
    )

    if manifest["outer_test_used"] is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE4_PLANNING_FORBIDDEN"
        )

    if not manifest["routes"]:
        raise RuntimeError(
            "PHASE4_NO_ROUTES_SURVIVED"
        )

    path = (
        output
        / "control/frozen_phase4_route_manifest.json"
    )
    _atomic_json(path, manifest)

    status = {
        "phase": "phase4",
        "status": "ROUTE_MANIFEST_FROZEN",
        "generated_route_count": (
            manifest["generated_route_count"]
        ),
        "budget_semantics": (
            manifest["budget_semantics"]
        ),
        "per_dataset_job_ceiling": (
            manifest["per_dataset_job_ceiling"]
        ),
        "campaign_job_ceiling": (
            manifest["campaign_job_ceiling"]
        ),
        "generated_jobs_by_dataset": (
            manifest["generated_jobs_by_dataset"]
        ),
        "unused_ceiling_by_dataset": (
            manifest["unused_ceiling_by_dataset"]
        ),
        "route_manifest": str(
            path.relative_to(output)
        ),
        "route_manifest_fingerprint": (
            manifest[
                "phase4_route_manifest_fingerprint"
            ]
        ),
        "execution_started": False,
        "outer_test_used": False,
        "phase5_release_allowed": False,
        "outer_release_allowed": False,
    }

    _atomic_json(
        output
        / "status/phase4_repair_planning.json",
        status,
    )

    return {
        "manifest": manifest,
        "status": status,
    }
