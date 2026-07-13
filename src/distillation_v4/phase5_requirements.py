from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .contracts import CampaignConfig


CONTROL_REQUIREMENTS = {
    "soil_direct": (
        "soil_representation_sensitivity_control",
    ),
    "missing_aware": (
        "missing_aware_ablation_suite_control",
    ),
    "fine_tune": (
        "active_compute_steps_control",
    ),
    "prediction_kd": (
        "shuffled_teacher_prediction_control",
    ),
    "representation_kd": (
        "random_representation_transfer_control",
    ),
    "combined_kd": (
        "ckd_balancing_sensitivity_control",
    ),
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def freeze_phase5_control_requirements(
    *,
    config: CampaignConfig,
    output: Path,
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    per_dataset_ceiling = int(
        config.budgets["phase5"]
    )

    datasets: dict[str, Any] = {}

    for dataset in config.primary_datasets:
        survival = aggregation[
            "route_survival"
        ][dataset]

        requirements: list[dict[str, Any]] = []

        for route, controls in (
            CONTROL_REQUIREMENTS.items()
        ):
            route_state = survival.get(route)

            if not route_state:
                continue

            if route_state.get("survived") is not True:
                continue

            for control in controls:
                requirements.append(
                    {
                        "control": control,
                        "required_by_route": route,
                        "generation_reason": (
                            "PREDECLARED_ROUTE_SURVIVED_"
                            "STRUCTURAL_COMPLETENESS"
                        ),
                        "effect_sign_used": False,
                        "outer_test_used": False,
                    }
                )

        if len(requirements) > per_dataset_ceiling:
            raise RuntimeError(
                "PHASE5_CONTROL_REQUIREMENT_CEILING_EXCEEDED:"
                f"{dataset}:{len(requirements)}:"
                f"{per_dataset_ceiling}"
            )

        datasets[dataset] = {
            "control_requirement_count": len(
                requirements
            ),
            "per_dataset_job_ceiling": (
                per_dataset_ceiling
            ),
            "control_type_requirement_count": len(
                requirements
            ),
            "unallocated_job_ceiling": (
                per_dataset_ceiling
            ),
            "requirements": requirements,
            "requirements_are_job_counts": False,
            "job_matrix_materialized": False,
            "job_allocation_status": (
                "REQUIRES_PREDECLARED_FOLD_SEED_"
                "ALLOCATION_CONTRACT"
            ),
        }

    payload = {
        "phase": "phase5",
        "status": (
            "CONTROL_REQUIREMENTS_FROZEN"
        ),
        "budget_semantics": (
            "PER_DATASET_MAXIMUM_NOT_REQUIRED_COUNT"
        ),
        "datasets": datasets,
        "source_phase4_route_fingerprint": (
            aggregation[
                "route_manifest_fingerprint"
            ]
        ),
        "generation_uses_phase4_effect_sign": False,
        "outer_test_used": False,
        "execution_started": False,
        "outer_release_allowed": False,
    }

    payload[
        "phase5_control_requirements_fingerprint"
    ] = _fingerprint(payload)

    path = (
        output
        / "control/frozen_phase5_control_requirements.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return payload
