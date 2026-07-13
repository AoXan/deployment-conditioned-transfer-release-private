from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


SUPPORTED_REPAIR_ROUTES = {
    "supervised",
    "soil_direct",
    "missing_aware",
    "fine_tune",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
}


@dataclass(frozen=True)
class RepairOuterAdapterCapability:
    route: str
    implementation_status: str
    requires_prediction_teacher: bool
    requires_representation_teacher: bool
    requires_phase4_loss_weights: bool
    deployable_at_inference: bool


CAPABILITIES = {
    "soil_direct": RepairOuterAdapterCapability(
        route="soil_direct",
        implementation_status="CONNECTED",
        requires_prediction_teacher=False,
        requires_representation_teacher=False,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "missing_aware": RepairOuterAdapterCapability(
        route="missing_aware",
        implementation_status="CONNECTED",
        requires_prediction_teacher=False,
        requires_representation_teacher=False,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "supervised": RepairOuterAdapterCapability(
        route="supervised",
        implementation_status="CONNECTED",
        requires_prediction_teacher=False,
        requires_representation_teacher=False,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "fine_tune": RepairOuterAdapterCapability(
        route="fine_tune",
        implementation_status="CONNECTED",
        requires_prediction_teacher=False,
        requires_representation_teacher=True,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "prediction_kd": RepairOuterAdapterCapability(
        route="prediction_kd",
        implementation_status="CONNECTED",
        requires_prediction_teacher=True,
        requires_representation_teacher=False,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "representation_kd": RepairOuterAdapterCapability(
        route="representation_kd",
        implementation_status="CONNECTED",
        requires_prediction_teacher=False,
        requires_representation_teacher=True,
        requires_phase4_loss_weights=False,
        deployable_at_inference=True,
    ),
    "combined_kd": RepairOuterAdapterCapability(
        route="combined_kd",
        implementation_status="CONNECTED",
        requires_prediction_teacher=True,
        requires_representation_teacher=True,
        requires_phase4_loss_weights=True,
        deployable_at_inference=True,
    ),
}


def capability_for_route(
    route: str,
) -> RepairOuterAdapterCapability:
    if route not in SUPPORTED_REPAIR_ROUTES:
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_UNSUPPORTED:"
            + route
        )

    return CAPABILITIES[route]


def _teacher_records(
    *,
    output: Path,
) -> list[dict[str, Any]]:
    path = (
        output
        / "control/frozen_teacher_registry.json"
    )

    if not path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_TEACHER_REGISTRY_MISSING"
        )

    payload = json.loads(path.read_text())
    teachers = payload.get("teachers")

    if not isinstance(teachers, list):
        raise RuntimeError(
            "REPAIR_OUTER_TEACHER_REGISTRY_INVALID"
        )

    return teachers


def validate_repair_outer_job_dependencies(
    *,
    output: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    route = str(job["route"])
    dataset = str(job["dataset"])
    fold = str(job["fold"])

    capability = capability_for_route(route)

    route_contract = job.get("route_contract")

    if not isinstance(route_contract, dict):
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_CONTRACT_MISSING"
        )

    if route_contract.get("dataset") != dataset:
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_CONTRACT_DATASET_MISMATCH"
        )

    if route_contract.get("route") != route:
        raise RuntimeError(
            "REPAIR_OUTER_ROUTE_CONTRACT_ROUTE_MISMATCH"
        )

    if job.get("selection_scope") != (
        "FROZEN_BEFORE_OUTER_TEST"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_JOB_SELECTION_SCOPE_INVALID"
        )

    if job.get("outer_test_role") != (
        "FINAL_ESTIMATION_ONLY"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_JOB_ROLE_INVALID"
        )

    if job.get(
        "may_enable_downstream_jobs"
    ) is not False:
        raise RuntimeError(
            "REPAIR_OUTER_JOB_DOWNSTREAM_MUTATION_FORBIDDEN"
        )

    teacher_matches: dict[str, list[dict[str, Any]]] = {
        "prediction": [],
        "representation": [],
    }

    if (
        capability.requires_prediction_teacher
        or capability.requires_representation_teacher
    ):
        teachers = _teacher_records(output=output)

        for record in teachers:
            if str(record.get("dataset")) != dataset:
                continue

            if str(record.get("fold")) != fold:
                continue

            role = str(record.get("role"))

            if role in teacher_matches:
                teacher_matches[role].append(record)

    if capability.requires_prediction_teacher:
        matches = teacher_matches["prediction"]

        if len(matches) != 1:
            raise RuntimeError(
                "REPAIR_OUTER_PREDICTION_TEACHER_"
                f"RESOLUTION_FAILED:{dataset}:{fold}:"
                f"{len(matches)}"
            )

    if capability.requires_representation_teacher:
        matches = teacher_matches["representation"]

        if len(matches) != 1:
            raise RuntimeError(
                "REPAIR_OUTER_REPRESENTATION_TEACHER_"
                f"RESOLUTION_FAILED:{dataset}:{fold}:"
                f"{len(matches)}"
            )

    phase4_manifest_path = (
        output
        / "control/frozen_phase4_route_manifest.json"
    )

    if not phase4_manifest_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_MANIFEST_MISSING"
        )

    phase4_manifest = json.loads(
        phase4_manifest_path.read_text()
    )

    phase4_routes = phase4_manifest.get("routes")

    if not isinstance(phase4_routes, list):
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_MANIFEST_INVALID"
        )

    matching_phase4 = [
        record
        for record in phase4_routes
        if (
            str(record.get("dataset")) == dataset
            and str(record.get("fold")) == fold
            and str(record.get("route")) == route
        )
    ]

    if not matching_phase4:
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_ROUTE_LINEAGE_MISSING:"
            f"{dataset}:{fold}:{route}"
        )

    seeds = {
        int(record["seed"])
        for record in matching_phase4
        if record.get("seed") is not None
    }

    if int(job["seed"]) not in seeds:
        raise RuntimeError(
            "REPAIR_OUTER_PHASE4_SEED_LINEAGE_MISSING:"
            f"{dataset}:{fold}:{route}:"
            f"{job['seed']}"
        )

    return {
        "job_id": str(job["job_id"]),
        "dataset": dataset,
        "route": route,
        "fold": fold,
        "seed": int(job["seed"]),
        "capability": {
            "implementation_status": (
                capability.implementation_status
            ),
            "requires_prediction_teacher": (
                capability.requires_prediction_teacher
            ),
            "requires_representation_teacher": (
                capability.requires_representation_teacher
            ),
            "requires_phase4_loss_weights": (
                capability.requires_phase4_loss_weights
            ),
            "deployable_at_inference": (
                capability.deployable_at_inference
            ),
        },
        "prediction_teacher_count": len(
            teacher_matches["prediction"]
        ),
        "representation_teacher_count": len(
            teacher_matches["representation"]
        ),
        "phase4_lineage_count": len(
            matching_phase4
        ),
        "dependency_status": (
            "DEPENDENCIES_VALIDATED"
        ),
    }
