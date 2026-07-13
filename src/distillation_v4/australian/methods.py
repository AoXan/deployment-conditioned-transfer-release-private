from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def recover_method_inventory(formal_root: Path) -> list[dict[str, Any]]:
    control = formal_root / "control"
    phase3 = json.loads((control / "phase3_repair_execution_matrix.json").read_text())
    phase4 = json.loads((control / "frozen_phase4_route_manifest.json").read_text())
    phase5 = json.loads((control / "frozen_phase5_job_matrix.json").read_text())
    inventory: dict[str, dict[str, Any]] = {}
    for dataset in phase3.get("datasets", {}).values():
        for job in dataset.get("jobs", []):
            identifier = str(job["candidate"])
            inventory.setdefault(identifier, {
                "component_id": identifier,
                "kind": "MODEL_ARCHITECTURE",
                "model_family": job.get("model_family"),
                "component": job.get("component"),
                "source_status": "EXECUTED_IN_PHASE3",
                "source_phase": "phase3",
            })
    for route in phase4.get("routes", []):
        identifier = str(route["route"])
        inventory.setdefault(identifier, {
            "component_id": identifier,
            "kind": "ROUTE",
            "source_status": "EXECUTED_IN_PHASE4",
            "source_phase": "phase4",
        })
    for route in phase4.get("blocked_candidates", []):
        identifier = str(route["route"])
        inventory.setdefault(identifier, {
            "component_id": identifier,
            "kind": "ROUTE",
            "source_status": "BLOCKED_IN_SOURCE_CAMPAIGN",
            "source_reason": route.get("reason"),
            "source_phase": "phase4",
        })
    for job in phase5.get("jobs", []):
        identifier = str(job["control"])
        inventory.setdefault(identifier, {
            "component_id": identifier,
            "kind": "CONTROL_OR_ABLATION",
            "required_by_route": job.get("required_by_route"),
            "source_status": "EXECUTED_IN_PHASE5",
            "source_phase": "phase5",
        })
        contract = job.get("soil_extension_contract", {})
        if identifier == "missing_aware_ablation_suite_control":
            variants = ["no_mask", "explicit_mask"]
            variants += [f"dropout_{value}" for value in contract.get("dropout_sensitivity", [])]
            variants += [f"soil_representation_{value}" for value in contract.get("representation_controls", [])]
            variants.append(f"dropout_{contract.get('primary_dropout_probability', 0.5)}")
            for variant in variants:
                inventory.setdefault(variant, {
                    "component_id": variant,
                    "kind": "ABLATION_VARIANT",
                    "required_by_route": "missing_aware",
                    "source_status": "DECLARED_OR_EXECUTED_IN_PHASE5",
                    "source_phase": "phase5",
                })
    # Required matched controls are derived from the recovered deployable/teacher
    # pairs; they are Australian adapted controls, not claims of exact V4 replay.
    if "hgb_deployable" in inventory and "shared_neural_receiver" in inventory:
        for identifier in ("matched_parameter_deployable_control", "matched_active_compute_deployable_control", "capacity_scaling_control"):
            inventory.setdefault(identifier, {
                "component_id": identifier,
                "kind": "ADAPTED_MATCHED_CONTROL",
                "source_status": "DERIVED_FROM_RECOVERED_PHASE3_MATCHED_CONTROLS",
                "source_phase": "australian_adaptation",
            })
    return [inventory[key] for key in sorted(inventory)]
