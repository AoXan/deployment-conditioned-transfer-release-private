from __future__ import annotations

import hashlib
import json
from typing import Any


RECOVERY_LADDER = (
    "EXACT_CHECKPOINT_CONTRACT",
    "FROZEN_VIEW_FIELD_MAPPING",
    "RAW_SOURCE_REBUILD",
    "MODALITY_RECOVERY",
    "METHOD_LEVEL_TARGET_RETRAINING",
    "TARGET_SPECIFIC_TEACHER",
    "SHARED_FEATURE_TRANSFER",
    "FROZEN_ENCODER_TARGET_HEAD",
    "LOW_LABEL_ADAPTATION",
    "SPATIAL_TEMPORAL_MECHANISM_DIAGNOSTIC",
)


def _id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _protocol(dataset: dict[str, Any], method: dict[str, Any]) -> tuple[str, str]:
    if not dataset.get("target_available"):
        return "PROVEN_UNTESTABLE", "NO_LEGAL_TARGET_AFTER_RECOVERY"
    identifier = str(method["component_id"])
    exact = set(dataset.get("exact_compatible_components", []))
    if identifier in exact:
        return "EXACT", "EXACT_INPUT_CHECKPOINT_AND_TASK_CONTRACT"
    transfer = set(dataset.get("transfer_components", []))
    if identifier in transfer or identifier in {"fine_tune", "prediction_kd", "representation_kd", "combined_kd"}:
        return "TRANSFER", "SOURCE_OR_TARGET_TEACHER_AND_TARGET_HEAD_REQUIRED"
    if dataset.get("adapted_training_supported", True):
        return "ADAPTED", "METHOD_LEVEL_RETRAINING_ON_NATIVE_SAMPLE_UNIT"
    return "DIAGNOSTIC", "MECHANISM_OR_TRANSPORTABILITY_ONLY"


def build_candidate_matrix(*, datasets: list[dict[str, Any]], methods: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for dataset in datasets:
        for method in methods:
            protocol, reason = _protocol(dataset, method)
            payload = {"dataset_id": dataset["dataset_id"], "component_id": method["component_id"], "protocols": dataset.get("protocols", [])}
            candidates.append({
                "candidate_id": _id(payload),
                **payload,
                "component_kind": method["kind"],
                "source_status": method.get("source_status"),
                "source_phase": method.get("source_phase"),
                "planned_protocol": protocol,
                "planning_reason": reason,
                "recovery_attempts": list(RECOVERY_LADDER),
                "final_status": "CONFIGURED_NOT_RUN" if protocol != "PROVEN_UNTESTABLE" else "PROVEN_UNTESTABLE",
            })
    observed = {item["component_id"] for item in candidates}
    expected = {item["component_id"] for item in methods}
    return {
        "schema_version": "stage8_v4_australian_candidate_matrix_v1",
        "candidates": candidates,
        "unresolved_components": sorted(expected - observed),
        "candidate_count": len(candidates),
    }
