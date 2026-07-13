"""Validation and deterministic bounded expansion for Stage 8 supplemental work."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import canonical_sha256

EVIDENCE_TIERS = {"PROTOCOL_COMPLETION", "CONFIRMATORY_ABLATION", "EXPLORATORY_EXTENSION", "POST_HOC_DIAGNOSTIC"}
PRIORITIES = {"P0_PROTOCOL_COMPLETION", "P1_HIGH_EVIDENCE_EXTENSION", "P2_TARGETED_EXPLORATION", "P3_OPTIONAL_DIAGNOSTIC"}


def validate_supplemental(campaign: dict[str, Any]) -> list[str]:
    errors = []
    if campaign.get("schema_version") != "stage8-supplemental-v1": errors.append("wrong_supplemental_schema_version")
    if campaign.get("outputs", {}).get("root") != "outputs/stage8/supplemental_v1": errors.append("wrong_supplemental_namespace")
    for job in campaign.get("jobs", []):
        job_id = str(job.get("id", ""))
        if job.get("evidence_tier") not in EVIDENCE_TIERS: errors.append(f"invalid_evidence_tier:{job_id}")
        if job.get("priority") not in PRIORITIES: errors.append(f"invalid_priority:{job_id}")
        for key in ("dataset", "view", "folds", "sample_unit", "target_unit", "seeds", "label_budgets", "models", "loss", "hyperparameters", "selection_split", "expected_artifacts", "acceptance_checks", "stop_condition", "resume_behavior", "checkpoint_frequency"):
            if key not in job: errors.append(f"missing_supplemental_field:{job_id}:{key}")
        command = [str(value) for value in job.get("command", [])]
        if "--job-id" in command:
            index = command.index("--job-id")
            if index + 1 >= len(command) or command[index + 1] != job_id:
                errors.append(f"command_job_id_mismatch:{job_id}")
        handler = job.get("handler", {})
        for key in ("id", "mode", "inputs", "outputs", "acceptance_checks"):
            if not handler.get(key):
                errors.append(f"missing_handler_binding:{job_id}:{key}")
        namespace = str(job.get("namespace", ""))
        if namespace and not namespace.startswith("outputs/stage8/supplemental_v1/"):
            errors.append(f"cross_namespace_output:{job_id}")
    return errors


def expand_experiments(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    root = campaign["outputs"]["root"]
    for matrix in campaign.get("experiment_matrices", []):
        axes = matrix.get("axes", {})
        keys = sorted(axes)
        reference = {key: axes[key][0] for key in keys}
        combinations: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for key in keys:
            for value in axes[key]:
                params = {**reference, key: value}
                digest = canonical_sha256(params)
                if digest not in seen:
                    combinations.append((key, params))
                    seen.add(digest)
        limit = int(matrix["max_search_configs"])
        if len(combinations) > limit: combinations = combinations[:limit]
        for index, (changed_axis, params) in enumerate(combinations):
            payload = {"matrix_id": matrix["id"], "index": index, "params": params}
            rows.append({**payload, "reference_params": reference, "changed_axis": changed_axis, "route": matrix["route"], "evidence_tier": matrix["evidence_tier"], "priority": matrix["priority"], "search_size": len(combinations), "max_search_configs": limit, "namespace": f"{root}/{matrix['evidence_tier'].lower()}/{matrix['route']}/{matrix['id']}/{canonical_sha256(payload)[:12]}"})
    return rows


def measured_route_throughput(progress_path: Path) -> list[dict[str, Any]]:
    """Summarise observed reruns_v2 completion throughput without invented runtime."""
    events: dict[str, list[datetime]] = defaultdict(list)
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("timestamp") and row.get("route"):
            events[str(row["route"])].append(datetime.fromisoformat(str(row["timestamp"])))
    rows = []
    for route, stamps in sorted(events.items()):
        stamps.sort()
        elapsed = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) > 1 else None
        rows.append({
            "route": route,
            "observed_completions": len(stamps),
            "observed_window_seconds": elapsed,
            "completions_per_hour": (3600 * (len(stamps) - 1) / elapsed) if elapsed and elapsed > 0 else None,
            "source": str(progress_path),
            "method": "first_to_last_completion_window",
        })
    return rows
