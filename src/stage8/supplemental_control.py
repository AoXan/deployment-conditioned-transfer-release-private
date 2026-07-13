"""Typed control-plane contracts for the Stage 8 Supplemental campaign."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    BLOCKED_INTERNAL = "BLOCKED_INTERNAL"
    SKIPPED_BY_GATE = "SKIPPED_BY_GATE"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"


class ArtifactStatus(str, Enum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    COMPLETE_UNVERIFIED = "COMPLETE_UNVERIFIED"
    ARTIFACT_ACCEPTED = "ARTIFACT_ACCEPTED"
    INVALIDATED = "INVALIDATED"


class ScientificStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    READY_FOR_FORMAL_RUN = "READY_FOR_FORMAL_RUN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SCIENTIFIC_GO = "SCIENTIFIC_GO"
    SCIENTIFIC_NO_GO = "SCIENTIFIC_NO_GO"


@dataclass(frozen=True)
class HandlerResult:
    execution_status: ExecutionStatus
    artifact_status: ArtifactStatus
    scientific_status: ScientificStatus
    reason_code: str
    outputs: tuple[str, ...] = ()
    acceptance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.reason_code.strip():
            raise ValueError("handler result requires a non-empty reason_code")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("execution_status", "artifact_status", "scientific_status"):
            value = payload[key]
            payload[key] = value.value if isinstance(value, Enum) else value
        payload["outputs"] = list(payload["outputs"])
        return payload


@dataclass(frozen=True)
class HandlerSpec:
    job_id: str
    handler_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    mode: str


def handler_registry(campaign: dict[str, Any]) -> dict[str, HandlerSpec]:
    registry: dict[str, HandlerSpec] = {}
    for job in campaign["jobs"]:
        handler = job.get("handler") or {}
        spec = HandlerSpec(
            job_id=str(job["id"]),
            handler_id=str(handler.get("id", "")),
            inputs=tuple(map(str, handler.get("inputs", []))),
            outputs=tuple(map(str, handler.get("outputs", job.get("expected_artifacts", [])))),
            acceptance_checks=tuple(map(str, handler.get("acceptance_checks", job.get("acceptance_checks", [])))),
            mode=str(handler.get("mode", "")),
        )
        if not spec.handler_id or not spec.inputs or not spec.outputs or not spec.acceptance_checks:
            raise ValueError(f"incomplete_handler_binding:{spec.job_id}")
        registry[spec.job_id] = spec
    return registry


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_failure_event(ledger: Path, current_registry: Path, event: dict[str, Any]) -> None:
    if not str(event.get("reason_code", "")).strip():
        raise ValueError("failure event requires reason_code")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    current = {"failures": []}
    if current_registry.exists():
        current = json.loads(current_registry.read_text())
    key = (event.get("job_id"), event.get("attempt_id"), event.get("reason_code"))
    retained = [row for row in current.get("failures", []) if (row.get("job_id"), row.get("attempt_id"), row.get("reason_code")) != key]
    retained.append(event)
    _atomic_json(current_registry, {"failures": retained})


def resume_decision(marker: Path, *, expected_namespace: Path, fingerprint_inputs: dict[str, str], artifact_hashes: dict[str, str], runtime_fingerprint: str | None = None) -> tuple[bool, str]:
    required = {"code", "config", "data", "view", "split", "feature"}
    if set(fingerprint_inputs) != required or any(not str(v) for v in fingerprint_inputs.values()):
        return False, "incomplete_expected_fingerprints"
    if not marker.exists():
        return False, "missing_acceptance_marker"
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False, "invalid_acceptance_marker"
    if payload.get("artifact_status") != ArtifactStatus.ARTIFACT_ACCEPTED.value:
        return False, "artifact_not_accepted"
    if Path(payload.get("namespace", "")).resolve() != expected_namespace.resolve():
        return False, "namespace_mismatch"
    if payload.get("fingerprint_inputs") != fingerprint_inputs:
        return False, "six_fingerprint_mismatch"
    if payload.get("artifact_hashes") != artifact_hashes:
        return False, "artifact_hash_mismatch"
    if runtime_fingerprint is not None and payload.get("runtime_fingerprint") != runtime_fingerprint:
        return False, "runtime_fingerprint_mismatch"
    return True, "exact_accepted_match"


HISTORICAL_REQUIRED_COLUMNS = {
    "BASELINE": {"sample_id", "y_true", "y_pred"},
    "T1": {"sample_id", "y_true", "y_pred"},
    "T2": {"sample_id", "y_true", "y_pred"},
    "M1": {"sample_id", "y_true", "y_pred"},
    "M2": {"sample_id", "y_true", "y_pred"},
    "R1": {"sample_id", "y_true", "y_pred", "validation_selected_pred", "nested_oof_ensemble_pred", "router_expert_index"},
    "U1": {"sample_id", "y_true", "y_pred", "lower", "upper", "uncertainty_score"},
    "U2": {"sample_id", "y_true", "y_pred", "lower", "upper", "missing_pattern", "uncertainty_score"},
}

REPAIRED_PROTOCOL_GAPS = {
    "T1": ["huber_modality_encoder", "aulc_delta_aulc_clustered_bootstrap"],
    "T2": ["label_budget_curve"],
    "M1": ["missing_token", "gated_modality_fusion"],
    "M2": ["single_shared_student", "four_loss_ledger"],
    "R1": ["soft_router", "persisted_oof_regret_gate_ledgers"],
    "U1": ["weighted_conformal", "finite_sample_and_ess_ledger", "winkler"],
    "U2": ["deployable_aurc", "random_rejection_control", "fallback_ledger"],
}


def aggregate_selected_replays(replay_root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for path in sorted(replay_root.glob("*/*.json")):
        payload = json.loads(path.read_text())
        if "source_run" not in payload:
            continue
        route = str(payload["route"])
        key = "BASELINE" if route.startswith("BASELINE_") else route
        source = Path(payload["source_run"])
        pred_path = source / "predictions.csv"
        columns: set[str] = set()
        if pred_path.exists():
            columns = set(pred_path.open().readline().strip().split(","))
        missing = sorted(HISTORICAL_REQUIRED_COLUMNS[key] - columns)
        base_equal = all(payload.get(k) is True for k in ("ids_equal", "targets_equal", "predictions_equal", "stable_fingerprint_inputs_equal"))
        commits_equal = payload.get("original_code") == payload.get("replay_code")
        if not base_equal:
            verdict = "REPLAY_FAILED"
        elif missing:
            verdict = "REPLAY_INCOMPLETE"
        elif commits_equal:
            verdict = "EXACT_CODE_REPLAY_VERIFIED"
        else:
            verdict = "NUMERIC_REPRODUCED_UNDER_CODE_DRIFT"
        items.append({
            "route": route,
            "seed": payload.get("seed"),
            "source_run": str(source),
            "source_commit": payload.get("original_code"),
            "replay_commit": payload.get("replay_code"),
            "code_drift": not commits_equal,
            "required_columns": sorted(HISTORICAL_REQUIRED_COLUMNS[key]),
            "missing_historical_required_columns": missing,
            "historical_replay_verdict": verdict,
            "stable_fingerprint_inputs_equal": payload.get("stable_fingerprint_inputs_equal"),
            "ids_equal": payload.get("ids_equal"),
            "targets_equal": payload.get("targets_equal"),
            "predictions_equal": payload.get("predictions_equal"),
            "repaired_protocol_gaps": REPAIRED_PROTOCOL_GAPS.get(key, []),
        })
    verdicts = {item["historical_replay_verdict"] for item in items}
    if "REPLAY_FAILED" in verdicts:
        batch = "BATCH_REPLAY_FAILED"
    elif "REPLAY_INCOMPLETE" in verdicts:
        batch = "BATCH_REPLAY_INCOMPLETE"
    elif verdicts == {"EXACT_CODE_REPLAY_VERIFIED"}:
        batch = "EXACT_BATCH_REPLAY_VERIFIED"
    else:
        batch = "NUMERIC_BATCH_REPRODUCED_WITH_CODE_DRIFT"
    return {"item_count": len(items), "items": items, "batch_verdict": batch, "retraining_triggered": False}
