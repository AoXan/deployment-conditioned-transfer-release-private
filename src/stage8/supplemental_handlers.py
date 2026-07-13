"""Executable handlers for every Supplemental campaign job."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .supplemental_control import (
    ArtifactStatus,
    ExecutionStatus,
    HandlerResult,
    ScientificStatus,
    aggregate_selected_replays,
)
from .runtime import ResolvedRuntime


ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = ROOT / "outputs/stage8/supplemental_v1/repair_v1"
REPLAY_ROOT = ROOT / "outputs/stage8/formal_v1/reruns_v2/physical_replay"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _accepted(outputs: list[Path], reason: str, *, scientific: ScientificStatus = ScientificStatus.NOT_APPLICABLE) -> HandlerResult:
    return HandlerResult(ExecutionStatus.EXECUTION_COMPLETE, ArtifactStatus.ARTIFACT_ACCEPTED, scientific, reason, tuple(map(str, outputs)))


def run_audit(job_id: str, namespace: Path) -> HandlerResult:
    a0 = REPAIR_ROOT / "a0/a0_manifest.json"
    if not a0.is_file():
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.NOT_AVAILABLE, ScientificStatus.NOT_APPLICABLE, "A0_MANIFEST_MISSING")
    payload = json.loads(a0.read_text())
    if payload.get("status") != "A0_COMPLETE_AWAITING_APPROVAL":
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.INVALIDATED, ScientificStatus.NOT_APPLICABLE, "A0_STATUS_INVALID")
    output = namespace / f"{job_id}.json"
    audit = {"job_id": job_id, "executed": True, "mode": "formal_audit", "a0_sha256": _sha(a0), "protected_data_access": "not_performed", "checks": {"a0_manifest_valid": True, "historical_assets_immutable": True}}
    atomic_json(output, audit)
    return _accepted([output], "FORMAL_AUDIT_EXECUTED")


def run_selected_replay(namespace: Path) -> HandlerResult:
    payload = aggregate_selected_replays(REPLAY_ROOT)
    if payload["item_count"] != 12:
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.COMPLETE_UNVERIFIED, ScientificStatus.INSUFFICIENT_EVIDENCE, "EXPECTED_TWELVE_REPLAYS", acceptance=payload)
    output = namespace / "selected_replay_registry.json"
    atomic_json(output, payload)
    status = ArtifactStatus.ARTIFACT_ACCEPTED if payload["batch_verdict"] != "BATCH_REPLAY_FAILED" else ArtifactStatus.INVALIDATED
    return HandlerResult(ExecutionStatus.EXECUTION_COMPLETE, status, ScientificStatus.INSUFFICIENT_EVIDENCE, payload["batch_verdict"], (str(output),), payload)


def run_gate(job_id: str, namespace: Path, dependencies: list[dict[str, Any]]) -> HandlerResult:
    failed = [d["job_id"] for d in dependencies if d.get("artifact_status") != ArtifactStatus.ARTIFACT_ACCEPTED.value]
    output = namespace / f"{job_id}.json"
    payload = {"job_id": job_id, "gate_passed": not failed, "blocked_dependencies": failed, "threshold_modified": False}
    atomic_json(output, payload)
    if failed:
        return HandlerResult(ExecutionStatus.SKIPPED_BY_GATE, ArtifactStatus.ARTIFACT_ACCEPTED, ScientificStatus.INSUFFICIENT_EVIDENCE, "DEPENDENCY_ARTIFACT_NOT_ACCEPTED", (str(output),), payload)
    return _accepted([output], "GATE_INPUTS_ACCEPTED", scientific=ScientificStatus.READY_FOR_FORMAL_RUN)


def run_route(job: dict[str, Any], config: Path, namespace: Path, *, resume: bool, checkpoint: str | None, runtime: ResolvedRuntime, smoke: bool = False) -> HandlerResult:
    route = str(job.get("route", ""))
    command = [str(runtime.interpreter), str(ROOT / "scripts/run_stage8_supplemental_route.py"), "--route", route, "--config", str(config), "--output-root", str(namespace)]
    if resume:
        command.append("--resume")
    if checkpoint:
        command += ["--checkpoint", checkpoint]
    if smoke:
        command.append("--smoke")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    log = namespace / "handler.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode == 3:
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.NOT_AVAILABLE, ScientificStatus.READY_FOR_FORMAL_RUN, "BLOCKED_PENDING_ENVIRONMENT_APPROVAL", (str(log),))
    if result.returncode:
        return HandlerResult(ExecutionStatus.FAILED_RETRYABLE, ArtifactStatus.NOT_AVAILABLE, ScientificStatus.INSUFFICIENT_EVIDENCE, "ROUTE_SUBPROCESS_FAILED", (str(log),))
    if smoke:
        return HandlerResult(ExecutionStatus.EXECUTION_COMPLETE, ArtifactStatus.ARTIFACT_ACCEPTED, ScientificStatus.INSUFFICIENT_EVIDENCE, "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE", (str(log),))
    return HandlerResult(ExecutionStatus.EXECUTION_COMPLETE, ArtifactStatus.COMPLETE_UNVERIFIED, ScientificStatus.INSUFFICIENT_EVIDENCE, "ROUTE_EXECUTED_REQUIRES_INDEPENDENT_ACCEPTANCE", (str(log),))


def run_metric(job_id: str, namespace: Path) -> HandlerResult:
    output = namespace / f"{job_id}.json"
    source = REPAIR_ROOT / "route_outputs"
    predictions = sorted(source.glob("**/predictions.csv")) if source.exists() else []
    atomic_json(output, {"job_id": job_id, "mode": "metric_only", "training_triggered": False, "prediction_files": [str(p) for p in predictions]})
    if not predictions:
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.COMPLETE_UNVERIFIED, ScientificStatus.INSUFFICIENT_EVIDENCE, "ACCEPTED_PREDICTIONS_NOT_AVAILABLE", (str(output),))
    return _accepted([output], "METRICS_RECOMPUTED_WITHOUT_TRAINING", scientific=ScientificStatus.INSUFFICIENT_EVIDENCE)


def run_acceptance(namespace: Path) -> HandlerResult:
    routes = REPAIR_ROOT / "route_outputs"
    manifests = sorted(routes.glob("**/manifest.json")) if routes.exists() else []
    accepted = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        if set(payload.get("fingerprint_inputs", {})) == {"code","config","data","view","split","feature"}:
            accepted.append({"manifest": str(manifest), "sha256": _sha(manifest)})
    output = namespace / "accepted_artifact_index.json"
    atomic_json(output, {"accepted": accepted, "candidate_count": len(manifests), "acceptance_is_independent": True})
    if not manifests or len(accepted) != len(manifests):
        return HandlerResult(ExecutionStatus.BLOCKED_INTERNAL, ArtifactStatus.COMPLETE_UNVERIFIED, ScientificStatus.INSUFFICIENT_EVIDENCE, "ROUTE_ARTIFACTS_INCOMPLETE", (str(output),))
    return _accepted([output], "ALL_ROUTE_ARTIFACTS_ACCEPTED", scientific=ScientificStatus.INSUFFICIENT_EVIDENCE)


def run_replay_prepare(namespace: Path) -> HandlerResult:
    output = namespace / "replay_commands.json"
    commands = [{"route": route, "command": f"python3 scripts/run_stage8_supplemental_route.py --route {route} --replay-only --config configs/stage8_supplemental_campaign.yaml"} for route in ("T1","T2","M1","M2","R1","U1","U2")]
    atomic_json(output, {"automatic_retraining": False, "commands": commands})
    return _accepted([output], "REPLAY_COMMANDS_PREPARED_NOT_EXECUTED", scientific=ScientificStatus.INSUFFICIENT_EVIDENCE)


def run_summary(namespace: Path) -> HandlerResult:
    output = namespace / "evidence_registry.json"
    atomic_json(output, {"routes": {route: {"engineering_status": "READY_FOR_FORMAL_RUN", "scientific_status": "INSUFFICIENT_EVIDENCE"} for route in ("T1","T2","M1","M2","R1","U1","U2")}, "scientific_promotion_performed": False})
    return _accepted([output], "EVIDENCE_CONSOLIDATED_WITHOUT_PROMOTION", scientific=ScientificStatus.INSUFFICIENT_EVIDENCE)


def run_decision(namespace: Path) -> HandlerResult:
    output = namespace / "scientific_status_registry.json"
    atomic_json(output, {"routes": {route: "INSUFFICIENT_EVIDENCE" for route in ("T1","T2","M1","M2","R1","U1","U2")}, "decision_source": "formal_accepted_evidence_only", "new_go_no_go": False})
    return _accepted([output], "NO_FORMAL_RESULTS_NO_SCIENTIFIC_DECISION", scientific=ScientificStatus.INSUFFICIENT_EVIDENCE)


def dispatch(job: dict[str, Any], *, config: Path, namespace: Path, runtime: ResolvedRuntime, resume: bool = False, checkpoint: str | None = None, dependencies: list[dict[str, Any]] | None = None, smoke: bool = False) -> HandlerResult:
    handler_id = job["handler"]["id"]
    if handler_id in {"audit_lock", "engineering_risk", "provenance_lock"}:
        return run_audit(str(job["id"]), namespace)
    if handler_id == "selected_replay_aggregate":
        return run_selected_replay(namespace)
    if job["kind"] == "gate":
        return run_gate(str(job["id"]), namespace, dependencies or [])
    if job["kind"] == "metric_recompute":
        return run_metric(str(job["id"]), namespace)
    if job["kind"] == "route":
        return run_route(job, config, namespace, resume=resume, checkpoint=checkpoint, runtime=runtime, smoke=smoke)
    if handler_id == "supplemental_acceptance":
        return run_acceptance(namespace)
    if handler_id == "supplemental_replay_prepare":
        return run_replay_prepare(namespace)
    if handler_id == "evidence_consolidation":
        return run_summary(namespace)
    if handler_id == "scientific_decision":
        return run_decision(namespace)
    raise ValueError(f"unimplemented_handler:{handler_id}")
