from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file, stable_hash


def heartbeat_path(output_root: Path) -> Path:
    return output_root / "status" / "heartbeat.json"


def stop_request_path(output_root: Path) -> Path:
    return output_root / "status" / "STOP_REQUESTED.json"


def failure_ledger_path(output_root: Path) -> Path:
    return output_root / "failures" / "failure_ledger.jsonl"


def write_heartbeat(
    output_root: Path,
    *,
    status: str,
    current_job: str | None,
    completed: int,
    failed: int,
    remaining: int,
    plan_fingerprint: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "remediation_heartbeat_v1",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "status": status,
        "current_job": current_job,
        "completed": int(completed),
        "failed": int(failed),
        "remaining": int(remaining),
        "plan_fingerprint": plan_fingerprint,
    }
    atomic_json(heartbeat_path(output_root), payload)
    return payload


def request_stop(
    output_root: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "remediation_stop_request_v1",
        "timestamp_unix": time.time(),
        "reason": reason,
        "requested": True,
    }
    atomic_json(stop_request_path(output_root), payload)
    return payload


def clear_stop_request(output_root: Path) -> None:
    stop_request_path(output_root).unlink(missing_ok=True)


def stop_requested(output_root: Path) -> bool:
    return stop_request_path(output_root).is_file()


def append_failure(
    output_root: Path,
    event: dict[str, Any],
) -> None:
    path = failure_ledger_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_unix": time.time(),
        **event,
    }

    with path.open("a") as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


def exact_resume_decision(
    job_directory: Path,
    *,
    expected_namespace: str,
    expected_plan_fingerprint: str,
    expected_config_hash: str,
    expected_code_hash: str,
    expected_data_hash: str,
    expected_split_hash: str,
    expected_feature_hash: str,
) -> tuple[bool, str]:
    required = {
        "manifest.json",
        "acceptance.json",
        "COMPLETED.json",
        "predictions.csv",
        "fold_ids.json",
        "metrics.json",
    }

    missing = sorted(
        name
        for name in required
        if not (job_directory / name).is_file()
    )
    if missing:
        return False, "REQUIRED_ARTIFACT_MISSING"

    manifest = json.loads(
        (job_directory / "manifest.json").read_text()
    )
    acceptance = json.loads(
        (job_directory / "acceptance.json").read_text()
    )
    completed = json.loads(
        (job_directory / "COMPLETED.json").read_text()
    )

    if acceptance.get("status") != "ARTIFACT_ACCEPTED":
        return False, "ARTIFACT_NOT_ACCEPTED"

    if completed.get("status") != "COMPLETED":
        return False, "COMPLETION_STATUS_INVALID"

    fingerprint = manifest.get("fingerprint")
    if not fingerprint:
        return False, "MANIFEST_FINGERPRINT_MISSING"

    if acceptance.get("fingerprint") != fingerprint:
        return False, "ACCEPTANCE_FINGERPRINT_MISMATCH"

    if completed.get("fingerprint") != fingerprint:
        return False, "COMPLETION_FINGERPRINT_MISMATCH"

    checks = {
        "namespace": expected_namespace,
        "plan_fingerprint": expected_plan_fingerprint,
        "resolved_config_hash": expected_config_hash,
        "code_tree_hash": expected_code_hash,
        "data_hash": expected_data_hash,
        "split_fingerprint": expected_split_hash,
        "feature_contract_hash": expected_feature_hash,
    }

    for key, expected in checks.items():
        observed = manifest.get(key)
        if observed != expected:
            return False, f"{key.upper()}_MISMATCH"

    artifact_hashes = acceptance.get(
        "artifact_sha256",
        {},
    )
    expected_artifacts = {
        "predictions": job_directory / "predictions.csv",
        "fold_ids": job_directory / "fold_ids.json",
        "metrics": job_directory / "metrics.json",
    }

    for name, path in expected_artifacts.items():
        expected_hash = artifact_hashes.get(name)
        if not expected_hash:
            return False, f"{name.upper()}_HASH_MISSING"
        if sha256_file(path) != expected_hash:
            return False, f"{name.upper()}_HASH_MISMATCH"

    return True, "EXACT_ACCEPTED_MATCH"


def runtime_contract_fingerprint(payload: dict[str, Any]) -> str:
    return stable_hash(payload)
