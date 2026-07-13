from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


NON_SCIENTIFIC_EXECUTION_STATES = {
    "DRY_RUN",
    "ACCOUNTING_ONLY",
    "CONTROL_ONLY",
    "PLANNED",
    "NOT_RUN",
    "SMOKE_ONLY",
    "SYNTHETIC_ONLY",
}

NON_SCIENTIFIC_BOOLEAN_FLAGS = {
    "dry_run",
    "accounting_only",
    "control_only",
    "smoke_only",
    "synthetic_only",
}

REQUIRED_JOB_ARTIFACTS = {
    "manifest.json",
    "acceptance.json",
    "COMPLETED.json",
    "predictions.csv",
    "fold_ids.json",
    "metrics.json",
}


@dataclass(frozen=True)
class JobAuditResult:
    accepted: bool
    engineering_completed: bool
    scientific_ready: bool
    reasons: tuple[str, ...]
    evidence: dict[str, Any]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as error:
        raise ValueError(
            f"INVALID_JSON:{path}:{type(error).__name__}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")

    return payload


def _nested_values(payload: Any, key_name: str) -> list[Any]:
    values: list[Any] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() == key_name.lower():
                values.append(value)
            values.extend(_nested_values(value, key_name))

    elif isinstance(payload, list):
        for value in payload:
            values.extend(_nested_values(value, key_name))

    return values


def _contains_non_scientific_execution(payload: Any) -> list[str]:
    reasons: list[str] = []

    if not isinstance(payload, (dict, list)):
        return reasons

    for key in (
        "status",
        "execution_status",
        "artifact_status",
        "scientific_status",
        "mode",
        "role",
    ):
        for value in _nested_values(payload, key):
            normalized = str(value).strip().upper()
            if normalized in NON_SCIENTIFIC_EXECUTION_STATES:
                reasons.append(
                    f"NON_SCIENTIFIC_EXECUTION_STATE:{key}:{normalized}"
                )

    for flag in NON_SCIENTIFIC_BOOLEAN_FLAGS:
        for value in _nested_values(payload, flag):
            if value is True:
                reasons.append(
                    f"NON_SCIENTIFIC_EXECUTION_FLAG:{flag}"
                )

    return sorted(set(reasons))


def audit_job_artifacts(
    job_directory: Path,
    *,
    expected_namespace: str | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_job_id: str | None = None,
    expected_source_route: str | None = None,
    expected_split_fingerprint: str | None = None,
    expected_dependency_checkpoint_hash: str | None = None,
    allow_smoke_only: bool = False,
) -> JobAuditResult:
    """Reject engineering-complete jobs lacking scientific execution evidence.

    This deliberately treats completion, zero failures, and artifact presence
    as insufficient unless acceptance, split, route, and checkpoint lineage
    are mutually consistent.
    """

    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "job_directory": str(job_directory),
        "allow_smoke_only": allow_smoke_only,
        "audit_scope": (
            "SMOKE_EXECUTION_VALIDATION"
            if allow_smoke_only
            else "FORMAL_SCIENTIFIC_ACCEPTANCE"
        ),
    }

    completed_path = job_directory / "COMPLETED.json"
    engineering_completed = completed_path.is_file()

    missing = sorted(
        name
        for name in REQUIRED_JOB_ARTIFACTS
        if not (job_directory / name).is_file()
    )

    if missing:
        reasons.append(
            "REQUIRED_SCIENTIFIC_ARTIFACTS_MISSING:"
            + ",".join(missing)
        )

    if not engineering_completed:
        reasons.append("COMPLETION_MARKER_MISSING")

    if missing:
        return JobAuditResult(
            accepted=False,
            engineering_completed=engineering_completed,
            scientific_ready=False,
            reasons=tuple(reasons),
            evidence=evidence,
        )

    manifest = _load_object(job_directory / "manifest.json")
    acceptance = _load_object(job_directory / "acceptance.json")
    completed = _load_object(completed_path)
    folds = _load_object(job_directory / "fold_ids.json")

    evidence.update(
        {
            "manifest": manifest,
            "acceptance": acceptance,
            "completed": completed,
            "folds": folds,
        }
    )

    execution_reasons = [
        *_contains_non_scientific_execution(manifest),
        *_contains_non_scientific_execution(acceptance),
        *_contains_non_scientific_execution(completed),
    ]

    if allow_smoke_only:
        execution_reasons = [
            reason
            for reason in execution_reasons
            if reason
            != "NON_SCIENTIFIC_EXECUTION_FLAG:smoke_only"
        ]

        smoke_flags = [
            *_nested_values(manifest, "smoke_only"),
            *_nested_values(acceptance, "smoke_only"),
            *_nested_values(completed, "smoke_only"),
        ]

        if not any(value is True for value in smoke_flags):
            reasons.append(
                "SMOKE_AUDIT_MODE_REQUIRES_SMOKE_ONLY_EVIDENCE"
            )

    reasons.extend(execution_reasons)

    if acceptance.get("status") != "ARTIFACT_ACCEPTED":
        reasons.append("ARTIFACT_NOT_ACCEPTED")

    if completed.get("status") != "COMPLETED":
        reasons.append("COMPLETION_STATUS_INVALID")

    manifest_fingerprint = manifest.get("fingerprint")
    if not manifest_fingerprint:
        reasons.append("MANIFEST_FINGERPRINT_MISSING")
    else:
        if acceptance.get("fingerprint") != manifest_fingerprint:
            reasons.append("ACCEPTANCE_FINGERPRINT_MISMATCH")

        if completed.get("fingerprint") != manifest_fingerprint:
            reasons.append("COMPLETION_FINGERPRINT_MISMATCH")

    if expected_namespace is not None:
        if manifest.get("namespace") != expected_namespace:
            reasons.append("NAMESPACE_MISMATCH")

    if expected_plan_fingerprint is not None:
        if (
            manifest.get("plan_fingerprint")
            != expected_plan_fingerprint
        ):
            reasons.append("PLAN_FINGERPRINT_MISMATCH")

    job_payload = manifest.get("job", {})
    if not isinstance(job_payload, dict):
        job_payload = {}

    if expected_job_id is not None:
        observed_job_id = (
            job_payload.get("candidate_id")
            or job_payload.get("job_id")
            or manifest.get("job_id")
        )
        if observed_job_id != expected_job_id:
            reasons.append("JOB_IDENTITY_MISMATCH")

    if expected_source_route is not None:
        observed_route = (
            job_payload.get("source_route")
            or job_payload.get("route")
            or manifest.get("source_route")
            or manifest.get("route")
        )
        if observed_route != expected_source_route:
            reasons.append("SOURCE_ROUTE_MISMATCH")

    if expected_split_fingerprint is not None:
        manifest_split = (
            manifest.get("split_fingerprint")
            or manifest.get("split_hash")
        )
        fold_split = (
            folds.get("split_fingerprint")
            or folds.get("split_hash")
        )

        if manifest_split != expected_split_fingerprint:
            reasons.append("MANIFEST_SPLIT_FINGERPRINT_MISMATCH")

        if fold_split != expected_split_fingerprint:
            reasons.append("FOLD_SPLIT_FINGERPRINT_MISMATCH")

        if manifest_split != fold_split:
            reasons.append("MANIFEST_FOLD_SPLIT_DISAGREEMENT")

    test_training_flags = [
        *_nested_values(manifest, "target_test_used_for_training"),
        *_nested_values(manifest, "target_test_used_for_selection"),
        *_nested_values(manifest, "test_labels_used_for_training"),
        *_nested_values(manifest, "outer_test_used_for_selection"),
    ]

    if any(value is True for value in test_training_flags):
        reasons.append("TARGET_OR_OUTER_TEST_LEAKAGE_DECLARED")

    if expected_dependency_checkpoint_hash is not None:
        declared_hashes = [
            manifest.get("source_checkpoint_hash"),
            manifest.get("dependency_checkpoint_hash"),
            manifest.get("expected_checkpoint_hash"),
        ]
        declared_hashes = [
            value for value in declared_hashes if value is not None
        ]

        actual_loaded_hashes = [
            *_nested_values(
                manifest,
                "actual_checkpoint_hash_loaded",
            ),
            *_nested_values(
                acceptance,
                "actual_checkpoint_hash_loaded",
            ),
        ]

        checkpoint_paths = [
            manifest.get("source_checkpoint"),
            manifest.get("dependency_checkpoint"),
        ]
        checkpoint_paths = [
            Path(value)
            for value in checkpoint_paths
            if isinstance(value, str) and value
        ]

        if (
            expected_dependency_checkpoint_hash
            not in declared_hashes
        ):
            reasons.append(
                "DECLARED_DEPENDENCY_CHECKPOINT_HASH_MISMATCH"
            )

        if not actual_loaded_hashes:
            reasons.append(
                "ACTUAL_LOADED_CHECKPOINT_HASH_NOT_RECORDED"
            )
        elif (
            expected_dependency_checkpoint_hash
            not in actual_loaded_hashes
        ):
            reasons.append(
                "ACTUAL_LOADED_CHECKPOINT_HASH_MISMATCH"
            )

        existing_checkpoint_paths = [
            path for path in checkpoint_paths if path.is_file()
        ]

        if checkpoint_paths and not existing_checkpoint_paths:
            reasons.append("DEPENDENCY_CHECKPOINT_FILE_MISSING")

        for checkpoint_path in existing_checkpoint_paths:
            if (
                sha256_file(checkpoint_path)
                != expected_dependency_checkpoint_hash
            ):
                reasons.append(
                    "DEPENDENCY_CHECKPOINT_FILE_HASH_MISMATCH"
                )

    artifact_hashes = acceptance.get(
        "artifact_sha256",
        {},
    )

    required_hashes = {
        "predictions": job_directory / "predictions.csv",
        "fold_ids": job_directory / "fold_ids.json",
        "metrics": job_directory / "metrics.json",
    }

    if not isinstance(artifact_hashes, dict):
        reasons.append("ARTIFACT_HASH_LEDGER_INVALID")
    else:
        for name, path in required_hashes.items():
            expected_hash = artifact_hashes.get(name)
            if not expected_hash:
                reasons.append(
                    f"{name.upper()}_HASH_MISSING"
                )
                continue

            if sha256_file(path) != expected_hash:
                reasons.append(
                    f"{name.upper()}_HASH_MISMATCH"
                )

    reasons = sorted(set(reasons))
    accepted = not reasons

    return JobAuditResult(
        accepted=accepted,
        engineering_completed=engineering_completed,
        scientific_ready=accepted,
        reasons=tuple(reasons),
        evidence=evidence,
    )


def audit_campaign_completion(
    campaign_root: Path,
) -> dict[str, Any]:
    """Separate completion-marker counts from accepted scientific jobs."""

    completed_markers = sorted(
        campaign_root.rglob("COMPLETED.json")
    )

    audited_directories: set[Path] = set()
    accepted_jobs = 0
    rejected_jobs = 0
    rejection_histogram: dict[str, int] = {}

    for marker in completed_markers:
        directory = marker.parent

        if directory in audited_directories:
            continue
        audited_directories.add(directory)

        result = audit_job_artifacts(directory)

        if result.accepted:
            accepted_jobs += 1
        else:
            rejected_jobs += 1
            for reason in result.reasons:
                rejection_histogram[reason] = (
                    rejection_histogram.get(reason, 0) + 1
                )

    engineering_completed = len(completed_markers)

    return {
        "schema_version": "campaign_completion_scientific_audit_v1",
        "campaign_root": str(campaign_root),
        "engineering_completion_markers": engineering_completed,
        "audited_job_directories": len(audited_directories),
        "scientifically_accepted_jobs": accepted_jobs,
        "scientifically_rejected_jobs": rejected_jobs,
        "completion_is_not_scientific_acceptance": True,
        "all_completed_is_insufficient": (
            engineering_completed > 0
            and accepted_jobs < len(audited_directories)
        ),
        "scientific_campaign_ready": (
            len(audited_directories) > 0
            and rejected_jobs == 0
            and accepted_jobs == len(audited_directories)
        ),
        "rejection_histogram": dict(
            sorted(
                rejection_histogram.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }
