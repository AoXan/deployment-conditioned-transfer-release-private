from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HISTORICAL_DISPOSITIONS = {
    "RETAIN_AS_IS",
    "RECOMPUTE_METRICS_ONLY",
    "RERUN_FROM_EXISTING_VIEW",
    "REBUILD_VIEW_AND_RERUN",
    "RETAIN_AS_POST_HOC_ONLY",
    "INVALIDATED",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def artifact_reuse_status(marker_path: Path, expected_fingerprint: str) -> tuple[bool, str]:
    if not marker_path.exists():
        return False, "missing_completion_marker"
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "invalid_completion_marker"
    if payload.get("status") != "VERIFIED":
        return False, "status_not_verified"
    if payload.get("fingerprint") != expected_fingerprint:
        return False, "fingerprint_mismatch"
    return True, "fingerprint_match"


def classify_historical_artifact(
    *,
    hashes_match: bool,
    predictions_valid: bool,
    metrics_valid: bool,
    view_valid: bool,
    leakage: bool,
    post_hoc: bool,
    irrecoverable: bool = False,
) -> str:
    """Return the smallest scientifically legal action for a historical result."""
    if irrecoverable or (leakage and view_valid and predictions_valid):
        return "INVALIDATED"
    if not view_valid:
        return "REBUILD_VIEW_AND_RERUN"
    if post_hoc:
        return "RETAIN_AS_POST_HOC_ONLY"
    if not hashes_match or not predictions_valid:
        return "RERUN_FROM_EXISTING_VIEW"
    if not metrics_valid:
        return "RECOMPUTE_METRICS_ONLY"
    return "RETAIN_AS_IS"


def write_tombstone(path: Path, *, job_id: str, reason: str, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "status": "INVALIDATED", "reason": reason, "fingerprint": fingerprint}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

