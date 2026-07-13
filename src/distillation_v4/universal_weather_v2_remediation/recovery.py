from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


AUTHORITATIVE_METHODS = (
    ("T1", "scripts/run_stage8_formal_adapter.py", "run_t1", "REUSE_WITH_ARTIFACT_HARDENING"),
    ("T2", "scripts/run_stage8_formal_adapter.py", "run_t2", "REUSE_WITH_ARTIFACT_HARDENING"),
    ("M1", "scripts/run_stage8_formal_adapter.py", "run_m1", "REUSE_WITH_ARTIFACT_HARDENING"),
    ("M2", "scripts/run_stage8_formal_adapter.py", "run_m2", "HISTORICAL_CONTROL_ONLY"),
    ("T1_HELPERS", "src/stage8/formal_routes.py", "T1_WINDOWS,deterministic_fraction_ids", "DIRECT_REUSE"),
    ("ARTIFACT_HELPERS", "src/stage8/formal_training.py", "write_artifacts", "REUSE_WITH_ARTIFACT_HARDENING"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_root: Path, path: str) -> str:
    try:
        value = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", path],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or "UNCOMMITTED_SOURCE"
    except Exception:
        return "UNVERIFIED_GIT_HISTORY"


def build_method_recovery_registry(repo_root: Path) -> dict:
    methods = []
    missing = []
    for method_id, relative_path, symbol, reuse_status in AUTHORITATIVE_METHODS:
        path = repo_root / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        methods.append(
            {
                "method_id": method_id,
                "source_path": relative_path,
                "symbol": symbol,
                "source_sha256": sha256_file(path),
                "source_commit": _source_commit(repo_root, relative_path),
                "reuse_status": reuse_status,
                "scientific_result_reused": False,
            }
        )
    return {
        "schema_version": "method_recovery_registry_v1",
        "methods": methods,
        "missing_required_sources": missing,
        "plotting_provenance": "UNVERIFIED_PLOTTING_PROVENANCE",
        "complete": not missing,
    }
