"""Immutable manifests for external Stage 8 inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORBIDDEN = ("nvt", "observed_values", "observed-target", "observed_target")


def _guard(path: Path) -> None:
    text = str(path).lower().replace(" ", "_")
    if any(token in text for token in FORBIDDEN):
        raise PermissionError(f"excluded asset path rejected: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot_manifest(roots: dict[str, Path], output: Path) -> dict:
    snapshots = {}
    for name, root in roots.items():
        root = root.resolve(); _guard(root)
        files = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            _guard(path)
            stat = path.stat()
            files.append({"relative_path": str(path.relative_to(root)), "absolute_path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(path)})
        snapshots[name] = {"root": str(root), "file_count": len(files), "files": files}
    payload = {"schema_version": "stage8-snapshot-v1", "snapshots": snapshots}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def validate_snapshot_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    errors = []
    for name, snapshot in payload["snapshots"].items():
        for record in snapshot["files"]:
            candidate = Path(record["absolute_path"]); _guard(candidate)
            if not candidate.is_file():
                errors.append({"snapshot": name, "path": str(candidate), "reason": "missing"})
            elif _sha256(candidate) != record["sha256"]:
                errors.append({"snapshot": name, "path": str(candidate), "reason": "sha256_mismatch"})
    return errors
