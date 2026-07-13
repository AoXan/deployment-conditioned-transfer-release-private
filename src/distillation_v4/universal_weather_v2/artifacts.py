from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    serialised = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return sha256_bytes(serialised)


def atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def atomic_torch_save(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
        )
    )

    os.close(file_descriptor)

    temporary = Path(temporary_name)

    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    atomic_torch_save(path, payload)

    manifest = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "metadata": metadata,
    }

    atomic_json(
        path.with_suffix(
            path.suffix + ".manifest.json"
        ),
        manifest,
    )

    return manifest


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(
        payload["state_dict"],
        strict=strict,
    )

    return payload


def completion_path(
    job_directory: Path,
) -> Path:
    return (
        job_directory
        / "COMPLETED.json"
    )


def job_is_complete(
    job_directory: Path,
    *,
    fingerprint: str,
) -> bool:
    marker = completion_path(
        job_directory
    )

    if not marker.is_file():
        return False

    try:
        payload = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return (
        payload.get("fingerprint")
        == fingerprint
        and payload.get("status")
        == "COMPLETED"
    )


def mark_complete(
    job_directory: Path,
    *,
    fingerprint: str,
    result: dict[str, Any],
) -> None:
    atomic_json(
        completion_path(
            job_directory
        ),
        {
            "status": "COMPLETED",
            "fingerprint": fingerprint,
            "result": result,
        },
    )


def record_failure(
    *,
    failure_ledger: Path,
    job: dict[str, Any],
    error: BaseException,
) -> None:
    failure_ledger.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing: list[dict[str, Any]] = []

    if failure_ledger.is_file():
        try:
            existing = json.loads(
                failure_ledger.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            existing = []

    existing.append(
        {
            "job": job,
            "failure_type": type(error).__name__,
            "reason": str(error),
        }
    )

    atomic_json(
        failure_ledger,
        existing,
    )
