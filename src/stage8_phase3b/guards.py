from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        _resolved(child).relative_to(_resolved(parent))
        return True
    except ValueError:
        return False


def validate_output_root(*, output_root: Path, formal_campaign: Path, writable_root: Path) -> Path:
    output = _resolved(output_root)
    formal = _resolved(formal_campaign)
    writable = _resolved(writable_root)

    if output == formal or is_relative_to(output, formal):
        raise ValueError(f"FORMAL_CAMPAIGN_OUTPUT_ROOT_FORBIDDEN:{output}")

    if not is_relative_to(output, writable):
        raise ValueError(f"OUTPUT_ROOT_OUTSIDE_WRITABLE_WORKTREE:{output}")

    return output


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

