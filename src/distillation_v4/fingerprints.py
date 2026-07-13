from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_tree_fingerprint(root: Path, paths: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    records = []
    for path in sorted({path.resolve() for path in paths}):
        records.append({"path": str(path.relative_to(root.resolve())), "sha256": sha256_file(path)})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, records


def mapping_fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
