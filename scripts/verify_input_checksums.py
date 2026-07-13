#!/usr/bin/env python3
"""Verify optional external publication inputs against a user-maintained checksum file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="sha256sum-compatible manifest")
    parser.add_argument("--root", type=Path, required=True, help="root containing the listed relative files")
    args = parser.parse_args()
    failures = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = args.root / relative.strip().lstrip("*")
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("All listed input checksums match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
