#!/usr/bin/env python3
"""Resolve environment-variable placeholders in a public YAML configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def resolve(value):
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if isinstance(value, str):
        if value == "${PYTHON_EXECUTABLE}":
            return sys.executable
        return os.path.expandvars(value)
    return value


def walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    else:
        yield value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    payload = resolve(yaml.safe_load(args.input.read_text()))
    unresolved = [str(item) for item in walk(payload) if isinstance(item, str) and "${" in item]
    if unresolved:
        raise SystemExit(f"unresolved environment placeholders: {unresolved}")
    if not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(payload, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
