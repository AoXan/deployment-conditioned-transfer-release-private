#!/usr/bin/env python3
"""Build or validate the allowlisted publication repository tree."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
from pathlib import Path

import yaml


IDENTITY_MARKERS = (b"/Users/", b"OneDrive", b"Adelaide", b"AIML")
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "OpenAI key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
}
SENSITIVE_FILENAMES = {".env", "id_rsa", "id_ed25519", "credentials.json"}


def excluded(relative: Path, patterns: list[str]) -> bool:
    text = relative.as_posix()
    return any(fnmatch.fnmatch(text, pattern) or text == pattern or text.startswith(pattern.rstrip("/") + "/") for pattern in patterns)


def sources(root: Path, patterns: list[str], excludes: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        matches = list(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"release include did not match: {pattern}")
        for match in matches:
            if match.is_dir():
                found.update(path for path in match.rglob("*") if path.is_file())
            elif match.is_file():
                found.add(match)
    return sorted(path for path in found if not excluded(path.relative_to(root), excludes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("configs/publication_release_manifest.yaml"))
    parser.add_argument("--output", type=Path, default=Path("dist/publication-release"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = yaml.safe_load(manifest_path.read_text())
    files = sources(root, manifest["include"], manifest["exclude"])
    findings = []
    detector_documents = {
        Path("scripts/build_publication_release.py"),
        Path("tests/test_publication_repro_cli.py"),
        Path("docs/reproducibility/release_scope.md"),
    }
    for path in files:
        if path.is_symlink():
            findings.append(f"symlink: {path.relative_to(root)}")
            continue
        relative = path.relative_to(root)
        if path.name in SENSITIVE_FILENAMES:
            findings.append(f"sensitive filename: {relative}")
        if relative not in detector_documents and path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".tex", ".bib", ".txt", ".csv"}:
            payload = path.read_bytes()
            for marker in IDENTITY_MARKERS:
                if marker in payload:
                    findings.append(f"identity/path marker {marker.decode(errors='replace')}: {path.relative_to(root)}")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(payload):
                    findings.append(f"{label}: {relative}")
    if findings:
        raise RuntimeError("release scan failed:\n" + "\n".join(findings))
    output = args.output if args.output.is_absolute() else root / args.output
    if not args.validate_only:
        if output.exists():
            shutil.rmtree(output)
        for source in files:
            destination = output / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(json.dumps({"status": "PASS", "files": len(files), "output": str(output), "written": not args.validate_only}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
