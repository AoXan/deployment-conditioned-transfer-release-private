#!/usr/bin/env python3
"""Inspect or acquire publication datasets without downloading restricted data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ZENODO_RECORD = "17279151"


def zenodo_record() -> dict:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=30) as response:
        return json.load(response)


def download_cybench(output_root: Path) -> Path:
    record = zenodo_record()
    item = next((entry for entry in record.get("files", []) if entry.get("key") == "cybench-data.zip"), None)
    if item is None:
        raise RuntimeError("cybench-data.zip is absent from the pinned Zenodo record")
    output_root.mkdir(parents=True, exist_ok=True)
    archive = output_root / "cybench-data.zip"
    with urllib.request.urlopen(item["links"]["self"], timeout=60) as source, archive.open("wb") as target:
        shutil.copyfileobj(source, target)
    algorithm, expected = item["checksum"].split(":", 1)
    digest = hashlib.new(algorithm)
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError("downloaded CY-Bench archive checksum does not match the Zenodo record")
    destination = output_root / "cybench/raw"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe path in CY-Bench archive: {member.filename}")
        bundle.extractall(destination)
    archive.unlink()
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cybench", "g2f", "roseworthy", "waite"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/external"))
    parser.add_argument("--download", action="store_true", help="Download only a source whose manifest permits it.")
    parser.add_argument("--validate-only", action="store_true", help="Inspect source metadata without downloading files.")
    args = parser.parse_args()
    if args.dataset == "cybench":
        record = zenodo_record()
        payload = {"dataset": "cybench", "record": record.get("id"), "doi": record.get("doi"), "downloaded": False}
        if args.download and not args.validate_only:
            payload["path"] = str(download_cybench(args.output_root))
            payload["downloaded"] = True
        print(json.dumps(payload, indent=2))
        return 0
    message = (
        f"{args.dataset} requires source-specific access or licence review; this command never downloads it. "
        "Follow docs/reproducibility/data_access.md and place an authorised copy under --output-root."
    )
    print(json.dumps({"dataset": args.dataset, "status": "MANUAL_ACCESS_REQUIRED", "message": message}, indent=2))
    return 2 if args.download else 0


if __name__ == "__main__":
    raise SystemExit(main())
