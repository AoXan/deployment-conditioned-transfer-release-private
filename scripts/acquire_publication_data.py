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
CSIRO_WAITE_COLLECTION = "https://data.csiro.au/dap/ws/v2/collections/39878"
CSIRO_WAITE_LICENCE = "Creative Commons Attribution 4.0 International Licence"
WAITE_FILENAME = "Waite_Trial_Data.xls"
WAITE_SHA256 = "a5b1b7f4c943a6533917e3b7c4fe51eaa030c77b9777cd34c0864ca3c8961c29"


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
            member_target = (destination / member.filename).resolve()
            if not member_target.is_relative_to(root):
                raise RuntimeError(f"unsafe path in CY-Bench archive: {member.filename}")
        bundle.extractall(destination)
    archive.unlink()
    return destination


def csiro_waite_records() -> tuple[dict, dict]:
    with urllib.request.urlopen(CSIRO_WAITE_COLLECTION, timeout=30) as response:
        collection = json.load(response)
    with urllib.request.urlopen(f"{CSIRO_WAITE_COLLECTION}/data", timeout=30) as response:
        data = json.load(response)
    if collection.get("doi") != "10.4225/08/55E5165EC0D29":
        raise RuntimeError("unexpected DOI returned by the CSIRO Waite collection")
    if collection.get("licence") != CSIRO_WAITE_LICENCE or data.get("licence") != CSIRO_WAITE_LICENCE:
        raise RuntimeError("the CSIRO Waite record no longer reports the expected CC BY 4.0 licence")
    return collection, data


def download_waite(output_root: Path) -> Path:
    _, data = csiro_waite_records()
    item = next((entry for entry in data.get("file", []) if entry.get("filename") == WAITE_FILENAME), None)
    if item is None:
        raise RuntimeError(f"{WAITE_FILENAME} is absent from the CSIRO Waite record")
    link = item.get("presignedLink", {}).get("href") or item.get("link", {}).get("href")
    if not link:
        raise RuntimeError("the CSIRO Waite record has no downloadable file link")
    destination = output_root / "waite"
    destination.mkdir(parents=True, exist_ok=True)
    workbook = destination / WAITE_FILENAME
    with urllib.request.urlopen(link, timeout=60) as source, workbook.open("wb") as target:
        shutil.copyfileobj(source, target)
    digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    if digest != WAITE_SHA256:
        workbook.unlink(missing_ok=True)
        raise RuntimeError("downloaded Waite workbook checksum does not match the accepted input")
    return workbook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cybench", "g2f", "roseworthy", "waite"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/external"))
    parser.add_argument("--download", action="store_true", help="Download only a source whose manifest permits it.")
    parser.add_argument(
        "--validate-only", action="store_true", help="Inspect source metadata without downloading files."
    )
    args = parser.parse_args()
    if args.dataset == "cybench":
        record = zenodo_record()
        payload = {"dataset": "cybench", "record": record.get("id"), "doi": record.get("doi"), "downloaded": False}
        if args.download and not args.validate_only:
            payload["path"] = str(download_cybench(args.output_root))
            payload["downloaded"] = True
        print(json.dumps(payload, indent=2))
        return 0
    if args.dataset == "waite":
        collection, data = csiro_waite_records()
        payload = {
            "dataset": "waite",
            "doi": collection.get("doi"),
            "version": collection.get("versionNumber"),
            "licence": collection.get("licence"),
            "filename": WAITE_FILENAME,
            "expected_sha256": WAITE_SHA256,
            "public_access": collection.get("accessLevel") == "Public" and collection.get("dataRestricted") == "FALSE",
            "downloaded": False,
        }
        if not any(entry.get("filename") == WAITE_FILENAME for entry in data.get("file", [])):
            raise RuntimeError(f"{WAITE_FILENAME} is absent from the CSIRO Waite record")
        if args.download and not args.validate_only:
            payload["path"] = str(download_waite(args.output_root))
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
