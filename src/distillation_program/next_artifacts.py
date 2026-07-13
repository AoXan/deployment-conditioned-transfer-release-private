from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


FINGERPRINT_KEYS = ("code", "config", "data", "view", "split", "feature", "model", "runtime")


def fingerprint_bundle(values: dict[str, Any]) -> dict[str, str]:
    missing = set(FINGERPRINT_KEYS) - set(values)
    if missing:
        raise ValueError(f"fingerprint_inputs_missing:{sorted(missing)}")
    return {key: hashlib.sha256(json.dumps(values[key], sort_keys=True, default=str).encode()).hexdigest() for key in FINGERPRINT_KEYS}


def validate_resume_artifact(root: Path, expected: dict[str, str]) -> str:
    manifest_path = root / "manifest.json"
    marker_path = root / "completion_marker.json"
    acceptance_path = root / "acceptance.json"
    if not all(path.is_file() for path in (manifest_path, marker_path, acceptance_path)):
        return "NO_REUSABLE_ARTIFACT"
    marker = json.loads(marker_path.read_text())
    if marker.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        return "REJECT_MANIFEST_HASH_MISMATCH"
    if marker.get("acceptance_sha256") != hashlib.sha256(acceptance_path.read_bytes()).hexdigest():
        return "REJECT_ACCEPTANCE_HASH_MISMATCH"
    actual = json.loads(manifest_path.read_text()).get("fingerprint_inputs", {})
    if set(actual) != set(FINGERPRINT_KEYS) or set(expected) != set(FINGERPRINT_KEYS):
        return "REJECT_INCOMPLETE_FINGERPRINTS"
    for key in FINGERPRINT_KEYS:
        if actual[key] != expected[key]:
            return f"REJECT_{key.upper()}_FINGERPRINT_MISMATCH"
    return "REUSABLE"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


class ArtifactWriter:
    def __init__(self, root: Path):
        self.root = root

    def write_manifest(self, payload: dict[str, Any]) -> None:
        inputs = payload.get("fingerprint_inputs", {})
        if set(inputs) != set(FINGERPRINT_KEYS):
            raise ValueError("manifest_requires_eight_fingerprints")
        _atomic_text(self.root / "manifest.json", json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    def write_predictions(self, frame: pd.DataFrame) -> None:
        required = {"sample_id", "y_true", "y_pred"}
        if frame.empty or not required <= set(frame):
            raise ValueError("prediction_contract_invalid")
        _atomic_text(self.root / "predictions.csv", frame.to_csv(index=False))

    def write_fold_ids(self, frame: pd.DataFrame) -> None:
        if frame.empty or not {"sample_id", "split"} <= set(frame):
            raise ValueError("fold_id_contract_invalid")
        _atomic_text(self.root / "fold_assignments.csv", frame.to_csv(index=False))

    def write_metrics(self, payload: dict[str, Any]) -> None:
        _atomic_text(self.root / "metrics.json", json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    def accept(self) -> None:
        required = ["manifest.json", "predictions.csv", "fold_assignments.csv", "metrics.json"]
        if not all((self.root / name).is_file() for name in required):
            raise RuntimeError("artifact_contract_incomplete")
        predictions = pd.read_csv(self.root / "predictions.csv")
        folds = pd.read_csv(self.root / "fold_assignments.csv")
        test_ids = set(folds.loc[folds.split.eq("test"), "sample_id"].astype(str))
        prediction_ids = set(predictions.sample_id.astype(str))
        if prediction_ids != test_ids:
            raise RuntimeError("prediction_fold_id_mismatch")
        _atomic_text(self.root / "acceptance.json", json.dumps({"status": "ARTIFACT_ACCEPTED_NOT_SCIENTIFIC_VERDICT"}, indent=2) + "\n")

    def write_completion_marker(self) -> None:
        acceptance = self.root / "acceptance.json"
        if not acceptance.is_file():
            raise RuntimeError("acceptance_required_before_completion")
        manifest = self.root / "manifest.json"
        manifest_payload = json.loads(manifest.read_text())
        marker = {
            "status": "ARTIFACT_ACCEPTED_NOT_SCIENTIFIC_VERDICT",
            "scientific_status": manifest_payload.get("scientific_status", "INSUFFICIENT_EVIDENCE"),
            "evidence_status": manifest_payload.get("evidence_status"),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
        }
        _atomic_text(self.root / "completion_marker.json", json.dumps(marker, indent=2, sort_keys=True) + "\n")
