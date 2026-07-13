from __future__ import annotations

from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ResumeVerdict(Enum):
    REUSABLE = "REUSABLE"
    NOT_COMPLETE = "NOT_COMPLETE"
    HASH_MISMATCH = "HASH_MISMATCH"
    FINGERPRINT_MISMATCH = "FINGERPRINT_MISMATCH"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"


class ArtifactStore:
    fingerprint_keys = (
        "code_tree", "config", "data", "view", "split", "feature", "model", "runtime"
    )

    def __init__(self, root: Path):
        self.root = root

    def _text(self, name: str, value: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(value)
        temp.replace(path)

    def write_predictions(self, frame: pd.DataFrame) -> None:
        required = {"sample_id", "y_true", "y_pred"}
        if frame.empty or not required <= set(frame):
            raise ValueError("PREDICTION_CONTRACT_INVALID")
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all():
            raise ValueError("PREDICTION_NONFINITE")
        self._text("predictions.csv", frame.to_csv(index=False))

    def write_fold_ids(self, frame: pd.DataFrame) -> None:
        if frame.empty or not {"sample_id", "split"} <= set(frame):
            raise ValueError("FOLD_CONTRACT_INVALID")
        self._text("fold_assignments.csv", frame.to_csv(index=False))

    def write_metrics(self, payload: dict[str, Any]) -> None:
        self._text("metrics.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def write_manifest(self, payload: dict[str, Any]) -> None:
        fingerprints = payload.get("fingerprints", {})
        if set(fingerprints) != set(self.fingerprint_keys):
            raise ValueError("MANIFEST_FINGERPRINTS_INCOMPLETE")
        self._text("manifest.json", json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    def accept(self) -> None:
        required = ["predictions.csv", "fold_assignments.csv", "metrics.json", "manifest.json"]
        if not all((self.root / name).is_file() for name in required):
            raise RuntimeError("ARTIFACT_CONTRACT_INCOMPLETE")
        predictions = pd.read_csv(self.root / "predictions.csv")
        folds = pd.read_csv(self.root / "fold_assignments.csv")
        prediction_ids = list(predictions.sample_id.astype(str))
        test_ids = list(folds.loc[folds.split.eq("test"), "sample_id"].astype(str))
        if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != set(test_ids):
            raise RuntimeError("PREDICTION_FOLD_ID_MISMATCH")
        self._text("acceptance.json", json.dumps({"engineering_status": "ENGINEERING_ACCEPTED"}, indent=2) + "\n")

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def complete(self) -> None:
        if not (self.root / "acceptance.json").is_file():
            raise RuntimeError("ACCEPTANCE_REQUIRED")
        payload = {
            "engineering_status": "ENGINEERING_ACCEPTED",
            "scientific_status": "NOT_ENCODED_IN_COMPLETION_MARKER",
            "manifest_sha256": self._sha(self.root / "manifest.json"),
            "acceptance_sha256": self._sha(self.root / "acceptance.json"),
        }
        self._text("completion_marker.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def resume_verdict(self, expected: dict[str, str]) -> ResumeVerdict:
        required = [self.root / name for name in ("manifest.json", "acceptance.json", "completion_marker.json")]
        if not all(path.is_file() for path in required):
            return ResumeVerdict.NOT_COMPLETE
        marker = json.loads((self.root / "completion_marker.json").read_text())
        if marker.get("manifest_sha256") != self._sha(self.root / "manifest.json"):
            return ResumeVerdict.HASH_MISMATCH
        actual = json.loads((self.root / "manifest.json").read_text()).get("fingerprints", {})
        if actual.get("runtime") != expected.get("runtime"):
            return ResumeVerdict.RUNTIME_MISMATCH
        return ResumeVerdict.REUSABLE if actual == expected else ResumeVerdict.FINGERPRINT_MISMATCH
