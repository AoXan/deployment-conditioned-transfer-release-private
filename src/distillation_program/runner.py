from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .official_methods import regression_distillation_loss


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_plan(config: dict[str, Any], *, matrix: str) -> dict[str, Any]:
    selected = [job for job in config["experiments"] if job["matrix"] == matrix]
    return {
        "program": config["program"],
        "matrix": matrix,
        "job_ids": [job["id"] for job in selected],
        "scientific_statuses": [job["status"] for job in selected],
        "plan_fingerprint": _hash(selected),
        "executed_jobs": 0,
    }


def resume_artifact_status(run_directory: Path, expected_config_fingerprint: str) -> str:
    marker_path = run_directory / "completion_marker.json"
    manifest_path = run_directory / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        return "NO_REUSABLE_ARTIFACT"
    marker = json.loads(marker_path.read_text())
    manifest_bytes = manifest_path.read_bytes()
    if marker.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        return "REJECT_MANIFEST_HASH_MISMATCH"
    manifest = json.loads(manifest_bytes)
    if manifest.get("fingerprint_inputs", {}).get("config") != expected_config_fingerprint:
        return "REJECT_CONFIG_FINGERPRINT_MISMATCH"
    return "REUSABLE"


class _Regressor(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(width, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, value):
        return self.layers(value).squeeze(1)


def run_safe_smoke(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    torch.manual_seed(101)
    rng = np.random.default_rng(101)
    n = 96
    weather = rng.normal(size=(n, 3)).astype("float32")
    soil = rng.normal(size=(n, 2)).astype("float32")
    target = (weather[:, 0] - 0.5 * weather[:, 1] + 0.7 * soil[:, 0]).astype("float32")
    train, validation, test = np.arange(64), np.arange(64, 80), np.arange(80, 96)
    rich = torch.from_numpy(np.column_stack([weather, soil]))
    deployable = torch.from_numpy(weather)
    y = torch.from_numpy(target)

    teacher = _Regressor(5)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.01)
    for _ in range(30):
        teacher_optimizer.zero_grad()
        loss = torch.nn.functional.huber_loss(teacher(rich[train]), y[train])
        loss.backward()
        teacher_optimizer.step()
    teacher.eval()

    student = _Regressor(3)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    for _ in range(30):
        optimizer.zero_grad()
        prediction = student(deployable[train])
        with torch.no_grad():
            teacher_prediction = teacher(rich[train])
        loss, _ = regression_distillation_loss(
            student_prediction=prediction,
            teacher_prediction=teacher_prediction,
            target=y[train],
            supervised_weight=0.5,
            distillation_weight=0.5,
        )
        loss.backward()
        optimizer.step()
    student.eval()
    with torch.no_grad():
        prediction = student(deployable[test]).numpy()
    if not np.isfinite(prediction).all() or np.std(prediction) == 0:
        raise RuntimeError("invalid smoke prediction")

    destination = output_root / "smoke"
    rows = pd.DataFrame({
        "sample_id": [f"synthetic_{index:03d}" for index in test],
        "y_true": target[test],
        "y_pred": prediction,
        "fold": "synthetic_test",
        "evidence_status": "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE",
    })
    folds = pd.DataFrame({
        "sample_id": [f"synthetic_{index:03d}" for index in range(n)],
        "split": np.where(np.arange(n) < 64, "train", np.where(np.arange(n) < 80, "validation", "test")),
    })
    metrics = {"mae": float(np.mean(np.abs(rows.y_true - rows.y_pred))), "n": len(rows)}
    _atomic_csv(destination / "predictions.csv", rows)
    _atomic_csv(destination / "fold_assignments.csv", folds)
    _atomic_json(destination / "metrics.json", metrics)
    manifest = {
        "program": config["program"],
        "evidence_status": "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "scientific_status": "NOT_APPLICABLE",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "fingerprint": _hash({"config": config, "seed": 101, "runtime": torch.__version__}),
        "validation_ids": [f"synthetic_{index:03d}" for index in validation],
    }
    _atomic_json(destination / "manifest.json", manifest)
    _atomic_json(destination / "completion_marker.json", {
        "status": "SMOKE_VERIFIED",
        "scientific_status": "NOT_APPLICABLE",
        "manifest_fingerprint": manifest["fingerprint"],
    })
    return {"status": "SMOKE_VERIFIED", "metrics": metrics, "output": str(destination)}
