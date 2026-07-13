from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import sys
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .artifacts import atomic_json, fingerprint, sha256_file


def accept_jobs(root: Path, *, config: dict[str, Any] | None = None, config_path: Path | None = None) -> dict[str, Any]:
    accepted, rejected = [], []
    for manifest_path in sorted((root / "jobs").glob("*/manifest.json")):
        job_root = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            completion = json.loads((job_root / "completion.json").read_text())
            if any(sha256_file(job_root / name) != digest for name, digest in completion.get("artifact_hashes", {}).items()): raise ValueError("ARTIFACT_HASH_MISMATCH")
            required = [job_root / name for name in (manifest["checkpoint"], manifest["preprocessor"], "predictions.csv", "metrics.json", "fold_test_ids.csv", "fold_assignments.csv", "split_contract.json", "manifest.json")]
            if any(not path.is_file() for path in required): raise ValueError("REQUIRED_ARTIFACT_MISSING")
            if (job_root / "completion.json").stat().st_mtime < max(path.stat().st_mtime for path in required): raise ValueError("COMPLETION_MARKER_ORDER_INVALID")
            pred = pd.read_csv(job_root / "predictions.csv")
            folds = pd.read_csv(job_root / "fold_test_ids.csv")
            metrics = json.loads((job_root / "metrics.json").read_text())
            if pred.empty or not np.isfinite(pred[["y_true", "y_pred"]].to_numpy()).all(): raise ValueError("EMPTY_OR_NONFINITE")
            if pred.sample_id.astype(str).tolist() != folds.sample_id.astype(str).tolist(): raise ValueError("FOLD_ID_ORDER_MISMATCH")
            recomputed = {"mae": mean_absolute_error(pred.y_true, pred.y_pred), "rmse": mean_squared_error(pred.y_true, pred.y_pred) ** 0.5, "r2": r2_score(pred.y_true, pred.y_pred)}
            if any(not np.isclose(float(metrics[k]), float(recomputed[k]), rtol=1e-8, atol=1e-9, equal_nan=True) for k in recomputed): raise ValueError("METRIC_RECOMPUTE_MISMATCH")
            if set(manifest.get("fingerprints", {})) != {"config", "data", "view", "split", "feature", "model", "runtime", "code"}: raise ValueError("FINGERPRINT_CONTRACT_INCOMPLETE")
            if config is not None and config_path is not None:
                try:
                    import torch
                    torch_version = torch.__version__
                except ImportError:
                    torch_version = "NOT_INSTALLED"
                code_files = [Path("src/distillation_v4/australian/execution.py"), Path("src/distillation_v4/australian/campaign.py"), Path("scripts/run_stage8_v4_australian.py")]
                expected = {
                    "config": fingerprint({"entry_file_sha256": sha256_file(config_path), "resolved_config": config}),
                    "runtime": fingerprint({"interpreter": str(Path(config["runtime"]["interpreter"]).resolve()), "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch_version}),
                    "code": fingerprint({str(path): sha256_file(path) for path in code_files}),
                    "split": fingerprint(json.loads((job_root / "split_contract.json").read_text())),
                    "model": fingerprint({"component": manifest["component"], "seed": manifest["seed"]}),
                }
                preprocessor = joblib.load(job_root / manifest["preprocessor"])
                expected["feature"] = fingerprint({"deployable": preprocessor["deployable_features"], "weather": [c for c in preprocessor["deployable_features"] if "weather" in c or "silo" in c or "rain" in c or "temp" in c], "soil": preprocessor["soil_features"]})
                inventory = {row["dataset_id"]: row for row in json.loads((root / "control/dataset_inventory.json").read_text())}
                expected["data"] = expected["view"] = inventory[manifest["dataset_id"]]["view_sha256"]
                if any(manifest["fingerprints"].get(key) != value for key, value in expected.items()): raise ValueError("INDEPENDENT_FINGERPRINT_RECOMPUTE_MISMATCH")
            accepted.append({"job_id": job_root.name, "dataset_id": manifest["dataset_id"], "component": manifest["component"], "fold": manifest["fold"], "seed": manifest["seed"], "sample_unit": manifest["sample_unit"], "target_unit": manifest["target_unit"], "protocol_tier": manifest["protocol_tier"], "metrics": metrics, "predictions": str(job_root / "predictions.csv")})
        except Exception as exc:
            rejected.append({"job_id": job_root.name, "reason": f"{type(exc).__name__}:{exc}"})
    payload = {"schema_version": "stage8_v4_australian_acceptance_v1", "accepted": accepted, "rejected": rejected, "accepted_count": len(accepted), "rejected_count": len(rejected), "strata": {tier: sum(row["protocol_tier"] == tier for row in accepted) for tier in ("EXACT", "ADAPTED", "TRANSFER", "DIAGNOSTIC")}}
    atomic_json(root / "acceptance/accepted_artifact_index.json", payload)
    return payload
