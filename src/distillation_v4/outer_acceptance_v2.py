from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .campaign import load_primary_frame
from .contracts import CampaignConfig
from .repair_outer_artifacts import sha256_file, validate_repair_outer_record
from .repair_outer_matrix import validate_frozen_repair_outer_job_matrix


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def expected_execution_fingerprint(*, job: dict[str, Any], matrix_fingerprint: str, training_contract: dict[str, Any]) -> str:
    return _fingerprint({
        "job": job,
        "matrix_fingerprint": matrix_fingerprint,
        "training_contract": training_contract,
        "code_contract": "stage8_v4_repair_outer_v1",
    })


def load_approved_role_registry(*, protocol: Path, approval: Path) -> dict[tuple[str, str], str]:
    approved = json.loads(approval.read_text())
    if approved.get("approved") is not True or approved.get("status") != "FORMALLY_APPROVED":
        raise RuntimeError("OUTER_APPROVAL_NOT_FORMAL")
    if sha256_file(protocol) != approved.get("protocol_sha256"):
        raise RuntimeError("PROTOCOL_HASH_MISMATCH")
    payload = json.loads(protocol.read_text())
    roles: dict[tuple[str, str], str] = {}
    for item in payload.get("route_matrix", []):
        key = (str(item["dataset"]), str(item["route"]))
        if key in roles:
            raise RuntimeError("PROTOCOL_ROLE_DUPLICATE:" + ":".join(key))
        roles[key] = str(item["role"])
    return roles


def aggregate_fold_first(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["fold"])].append(float(pair["effect"]))
    fold_effects = {fold: float(np.mean(values)) for fold, values in sorted(grouped.items())}
    values = list(fold_effects.values())
    seed_values = [float(pair["effect"]) for pair in pairs]
    return {
        "fold_count": len(values),
        "seed_pair_count": len(pairs),
        "fold_effects": fold_effects,
        "mean_effect": float(np.mean(values)) if values else None,
        "fold_effect_range": [float(np.min(values)), float(np.max(values))] if values else None,
        "fold_consistency": f"{sum(value > 0 for value in values)}/{len(values)}",
        "fold_negative_transfer_rate": float(np.mean([value < 0 for value in values])) if values else None,
        "job_negative_transfer_rate": float(np.mean([value < 0 for value in seed_values])) if seed_values else None,
        "seed_sensitivity_sd": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
        "inference_boundary": "THREE_FOLDS_DESCRIPTIVE_NOT_SUFFICIENT_FOR_STRONG_SIGNIFICANCE_CLAIM" if len(values) <= 3 else "FOLD_LEVEL_DESCRIPTIVE",
    }


def _performance_status(role: str, summary: dict[str, Any]) -> str:
    effects = list(summary["fold_effects"].values())
    mean = float(summary["mean_effect"])
    all_positive = bool(effects) and all(value > 0 for value in effects)
    all_negative = bool(effects) and all(value < 0 for value in effects)
    if role == "PRIMARY_CONFIRMATORY":
        if all_positive:
            return "PRIMARY_DIRECTIONALLY_POSITIVE_LIMITED_FOLD_EVIDENCE"
        if all_negative:
            return "PRIMARY_DIRECTIONALLY_NEGATIVE_LIMITED_FOLD_EVIDENCE"
        return "PRIMARY_MIXED_LIMITED_FOLD_EVIDENCE"
    if role == "SECONDARY_CONFIRMATORY":
        return "SECONDARY_SUPPORTIVE_LIMITED_FOLD_EVIDENCE" if all_positive else "SECONDARY_MIXED_OR_NEGATIVE_LIMITED_FOLD_EVIDENCE"
    if mean > 0 and all_positive:
        return "EXPLORATORY_PERFORMANCE_POSITIVE"
    if mean > 0:
        return "EXPLORATORY_PERFORMANCE_CONDITIONAL"
    return "EXPLORATORY_PERFORMANCE_NEGATIVE"


def campaign_status(routes: list[dict[str, Any]]) -> str:
    primary = [route for route in routes if route.get("role") == "PRIMARY_CONFIRMATORY"]
    return primary[0]["performance_claim_status"] if len(primary) == 1 else "INSUFFICIENT_EVIDENCE_PRIMARY_ROUTE_NOT_UNIQUE"


def _validate_preprocessor(record: dict[str, Any], output: Path, frame: pd.DataFrame, fold: pd.DataFrame, manifest: dict[str, Any]) -> None:
    payload = joblib.load(output / str(record["preprocessor"]))
    pipeline = payload.get("weather_preprocessor") if isinstance(payload, dict) else payload
    features = manifest.get("weather_features", manifest.get("features", []))
    if list(getattr(pipeline, "feature_names_in_", [])) != list(features):
        raise RuntimeError("PREPROCESSOR_FEATURE_LINEAGE_MISMATCH")
    train_ids = set(fold.loc[fold["split"].eq("train"), "sample_id"].astype(str))
    train = frame.loc[frame["sample_id"].astype(str).isin(train_ids), features]
    if len(train) != len(train_ids):
        raise RuntimeError("PREPROCESSOR_TRAIN_ID_REBUILD_MISMATCH")
    expected = np.nanmedian(train.to_numpy(float), axis=0)
    observed = np.asarray(pipeline.named_steps["impute"].statistics_, dtype=float)
    if not np.allclose(expected, observed, equal_nan=True, rtol=1e-10, atol=1e-10):
        raise RuntimeError("PREPROCESSOR_NOT_FIT_ON_DECLARED_TRAIN")


def _validate_early_stopping(record: dict[str, Any], output: Path) -> None:
    ledger = json.loads((output / str(record["loss_ledger"])).read_text())
    epochs = ledger.get("epochs", [])
    if not epochs:
        raise RuntimeError("EARLY_STOPPING_LEDGER_EMPTY")
    best = min(epochs, key=lambda row: float(row["validation_scaled_mae"]))
    if int(float(best["epoch"])) != int(record["selected_epoch"]):
        raise RuntimeError("EARLY_STOPPING_SELECTED_EPOCH_MISMATCH")


def _worst_group_mae(predictions: pd.DataFrame, dataset: str) -> float:
    ids = predictions["sample_id"].astype(str)
    groups = ids.str.split("|").str[-1].str.rsplit("_", n=1).str[0] if dataset == "PRIMARY_G2F_MAIZE" else ids.str.split("|").str[0].str.rsplit("-", n=1).str[0]
    errors = (predictions["y_true"] - predictions["y_pred"]).abs()
    return float(pd.DataFrame({"group": groups, "error": errors}).groupby("group").error.mean().max())


def run_outer_acceptance_v2(*, config: CampaignConfig, output: Path, protocol: Path, destination: Path) -> dict[str, Any]:
    matrix = validate_frozen_repair_outer_job_matrix(output=output, path=output / "control/frozen_repair_outer_job_matrix.json")
    roles = load_approved_role_registry(protocol=protocol, approval=output / "control/formal_outer_approval.json")
    matrix_keys = {(str(job["dataset"]), str(job["route"])) for job in matrix["jobs"]}
    if matrix_keys != set(roles):
        raise RuntimeError("PROTOCOL_MATRIX_ROLE_COVERAGE_MISMATCH")
    frames = {dataset: load_primary_frame(config, output, dataset) for dataset in config.primary_datasets}
    rows: list[dict[str, Any]] = []
    records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for job in matrix["jobs"]:
        root = output / "outer_test_repair" / str(job["dataset"]) / str(job["job_id"])
        expected = expected_execution_fingerprint(job=job, matrix_fingerprint=str(matrix["repair_outer_job_matrix_fingerprint"]), training_contract=config.raw.get("training", {}))
        record = validate_repair_outer_record(output=output, record_path=root / "job_record.json", expected_job=job, expected_execution_fingerprint=expected)
        for extra in ("calibration_predictions", "loss_ledger", "test_representation", "scenario_metrics", "scenario_predictions"):
            if record.get(extra):
                artifact = output / str(record[extra]); expected_hash = record.get(extra + "_sha256")
                if not artifact.is_file() or not expected_hash or sha256_file(artifact) != expected_hash:
                    raise RuntimeError("OUTER_V2_AUXILIARY_ARTIFACT_INVALID:" + extra + ":" + str(job["job_id"]))
        predictions = pd.read_csv(output / str(record["predictions"]))
        folds = pd.read_csv(output / str(record["fold_assignments"]))
        manifest = json.loads((output / str(record["manifest"])).read_text())
        frame = frames[str(job["dataset"])]
        rebuilt_ids = set(frame.loc[frame["year"].eq(int(job["outer_test_year"])), "sample_id"].astype(str))
        if rebuilt_ids != set(predictions["sample_id"].astype(str)):
            raise RuntimeError("INDEPENDENT_SOURCE_TEST_ID_MISMATCH:" + str(job["job_id"]))
        _validate_preprocessor(record, output, frame, folds, manifest)
        _validate_early_stopping(record, output)
        required = [output / str(record[name]) for name in ("predictions", "metrics", "fold_assignments", "checkpoint", "preprocessor", "manifest")]
        marker = root / "completion_marker.json"
        marker_warning = marker.stat().st_mtime < max(path.stat().st_mtime for path in required)
        warning = "MARKER_MTIME_PRECEDES_REQUIRED_ARTIFACT" if marker_warning else ("LEGACY_OUTER_TEST_USED_FIELD_ABSENT_CONFIRMED_BY_MARKER_AND_MANIFEST" if "outer_test_used" not in record else "")
        row = {"job_id": job["job_id"], "dataset": job["dataset"], "route": job["route"], "role": roles[(str(job["dataset"]), str(job["route"]))], "fold": job["fold"], "seed": job["seed"], "engineering_status": "ENGINEERING_ACCEPTED_WITH_WARNING" if warning else "ENGINEERING_ACCEPTED", "warning": warning, "mae": float(record["mae"]), "rmse": float(record["rmse"]), "r2": record["r2"], "worst_group_mae": _worst_group_mae(predictions, str(job["dataset"])), "execution_fingerprint": expected, "predictions": record["predictions"], "metrics": record["metrics"]}
        rows.append(row)
        records[(str(job["dataset"]), str(job["fold"]), int(job["seed"]), str(job["route"]))] = row
    route_results: list[dict[str, Any]] = []
    for dataset, route in sorted({(row["dataset"], row["route"]) for row in rows if row["route"] != "supervised"}):
        pairs = []
        for key, row in records.items():
            if key[0] == dataset and key[3] == route:
                baseline = records[(key[0], key[1], key[2], "supervised")]
                pairs.append({"fold": key[1], "seed": key[2], "effect": float(baseline["mae"]) - float(row["mae"]), "worst_group_effect": float(baseline["worst_group_mae"]) - float(row["worst_group_mae"])})
        summary = aggregate_fold_first(pairs)
        role = roles[(dataset, route)]
        route_results.append({"dataset": dataset, "route": route, "role": role, **summary, "worst_group_effect_mean": float(np.mean([p["worst_group_effect"] for p in pairs])), "worst_group_effect_min": float(np.min([p["worst_group_effect"] for p in pairs])), "performance_claim_status": _performance_status(role, summary), "mechanism_claim_status": "INSUFFICIENT_EVIDENCE_MECHANISM_NOT_IDENTIFIED_BY_OUTER_PERFORMANCE" if route == "missing_aware" else "MECHANISM_EVIDENCE_REQUIRES_SEPARATE_DEVELOPMENT_CONTROLS", "claim_boundary": "FROZEN_OUTER_ENDPOINTS_ONLY_NO_ROUTE_SELECTION"})
    payload = {"schema_version": "stage8_v4_outer_acceptance_v2", "engineering_status": "ENGINEERING_ACCEPTED_WITH_PROVENANCE_WARNINGS" if any(row["warning"] for row in rows) else "ENGINEERING_ACCEPTED", "job_count": len(matrix["jobs"]), "accepted_job_count": len(rows), "role_registry_hash": sha256_file(protocol), "campaign_scientific_status": campaign_status(route_results), "fold_is_primary_evidence_unit": True, "seed_is_independent_replication": False, "legacy_status_superseded_for_interpretation": "reports/repair_final_evidence_registry.json", "routes": route_results}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "job_acceptance_registry.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_csv(destination / "job_acceptance_registry.csv", index=False)
    (destination / "route_summary.json").write_text(json.dumps(route_results, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{**{k: v for k, v in row.items() if k != "fold_effects"}, "fold_effects": json.dumps(row["fold_effects"], sort_keys=True)} for row in route_results]).to_csv(destination / "route_summary.csv", index=False)
    (destination / "campaign_scientific_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Stage 8 V4 Outer Acceptance V2", "", f"- Engineering: `{payload['engineering_status']}`", f"- Campaign science: `{payload['campaign_scientific_status']}`", "- Fold is the primary evidence unit; seeds quantify stability only.", "- Three folds do not support a strong statistical-significance claim.", ""]
    for row in route_results:
        lines.extend([f"## {row['dataset']} / {row['route']}", "", f"- Role: `{row['role']}`", f"- Performance: `{row['performance_claim_status']}`", f"- Mechanism: `{row['mechanism_claim_status']}`", f"- Mean fold effect: `{row['mean_effect']}`", f"- Fold consistency: `{row['fold_consistency']}`", ""])
    (destination / "outer_acceptance_report.md").write_text("\n".join(lines))
    return payload
