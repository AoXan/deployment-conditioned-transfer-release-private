#!/usr/bin/env python3
"""Stage 7 unified nursery experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage7_all_experiments.yaml"
STAGE_ORDER = ["7a", "7b", "7c"]
PREDICTION_SOURCES = [
    ROOT / "outputs/stage4_expert_set_pressure/stage4_predictions.csv",
    ROOT / "outputs/cross_dataset_validation/cybench/cybench_maize_us_predictions.csv",
    ROOT / "outputs/stage4_expert_set_pressure/stage4_g2f_predictions.csv",
]
PREDICTION_COLUMN_ALIASES = {
    "xgboost_if_existing": "pred_official_xgboost",
    "best_single": "pred_oracle_best_expert",
    "stacking": "pred_stacking_ridge",
    "oracle_best_expert_diagnostic": "pred_oracle_best_expert",
    "learned_gate_diagnostic": "pred_calibrated_logistic_gate_router",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"Config did not parse to a mapping: {path}")
    return config


def split_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def read_matrix(config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve_path(config["inputs"]["frozen_experiment_matrix"]))


def read_registry(config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(resolve_path(config["inputs"]["registry"]))


def validate_static_paths(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, value in config["inputs"].items():
        path = resolve_path(value)
        checks.append(
            {
                "check": f"input_path:{key}",
                "status": "pass" if path.exists() else "fail",
                "severity": "blocker" if key in {"registry", "frozen_experiment_matrix", "folds", "nursery_release"} and not path.exists() else "warning",
                "details": relpath(path),
            }
        )
    for key, value in config["outputs"].items():
        if key.endswith("_dir") or key in {"root", "results_dir", "temp_dir", "resume_dir", "log_dir", "status_dir"}:
            path = resolve_path(value)
            path.mkdir(parents=True, exist_ok=True)
            internal = str(path.resolve()).startswith(str(ROOT.resolve()))
            checks.append({"check": f"output_internal:{key}", "status": "pass" if internal else "fail", "severity": "blocker" if not internal else "info", "details": relpath(path)})
    free_gib = shutil.disk_usage(resolve_path(config["outputs"]["root"])).free / (1024**3)
    hard_stop = float(config["safety"]["disk_free_hard_stop_gib"])
    checks.append({"check": "disk_free_above_hard_stop", "status": "pass" if free_gib > hard_stop else "fail", "severity": "blocker" if free_gib <= hard_stop else "info", "details": f"free_gib={free_gib:.1f}; hard_stop_gib={hard_stop:.1f}"})
    matrix_checks = validate_frozen_matrix_inputs(config)
    checks.extend(
        {
            "check": f"frozen_matrix_input:{row['experiment_id']}",
            "status": row["status"],
            "severity": "blocker" if row["status"] != "pass" and row["execution_stage"] not in {"future_boundary", "audit_only"} else "warning",
            "details": row["details"],
        }
        for row in matrix_checks
    )
    return checks


def path_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256_or_missing(path: Path) -> str:
    if path.is_file():
        return path_sha256(path)
    if path.is_dir():
        return f"directory:{relpath(path)}"
    return f"missing:{relpath(path)}"


def model_parameters(model: str, seed: int) -> dict[str, Any]:
    try:
        estimator = make_estimator(model, seed)
    except ValueError:
        pred_col = PREDICTION_COLUMN_ALIASES.get(model, f"pred_{model}")
        return {"adapter": "existing_prediction", "model": model, "prediction_column": pred_col}
    if estimator is None:
        return {"adapter": "mean_or_existing_prediction", "model": model}
    return {"adapter": type(estimator).__name__, "model": model, "params": estimator.get_params(deep=False)}


def input_hashes_for_job(config: dict[str, Any], job: dict[str, Any]) -> list[dict[str, str]]:
    registry = read_registry(config)
    row = registry[registry["view_id"].astype(str).eq(str(job["view_id"]))]
    paths: list[str] = []
    if not row.empty and str(row.iloc[0].get("path", "")).strip():
        paths.extend(part.strip() for part in str(row.iloc[0]["path"]).split(";") if part.strip())
    elif str(job.get("input_path", "")).strip():
        paths.extend(part.strip() for part in str(job["input_path"]).split(";") if part.strip())
    if str(job.get("view_id", "")).startswith("stage4_"):
        paths.extend(str(relpath(path)) for path in PREDICTION_SOURCES if path.exists())
    return [{"path": relpath(resolve_path(path)), "sha256": file_sha256_or_missing(resolve_path(path))} for path in paths]


def job_fingerprint(config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    fingerprint_config = resolve_path(config.get("execution", {}).get("fingerprint_config_path", config.get("_config_path", DEFAULT_CONFIG)))
    payload = {
        "fingerprint_schema": "stage7_job_fingerprint_v1",
        "job_id": job.get("job_id", ""),
        "stage": job.get("stage", ""),
        "experiment_id": job.get("experiment_id", ""),
        "runner_code_sha256": file_sha256_or_missing(Path(__file__).resolve()),
        "yaml_sha256": file_sha256_or_missing(fingerprint_config),
        "frozen_matrix_row_sha256": job.get("frozen_matrix_row_sha256", ""),
        "registry_sha256": file_sha256_or_missing(resolve_path(config["inputs"]["registry"])),
        "dataset_view_hashes": input_hashes_for_job(config, job),
        "target": job.get("target", ""),
        "feature_groups": job.get("feature_groups", ""),
        "validation_axis": job.get("validation_axis", ""),
        "fold": job.get("fold", ""),
        "fold_manifest_sha256": file_sha256_or_missing(resolve_path(config["inputs"]["folds"])),
        "model_parameters": model_parameters(str(job["model"]), int(job["seed"])),
        "seed": int(job["seed"]),
    }
    return {"hash": stable_json_hash(payload), "payload": payload}


def target_candidates(target: str) -> list[str]:
    aliases = {
        "yield": ["yield", "target_yield"],
        "Yield_Mg_ha": ["Yield_Mg_ha", "target_yield"],
        "environment_mean_Yield_Mg_ha_balanced_hybrid_subset": [
            "environment_mean_Yield_Mg_ha_balanced_hybrid_subset",
            "target_yield",
        ],
        "track_defined_yield": ["track_defined_yield", "target_value", "target_yield"],
        "track_specific_yield": [
            "track_specific_yield",
            "yield_t_ha",
            "observed_yield_t_ha",
            "target_yield",
            "target_value",
        ],
    }
    return aliases.get(target, [target, "target_yield", "yield_t_ha", "observed_yield_t_ha", "Yield_Mg_ha"])


def validate_frame_target(path: Path, target: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing_path:{relpath(path)}"
    if path.is_dir():
        return False, f"directory_path_not_allowed:{relpath(path)}"
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return False, f"parse_error:{type(exc).__name__}:{exc}"
    candidates = target_candidates(target)
    if not any(candidate in columns for candidate in candidates):
        return False, f"target_missing:{target}; candidates={candidates}; path={relpath(path)}"
    feature_count = len([c for c in columns if c not in set(candidates)])
    if feature_count <= 0:
        return False, f"no_features:{relpath(path)}"
    return True, f"path={relpath(path)} columns={len(columns)}"


def validate_frozen_matrix_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = read_matrix(config)
    registry = read_registry(config)
    registry_by_view = {str(row["view_id"]): row for _, row in registry.iterrows()}
    rows: list[dict[str, Any]] = []
    for _, exp in matrix.iterrows():
        view_id = str(exp["view_id"])
        execution_stage = str(exp.get("execution_stage", ""))
        target = str(exp.get("target", ""))
        status = "pass"
        details = ""
        if execution_stage in {"future_boundary", "audit_only"}:
            details = f"non_training_boundary:{execution_stage}"
        elif view_id == "waite_and_regional_modality_views":
            manifest_path = resolve_path(str(exp["input_path"]))
            if not manifest_path.exists():
                status, details = "fail", f"missing_modality_manifest:{relpath(manifest_path)}"
            else:
                manifest = pd.read_csv(manifest_path)
                failures = []
                for _, mrow in manifest.iterrows():
                    if bool(mrow.get("registered", True)):
                        ok, msg = validate_frame_target(resolve_path(str(mrow["path"])), target)
                        if not ok:
                            failures.append(f"{mrow['view_id']}:{msg}")
                status = "pass" if not failures else "fail"
                details = "modality_views_valid" if not failures else "; ".join(failures[:5])
        elif view_id in registry_by_view:
            reg = registry_by_view[view_id]
            source_boundary = str(reg.get("source_boundary", ""))
            path_text = str(reg.get("path", "") or exp.get("input_path", ""))
            if ";" in path_text or "existing_result_or_external_track" in source_boundary or "design_input" in source_boundary:
                source_failures = [relpath(resolve_path(part.strip())) for part in path_text.split(";") if part.strip() and not resolve_path(part.strip()).exists()]
                status = "pass" if not source_failures else "fail"
                details = "design_or_existing_result_input_valid" if not source_failures else f"missing_sources:{source_failures}"
            elif not path_text:
                status, details = "fail", "empty_registry_path"
            else:
                ok, msg = validate_frame_target(resolve_path(path_text), target)
                status, details = ("pass" if ok else "fail"), msg
        else:
            status, details = "fail", f"view_not_registered:{view_id}"
        rows.append(
            {
                "experiment_id": str(exp["experiment_id"]),
                "view_id": view_id,
                "execution_stage": execution_stage,
                "status": status,
                "details": details,
            }
        )
    return rows


def choose_stage_rows(matrix: pd.DataFrame, stage_config: dict[str, Any]) -> pd.DataFrame:
    rows = matrix.copy()
    ids = stage_config.get("experiment_ids") or []
    if ids:
        rows = rows[rows["experiment_id"].isin(ids)].copy()
    else:
        include_status = set(stage_config.get("include_status", []))
        if include_status:
            rows = rows[rows["status"].isin(include_status)].copy()
    exclude = set(stage_config.get("exclude_experiment_ids", []))
    if exclude:
        rows = rows[~rows["experiment_id"].isin(exclude)].copy()
    include_exec = set(stage_config.get("include_execution_stages", []))
    if include_exec:
        rows = pd.concat([rows, matrix[matrix["execution_stage"].isin(include_exec)]], ignore_index=True).drop_duplicates("experiment_id")
    return rows


def axes_for(exp: pd.Series, stage_config: dict[str, Any]) -> list[str]:
    overrides = stage_config.get("axis_overrides", {})
    if exp["experiment_id"] in overrides:
        return split_list(overrides[exp["experiment_id"]])
    axes = split_list(exp.get("validation_axis"))
    return axes if axes and axes != ["not_run"] else ["not_applicable"]


def models_for(exp: pd.Series, stage_config: dict[str, Any]) -> list[str]:
    overrides = stage_config.get("model_overrides", {})
    if exp["experiment_id"] in overrides:
        return split_list(overrides[exp["experiment_id"]])
    models = split_list(exp.get("baseline_models"))
    return models if models and models != ["none"] else ["none"]


def seeds_for(exp: pd.Series, stage_config: dict[str, Any], axis: str, config: dict[str, Any]) -> list[int]:
    view_id = str(exp.get("view_id", ""))
    if view_id.startswith("stage4_cybench") or view_id == "stage4_cybench_ensemble_and_oracle":
        axis_seeds = config.get("execution", {}).get("existing_prediction_axis_seeds", {}).get(
            "CY-Bench",
            {
            "random": [101, 202, 303],
            "unseen_location": [700, 701, 702],
            "temporal_forward": [2921, 2922, 2923],
            "spatiotemporal": [1200, 1201, 1202],
            },
        )
        if axis in axis_seeds:
            return [int(s) for s in axis_seeds[axis]]
    return [int(s) for s in stage_config.get("seeds", config_default_seeds(stage_config))]


def config_default_seeds(stage_config: dict[str, Any]) -> list[int]:
    return [101] if "seeds" not in stage_config else [int(s) for s in stage_config["seeds"]]


def stages_to_run(stage: str, config: dict[str, Any]) -> list[str]:
    if stage == "all":
        return list(config.get("stage_ordering", {}).get("all", STAGE_ORDER))
    return [stage]


def build_job_manifest(config: dict[str, Any], stage: str, experiment_id: str | None = None, max_workers: int | None = None, timeout_per_job: int | None = None) -> pd.DataFrame:
    matrix = read_matrix(config)
    registry = read_registry(config)
    registry_ids = set(registry["view_id"].dropna().astype(str))
    jobs: list[dict[str, Any]] = []
    for stage_name in stages_to_run(stage, config):
        stage_config = config["stages"][stage_name]
        rows = choose_stage_rows(matrix, stage_config)
        if experiment_id:
            rows = rows[rows["experiment_id"].eq(experiment_id)]
        timeout = int(timeout_per_job or stage_config.get("timeout_per_job_seconds") or config["execution"]["timeout_per_job_seconds"])
        workers = int(max_workers or config["execution"]["max_workers"])
        for _, exp in rows.iterrows():
            frozen_row_sha256 = stable_json_hash(exp.to_dict())
            for axis in axes_for(exp, stage_config):
                for model in models_for(exp, stage_config):
                    for fold in split_list(stage_config.get("fold_tokens", ["fold_001"])) or ["fold_001"]:
                        for seed in seeds_for(exp, stage_config, axis, config):
                            base = f"{stage_name}-{exp['experiment_id']}-{model}-{axis}-{fold}-s{seed}"
                            digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
                            jobs.append(
                                {
                                    "job_id": f"{stage_name.upper()}-{len(jobs)+1:04d}-{digest}",
                                    "stage": stage_name,
                                    "stage_enabled": bool(stage_config.get("enabled", False)),
                                    "formal_authorized": bool(stage_config.get("formal_authorized", False)),
                                    "experiment_id": exp["experiment_id"],
                                    "experiment_status": exp["status"],
                                    "execution_stage": exp.get("execution_stage", ""),
                                    "track": exp["track"],
                                    "track_role": exp["track_role"],
                                    "view_id": exp["view_id"],
                                    "registered_view": str(exp["view_id"]) in registry_ids,
                                    "input_path": exp.get("input_path", ""),
                                    "sample_unit": exp["sample_unit"],
                                    "target": exp["target"],
                                    "feature_groups": exp.get("allowed_feature_groups", exp.get("feature_groups", "")),
                                    "frozen_matrix_row_sha256": frozen_row_sha256,
                                    "model": model,
                                    "validation_axis": axis,
                                    "fold": fold,
                                    "seed": int(seed),
                                    "claim_served": exp["claim_served"],
                                    "rq_served": exp["rq_served"],
                                    "story_alignment": exp["story_alignment"],
                                    "stop_rule": exp.get("stop_rule", ""),
                                    "classification": stage_config.get("classification", ""),
                                    "timeout_seconds": timeout,
                                    "max_workers": workers,
                                    "status": "planned" if stage_config.get("enabled", False) else "disabled",
                                    "output_location": exp["output_location"],
                                }
                            )
    manifest = pd.DataFrame(jobs)
    job_list_csv = config.get("execution", {}).get("rerun_job_list_csv")
    if job_list_csv and not manifest.empty:
        job_list_path = resolve_path(str(job_list_csv))
        rerun_jobs = pd.read_csv(job_list_path)
        allowed = set(rerun_jobs["job_id"].dropna().astype(str))
        manifest = manifest[manifest["job_id"].astype(str).isin(allowed)].copy()
    return manifest


def write_status(config: dict[str, Any], stage: str, manifest: pd.DataFrame, mode: str) -> None:
    payload = {
        "generated_at": now_iso(),
        "stage": stage,
        "stage_order": stages_to_run(stage, config),
        "mode": mode,
        "job_count": int(len(manifest)),
        "enabled_jobs": int(manifest["stage_enabled"].sum()) if len(manifest) else 0,
        "output_root": relpath(resolve_path(config["outputs"]["root"])),
    }
    atomic_write_json(resolve_path(config["outputs"]["status_json"]), payload)
    atomic_write_json(resolve_path(config["outputs"]["heartbeat_file"]), {"ts": now_iso(), "stage": stage, "mode": mode, "status": "alive"})


def formal_guard(config: dict[str, Any], stage: str) -> None:
    auth = config["formal_authorization"]
    if not auth.get("formal_execution_enabled", False):
        raise SystemExit("Formal execution is disabled by config.")
    if stage not in set(auth.get("authorized_stages", [])):
        raise SystemExit(f"Stage {stage} is not authorized by config.")
    for stage_name in stages_to_run(stage, config):
        stage_config = config["stages"][stage_name]
        if not stage_config.get("enabled", False) or not stage_config.get("formal_authorized", False):
            raise SystemExit(f"Stage {stage_name} is not enabled and formally authorized.")


def marker_path(config: dict[str, Any], job: dict[str, Any]) -> Path:
    return resolve_path(config["outputs"]["completion_marker_dir"]) / job["stage"] / f"{job['job_id']}.done.json"


def marker_payload(config: dict[str, Any], job: dict[str, Any], metrics_path: Path, predictions_path: Path) -> dict[str, Any]:
    fingerprint = job_fingerprint(config, job)
    return {
        "ts": now_iso(),
        "job_id": job["job_id"],
        "metrics_path": relpath(metrics_path),
        "predictions_path": relpath(predictions_path),
        "fingerprint_hash": fingerprint["hash"],
        "fingerprint": fingerprint["payload"],
    }


def resume_marker_status(config: dict[str, Any], job: dict[str, Any]) -> tuple[bool, str]:
    return marker_fingerprint_status(config, job)


def marker_fingerprint_status(config: dict[str, Any], job: dict[str, Any]) -> tuple[bool, str]:
    marker = marker_path(config, job)
    if not marker.exists():
        return False, "missing_marker"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable_marker:{type(exc).__name__}"
    expected = job_fingerprint(config, job)
    expected_hash = expected["hash"]
    actual = payload.get("fingerprint_hash")
    if not actual:
        return False, "missing_fingerprint"
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        return False, "missing_fingerprint_payload"
    if actual != stable_json_hash(fingerprint):
        return False, "fingerprint_hash_mismatch"
    if actual != expected_hash:
        allowed_yaml = set(str(x) for x in config.get("execution", {}).get("accepted_equivalent_yaml_sha256", []))
        allowed_runner = set(str(x) for x in config.get("execution", {}).get("accepted_equivalent_runner_code_sha256", []))
        fp_yaml = str(fingerprint.get("yaml_sha256", ""))
        fp_runner = str(fingerprint.get("runner_code_sha256", ""))
        equivalent = dict(expected["payload"])
        equivalent["yaml_sha256"] = fp_yaml
        equivalent["runner_code_sha256"] = fp_runner
        if fp_yaml in allowed_yaml and fp_runner in allowed_runner and fingerprint == equivalent:
            return True, "fingerprint_equivalent_authorized_rerun"
        return False, "fingerprint_mismatch"
    return True, "fingerprint_match"


def result_dir(config: dict[str, Any], job: dict[str, Any]) -> Path:
    return resolve_path(config["outputs"]["results_dir"]) / job["stage"] / job["experiment_id"] / job["job_id"]


def load_registered_frame(config: dict[str, Any], job: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    registry = read_registry(config)
    row = registry[registry["view_id"].astype(str).eq(str(job["view_id"]))]
    path_text = str(job.get("input_path") or "")
    if not row.empty and str(row.iloc[0].get("path", "")).strip():
        path_text = str(row.iloc[0]["path"])
    if ";" in path_text:
        raise ValueError("multi_source_design_input_requires_existing_prediction_adapter")
    path = resolve_path(path_text)
    if path.is_dir():
        raise ValueError(f"directory_path_not_allowed:{relpath(path)}")
    if not path.exists():
        raise FileNotFoundError(relpath(path))
    return pd.read_csv(path), relpath(path)


def registry_target_for_view(config: dict[str, Any], view_id: str, fallback: str) -> str:
    registry = read_registry(config)
    row = registry[registry["view_id"].astype(str).eq(str(view_id))]
    if not row.empty:
        for key in ["target_column", "target"]:
            value = row.iloc[0].get(key, "")
            if not pd.isna(value) and str(value).strip():
                return str(value).strip()
    return fallback


def target_column(job: dict[str, Any], frame: pd.DataFrame) -> str:
    target = str(job["target"])
    for candidate in target_candidates(target):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"target column not found: {target}")


def split_frame(frame: pd.DataFrame, job: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    seed = int(job["seed"])
    axis = str(job["validation_axis"])
    df = frame.copy()
    if axis in {"year_forward_small_smoke", "temporal_forward_smoke"} and "year" in df.columns:
        return df[df["year"] <= 1983], df[df["year"] >= 1984], "frozen_waite_smoke_year_forward"
    if "temporal" in axis and "year" in df.columns:
        years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().unique())
        if len(years) >= 4:
            cutoff = years[max(1, int(len(years) * 0.75) - 1)]
            return df[df["year"] <= cutoff], df[df["year"] > cutoff], f"temporal_forward_cutoff_{cutoff}"
    group_col = None
    for candidate in ["location_id", "state", "Env", "Hybrid", "adm_id", "plot"]:
        if candidate in df.columns and ("holdout" in axis or "group" in axis or "unseen" in axis):
            group_col = candidate
            break
    if group_col and df[group_col].nunique(dropna=True) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        groups = df[group_col].astype("object").where(df[group_col].notna(), "__MISSING_GROUP__").astype(str)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx], df.iloc[test_idx], f"group_shuffle_{group_col}"
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    return train, test, "random_holdout"


def make_estimator(model: str, seed: int) -> Any:
    if model in {"naive_mean", "official_average_yield", "none"}:
        return None
    if model in {"ridge", "ridge_smoke", "official_linear_trend"}:
        return Ridge(alpha=1.0, solver="lsqr")
    if model in {"random_forest", "random_forest_light", "random_forest_light_if_existing"}:
        trees = 50 if model == "random_forest_light" else 150
        if model == "random_forest_light_if_existing":
            trees = 50
        return RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1, min_samples_leaf=2)
    if model in {"hist_gradient_boosting", "hist_gradient_boosting_light"}:
        return HistGradientBoostingRegressor(random_state=seed, max_iter=50 if model.endswith("_light") else 150)
    if model in {"mean_ensemble", "validation_weighted_ensemble", "stacking", "oracle_best_expert_diagnostic", "learned_gate_diagnostic", "best_single"}:
        return None
    raise ValueError(f"unsupported_model_adapter:{model}")


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    blocked = {
        target,
        "grain_yield_kg_ha",
        "target_yield",
        "yield",
        "Yield_Mg_ha",
        "observed_yield_t_ha",
        "sample_id",
        "row_status",
        "source_file",
        "view_id",
    }
    cols = [c for c in frame.columns if c not in blocked and not c.lower().startswith("pred_")]
    return cols


def finite_target(series: pd.Series, job: dict[str, Any], column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = numeric.isna()
    if bad.all():
        raise ValueError(f"nonfinite_target view={job['view_id']} fold={job['fold']} model={job['model']} column={column} all_rows")
    return numeric


def prepare_fold_features(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], job: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    X_train = train[cols].replace([np.inf, -np.inf], np.nan).copy()
    X_test = test[cols].replace([np.inf, -np.inf], np.nan).copy()
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical = [c for c in cols if c not in numeric and X_train[c].astype(str).nunique(dropna=True) <= 50]
    all_missing = [c for c in numeric if X_train[c].isna().all()]
    if all_missing:
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)
        numeric = [c for c in numeric if c not in set(all_missing)]
    if not numeric and not categorical:
        raise ValueError(f"no_usable_features_after_finite_filter view={job['view_id']} fold={job['fold']} model={job['model']}")
    return X_train, X_test, numeric, categorical, all_missing


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def ensemble_table_predictions(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, numeric: list[str], categorical: list[str], model: str, seed: int) -> np.ndarray:
    idx = np.arange(len(X_train))
    sub_idx, meta_idx = train_test_split(idx, test_size=0.25, random_state=seed)
    if categorical:
        base_specs = [
            ("ridge_alpha_1", Ridge(alpha=1.0, solver="lsqr")),
            ("ridge_alpha_10", Ridge(alpha=10.0, solver="lsqr")),
        ]
    else:
        base_specs = [
            ("ridge", Ridge(alpha=1.0, solver="lsqr")),
            ("hist_gradient_boosting_light", HistGradientBoostingRegressor(random_state=seed, max_iter=30)),
        ]
    meta_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    meta_y = y_train.iloc[meta_idx]
    for _, estimator in base_specs:
        pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)), ("model", estimator)])
        pipe.fit(X_train.iloc[sub_idx], y_train.iloc[sub_idx])
        meta_preds.append(np.asarray(pipe.predict(X_train.iloc[meta_idx]), dtype=float))
        test_preds.append(np.asarray(pipe.predict(X_test), dtype=float))
    meta_matrix = np.column_stack(meta_preds)
    test_matrix = np.column_stack(test_preds)
    if model == "mean_ensemble":
        return np.mean(test_matrix, axis=1)
    meta_mae = np.array([mean_absolute_error(meta_y, meta_matrix[:, i]) for i in range(meta_matrix.shape[1])])
    if model == "validation_weighted_ensemble":
        weights = 1.0 / np.maximum(meta_mae, 1e-9)
        weights = weights / weights.sum()
        return test_matrix @ weights
    if model == "stacking":
        meta_model = Ridge(alpha=1.0, solver="lsqr")
        meta_model.fit(meta_matrix, meta_y)
        return np.asarray(meta_model.predict(test_matrix), dtype=float)
    if model == "oracle_best_expert_diagnostic":
        best = np.argmin(np.abs(test_matrix - y_test.to_numpy()[:, None]), axis=1)
        return test_matrix[np.arange(len(test_matrix)), best]
    if model == "learned_gate_diagnostic":
        best_model = int(np.argmin(meta_mae))
        return test_matrix[:, best_model]
    raise ValueError(f"unsupported_table_ensemble_adapter:{model}")


def train_table_adapter(config: dict[str, Any], job: dict[str, Any], write_outputs: bool = True) -> dict[str, Any]:
    frame, source = load_registered_frame(config, job)
    target = target_column(job, frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[target] = finite_target(frame[target], job, target)
    frame = frame[frame[target].notna()].copy()
    if frame.empty:
        raise ValueError("empty_data_after_target_filter")
    train, test, split_note = split_frame(frame, job)
    min_train = int(config["execution"].get("min_train_rows", 20))
    min_test = int(config["execution"].get("min_test_rows", 5))
    if len(train) < min_train or len(test) < min_test:
        raise ValueError(f"too_small_fold train={len(train)} test={len(test)}")
    y_train = finite_target(train[target], job, target)
    y_test = finite_target(test[target], job, target)
    model = str(job["model"])
    if model in {"mean_ensemble", "validation_weighted_ensemble", "stacking", "oracle_best_expert_diagnostic", "learned_gate_diagnostic"}:
        cols = feature_columns(frame, target)
        X_train, X_test, numeric, categorical, all_missing = prepare_fold_features(train, test, cols, job)
        pred = ensemble_table_predictions(X_train, y_train, X_test, y_test, numeric, categorical, model, int(job["seed"]))
    else:
        estimator = make_estimator(model, int(job["seed"]))
        if estimator is None:
            pred = np.repeat(float(y_train.mean()), len(test))
        else:
            cols = feature_columns(frame, target)
            X_train, X_test, numeric, categorical, all_missing = prepare_fold_features(train, test, cols, job)
            pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)), ("model", estimator)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
    return write_job_outputs(config, job, source, target, test, y_test.to_numpy(), np.asarray(pred, dtype=float), split_note, write_outputs=write_outputs)


def existing_prediction_adapter(config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    pred_col = PREDICTION_COLUMN_ALIASES.get(str(job["model"]), f"pred_{job['model']}")
    pieces = []
    for source in PREDICTION_SOURCES:
        if source.exists():
            df = pd.read_csv(source)
            if pred_col in df.columns:
                pieces.append((source, df))
    if not pieces:
        raise ValueError(f"existing_prediction_column_missing:{pred_col}")
    source, frame = choose_prediction_source(pieces, job)
    axis = str(job["validation_axis"])
    seed = int(job["seed"])
    if "axis" in frame.columns and axis != "not_applicable":
        frame = frame[frame["axis"].astype(str).eq(axis)]
    if "seed" in frame.columns:
        frame = frame[pd.to_numeric(frame["seed"], errors="coerce").eq(seed)]
    dataset = str(job["track"]).split("_fixed_experts")[0]
    if "dataset" in frame.columns:
        filtered = frame[frame["dataset"].astype(str).eq(dataset)]
        if not filtered.empty:
            frame = filtered
    elif "subset" in frame.columns:
        token = "maize_US" if "maize" in dataset else ("wheat_US" if "wheat" in dataset else "")
        if token:
            filtered = frame[frame["subset"].astype(str).eq(token)]
            if not filtered.empty:
                frame = filtered
    if frame.empty:
        raise ValueError("empty_existing_prediction_slice")
    target = "target_yield" if "target_yield" in frame.columns else target_column(job, frame)
    y_true = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_pred = pd.to_numeric(frame[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() < int(config["execution"].get("min_test_rows", 5)):
        raise ValueError(f"too_few_existing_predictions n={int(valid.sum())}")
    return write_job_outputs(config, job, relpath(source), target, frame[valid], y_true[valid].to_numpy(), y_pred[valid].to_numpy(), "existing_oof_predictions")


def choose_prediction_source(pieces: list[tuple[Path, pd.DataFrame]], job: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    track = str(job["track"]).lower()
    if "maize" in track:
        preferred = "cybench_maize_us_predictions"
    elif "g2f" in track:
        preferred = "stage4_g2f_predictions"
    else:
        preferred = "stage4_predictions"
    for source, frame in pieces:
        if preferred in source.name or preferred in str(source):
            return source, frame
    return pieces[0]


def modality_manifest_adapter(config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    manifest = pd.read_csv(resolve_path(str(job["input_path"])))
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        if not bool(row.get("registered", True)):
            continue
        sub_job = dict(job)
        sub_job["view_id"] = str(row["view_id"])
        sub_job["input_path"] = str(row["path"])
        default_target = "observed_yield_t_ha" if str(row["view_id"]).startswith("waite_") else "yield_t_ha"
        sub_job["target"] = registry_target_for_view(config, str(row["view_id"]), default_target)
        metrics = train_table_adapter(config, sub_job, write_outputs=False)
        metrics["sub_view_id"] = str(row["view_id"])
        prediction_rows.append(metrics["_predictions_frame"])
        metrics_rows.append(metrics)
    if not metrics_rows:
        raise ValueError("empty_modality_manifest")
    out = result_dir(config, job)
    combined_predictions = pd.concat(prediction_rows, ignore_index=True)
    clean_metrics = [{k: v for k, v in row.items() if k != "_predictions_frame"} for row in metrics_rows]
    atomic_write_csv(out / "predictions.csv", combined_predictions)
    atomic_write_json(out / "metrics.json", {"job_id": job["job_id"], "stage": job["stage"], "experiment_id": job["experiment_id"], "model": job["model"], "subview_metrics": clean_metrics})
    atomic_write_json(marker_path(config, job), marker_payload(config, job, out / "metrics.json", out / "predictions.csv"))
    mae = float(np.mean([row["MAE"] for row in clean_metrics]))
    return {"job_id": job["job_id"], "stage": job["stage"], "experiment_id": job["experiment_id"], "model": job["model"], "MAE": mae, "subview_count": len(clean_metrics), "runtime_seconds": None}


def write_job_outputs(
    config: dict[str, Any],
    job: dict[str, Any],
    source: str,
    target: str,
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_note: str,
    write_outputs: bool = True,
) -> dict[str, Any]:
    if not np.isfinite(y_true).all():
        raise ValueError(f"nonfinite_target view={job['view_id']} fold={job['fold']} model={job['model']} column={target}")
    if not np.isfinite(y_pred).all():
        raise ValueError(f"nonfinite_prediction view={job['view_id']} fold={job['fold']} model={job['model']} column=y_pred")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "experiment_id": job["experiment_id"],
        "model": job["model"],
        "validation_axis": job["validation_axis"],
        "fold": job["fold"],
        "seed": int(job["seed"]),
        "source": source,
        "target": target,
        "split_note": split_note,
        "n_train_or_source": None,
        "n_test": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "runtime_seconds": None,
    }
    finite_metrics = [v for k, v in metrics.items() if k in {"MAE", "RMSE", "R2"} and v is not None]
    if any(not math.isfinite(float(v)) for v in finite_metrics if not (isinstance(v, float) and math.isnan(v))):
        raise ValueError("nonfinite_metric")
    pred_frame = pd.DataFrame(
        {
            "sample_id": test["sample_id"].astype(str).to_numpy() if "sample_id" in test.columns else np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "model": job["model"],
            "view_id": job["view_id"],
            "validation_axis": job["validation_axis"],
            "fold": job["fold"],
            "seed": int(job["seed"]),
        }
    )
    if write_outputs:
        out = result_dir(config, job)
        atomic_write_csv(out / "predictions.csv", pred_frame)
        atomic_write_json(out / "metrics.json", metrics)
        marker = marker_path(config, job)
        atomic_write_json(marker, marker_payload(config, job, out / "metrics.json", out / "predictions.csv"))
    else:
        metrics["_predictions_frame"] = pred_frame
    return metrics


def run_single_job(config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    atomic_write_json(resolve_path(config["outputs"]["heartbeat_dir"]) / f"{job['job_id']}.json", {"ts": now_iso(), "job_id": job["job_id"], "status": "running"})
    try:
        metrics_already_written = False
        if str(job["view_id"]) == "waite_and_regional_modality_views":
            metrics = modality_manifest_adapter(config, job)
            metrics_already_written = True
        else:
            path_text = str(job.get("input_path", ""))
            design_like = str(job.get("view_id", "")).startswith("stage4_") or "stage4_" in str(job.get("view_id", "")) or ";" in path_text
            table_like = str(job["view_id"]).startswith(("waite", "regional", "g2f"))
            if design_like:
                metrics = existing_prediction_adapter(config, job)
            elif table_like:
                metrics = train_table_adapter(config, job)
            else:
                metrics = existing_prediction_adapter(config, job)
        metrics["runtime_seconds"] = round(time.time() - start, 3)
        if not metrics_already_written:
            atomic_write_json(result_dir(config, job) / "metrics.json", metrics)
        return {"status": "done", "metrics": metrics}
    except Exception as exc:
        tb = traceback.format_exc()
        out = result_dir(config, job)
        atomic_write_text(out / "traceback.txt", tb)
        return {"status": "failed", "failure_type": type(exc).__name__, "message": str(exc), "traceback_path": relpath(out / "traceback.txt")}


def run_child(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    job = json.loads(args.job_json)
    result = run_single_job(config, job)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "done" else 1


def record_failure(config: dict[str, Any], job: dict[str, Any], failure_type: str, message: str, traceback_path: str = "") -> None:
    append_csv(
        resolve_path(config["outputs"]["failure_ledger_csv"]),
        {"ts": now_iso(), "job_id": job["job_id"], "stage": job["stage"], "experiment_id": job["experiment_id"], "model": job["model"], "fold": job["fold"], "seed": job["seed"], "failure_type": failure_type, "message": message, "traceback_path": traceback_path},
    )


def record_invalidation(config: dict[str, Any], job: dict[str, Any], reason: str) -> None:
    append_csv(
        resolve_path(config["outputs"].get("invalidation_ledger_csv", config["outputs"]["failure_ledger_csv"])),
        {"ts": now_iso(), "job_id": job["job_id"], "stage": job["stage"], "experiment_id": job["experiment_id"], "model": job["model"], "fold": job["fold"], "seed": job["seed"], "reason": reason, "marker_path": relpath(marker_path(config, job))},
    )


def archive_existing_job_output(config: dict[str, Any], job: dict[str, Any], reason: str) -> None:
    archive_root = resolve_path(config["outputs"]["root"]) / "invalidated"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = archive_root / job["stage"] / f"{job['job_id']}.{stamp}.{reason}"
    moved: list[str] = []
    source_result = result_dir(config, job)
    if source_result.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / "result_dir"
        source_result.replace(target)
        moved.append(relpath(target))
    source_marker = marker_path(config, job)
    if source_marker.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / source_marker.name
        source_marker.replace(target)
        moved.append(relpath(target))
    if moved:
        atomic_write_json(archive_dir / "invalidation.json", {"ts": now_iso(), "job_id": job["job_id"], "reason": reason, "moved": moved})


def run_formal(config: dict[str, Any], stage: str, manifest: pd.DataFrame, resume: bool, timeout_override: int | None) -> int:
    jobs = manifest[manifest["stage_enabled"] & manifest["formal_authorized"]].to_dict("records")
    total = len(jobs)
    retry_limit = int(config["execution"].get("max_retries", 0))
    exit_code = 0
    for idx, job in enumerate(jobs, start=1):
        if float(shutil.disk_usage(resolve_path(config["outputs"]["root"])).free / (1024**3)) <= float(config["safety"]["disk_free_hard_stop_gib"]):
            print(f"[{idx}/{total}] FAILED {job['job_id']} disk_hard_stop recorded_and_stopping", flush=True)
            record_failure(config, job, "disk_hard_stop", "free disk below hard stop")
            return 2
        marker_valid, marker_reason = resume_marker_status(config, job)
        if marker_valid:
            reason = "skipped_resume_fingerprint_match" if resume else "existing_not_overwritten_fingerprint_match"
            print(f"[{idx}/{total}] DONE {job['job_id']} {reason}", flush=True)
            continue
        if marker_path(config, job).exists():
            record_invalidation(config, job, marker_reason)
            archive_existing_job_output(config, job, marker_reason)
            print(f"[{idx}/{total}] START {job['job_id']} resume_marker_invalidated={marker_reason}", flush=True)
        elif result_dir(config, job).exists():
            archive_existing_job_output(config, job, "rerun_preserve_existing_result_without_valid_marker")
        print(f"[{idx}/{total}] START {job['job_id']} exp={job['experiment_id']} model={job['model']} axis={job['validation_axis']}", flush=True)
        append_jsonl(resolve_path(config["outputs"]["progress_jsonl"]), {"ts": now_iso(), "event": "start", "current": idx, "total": total, **job})
        timeout = int(timeout_override or job["timeout_seconds"])
        attempt_result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(retry_limit + 1):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(DEFAULT_CONFIG if False else args_config_path(config)), "--run-single-job", "--job-json", json.dumps(job)]
            try:
                attempt_result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"[{idx}/{total}] TIMEOUT {job['job_id']} recorded_and_skipped", flush=True)
                record_failure(config, job, "timeout", f"timeout_seconds={timeout}")
                append_jsonl(resolve_path(config["outputs"]["progress_jsonl"]), {"ts": now_iso(), "event": "timeout", "current": idx, "total": total, **job})
                exit_code = 1
                break
            if attempt_result.returncode == 0:
                print(attempt_result.stdout.strip(), flush=True)
                print(f"[{idx}/{total}] DONE {job['job_id']}", flush=True)
                append_jsonl(resolve_path(config["outputs"]["progress_jsonl"]), {"ts": now_iso(), "event": "done", "current": idx, "total": total, **job})
                break
            if attempt >= retry_limit:
                detail = attempt_result.stdout.strip() or attempt_result.stderr.strip()
                failure_type = "job_failed"
                traceback_path = ""
                try:
                    parsed = json.loads(attempt_result.stdout.strip().splitlines()[-1])
                    failure_type = parsed.get("failure_type", failure_type)
                    detail = parsed.get("message", detail)
                    traceback_path = parsed.get("traceback_path", "")
                except Exception:
                    pass
                print(f"[{idx}/{total}] FAILED {job['job_id']} recorded_and_continuing", flush=True)
                record_failure(config, job, failure_type, detail, traceback_path)
                append_jsonl(resolve_path(config["outputs"]["progress_jsonl"]), {"ts": now_iso(), "event": "failed", "current": idx, "total": total, **job, "message": detail})
                exit_code = 1
        write_status(config, stage, manifest, mode="formal_running")
    for stage_name in stages_to_run(stage, config):
        stage_jobs = [j for j in jobs if j["stage"] == stage_name]
        if stage_jobs and all(resume_marker_status(config, j)[0] for j in stage_jobs):
            atomic_write_json(resolve_path(config["outputs"]["completion_marker_dir"]) / f"{stage_name}.complete.json", {"ts": now_iso(), "stage": stage_name, "job_count": len(stage_jobs)})
    write_status(config, stage, manifest, mode="formal_finished")
    return exit_code


def args_config_path(config: dict[str, Any]) -> str:
    return str(resolve_path(config.get("_config_path", DEFAULT_CONFIG)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["7a", "7b", "7c", "all"], default="7a")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-formal", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--timeout-per-job", type=int)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-single-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_single_job:
        return run_child(args)
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    preflight_rows = validate_static_paths(config)
    atomic_write_csv(resolve_path(config["outputs"]["preexecution_check_csv"]), pd.DataFrame(preflight_rows))
    blockers = [row for row in preflight_rows if row["status"] == "fail" and row["severity"] == "blocker"]
    if blockers:
        raise SystemExit(f"Preflight blockers remain: {blockers}")
    manifest = build_job_manifest(config, args.stage, args.experiment_id, args.max_workers, args.timeout_per_job)
    atomic_write_csv(resolve_path(config["outputs"]["job_manifest_csv"]), manifest)
    if args.run_formal and not args.dry_run and not args.preflight_only:
        formal_guard(config, args.stage)
    mode = "dry_run" if args.dry_run or not args.run_formal else "formal_requested"
    write_status(config, args.stage, manifest, mode=mode)
    append_jsonl(resolve_path(config["outputs"]["progress_jsonl"]), {"ts": now_iso(), "event": mode, "stage": args.stage, "stage_order": stages_to_run(args.stage, config), "job_count": int(len(manifest))})
    if args.preflight_only:
        print(json.dumps({"stage": args.stage, "stage_order": stages_to_run(args.stage, config), "preflight_rows": len(preflight_rows), "job_count": len(manifest), "mode": "preflight_only"}, indent=2), flush=True)
        return 0
    if not args.run_formal or args.dry_run:
        print(json.dumps({"stage": args.stage, "stage_order": stages_to_run(args.stage, config), "job_count": len(manifest), "mode": "dry_run", "manifest": relpath(resolve_path(config["outputs"]["job_manifest_csv"]))}, indent=2), flush=True)
        return 0
    return run_formal(config, args.stage, manifest, args.resume, args.timeout_per_job)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    sys.exit(main())
