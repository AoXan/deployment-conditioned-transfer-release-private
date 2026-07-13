#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.cybench_loader import get_available_subsets, process_cybench_subset
from src.data.data_loader import load_nvt_canola
from src.data.nvt_aggregation import aggregate_nvt_town_year
from src.features.smart_temporal import extract_smart_features
from src.features.temporal_builder import CLIMATE_FEATURES, filter_subset, fuse_and_tag_drought, split_by_resolution
from src.models.publication_evaluator import (
    Combo,
    STATUS_FAILED,
    STATUS_INSUFFICIENT_SAMPLES,
    STATUS_SKIPPED,
    STATUS_VALID,
    aggregate_results,
    bootstrap_mean_ci,
    dependency_status,
    evaluate_combo,
    feature_granularity_table,
    legacy_result_audit,
    make_splits,
    random_spatial_gap,
    select_feature_columns,
    stable_hash_frame,
    target_status,
)
from src.models.weather_fingerprint import run_weather_fingerprint_diagnostics

RUN_ROOT = BASE_DIR / "outputs" / "publication_experiments"
DEFAULT_NVT_EXTERNAL_DIR = BASE_DIR / "new data"
DEFAULT_NVT_FUSED_PATH = BASE_DIR / "Seabrook_Offline_Results" / "04_task3_micro_canola_prediction" / "fused_multivariate_dataset.csv"

MODE_CONFIG = {
    "small": {
        "feature_sets": ["smart_agronomic", "seasonal_weather", "daily_high_dim"],
        "split_strategies": ["random", "spatial"],
        "models": ["naive_mean", "ridge", "random_forest"],
        "max_datasets": 3,
        "n_splits": 3,
    },
    "paper": {
        "feature_sets": ["smart_agronomic"],
        "split_strategies": ["random", "spatial", "temporal_forward", "spatiotemporal"],
        "models": ["naive_mean", "ridge", "random_forest"],
        "max_datasets": None,
        "n_splits": 3,
    },
    "full": {
        "feature_sets": ["smart_agronomic", "seasonal_weather", "monthly_weather", "weekly_weather", "daily_high_dim"],
        "split_strategies": ["random", "spatial", "group"],
        "models": ["naive_mean", "ridge", "svr", "random_forest", "xgboost"],
        "max_datasets": None,
        "n_splits": 5,
    },
}

AGGREGATED_SMALL_SPLITS = ["random", "spatial", "temporal_forward", "spatiotemporal"]


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def parse_args_for_tests(argv: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication-grade AJCAI cross-evaluation experiments.")
    parser.add_argument("--mode", choices=["small", "paper", "full"], default="small")
    parser.add_argument("--dry-run", action="store_true", help="Build audit and planned matrix without fitting models.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-combo results when present.")
    parser.add_argument("--force", action="store_true", help="Rerun combos even if cached results exist.")
    parser.add_argument("--rerun", choices=["failed"], default=None, help="With --resume, rerun only failed cached combos.")
    parser.add_argument("--run-id", default=None, help="Stable run id. Defaults to timestamp.")
    parser.add_argument("--cybench-dir", type=Path, default=None, help="Root containing CY-Bench crop/country folders.")
    parser.add_argument("--nvt-data-dir", type=Path, default=DEFAULT_NVT_EXTERNAL_DIR, help="Repository or external NVT data directory.")
    parser.add_argument("--nvt-fused-path", type=Path, default=DEFAULT_NVT_FUSED_PATH, help="Optional model-ready NVT fused dataset.")
    parser.add_argument("--include-nvt", action="store_true", default=True)
    parser.add_argument("--exclude-nvt", action="store_false", dest="include_nvt")
    parser.add_argument("--max-runtime-per-combo", type=float, default=None)
    parser.add_argument(
        "--task-definition",
        choices=["variety_level_yield", "town_year_aggregated_yield"],
        default="variety_level_yield",
    )
    parser.add_argument("--aggregation-method", choices=["arithmetic_mean"], default="arithmetic_mean")
    return parser


def main() -> int:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{args.mode}_%Y%m%d_%H%M%S")
    output_dir = RUN_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["results", "predictions", "splits", "diagnostics", "figures"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    datasets, audit_rows = load_publication_datasets(args, config)
    combos = build_matrix(datasets, config, task_definition=args.task_definition)
    write_manifest(args, config, output_dir, datasets, combos)
    write_dataset_artifacts(output_dir, datasets)
    write_aggregated_split_audit(output_dir, datasets, config)
    write_frozen_protocol_artifacts(output_dir, datasets, combos, config)

    audit_df = pd.DataFrame(audit_rows)
    planned_columns = [
        "dataset",
        "task_definition",
        "target_definition",
        "aggregation_method",
        "feature_set",
        "split_strategy",
        "model",
        "n_splits",
        "combo_id",
    ]
    planned_df = pd.DataFrame([combo.__dict__ | {"combo_id": combo.combo_id} for combo, _ in combos], columns=planned_columns)
    audit_df.to_csv(output_dir / "data_asset_audit.csv", index=False)
    planned_df.to_csv(output_dir / "planned_matrix.csv", index=False)
    if args.task_definition == "town_year_aggregated_yield":
        planned_df.to_csv(output_dir / "aggregated_small_smoke_matrix.csv", index=False)
    build_dependency_status(config["models"]).to_csv(output_dir / "dependency_status.csv", index=False)

    legacy_path = BASE_DIR / "outputs" / "cross_evaluation" / "global_evaluation_metrics_matrix.csv"
    legacy_result_audit(legacy_path).to_csv(output_dir / "legacy_result_audit.csv", index=False)

    if args.dry_run:
        write_dry_run_summary(output_dir, audit_df, planned_df)
        print(f"Dry-run complete. Outputs written to {output_dir}")
        return 0

    run_fingerprint_for_aggregated_datasets(output_dir, datasets, config, resume=args.resume, force=args.force)

    all_metrics = []
    all_predictions = []
    for combo, payload in combos:
        print(f"Running {combo.combo_id}")
        metrics, predictions = evaluate_combo(
            combo=combo,
            df=payload["df"],
            output_dir=output_dir,
            location_df=payload.get("location_df"),
            resume=args.resume,
            force=args.force,
            rerun=args.rerun,
            max_runtime_sec=args.max_runtime_per_combo,
        )
        all_metrics.append(metrics)
        if not predictions.empty:
            all_predictions.append(predictions)

    fold_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    evaluation = aggregate_results(fold_metrics)
    evaluation.to_csv(output_dir / "evaluation_matrix.csv", index=False)
    write_diagnostics(output_dir, audit_df, evaluation)
    if all_predictions:
        predictions_all = pd.concat(all_predictions, ignore_index=True)
        predictions_all.to_csv(output_dir / "predictions_all.csv", index=False)
        write_prediction_level_diagnostics(output_dir, predictions_all)
    print(f"Publication evaluation complete. Outputs written to {output_dir}")
    return 0


def load_publication_datasets(args: argparse.Namespace, config: dict) -> tuple[list[dict], list[dict]]:
    datasets: list[dict] = []
    audit_rows: list[dict] = []
    raw_dir = BASE_DIR / "data" / "raw"

    if args.include_nvt:
        nvt_payload, nvt_audit = load_nvt_dataset(
            args.nvt_data_dir,
            raw_dir,
            fused_path=args.nvt_fused_path,
            task_definition=args.task_definition,
            aggregation_method=args.aggregation_method,
        )
        audit_rows.append(nvt_audit)
        if nvt_payload is not None:
            datasets.append(nvt_payload)

    cybench_dir = resolve_cybench_dir(args.cybench_dir, raw_dir)
    if cybench_dir is None:
        audit_rows.append(
            {
                "dataset": "CY-Bench",
                "source_type": "repository",
                "location": "",
                "status": STATUS_SKIPPED,
                "status_reason": "CY-Bench directory not found",
            }
        )
    else:
        cybench_datasets, cybench_audit = load_cybench_datasets(cybench_dir)
        audit_rows.extend(cybench_audit)
        datasets.extend(cybench_datasets)

    max_datasets = config.get("max_datasets")
    if max_datasets is not None and len(datasets) > max_datasets:
        datasets = datasets[:max_datasets]
    return datasets, audit_rows


def load_nvt_dataset(
    nvt_dir: Path,
    raw_dir: Path,
    fused_path: Path | None = None,
    task_definition: str = "variety_level_yield",
    aggregation_method: str = "arithmetic_mean",
) -> tuple[dict | None, dict]:
    audit = {
        "dataset": "NVT_Canola_Australia",
        "source_type": "external",
        "location": str(nvt_dir),
        "external_data": True,
        "status": STATUS_SKIPPED,
        "status_reason": "",
    }
    if fused_path is not None and fused_path.exists():
        try:
            df = pd.read_csv(fused_path)
            payload_extra = {
                "task_definition": "variety_level_yield",
                "target_definition": "variety_yield",
                "aggregation_method": "none",
            }
            if task_definition == "town_year_aggregated_yield":
                df, provenance, consistency, target_summary = aggregate_nvt_town_year(df, aggregation_method=aggregation_method)
                payload_extra = {
                    "task_definition": "town_year_aggregated_yield",
                    "target_definition": "town_year_mean_yield",
                    "aggregation_method": aggregation_method,
                    "aggregation_provenance": provenance,
                    "environment_feature_consistency_audit": consistency,
                    "aggregated_target_summary": target_summary,
                }
            status, reason = audit_dataset_frame(df)
            audit.update(dataset_stats(df))
            audit.update(
                {
                    "location": str(fused_path),
                    "source_type": "repository_fused",
                    "external_data": False,
                    "status": status,
                    "status_reason": reason,
                    "file_hash": hash_files(fused_path.parent),
                    "reproducibility_note": "model-ready fused table migrated into repository; raw NVT wide files retained separately",
                }
            )
            if status != STATUS_VALID:
                return None, audit
            return {"name": "NVT_Canola_Australia", "df": df, "location_df": None, **payload_extra}, audit
        except Exception as exc:
            audit.update({"status": STATUS_FAILED, "status_reason": f"fused NVT table could not be read: {exc.__class__.__name__}: {exc}"})
            return None, audit

    if not nvt_dir.exists():
        audit["status_reason"] = "external NVT directory not found"
        return None, audit
    required_patterns = [
        "crop_yield/grdc_nvt/NVT_single-site-single-site-yield_canola*.csv",
        "NVT_single-site-single-site-yield_canola*.csv",
    ]
    if not any(list(nvt_dir.glob(pattern)) for pattern in required_patterns):
        audit["status_reason"] = "external NVT directory exists but required canola yield CSV was not found"
        return None, audit
    try:
        df_nvt = load_nvt_canola(nvt_dir)
        required_cols = {"Town", "Year", "Yield_t_ha"}
        if df_nvt.empty or not required_cols.issubset(df_nvt.columns):
            missing = sorted(required_cols.difference(df_nvt.columns))
            audit.update(
                {
                    "status": STATUS_SKIPPED,
                    "status_reason": f"NVT parsed table is not model-ready; missing columns={missing}; rows={len(df_nvt)}",
                }
            )
            return None, audit
        df_fused = fuse_and_tag_drought(df_nvt, raw_dir)
        datasets_by_resolution = split_by_resolution(df_fused)
        df_daily = datasets_by_resolution["Daily"]
        df_daily_c = filter_subset(df_daily, CLIMATE_FEATURES)
        df_smart = extract_smart_features(df_daily_c)
        smart_cols = ["GDD", "Max_Dry_Spell", "Heat_Stress_Days", "Total_Rain", "Town", "Year", "Variety"]
        df = pd.merge(df_daily, df_smart[smart_cols], on=["Town", "Year", "Variety"], how="inner")
        status, reason = audit_dataset_frame(df)
        audit.update(dataset_stats(df))
        audit.update({"status": status, "status_reason": reason, "file_hash": hash_files(nvt_dir)})
        if status != STATUS_VALID:
            return None, audit
        return {"name": "NVT_Canola_Australia", "df": df, "location_df": None}, audit
    except Exception as exc:
        status = STATUS_SKIPPED if isinstance(exc, KeyError) else STATUS_FAILED
        audit.update({"status": status, "status_reason": f"NVT external data could not be parsed into model-ready schema: {exc.__class__.__name__}: {exc}"})
        return None, audit


def load_cybench_datasets(cybench_dir: Path) -> tuple[list[dict], list[dict]]:
    datasets: list[dict] = []
    audit_rows: list[dict] = []
    subsets = get_available_subsets(cybench_dir)
    if not subsets:
        audit_rows.append(
            {
                "dataset": "CY-Bench",
                "source_type": "repository",
                "location": str(cybench_dir),
                "status": STATUS_SKIPPED,
                "status_reason": "no crop/country subsets with required files",
            }
        )
        return datasets, audit_rows
    for subset in subsets:
        crop = subset["crop"]
        country = subset["country"]
        name = f"CY-Bench_{crop}_{country}"
        country_dir = cybench_dir / crop / country
        location_file = country_dir / f"location_{crop}_{country}.csv"
        audit = {
            "dataset": name,
            "source_type": "repository",
            "location": str(country_dir),
            "external_data": False,
            "status": STATUS_SKIPPED,
            "status_reason": "",
        }
        if not location_file.exists():
            audit["status_reason"] = "location file not found"
            audit_rows.append(audit)
            continue
        try:
            location_df = pd.read_csv(location_file)
            df = process_cybench_subset(crop, country, cybench_dir)
            status, reason = audit_dataset_frame(df)
            audit.update(dataset_stats(df))
            audit.update({"status": status, "status_reason": reason, "file_hash": hash_files(country_dir)})
            if status == STATUS_VALID:
                datasets.append({"name": name, "df": df, "location_df": location_df})
        except Exception as exc:
            audit.update({"status": STATUS_FAILED, "status_reason": f"{exc.__class__.__name__}: {exc}"})
        audit_rows.append(audit)
    return datasets, audit_rows


def audit_dataset_frame(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return STATUS_INSUFFICIENT_SAMPLES, "empty processed frame"
    if "Yield_t_ha" not in df.columns:
        return STATUS_FAILED, "missing Yield_t_ha target"
    status, reason = target_status(df["Yield_t_ha"])
    if status != STATUS_VALID:
        return status, reason
    if not any(select_feature_columns(df, feature_set) for feature_set in MODE_CONFIG["paper"]["feature_sets"]):
        return STATUS_SKIPPED, "no usable feature columns"
    return STATUS_VALID, "ok"


def dataset_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_samples": 0}
    out = {
        "n_samples": int(len(df)),
        "n_columns": int(df.shape[1]),
        "data_hash": stable_hash_frame(df),
    }
    if "Year" in df:
        years = pd.to_numeric(df["Year"], errors="coerce").dropna()
        if not years.empty:
            out["year_min"] = int(years.min())
            out["year_max"] = int(years.max())
    if "Town" in df:
        out["n_locations"] = int(df["Town"].nunique())
    if "Yield_t_ha" in df:
        y = pd.to_numeric(df["Yield_t_ha"], errors="coerce")
        out["target_mean"] = float(y.mean())
        out["target_variance"] = float(y.var(ddof=0))
    return out


def build_matrix(
    datasets: list[dict],
    config: dict,
    task_definition: str = "variety_level_yield",
) -> list[tuple[Combo, dict]]:
    combos: list[tuple[Combo, dict]] = []
    for payload in datasets:
        split_strategies = AGGREGATED_SMALL_SPLITS if payload.get("task_definition") == "town_year_aggregated_yield" else config["split_strategies"]
        for feature_set in config["feature_sets"]:
            if not select_feature_columns(payload["df"], feature_set):
                continue
            for split_strategy in split_strategies:
                for model in config["models"]:
                    combo = Combo(
                        dataset=payload["name"],
                        feature_set=feature_set,
                        split_strategy=split_strategy,
                        model=model,
                        n_splits=config["n_splits"],
                        task_definition=payload.get("task_definition", task_definition),
                        target_definition=payload.get("target_definition", "variety_yield"),
                        aggregation_method=payload.get("aggregation_method", "none"),
                    )
                    combos.append((combo, payload))
    return combos


def resolve_cybench_dir(cli_value: Path | None, raw_dir: Path) -> Path | None:
    candidates = []
    if cli_value is not None:
        candidates.append(cli_value)
    candidates.extend(
        [
            raw_dir / "cybench_extracted" / "cybench-data",
            raw_dir / "cybench_extracted",
            raw_dir / "cybench_sample",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_manifest(args: argparse.Namespace, config: dict, output_dir: Path, datasets: list[dict], combos: list[tuple[Combo, dict]]) -> None:
    manifest = {
        "run_id": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "rerun": args.rerun,
        "max_runtime_per_combo": args.max_runtime_per_combo,
        "config": config,
        "task_definition": args.task_definition,
        "target_definition": "town_year_mean_yield" if args.task_definition == "town_year_aggregated_yield" else "variety_yield",
        "aggregation_method": args.aggregation_method if args.task_definition == "town_year_aggregated_yield" else "none",
        "protocol_status": "frozen" if args.mode == "paper" and args.task_definition == "town_year_aggregated_yield" else "development",
        "primary_metric": "mae",
        "datasets_included": [payload["name"] for payload in datasets],
        "n_combos": len(combos),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "feature_schema_hash": feature_schema_hash(datasets),
        "notes": [
            "NVT is treated as external data when included.",
            "LSTM is excluded from main matrix unless separately validated.",
            "Legacy cross-evaluation results are audited as reference-only.",
        ],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_frozen_protocol_artifacts(output_dir: Path, datasets: list[dict], combos: list[tuple[Combo, dict]], config: dict) -> None:
    matrix_rows = []
    for combo, payload in combos:
        role = "primary"
        if combo.split_strategy == "random":
            role = "diagnostic_optimism_reference"
        elif combo.split_strategy == "spatiotemporal":
            role = "diagnostic_robustness"
        matrix_rows.append(
            {
                **combo.__dict__,
                "combo_id": combo.combo_id,
                "role": role,
                "primary_metric": "mae",
                "feature_schema_hash": stable_hash_frame(payload["df"][select_feature_columns(payload["df"], combo.feature_set)])
                if select_feature_columns(payload["df"], combo.feature_set)
                else "none",
            }
        )
    pd.DataFrame(matrix_rows).to_csv(output_dir / "frozen_paper_experiment_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "hypothesis_id": "H1_primary",
                "hypothesis": "Random validation overestimates performance relative to strict unseen-town spatial validation.",
                "comparison": "random_vs_spatial",
                "primary_metric": "mae",
                "status": "confirmatory",
            },
            {
                "hypothesis_id": "H2_secondary",
                "hypothesis": "Random validation overestimates performance relative to future-year temporal validation.",
                "comparison": "random_vs_temporal_forward",
                "primary_metric": "mae",
                "status": "secondary",
            },
            {
                "hypothesis_id": "H3_diagnostic",
                "hypothesis": "Weather/agronomic features contain location- or year-identifying signal.",
                "comparison": "fingerprint_vs_majority_baseline",
                "primary_metric": "balanced_accuracy",
                "status": "diagnostic",
            },
        ]
    ).to_csv(output_dir / "hypothesis_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "component": "configuration",
                "status": "frozen" if config.get("models") == ["naive_mean", "ridge", "random_forest"] else STATUS_SKIPPED,
                "status_reason": "NVT aggregated AJCAI core matrix" if config.get("models") == ["naive_mean", "ridge", "random_forest"] else "non-core config",
            }
        ]
    ).to_csv(output_dir / "exclusion_registry.csv", index=False)


def run_fingerprint_for_aggregated_datasets(
    output_dir: Path,
    datasets: list[dict],
    config: dict,
    resume: bool = False,
    force: bool = False,
) -> None:
    expected = [
        output_dir / "weather_fingerprint_fold_metrics.csv",
        output_dir / "weather_fingerprint_predictions.csv",
        output_dir / "weather_fingerprint_splits.csv",
    ]
    if resume and not force and all(path.exists() for path in expected):
        return
    for payload in datasets:
        if payload.get("task_definition") != "town_year_aggregated_yield":
            continue
        run_weather_fingerprint_diagnostics(
            payload["df"],
            output_dir,
            feature_set="smart_agronomic",
            split_strategies=["random", "spatial", "temporal_forward", "spatiotemporal"],
            targets=["Town", "Year", "spatial_block"],
            n_splits=config["n_splits"],
        )


def write_dataset_artifacts(output_dir: Path, datasets: list[dict]) -> None:
    for payload in datasets:
        if payload.get("task_definition") != "town_year_aggregated_yield":
            continue
        payload["df"].to_csv(output_dir / "nvt_town_year_aggregated_dataset.csv", index=False)
        payload.get("aggregation_provenance", pd.DataFrame()).to_csv(output_dir / "aggregation_provenance.csv", index=False)
        payload.get("environment_feature_consistency_audit", pd.DataFrame()).to_csv(
            output_dir / "environment_feature_consistency_audit.csv",
            index=False,
        )
        payload.get("aggregated_target_summary", pd.DataFrame()).to_csv(output_dir / "aggregated_target_summary.csv", index=False)


def write_aggregated_split_audit(output_dir: Path, datasets: list[dict], config: dict) -> None:
    rows = []
    for payload in datasets:
        if payload.get("task_definition") != "town_year_aggregated_yield":
            continue
        df = payload["df"].copy()
        if {"total_rain", "avg_max_temp", "max_temp_extreme"}.issubset(df.columns):
            df["weather_hash"] = pd.util.hash_pandas_object(df[["total_rain", "avg_max_temp", "max_temp_extreme"]], index=False).astype(str)
        for split_strategy in AGGREGATED_SMALL_SPLITS:
            status, splits, split_df = make_splits(df, split_strategy, config["n_splits"])
            if status != STATUS_VALID:
                rows.append(
                    {
                        "dataset": payload["name"],
                        "split_strategy": split_strategy,
                        "fold": -1,
                        "status": status,
                        "status_reason": "split construction failed or insufficient samples",
                    }
                )
                continue
            for fold, (train_idx, test_idx) in enumerate(splits):
                train = split_df.iloc[train_idx]
                test = split_df.iloc[test_idx]
                rows.append(
                    {
                        "dataset": payload["name"],
                        "split_strategy": split_strategy,
                        "fold": fold,
                        "status": STATUS_VALID,
                        "status_reason": "ok",
                        "n_train": len(train),
                        "n_test": len(test),
                        "train_towns": "|".join(sorted(train["Town"].astype(str).unique())),
                        "test_towns": "|".join(sorted(test["Town"].astype(str).unique())),
                        "train_years": "|".join(sorted(train["Year"].astype(str).unique())),
                        "test_years": "|".join(sorted(test["Year"].astype(str).unique())),
                        "town_overlap_count": len(set(train["Town"].astype(str)).intersection(set(test["Town"].astype(str)))),
                        "year_overlap_count": len(set(train["Year"].astype(str)).intersection(set(test["Year"].astype(str)))),
                        "weather_hash_overlap_count": len(set(train.get("weather_hash", pd.Series(dtype=str)).astype(str)).intersection(set(test.get("weather_hash", pd.Series(dtype=str)).astype(str)))),
                        "train_target_mean": train["Yield_t_ha"].mean(),
                        "test_target_mean": test["Yield_t_ha"].mean(),
                        "train_target_variance": train["Yield_t_ha"].var(ddof=0),
                        "test_target_variance": test["Yield_t_ha"].var(ddof=0),
                        "split_hash": stable_hash_frame(pd.DataFrame({"test_idx": test_idx})),
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "aggregated_split_overlap_audit.csv", index=False)


def write_dry_run_summary(output_dir: Path, audit_df: pd.DataFrame, planned_df: pd.DataFrame) -> None:
    summary = {
        "n_audit_rows": int(len(audit_df)),
        "n_included_datasets": int((audit_df.get("status") == STATUS_VALID).sum()) if not audit_df.empty else 0,
        "n_planned_combos": int(len(planned_df)),
        "estimated_fold_fits": int(planned_df["n_splits"].sum()) if "n_splits" in planned_df else 0,
        "planned_feature_sets": sorted(planned_df["feature_set"].dropna().unique().tolist()) if "feature_set" in planned_df else [],
        "planned_models": sorted(planned_df["model"].dropna().unique().tolist()) if "model" in planned_df else [],
        "planned_split_strategies": sorted(planned_df["split_strategy"].dropna().unique().tolist()) if "split_strategy" in planned_df else [],
    }
    with open(output_dir / "dry_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def build_dependency_status(models: list[str]) -> pd.DataFrame:
    rows = []
    for model in sorted(set(models + ["lstm"])):
        status, reason = dependency_status(model)
        rows.append({"model": model, "status": status, "status_reason": reason})
    return pd.DataFrame(rows)


def write_diagnostics(output_dir: Path, audit_df: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    excluded = audit_df[audit_df["status"] != STATUS_VALID].copy() if not audit_df.empty and "status" in audit_df else pd.DataFrame()
    excluded.to_csv(output_dir / "excluded_dataset_report.csv", index=False)
    gap = random_spatial_gap(evaluation)
    gap.to_csv(output_dir / "random_spatial_gap.csv", index=False)
    feature_granularity_table(evaluation).to_csv(output_dir / "feature_granularity_table.csv", index=False)
    ci = bootstrap_mean_ci(gap["random_minus_spatial"]) if not gap.empty else {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    robustness = pd.DataFrame([{"metric": "random_minus_spatial_r2", **ci}])
    robustness.to_csv(output_dir / "robustness_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "component": "lstm",
                "status": STATUS_SKIPPED,
                "status_reason": "diagnostic-only; excluded from main matrix until torch dependency and non-silent training checks pass",
            }
        ]
    ).to_csv(output_dir / "lstm_validity_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "diagnostic": "weather_fingerprint",
                "status": STATUS_VALID if (output_dir / "weather_fingerprint_fold_metrics.csv").exists() else STATUS_SKIPPED,
                "status_reason": "see weather_fingerprint_fold_metrics.csv" if (output_dir / "weather_fingerprint_fold_metrics.csv").exists() else "not run",
            }
        ]
    ).to_csv(output_dir / "weather_fingerprint_table.csv", index=False)


def write_prediction_level_diagnostics(output_dir: Path, predictions: pd.DataFrame) -> None:
    pooled = build_pooled_oof_metrics(predictions)
    matched = build_matched_validation_comparisons(predictions)
    stats = build_bootstrap_permutation_results(matched)
    pooled.to_csv(output_dir / "pooled_oof_metrics.csv", index=False)
    matched.to_csv(output_dir / "matched_validation_comparison.csv", index=False)
    stats.to_csv(output_dir / "bootstrap_permutation_results.csv", index=False)


def build_pooled_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    group_cols = ["dataset", "task_definition", "target_definition", "aggregation_method", "feature_set", "split_strategy", "model"]
    for keys, grp in predictions.groupby(group_cols, dropna=False):
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "prediction_count": int(len(grp)),
                "r2_pooled": float(r2_score(grp["y_true"], grp["y_pred"])),
                "rmse_pooled": float(np.sqrt(mean_squared_error(grp["y_true"], grp["y_pred"]))),
                "mae_pooled": float(mean_absolute_error(grp["y_true"], grp["y_pred"])),
            }
        )
    return pd.DataFrame(rows)


def build_matched_validation_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    comparable = predictions.copy()
    comparable["absolute_error"] = (comparable["y_true"] - comparable["y_pred"]).abs()
    pairs = [("random", "spatial"), ("random", "temporal_forward"), ("spatial", "temporal_forward")]
    group_cols = ["dataset", "task_definition", "target_definition", "aggregation_method", "feature_set", "model"]
    for group_keys, grp in comparable.groupby(group_cols, dropna=False):
        for left, right in pairs:
            left_df = grp[grp["split_strategy"] == left][["sample_id", "absolute_error", "y_true", "y_pred"]].rename(
                columns={"absolute_error": "left_absolute_error", "y_true": "left_y_true", "y_pred": "left_y_pred"}
            )
            right_df = grp[grp["split_strategy"] == right][["sample_id", "absolute_error", "y_true", "y_pred"]].rename(
                columns={"absolute_error": "right_absolute_error", "y_true": "right_y_true", "y_pred": "right_y_pred"}
            )
            matched = left_df.merge(right_df, on="sample_id", how="inner")
            for _, row in matched.iterrows():
                rows.append(
                    {
                        **dict(zip(group_cols, group_keys)),
                        "comparison": f"{left}_vs_{right}",
                        "left_split": left,
                        "right_split": right,
                        "matched_on": "sample_id",
                        "sample_id": row["sample_id"],
                        "left_absolute_error": row["left_absolute_error"],
                        "right_absolute_error": row["right_absolute_error"],
                        "absolute_error_difference": row["right_absolute_error"] - row["left_absolute_error"],
                    }
                )
    return pd.DataFrame(rows)


def build_bootstrap_permutation_results(
    matched: pd.DataFrame,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    rows = []
    group_cols = ["dataset", "task_definition", "target_definition", "aggregation_method", "feature_set", "model", "comparison"]
    for keys, grp in matched.groupby(group_cols, dropna=False):
        diffs = pd.to_numeric(grp["absolute_error_difference"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(diffs) == 0:
            continue
        boot = rng.choice(diffs, size=(n_bootstrap, len(diffs)), replace=True).mean(axis=1)
        observed = float(diffs.mean())
        signs = rng.choice([-1, 1], size=(n_permutations, len(diffs)))
        perm = (signs * diffs).mean(axis=1)
        p_value = float((np.abs(perm) >= abs(observed)).mean())
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "metric": "absolute_error_difference",
                "n_matched": int(len(diffs)),
                "mean_difference": observed,
                "median_difference": float(np.median(diffs)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "permutation_p": p_value,
                "test_family": "paired_bootstrap_and_sign_permutation",
            }
        )
    return pd.DataFrame(rows)


def hash_files(path: Path) -> str:
    hasher = hashlib.sha1()
    files = sorted([p for p in path.rglob("*") if p.is_file()]) if path.exists() else []
    for file_path in files[:500]:
        try:
            hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
            hasher.update(file_path.read_bytes()[:1024 * 1024])
        except Exception:
            continue
    return hasher.hexdigest()


def git_head() -> str:
    head = BASE_DIR / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref_path = BASE_DIR / ".git" / text.split(" ", 1)[1]
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else text
    return text


def git_dirty() -> bool:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=BASE_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def feature_schema_hash(datasets: list[dict]) -> str:
    payload = []
    for dataset in datasets:
        df = dataset["df"]
        for feature_set in MODE_CONFIG["paper"]["feature_sets"]:
            cols = select_feature_columns(df, feature_set)
            if cols:
                payload.append({"dataset": dataset["name"], "feature_set": feature_set, "columns": cols})
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())
