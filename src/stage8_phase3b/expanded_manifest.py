from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .guards import stable_json_hash, validate_output_root
from .manifest import KEY_COLUMNS, PROHIBITED_HIGH_LIMIT_CLAIMS, write_manifest


BASELINE_COLUMNS = ["dataset_id", "split_id", "deployment_condition"]
EPS = 1e-9


@dataclass(frozen=True)
class ExpandedBuildResult:
    manifest: dict[str, Any]
    coverage_rows: list[dict[str, Any]]


def _string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _metric_bundle(frame: pd.DataFrame) -> dict[str, float | None]:
    return {
        "mae": _mean(frame, "mae"),
        "rmse": _mean(frame, "rmse"),
        "r2": _mean(frame, "r2"),
    }


def _metric_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _improves(candidate: float | None, baseline: float | None, metric: str) -> bool:
    if candidate is None or baseline is None:
        return False
    if metric in {"mae", "rmse"}:
        return candidate < baseline - EPS
    if metric == "r2":
        return candidate > baseline + EPS
    return False


def _positivity_tier(candidate: dict[str, float | None], baseline: dict[str, float | None]) -> str:
    improves = {
        "mae": _improves(candidate.get("mae"), baseline.get("mae"), "mae"),
        "rmse": _improves(candidate.get("rmse"), baseline.get("rmse"), "rmse"),
        "r2": _improves(candidate.get("r2"), baseline.get("r2"), "r2"),
    }
    if all(improves.values()):
        return "strong_relative_positive"
    if improves["mae"] or improves["rmse"]:
        return "relative_positive"
    if any(improves.values()):
        return "partial_positive"
    return "no_improvement"


def _mixed_metric_class(candidate: dict[str, float | None], baseline: dict[str, float | None]) -> str:
    improves = [
        _improves(candidate.get("mae"), baseline.get("mae"), "mae"),
        _improves(candidate.get("rmse"), baseline.get("rmse"), "rmse"),
        _improves(candidate.get("r2"), baseline.get("r2"), "r2"),
    ]
    if all(improves):
        return "positive_all_metrics"
    if improves[0] and improves[1]:
        return "positive_error_metrics_mixed_r2"
    if any(improves):
        return "mixed_or_partial"
    return "negative_or_no_improvement"


def _filter(frame: pd.DataFrame, **criteria: Any) -> pd.DataFrame:
    result = frame
    for column, expected in criteria.items():
        if expected in {None, "any"}:
            continue
        result = result[result[column].fillna("").astype(str) == str(expected)]
    return result


def _baseline_jobs(run_inventory: pd.DataFrame, *, dataset_id: str, split_id: str, condition: str) -> pd.DataFrame:
    return _filter(
        run_inventory,
        dataset_id=dataset_id,
        split_id=split_id,
        deployment_condition=condition,
        transfer_strategy="target_scratch",
        missing_modality_method="imputation",
    ).sort_values("seed")


def _matched_baselines_by_seed(run_inventory: pd.DataFrame, jobs: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    matched_frames: list[pd.DataFrame] = []
    matched_ids: list[str] = []
    for _, job in jobs.sort_values("seed").iterrows():
        baseline = _filter(
            run_inventory,
            dataset_id=job["dataset_id"],
            split_id=job["split_id"],
            deployment_condition=job["deployment_condition"],
            transfer_strategy="target_scratch",
            missing_modality_method="imputation",
            seed=job["seed"],
        )
        if baseline.empty:
            return [], pd.DataFrame()
        baseline = baseline.iloc[[0]]
        matched_frames.append(baseline)
        matched_ids.append(str(baseline.iloc[0]["candidate_id"]))
    return matched_ids, pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[str, ...], pd.Series]:
    lookup: dict[tuple[str, ...], pd.Series] = {}
    for _, row in candidates.iterrows():
        key = tuple(_string(row.get(column)) for column in KEY_COLUMNS)
        lookup[key] = row
    return lookup


def _group_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(_string(row.get(column)) for column in KEY_COLUMNS)


def _stable_group_id(prefix: str, jobs: pd.DataFrame) -> str:
    first = jobs.sort_values("seed").iloc[0]
    payload = {column: _string(first.get(column)) for column in KEY_COLUMNS}
    return f"{prefix}::{stable_json_hash(payload)[:12]}"


def _high_limit_from_formal(jobs: pd.DataFrame, *, candidate_row: pd.Series | None) -> tuple[bool, str]:
    if candidate_row is not None and _string(candidate_row.get("high_limit_reasons")):
        return _bool(candidate_row.get("high_limit_flag")), _string(candidate_row.get("high_limit_reasons"))
    if candidate_row is not None and _bool(candidate_row.get("high_limit_flag")):
        return True, "candidate_table_high_limit"
    reasons: list[str] = []
    dataset = _string(jobs.iloc[0].get("dataset_id")) if not jobs.empty else ""
    split = _string(jobs.iloc[0].get("split_id")) if not jobs.empty else ""
    condition = _string(jobs.iloc[0].get("deployment_condition")) if not jobs.empty else ""
    if dataset == "waite":
        reasons.append("waite_constant_or_low_granularity_soil_boundary")
    if dataset == "ROSEWORTHY_HISTORICAL_FARM" or split.startswith("ROLLING_"):
        reasons.append("rolling_or_tiny_test_boundary")
    if condition == "synthetic_no_weather" and jobs["prediction_constant"].astype(str).str.lower().isin(["true", "1"]).any():
        reasons.append("synthetic_no_weather_constant_prediction")
    return bool(reasons), ";".join(reasons)


def _paper_usage(tier: str, high_limit: bool) -> str:
    if high_limit:
        return "boundary_high_limit"
    if tier == "no_improvement":
        return "negative_or_no_improvement"
    return "expanded_replay_candidate"


def _scientific_use(tier: str, high_limit: bool) -> str:
    if high_limit:
        return "boundary_evidence_only"
    if tier == "no_improvement":
        return "negative_or_no_improvement_control"
    return "candidate_method_evidence"


def _cell_from_jobs(
    *,
    jobs: pd.DataFrame,
    run_inventory: pd.DataFrame,
    candidate_lookup: dict[tuple[str, ...], pd.Series],
    user_scenes: Iterable[str],
    scenario_category: str,
) -> dict[str, Any] | None:
    jobs = jobs.sort_values("seed")
    if jobs.empty:
        return None
    matched_ids, baselines = _matched_baselines_by_seed(run_inventory, jobs)
    if not matched_ids or baselines.empty:
        return None
    candidate_metrics = _metric_bundle(jobs)
    baseline_metrics = _metric_bundle(baselines)
    key = _group_key(jobs.iloc[0])
    candidate_row = candidate_lookup.get(key)
    if candidate_row is not None:
        candidate_group_id = _string(candidate_row.get("candidate_group_id"))
        tier = _string(candidate_row.get("positivity_tier")) or _positivity_tier(candidate_metrics, baseline_metrics)
        paper_usage = _string(candidate_row.get("paper_usage")) or _paper_usage(tier, False)
        seed_consistency = _safe_float(candidate_row.get("seed_consistency_mae"))
    else:
        candidate_group_id = _stable_group_id("expanded", jobs)
        tier = _positivity_tier(candidate_metrics, baseline_metrics)
        paper_usage = _paper_usage(tier, False)
        seed_consistency = None
    high_limit, high_limit_reasons = _high_limit_from_formal(jobs, candidate_row=candidate_row)
    if high_limit:
        paper_usage = "boundary_high_limit"
    return {
        "cell_type": "candidate",
        "candidate_group_id": candidate_group_id,
        "all_seed_candidate_ids": jobs["candidate_id"].astype(str).tolist(),
        "seeds": [int(seed) for seed in jobs["seed"].tolist()],
        "dataset_id": _string(jobs.iloc[0]["dataset_id"]),
        "split_id": _string(jobs.iloc[0]["split_id"]),
        "deployment_condition": _string(jobs.iloc[0]["deployment_condition"]),
        "transfer_strategy": _string(jobs.iloc[0]["transfer_strategy"]),
        "missing_modality_method": _string(jobs.iloc[0]["missing_modality_method"]),
        "source_dataset": _string(jobs.iloc[0]["source_dataset"]),
        "source_route": _string(jobs.iloc[0]["source_route"]),
        "scenario_category": scenario_category,
        "user_scenes": sorted(set(str(item) for item in user_scenes)),
        "positivity_tier": tier,
        "mixed_metric_class": _mixed_metric_class(candidate_metrics, baseline_metrics),
        "paper_usage": paper_usage,
        "high_limit_flag": high_limit,
        "high_limit_reasons": high_limit_reasons,
        "original_metrics": {
            **candidate_metrics,
            "base_mae": baseline_metrics.get("mae"),
            "base_rmse": baseline_metrics.get("rmse"),
            "base_r2": baseline_metrics.get("r2"),
            "delta_mae_vs_baseline": _metric_delta(candidate_metrics.get("mae"), baseline_metrics.get("mae")),
            "delta_rmse_vs_baseline": _metric_delta(candidate_metrics.get("rmse"), baseline_metrics.get("rmse")),
            "delta_r2_vs_baseline": _metric_delta(candidate_metrics.get("r2"), baseline_metrics.get("r2")),
            "seed_consistency_mae": seed_consistency,
        },
        "matched_baseline_ids": matched_ids,
        "matched_baseline_metrics": baseline_metrics,
        "replay_scientific_use": _scientific_use(tier, high_limit),
        "boundary_evidence_only": high_limit,
        "prohibited_claims": PROHIBITED_HIGH_LIMIT_CLAIMS if high_limit else [],
        "formal_jobs": jobs.to_dict(orient="records"),
    }


def _baseline_cell(jobs: pd.DataFrame, *, user_scenes: Iterable[str]) -> dict[str, Any]:
    jobs = jobs.sort_values("seed")
    metrics = _metric_bundle(jobs)
    first = jobs.iloc[0]
    return {
        "cell_type": "matched_baseline",
        "candidate_group_id": f"baseline::{first['dataset_id']}::{first['split_id']}::{first['deployment_condition']}",
        "all_seed_candidate_ids": jobs["candidate_id"].astype(str).tolist(),
        "seeds": [int(seed) for seed in jobs["seed"].tolist()],
        "dataset_id": _string(first["dataset_id"]),
        "split_id": _string(first["split_id"]),
        "deployment_condition": _string(first["deployment_condition"]),
        "transfer_strategy": "target_scratch",
        "missing_modality_method": "imputation",
        "source_dataset": "",
        "source_route": "",
        "scenario_category": "matched_baseline",
        "user_scenes": sorted(set(str(item) for item in user_scenes)),
        "positivity_tier": "matched_baseline",
        "mixed_metric_class": "matched_baseline",
        "paper_usage": "matched_baseline",
        "high_limit_flag": False,
        "high_limit_reasons": "",
        "original_metrics": metrics,
        "matched_baseline_ids": jobs["candidate_id"].astype(str).tolist(),
        "matched_baseline_metrics": metrics,
        "replay_scientific_use": "matched_baseline",
        "boundary_evidence_only": False,
        "prohibited_claims": [],
        "formal_jobs": jobs.to_dict(orient="records"),
    }


def _coverage_row(
    *,
    user_scene: str,
    required: bool,
    scenario_category: str,
    dataset_id: str,
    split_id: str,
    condition: str,
    transfer_strategy: str,
    method: str,
    source_dataset: str = "",
    source_route: str = "",
    status: str,
    jobs: pd.DataFrame | None = None,
    baselines: pd.DataFrame | None = None,
    cell: dict[str, Any] | None = None,
    searched_aliases: str = "",
    caveats: str = "",
) -> dict[str, Any]:
    jobs = pd.DataFrame() if jobs is None else jobs
    baselines = pd.DataFrame() if baselines is None else baselines
    metrics = _metric_bundle(jobs) if not jobs.empty else {"mae": None, "rmse": None, "r2": None}
    baseline_metrics = _metric_bundle(baselines) if not baselines.empty else {"mae": None, "rmse": None, "r2": None}
    return {
        "user_scene": user_scene,
        "required": required,
        "scenario_category": scenario_category,
        "found_status": status,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "deployment_condition": condition,
        "transfer_strategy": transfer_strategy,
        "missing_modality_method": method,
        "source_dataset": source_dataset,
        "source_route": source_route,
        "searched_aliases": searched_aliases,
        "candidate_group_id": "" if cell is None else cell.get("candidate_group_id", ""),
        "candidate_ids": ";".join(jobs["candidate_id"].astype(str).tolist()) if not jobs.empty else "",
        "matched_baseline_ids": ";".join(baselines["candidate_id"].astype(str).tolist()) if not baselines.empty else "",
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "baseline_mae": baseline_metrics.get("mae"),
        "baseline_rmse": baseline_metrics.get("rmse"),
        "baseline_r2": baseline_metrics.get("r2"),
        "positivity_tier": "" if cell is None else cell.get("positivity_tier", ""),
        "mixed_metric_class": "" if cell is None else cell.get("mixed_metric_class", ""),
        "high_limit_flag": "" if cell is None else cell.get("high_limit_flag", ""),
        "high_limit_reasons": "" if cell is None else cell.get("high_limit_reasons", ""),
        "caveats": caveats,
    }


def _append_cell(cells_by_key: dict[tuple[str, ...], dict[str, Any]], cell: dict[str, Any] | None) -> None:
    if cell is None:
        return
    key = tuple(str(cell.get(column, "")) for column in KEY_COLUMNS)
    if key in cells_by_key:
        existing = cells_by_key[key]
        existing["user_scenes"] = sorted(set(existing.get("user_scenes", [])) | set(cell.get("user_scenes", [])))
        return
    cells_by_key[key] = cell


def _add_baseline(
    *,
    baseline_cells_by_key: dict[tuple[str, str, str], dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    run_inventory: pd.DataFrame,
    user_scene: str,
    dataset_id: str,
    split_id: str,
    condition: str,
    required: bool,
) -> pd.DataFrame:
    jobs = _baseline_jobs(run_inventory, dataset_id=dataset_id, split_id=split_id, condition=condition)
    key = (dataset_id, split_id, condition)
    if jobs.empty:
        coverage_rows.append(
            _coverage_row(
                user_scene=f"{user_scene}::matched_baseline",
                required=required,
                scenario_category="matched_baseline",
                dataset_id=dataset_id,
                split_id=split_id,
                condition=condition,
                transfer_strategy="target_scratch",
                method="imputation",
                status="missing_baseline",
                searched_aliases=f"{dataset_id}|{split_id}|{condition}|target_scratch|imputation",
            )
        )
        return jobs
    if key not in baseline_cells_by_key:
        baseline_cells_by_key[key] = _baseline_cell(jobs, user_scenes=[user_scene])
    else:
        baseline_cells_by_key[key]["user_scenes"] = sorted(set(baseline_cells_by_key[key]["user_scenes"]) | {user_scene})
    coverage_rows.append(
        _coverage_row(
            user_scene=f"{user_scene}::matched_baseline",
            required=required,
            scenario_category="matched_baseline",
            dataset_id=dataset_id,
            split_id=split_id,
            condition=condition,
            transfer_strategy="target_scratch",
            method="imputation",
            status="found",
            jobs=jobs,
            baselines=jobs,
            cell=baseline_cells_by_key[key],
        )
    )
    return jobs


def _group_transfer_rows(frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
    columns = KEY_COLUMNS
    if frame.empty:
        return []
    return [group.sort_values("seed") for _, group in frame.groupby(columns, dropna=False)]


def _source_routes(frame: pd.DataFrame, configured: Any) -> list[str]:
    routes = sorted(frame["source_route"].fillna("").astype(str).unique().tolist())
    if configured in {None, "all"}:
        return routes
    requested = [str(item) for item in configured]
    return [route for route in routes if route in requested]


def _build_local_scenes(
    *,
    config: dict[str, Any],
    run_inventory: pd.DataFrame,
    candidate_lookup: dict[tuple[str, ...], pd.Series],
    cells_by_key: dict[tuple[str, ...], dict[str, Any]],
    baseline_cells_by_key: dict[tuple[str, str, str], dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    local = config.get("local", {})
    dataset_id = str(local["dataset_id"])
    methods = dict(local.get("methods", {}))
    for scene in local.get("scenes", []):
        user_scene = str(scene["user_scene"])
        split_id = str(scene["split_id"])
        condition = str(scene["deployment_condition"])
        baseline_jobs = _add_baseline(
            baseline_cells_by_key=baseline_cells_by_key,
            coverage_rows=coverage_rows,
            run_inventory=run_inventory,
            user_scene=user_scene,
            dataset_id=dataset_id,
            split_id=split_id,
            condition=condition,
            required=True,
        )
        for method_label in scene.get("methods", []):
            method = str(methods.get(method_label, method_label))
            jobs = _filter(
                run_inventory,
                dataset_id=dataset_id,
                split_id=split_id,
                deployment_condition=condition,
                transfer_strategy="target_scratch",
                missing_modality_method=method,
            ).sort_values("seed")
            if jobs.empty or baseline_jobs.empty:
                coverage_rows.append(
                    _coverage_row(
                        user_scene=user_scene,
                        required=True,
                        scenario_category="local",
                        dataset_id=dataset_id,
                        split_id=split_id,
                        condition=condition,
                        transfer_strategy="target_scratch",
                        method=method,
                        status="missing_candidate" if jobs.empty else "missing_baseline",
                        jobs=jobs,
                        baselines=baseline_jobs,
                        searched_aliases=f"{dataset_id}|{split_id}|{condition}|target_scratch|{method}",
                    )
                )
                continue
            cell = _cell_from_jobs(
                jobs=jobs,
                run_inventory=run_inventory,
                candidate_lookup=candidate_lookup,
                user_scenes=[user_scene],
                scenario_category="local",
            )
            _append_cell(cells_by_key, cell)
            coverage_rows.append(
                _coverage_row(
                    user_scene=user_scene,
                    required=True,
                    scenario_category="local",
                    dataset_id=dataset_id,
                    split_id=split_id,
                    condition=condition,
                    transfer_strategy="target_scratch",
                    method=method,
                    status="found" if cell else "missing_matched_baseline",
                    jobs=jobs,
                    baselines=baseline_jobs,
                    cell=cell,
                )
            )


def _add_transfer_scene(
    *,
    user_scene: str,
    required: bool,
    scenario_category: str,
    dataset_id: str,
    split_id: str,
    condition: str,
    source_dataset: str,
    transfer_strategy: str,
    missing_modality_method: str,
    route_filter: Any,
    run_inventory: pd.DataFrame,
    candidate_lookup: dict[tuple[str, ...], pd.Series],
    cells_by_key: dict[tuple[str, ...], dict[str, Any]],
    baseline_cells_by_key: dict[tuple[str, str, str], dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    caveats: str = "",
) -> None:
    baseline_jobs = _add_baseline(
        baseline_cells_by_key=baseline_cells_by_key,
        coverage_rows=coverage_rows,
        run_inventory=run_inventory,
        user_scene=user_scene,
        dataset_id=dataset_id,
        split_id=split_id,
        condition=condition,
        required=required,
    )
    frame = _filter(
        run_inventory,
        dataset_id=dataset_id,
        split_id=split_id,
        deployment_condition=condition,
        transfer_strategy=transfer_strategy,
        missing_modality_method=missing_modality_method,
        source_dataset=source_dataset,
    )
    routes = _source_routes(frame, route_filter)
    if not routes:
        coverage_rows.append(
            _coverage_row(
                user_scene=user_scene,
                required=required,
                scenario_category=scenario_category,
                dataset_id=dataset_id,
                split_id=split_id,
                condition=condition,
                transfer_strategy=transfer_strategy,
                method=missing_modality_method,
                source_dataset=source_dataset,
                status="missing_candidate",
                jobs=frame,
                baselines=baseline_jobs,
                searched_aliases=f"{dataset_id}|{split_id}|{condition}|{transfer_strategy}|{missing_modality_method}|{source_dataset}|{route_filter}",
                caveats=caveats,
            )
        )
        return
    for route in routes:
        jobs = frame[frame["source_route"].fillna("").astype(str) == route].sort_values("seed")
        cell = None if baseline_jobs.empty else _cell_from_jobs(
            jobs=jobs,
            run_inventory=run_inventory,
            candidate_lookup=candidate_lookup,
            user_scenes=[user_scene],
            scenario_category=scenario_category,
        )
        _append_cell(cells_by_key, cell)
        coverage_rows.append(
            _coverage_row(
                user_scene=user_scene,
                required=required,
                scenario_category=scenario_category,
                dataset_id=dataset_id,
                split_id=split_id,
                condition=condition,
                transfer_strategy=transfer_strategy,
                method=missing_modality_method,
                source_dataset=source_dataset,
                source_route=route,
                status="found" if cell else "missing_matched_baseline",
                jobs=jobs,
                baselines=baseline_jobs,
                cell=cell,
                caveats=caveats,
            )
        )


def _build_transfer_scenes(
    *,
    config: dict[str, Any],
    run_inventory: pd.DataFrame,
    candidate_lookup: dict[tuple[str, ...], pd.Series],
    cells_by_key: dict[tuple[str, ...], dict[str, Any]],
    baseline_cells_by_key: dict[tuple[str, str, str], dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    transfer = config.get("transfer", {})
    source_dataset = str(transfer["source_dataset"])
    transfer_strategy = str(transfer.get("transfer_strategy", "ordinary_transfer"))
    method = str(transfer.get("missing_modality_method", "missing_aware"))
    route_filter = transfer.get("include_source_routes", "all")
    cy = transfer.get("cybench_wheat_au", {})
    for item in cy.get("required", []):
        _add_transfer_scene(
            user_scene=str(item["user_scene"]),
            required=True,
            scenario_category="transfer_required",
            dataset_id=str(cy["dataset_id"]),
            split_id=str(item["split_id"]),
            condition=str(item["deployment_condition"]),
            source_dataset=source_dataset,
            transfer_strategy=transfer_strategy,
            missing_modality_method=method,
            route_filter=route_filter,
            run_inventory=run_inventory,
            candidate_lookup=candidate_lookup,
            cells_by_key=cells_by_key,
            baseline_cells_by_key=baseline_cells_by_key,
            coverage_rows=coverage_rows,
        )
    for item in cy.get("negative_or_no_improvement", []):
        split_id = str(item["split_id"])
        conditions = (
            sorted(
                run_inventory[
                    (run_inventory["dataset_id"].astype(str) == str(cy["dataset_id"]))
                    & (run_inventory["split_id"].astype(str) == split_id)
                ]["deployment_condition"].fillna("").astype(str).unique().tolist()
            )
            if str(item.get("deployment_condition")) == "any"
            else [str(item["deployment_condition"])]
        )
        for condition in conditions:
            _add_transfer_scene(
                user_scene=str(item["user_scene"]),
                required=True,
                scenario_category="transfer_negative_or_no_improvement",
                dataset_id=str(cy["dataset_id"]),
                split_id=split_id,
                condition=condition,
                source_dataset=source_dataset,
                transfer_strategy=transfer_strategy,
                missing_modality_method=method,
                route_filter=route_filter,
                run_inventory=run_inventory,
                candidate_lookup=candidate_lookup,
                cells_by_key=cells_by_key,
                baseline_cells_by_key=baseline_cells_by_key,
                coverage_rows=coverage_rows,
                caveats="selected_by_formal_negative_or_no_improvement_scene_request",
            )
    rose = transfer.get("roseworthy_e5_point", {})
    for forbidden in rose.get("forbidden_dataset_ids", []):
        if any(row.get("dataset_id") == forbidden for row in coverage_rows):
            raise ValueError(f"FORBIDDEN_ROSEWORTHY_DATASET_IN_COVERAGE:{forbidden}")
    for item in rose.get("required", []):
        _add_transfer_scene(
            user_scene=str(item["user_scene"]),
            required=True,
            scenario_category="transfer_roseworthy_e5_point",
            dataset_id=str(rose["dataset_id"]),
            split_id=str(item["split_id"]),
            condition=str(item["deployment_condition"]),
            source_dataset=source_dataset,
            transfer_strategy=transfer_strategy,
            missing_modality_method=method,
            route_filter=route_filter,
            run_inventory=run_inventory,
            candidate_lookup=candidate_lookup,
            cells_by_key=cells_by_key,
            baseline_cells_by_key=baseline_cells_by_key,
            coverage_rows=coverage_rows,
            caveats="roseworthy_e5_point_only_not_historical_farm",
        )
    for item in rose.get("include_if_present", []):
        _add_transfer_scene(
            user_scene=str(item["user_scene"]),
            required=False,
            scenario_category="transfer_roseworthy_e5_point",
            dataset_id=str(rose["dataset_id"]),
            split_id=str(item["split_id"]),
            condition=str(item["deployment_condition"]),
            source_dataset=source_dataset,
            transfer_strategy=transfer_strategy,
            missing_modality_method=method,
            route_filter=route_filter,
            run_inventory=run_inventory,
            candidate_lookup=candidate_lookup,
            cells_by_key=cells_by_key,
            baseline_cells_by_key=baseline_cells_by_key,
            coverage_rows=coverage_rows,
            caveats="optional_if_formally_present;roseworthy_e5_point_only_not_historical_farm",
        )
    waite = transfer.get("waite", {})
    waite_caveats = ";".join(str(item) for item in waite.get("caveats", []))
    for item in waite.get("required", []):
        _add_transfer_scene(
            user_scene=str(item["user_scene"]),
            required=True,
            scenario_category="transfer_waite_boundary",
            dataset_id=str(waite["dataset_id"]),
            split_id=str(waite["split_id"]),
            condition=str(item["deployment_condition"]),
            source_dataset=source_dataset,
            transfer_strategy=transfer_strategy,
            missing_modality_method=method,
            route_filter=route_filter,
            run_inventory=run_inventory,
            candidate_lookup=candidate_lookup,
            cells_by_key=cells_by_key,
            baseline_cells_by_key=baseline_cells_by_key,
            coverage_rows=coverage_rows,
            caveats=waite_caveats,
        )


def _select_smoke_ids(manifest: dict[str, Any]) -> list[str]:
    rows: list[dict[str, Any]] = [*manifest.get("baseline_cells", []), *manifest.get("cells", [])]
    selected: list[str] = []

    def add_first(predicate) -> None:
        for cell in rows:
            if predicate(cell):
                for cid in cell.get("all_seed_candidate_ids", []):
                    if cid not in selected:
                        selected.append(str(cid))
                        return

    add_first(lambda cell: cell.get("cell_type") == "matched_baseline" and cell.get("dataset_id") == "CY-Bench_wheat_AU")
    add_first(lambda cell: cell.get("transfer_strategy") == "target_scratch" and cell.get("missing_modality_method") in {"late_fusion", "teacher_student_m2"})
    add_first(lambda cell: cell.get("transfer_strategy") == "ordinary_transfer" and cell.get("dataset_id") == "CY-Bench_wheat_AU")
    add_first(lambda cell: cell.get("dataset_id") == "ROSEWORTHY_E5_POINT")
    add_first(lambda cell: cell.get("dataset_id") == "waite")
    return selected


def _write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_expanded_manifest_from_sources(
    *,
    scenario_config: Path,
    candidate_csv: Path,
    run_inventory_csv: Path,
    formal_campaign: Path,
    output_root: Path,
    writable_root: Path | None = None,
) -> ExpandedBuildResult:
    writable = Path(writable_root) if writable_root is not None else Path(__file__).resolve().parents[2]
    output = validate_output_root(output_root=Path(output_root), formal_campaign=Path(formal_campaign), writable_root=writable)
    config = yaml.safe_load(Path(scenario_config).read_text())
    run_inventory = pd.read_csv(run_inventory_csv).fillna("")
    candidates = pd.read_csv(candidate_csv).fillna("")
    candidate_lookup = _candidate_lookup(candidates)
    cells_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    baseline_cells_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []

    _build_local_scenes(
        config=config,
        run_inventory=run_inventory,
        candidate_lookup=candidate_lookup,
        cells_by_key=cells_by_key,
        baseline_cells_by_key=baseline_cells_by_key,
        coverage_rows=coverage_rows,
    )
    _build_transfer_scenes(
        config=config,
        run_inventory=run_inventory,
        candidate_lookup=candidate_lookup,
        cells_by_key=cells_by_key,
        baseline_cells_by_key=baseline_cells_by_key,
        coverage_rows=coverage_rows,
    )

    cells = list(cells_by_key.values())
    baseline_cells = list(baseline_cells_by_key.values())
    all_ids = list(dict.fromkeys([cid for cell in [*cells, *baseline_cells] for cid in cell.get("all_seed_candidate_ids", [])]))
    high_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in cells if cell.get("high_limit_flag"))
    non_high_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in cells if not cell.get("high_limit_flag"))
    baseline_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in baseline_cells)
    manifest = {
        "schema_version": "stage8_phase3b_expanded_replay_manifest_v1",
        "matrix": "expanded_user_specified",
        "formal_campaign": str(Path(formal_campaign).resolve()),
        "output_root": str(output),
        "scenario_config": str(Path(scenario_config).resolve()),
        "candidate_csv": str(Path(candidate_csv).resolve()),
        "run_inventory_csv": str(Path(run_inventory_csv).resolve()),
        "coverage_csv": str((writable / "docs/stage8_paper/phase4b_user_scenario_coverage.csv").resolve()),
        "expected_target_jobs": len(all_ids),
        "job_counts": {
            "candidate_cells": len(cells),
            "baseline_cells": len(baseline_cells),
            "high_limit_jobs": high_jobs,
            "non_high_limit_jobs": non_high_jobs,
            "baseline_jobs": baseline_jobs,
            "unique_target_jobs": len(all_ids),
            "coverage_rows": len(coverage_rows),
        },
        "cells": cells,
        "baseline_cells": baseline_cells,
        "all_formal_candidate_ids": all_ids,
        "subsets": {},
        "implementation_fidelity_requirements": {
            "must_call_original_stage8_v4_handlers": True,
            "algorithm_reimplementation_allowed": False,
            "persistence_hooks_must_not_change_numeric_execution": True,
            "expanded_builder_only_selects_formal_jobs": True,
        },
    }
    manifest["subsets"]["smoke"] = _select_smoke_ids(manifest)
    manifest["manifest_hash"] = stable_json_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    return ExpandedBuildResult(manifest=manifest, coverage_rows=coverage_rows)


def write_expanded_outputs(result: ExpandedBuildResult, *, output_root: Path, coverage_csv: Path) -> tuple[Path, Path, Path]:
    manifest_path, hash_path = write_manifest(result.manifest, output_root)
    _write_coverage_csv(coverage_csv, result.coverage_rows)
    return manifest_path, hash_path, coverage_csv


def validate_expanded_manifest(manifest: dict[str, Any], coverage_rows: list[dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "stage8_phase3b_expanded_replay_manifest_v1":
        errors.append("schema_version_not_expanded_v1")
    if not manifest.get("subsets", {}).get("smoke"):
        errors.append("smoke_subset_missing")
    ids = set(str(item) for item in manifest.get("all_formal_candidate_ids", []))
    for item in manifest.get("subsets", {}).get("smoke", []):
        if str(item) not in ids:
            errors.append(f"smoke_id_not_in_manifest:{item}")
    for cell in manifest.get("cells", []):
        if cell.get("dataset_id") == "ROSEWORTHY_HISTORICAL_FARM":
            errors.append("forbidden_roseworthy_historical_farm_cell")
        if len(cell.get("all_seed_candidate_ids", [])) != len(cell.get("matched_baseline_ids", [])):
            errors.append(f"baseline_count_mismatch:{cell.get('candidate_group_id')}")
        if cell.get("transfer_strategy") == "ordinary_transfer" and not cell.get("source_route"):
            errors.append(f"missing_source_route:{cell.get('candidate_group_id')}")
    if coverage_rows is not None:
        for row in coverage_rows:
            if _bool(row.get("required")) and str(row.get("found_status")) != "found":
                errors.append(f"required_scene_missing:{row.get('user_scene')}:{row.get('found_status')}")
            if row.get("dataset_id") == "ROSEWORTHY_HISTORICAL_FARM":
                errors.append(f"forbidden_roseworthy_historical_farm_coverage:{row.get('user_scene')}")
    classes = {str(cell.get("mixed_metric_class")) for cell in manifest.get("cells", [])}
    if "negative_or_no_improvement" not in classes:
        errors.append("negative_or_no_improvement_cells_missing")
    if not any(cell.get("high_limit_flag") for cell in manifest.get("cells", [])):
        errors.append("high_limit_cells_missing")
    return errors
