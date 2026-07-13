from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .explanation_consistency import (
    ExplanationObjectiveConfig,
    train_student_with_explanation_objective,
    write_json_atomic,
)
from .formal_bridge import FormalCodeBridge, ensure_source_dependency_alias
from .guards import stable_json_hash, validate_output_root
from .status import request_stop, stop_requested


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_CAMPAIGN = (
    ROOT.parent
    / "stage8-parallel-execution"
    / "outputs"
    / "distillation_program"
    / "stage8_v4_australian_narrative_v1"
    / "universal_weather_full_campaign_v2_parallel_v1"
)
DEFAULT_RUN_INVENTORY = ROOT.parent / "stage8-parallel-execution/docs/stage8_final_result_audit/run_inventory.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/stage8_explanation_consistency_optimisation_v2_selective"
TMP_ROOT = Path("/private/tmp")
V1_VARIANTS = {"prediction_only", "gradient_regularisation", "explanation_consistency", "sparsity_only", "consistency_sparsity"}
V2_VARIANTS = {
    "prediction_only",
    "gradient_regularisation",
    "explanation_consistency",
    "sparsity_only",
    "consistency_sparsity",
    "fidelity_weighted_consistency",
    "prediction_aware_selective_consistency",
    "deployment_contract_aware_consistency",
    "fidelity_selective_consistency",
    "deployment_fidelity_consistency",
}
V2_VIEWS = ["gaussian", "dropout", "modality_respecting"]
SMOKE_VARIANTS = {
    "prediction_only",
    "explanation_consistency",
    "fidelity_weighted_consistency",
    "prediction_aware_selective_consistency",
    "fidelity_selective_consistency",
}


def _scenario_items(matrix: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    scenarios = list(matrix["scenarios"])
    return scenarios[:2] if smoke else scenarios


def _variant_items(matrix: dict[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    variants = list(matrix["objective_variants"])
    if not smoke:
        return variants
    return [variant for variant in variants if variant.get("variant") in SMOKE_VARIANTS]


@dataclass(frozen=True)
class RunSpec:
    phase: str
    scenario_id: str
    dataset_id: str
    split_id: str
    deployment_condition: str
    initialisation: str
    source_dataset: str | None
    source_route: str | None
    seed: int
    variant: str
    objective: dict[str, Any]

    @property
    def run_id(self) -> str:
        return stable_json_hash(asdict(self))[:16]


def read_json(path: Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def matrix_hash(matrix: dict[str, Any]) -> str:
    return stable_json_hash(matrix)


def _route_candidates(inventory: pd.DataFrame, scenario: dict[str, str]) -> pd.DataFrame:
    subset = inventory[
        (inventory["dataset_id"].astype(str) == scenario["dataset_id"])
        & (inventory["split_id"].astype(str) == scenario["split_id"])
        & (inventory["deployment_condition"].astype(str) == scenario["deployment_condition"])
        & (inventory["transfer_strategy"].astype(str) == "ordinary_transfer")
        & (inventory["missing_modality_method"].astype(str) == "missing_aware")
        & (inventory["metrics_valid"].astype(bool))
    ]
    grouped = (
        subset.groupby(["source_dataset", "source_route"], dropna=False)
        .agg(seed_count=("seed", "nunique"), mean_mae=("mae", "mean"), candidate_ids=("candidate_id", lambda values: ";".join(map(str, values))))
        .reset_index()
    )
    grouped = grouped[grouped["seed_count"] >= 3].sort_values(["mean_mae", "source_dataset", "source_route"])
    if grouped.empty:
        raise ValueError(f"NO_SOURCE_PRETRAINED_ROUTE_FOR_SCENARIO:{scenario['scenario_id']}")
    return grouped


def _has_target_baseline(inventory: pd.DataFrame, scenario: dict[str, str]) -> bool:
    subset = inventory[
        (inventory["dataset_id"].astype(str) == scenario["dataset_id"])
        & (inventory["split_id"].astype(str) == scenario["split_id"])
        & (inventory["deployment_condition"].astype(str) == scenario["deployment_condition"])
        & (inventory["transfer_strategy"].astype(str) == "target_scratch")
        & (inventory["missing_modality_method"].astype(str).isin(["imputation", "missing_indicators"]))
        & (inventory["metrics_valid"].astype(bool))
    ]
    return int(subset["seed"].nunique()) >= 3


def _lock_third_scenario(inventory: pd.DataFrame, existing: list[dict[str, str]]) -> dict[str, str]:
    existing_keys = {(item["dataset_id"], item["split_id"], item["deployment_condition"]) for item in existing}
    candidates = [
        {
            "scenario_id": "cybench_wheat_au_temporal_complete",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "ROLLING_2018",
            "deployment_condition": "complete",
            "role": "auto_locked_temporal_if_formally_available",
        },
        {
            "scenario_id": "cybench_wheat_au_spatial_random_missing",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "SPATIAL",
            "deployment_condition": "synthetic_random_0_15",
            "role": "auto_locked_missing_modality_contract",
        },
        {
            "scenario_id": "cybench_wheat_au_group_random_missing",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "GROUP",
            "deployment_condition": "synthetic_random_0_15",
            "role": "auto_locked_missing_modality_contract",
        },
        {
            "scenario_id": "cybench_wheat_au_spatial_no_soil",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "SPATIAL",
            "deployment_condition": "synthetic_no_soil",
            "role": "auto_locked_missing_modality_contract",
        },
    ]
    for candidate in candidates:
        if (candidate["dataset_id"], candidate["split_id"], candidate["deployment_condition"]) in existing_keys:
            continue
        if "waite" in candidate["dataset_id"].lower() or "roseworthy" in candidate["dataset_id"].lower():
            continue
        try:
            _route_candidates(inventory, candidate)
        except ValueError:
            continue
        if _has_target_baseline(inventory, candidate):
            return candidate
    raise ValueError("NO_LEGAL_THIRD_DEPLOYMENT_SCENARIO_FOUND")


def _grid(*, variant: str, views: list[str], matrix_version: str) -> list[dict[str, Any]]:
    if matrix_version == "v1":
        if variant == "prediction_only":
            return [{"lambda_g": 0.0, "lambda_c": 0.0, "lambda_s": 0.0, "perturbation_scale": 0.02}]
        if variant == "gradient_regularisation":
            return [
                {"lambda_g": 1e-4, "lambda_c": 0.0, "lambda_s": 0.0, "perturbation_scale": 0.02},
                {"lambda_g": 1e-3, "lambda_c": 0.0, "lambda_s": 0.0, "perturbation_scale": 0.02},
            ]
        if variant == "explanation_consistency":
            return [
                {"lambda_g": 0.0, "lambda_c": 0.1, "lambda_s": 0.0, "perturbation_scale": 0.01},
                {"lambda_g": 0.0, "lambda_c": 0.25, "lambda_s": 0.0, "perturbation_scale": 0.02},
            ]
        if variant == "sparsity_only":
            return [
                {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_s": 0.01, "perturbation_scale": 0.02},
                {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_s": 0.05, "perturbation_scale": 0.02},
            ]
        if variant == "consistency_sparsity":
            return [
                {"lambda_g": 0.0, "lambda_c": 0.1, "lambda_s": 0.01, "perturbation_scale": 0.02},
                {"lambda_g": 0.0, "lambda_c": 0.25, "lambda_s": 0.01, "perturbation_scale": 0.02},
            ]
    base_by_variant = {
        "prediction_only": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0},
        "gradient_regularisation": {"lambda_g": 1e-4, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0},
        "explanation_consistency": {"lambda_g": 0.0, "lambda_c": 0.1, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0},
        "sparsity_only": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.01},
        "consistency_sparsity": {"lambda_g": 0.0, "lambda_c": 0.1, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.01},
        "fidelity_weighted_consistency": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.1, "lambda_p": 0.0, "lambda_s": 0.0},
        "prediction_aware_selective_consistency": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.1, "lambda_s": 0.0},
        "deployment_contract_aware_consistency": {"lambda_g": 0.0, "lambda_c": 0.1, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0},
        "fidelity_selective_consistency": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.1, "lambda_p": 0.1, "lambda_s": 0.0},
        "deployment_fidelity_consistency": {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.1, "lambda_p": 0.0, "lambda_s": 0.0},
    }
    scales = [0.01, 0.02, 0.05]
    grid: list[dict[str, Any]] = []
    for index, view in enumerate(views):
        params = dict(base_by_variant[variant])
        params.update(
            {
                "view": view,
                "perturbation_scale": scales[index],
                "fidelity_temperature": 1.0,
                "prediction_temperature": 1.0,
                "grid_index": index,
            }
        )
        grid.append(params)
    return grid


def build_default_matrix(*, run_inventory_csv: Path = DEFAULT_RUN_INVENTORY, matrix_version: str = "v2") -> dict[str, Any]:
    inventory = pd.read_csv(run_inventory_csv)
    scenarios = [
        {
            "scenario_id": "cybench_wheat_au_spatial_complete",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "SPATIAL",
            "deployment_condition": "complete",
            "role": "primary_spatial_weather_available",
        },
        {
            "scenario_id": "cybench_wheat_au_group_no_soil",
            "dataset_id": "CY-Bench_wheat_AU",
            "split_id": "GROUP",
            "deployment_condition": "synthetic_no_soil",
            "role": "second_fixed_deployment_contract",
        },
    ]
    if matrix_version == "v2":
        scenarios.append(_lock_third_scenario(inventory, scenarios))
    selected_routes: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        best = _route_candidates(inventory, scenario).iloc[0].to_dict()
        selected_routes[scenario["scenario_id"]] = {
            "source_dataset": str(best["source_dataset"]),
            "source_route": str(best["source_route"]),
            "formal_candidate_ids": str(best["candidate_ids"]).split(";"),
            "formal_mean_mae": float(best["mean_mae"]),
            "selection_scope": "pre_existing_formal_inventory_validation_proxy",
        }
    variant_names = list(V1_VARIANTS) if matrix_version == "v1" else [
        "prediction_only",
        "gradient_regularisation",
        "explanation_consistency",
        "sparsity_only",
        "consistency_sparsity",
        "fidelity_weighted_consistency",
        "prediction_aware_selective_consistency",
        "deployment_contract_aware_consistency",
        "fidelity_selective_consistency",
        "deployment_fidelity_consistency",
    ]
    views = ["gaussian"] if matrix_version == "v1" else V2_VIEWS
    variants = [{"variant": name, "tuning_grid": _grid(variant=name, views=views, matrix_version=matrix_version)} for name in variant_names]
    matrix = {
        "schema_version": f"stage8_explanation_consistency_matrix_{matrix_version}",
        "formal_run_inventory": str(Path(run_inventory_csv).resolve()),
        "scenarios": scenarios,
        "view_constructions": views,
        "initialisations": [
            {"initialisation": "target_scratch", "selected_routes": {}},
            {"initialisation": "source_pretrained", "selected_routes": selected_routes},
        ],
        "objective_variants": variants,
        "seeds": [101, 202, 303],
        "training_budget": {
            "epochs": 80,
            "patience": 12,
            "batch_size": 256,
            "target_scratch_learning_rate": 1e-3,
            "source_pretrained_learning_rate": 2e-4,
            "weight_decay": 1e-4,
        },
        "tuning_policy": {
            "selection_split": "validation",
            "selection_metric": "validation_rmse_squared",
            "test_used_for_selection": False,
            "same_grid_budget_per_variant": matrix_version == "v2",
            "selection_rule": "seed_aggregated_mean_validation_rmse_squared",
            "note": "V2 uses equal compact grids and selects by mean validation score across available tuning seeds.",
        },
    }
    matrix["matrix_hash"] = matrix_hash(matrix)
    return matrix


def write_matrix(matrix: dict[str, Any], output_root: Path) -> Path:
    control = Path(output_root) / "control"
    control.mkdir(parents=True, exist_ok=True)
    path = control / "explanation_consistency_matrix.json"
    payload = dict(matrix)
    payload["matrix_hash"] = matrix_hash({key: value for key, value in matrix.items() if key != "matrix_hash"})
    write_json_atomic(path, payload)
    (control / "explanation_consistency_matrix.sha256").write_text(payload["matrix_hash"] + "\n")
    return path


def load_or_build_matrix(output_root: Path, run_inventory_csv: Path = DEFAULT_RUN_INVENTORY) -> dict[str, Any]:
    path = Path(output_root) / "control" / "explanation_consistency_matrix.json"
    if path.is_file():
        return read_json(path)
    matrix = build_default_matrix(run_inventory_csv=run_inventory_csv)
    write_matrix(matrix, output_root)
    return matrix


def validate_matrix(matrix: dict[str, Any], *, formal_campaign: Path, output_root: Path, writable_root: Path) -> dict[str, Any]:
    output = validate_output_root(output_root=output_root, formal_campaign=formal_campaign, writable_root=writable_root)
    errors: list[str] = []
    schema = str(matrix.get("schema_version"))
    if schema not in {"stage8_explanation_consistency_matrix_v1", "stage8_explanation_consistency_matrix_v2"}:
        errors.append("BAD_MATRIX_SCHEMA")
    scenarios = matrix.get("scenarios", [])
    expected_scenarios = 3 if schema.endswith("_v2") else 2
    if len(scenarios) != expected_scenarios:
        errors.append(f"EXPECTED_{expected_scenarios}_SCENARIOS")
    if schema.endswith("_v2"):
        for scenario in scenarios:
            dataset = str(scenario.get("dataset_id", "")).lower()
            if "waite" in dataset or "roseworthy" in dataset:
                errors.append("V2_MAIN_OPTIMISATION_FORBIDS_WAITE_OR_ROSEWORTHY")
        if set(matrix.get("view_constructions", [])) != set(V2_VIEWS):
            errors.append("V2_VIEW_CONSTRUCTIONS_MISMATCH")
    if matrix.get("seeds") != [101, 202, 303]:
        errors.append("EXPECTED_FIXED_SEEDS_101_202_303")
    variants = {item.get("variant") for item in matrix.get("objective_variants", [])}
    expected = V2_VARIANTS if schema.endswith("_v2") else V1_VARIANTS
    if variants != expected:
        errors.append("OBJECTIVE_VARIANT_SET_MISMATCH")
    if schema.endswith("_v2"):
        grid_lengths = {len(item.get("tuning_grid", [])) for item in matrix.get("objective_variants", [])}
        if len(grid_lengths) != 1:
            errors.append("V2_REQUIRES_EQUAL_GRID_BUDGETS")
    source_routes = next(item for item in matrix["initialisations"] if item["initialisation"] == "source_pretrained")["selected_routes"]
    missing_sources: list[str] = []
    for scenario in scenarios:
        route = source_routes[scenario["scenario_id"]]
        for seed in matrix["seeds"]:
            source_dir = (
                Path(formal_campaign)
                / "jobs"
                / "source"
                / str(route["source_dataset"])
                / str(route["source_route"])
                / f"seed_{int(seed)}"
            )
            if not (source_dir / "student.pt").is_file():
                missing_sources.append(str(source_dir))
    status = "PASS" if not errors and not missing_sources else "FAIL"
    return {
        "status": status,
        "output_root": str(output),
        "errors": errors,
        "missing_source_checkpoints": missing_sources,
        "scenario_count": len(scenarios),
        "final_run_count": len(scenarios) * len(matrix["initialisations"]) * len(matrix["objective_variants"]) * len(matrix["seeds"]),
        "matrix_version": "v2" if schema.endswith("_v2") else "v1",
    }


def writable_root_for_output(output_root: Path, *, allow_tmp: bool = False) -> Path:
    resolved = Path(output_root).expanduser().resolve()
    if allow_tmp:
        try:
            resolved.relative_to(TMP_ROOT)
            return TMP_ROOT
        except ValueError:
            pass
    return ROOT


def _objective_config(variant: str, params: dict[str, Any]) -> ExplanationObjectiveConfig:
    return ExplanationObjectiveConfig(
        variant=variant,
        lambda_g=float(params.get("lambda_g", 0.0)),
        lambda_c=float(params.get("lambda_c", 0.0)),
        lambda_f=float(params.get("lambda_f", 0.0)),
        lambda_p=float(params.get("lambda_p", 0.0)),
        lambda_s=float(params.get("lambda_s", 0.0)),
        perturbation_scale=float(params.get("perturbation_scale", 0.02)),
        view=str(params.get("view", "gaussian")),
        fidelity_temperature=float(params.get("fidelity_temperature", 1.0)),
        prediction_temperature=float(params.get("prediction_temperature", 1.0)),
    )


def _tuning_specs(matrix: dict[str, Any], *, smoke: bool) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seeds = [101] if smoke else list(matrix["seeds"])
    source_init = next((item for item in matrix["initialisations"] if item["initialisation"] == "source_pretrained"), {"selected_routes": {}})
    source_routes = source_init.get("selected_routes", {})
    for scenario in _scenario_items(matrix, smoke=smoke):
        for initialisation in matrix["initialisations"]:
            route = source_routes.get(scenario["scenario_id"], {}) if initialisation["initialisation"] == "source_pretrained" else {}
            for variant in _variant_items(matrix, smoke=smoke):
                grid = variant["tuning_grid"][:2] if smoke else variant["tuning_grid"]
                for grid_index, params in enumerate(grid):
                    for seed in seeds:
                        specs.append(
                            RunSpec(
                                phase="tuning",
                                scenario_id=scenario["scenario_id"],
                                dataset_id=scenario.get("dataset_id", "UNKNOWN_DATASET"),
                                split_id=scenario.get("split_id", "UNKNOWN_SPLIT"),
                                deployment_condition=scenario.get("deployment_condition", "UNKNOWN_CONDITION"),
                                initialisation=initialisation["initialisation"],
                                source_dataset=route.get("source_dataset"),
                                source_route=route.get("source_route"),
                                seed=int(seed),
                                variant=variant["variant"],
                                objective={**params, "grid_index": grid_index},
                            )
                        )
    return specs


def _locked_params_from_tuning(output_root: Path, matrix: dict[str, Any], *, smoke: bool) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for spec in _tuning_specs(matrix, smoke=smoke):
        metrics_path = _run_dir(output_root, spec) / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = read_json(metrics_path)
        objective_key = stable_json_hash({key: value for key, value in spec.objective.items() if key != "grid_index"})
        key = (spec.scenario_id, spec.initialisation, spec.variant, objective_key)
        score = float(metrics.get("validation", {}).get("rmse", np.inf)) ** 2
        current = grouped.setdefault(
            key,
            {
                "scores": [],
                "objective": spec.objective,
                "selected_from_run_ids": [],
            },
        )
        current["scores"].append(score)
        current["selected_from_run_ids"].append(spec.run_id)
    locked: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (scenario_id, initialisation, variant, _objective_key), payload in grouped.items():
        key = (scenario_id, initialisation, variant)
        mean_score = float(np.mean(payload["scores"]))
        current = locked.get(key)
        if current is None or mean_score < current["selection_score"]:
            locked[key] = {
                "selection_score": mean_score,
                "selection_scores": payload["scores"],
                "objective": payload["objective"],
                "selected_from_run_ids": payload["selected_from_run_ids"],
                "selection_rule": "seed_aggregated_mean_validation_rmse_squared",
            }
    return locked


def _final_specs(matrix: dict[str, Any], output_root: Path, *, smoke: bool) -> list[RunSpec]:
    locked = _locked_params_from_tuning(output_root, matrix, smoke=smoke)
    specs: list[RunSpec] = []
    seeds = [101] if smoke else list(matrix["seeds"])
    source_init = next((item for item in matrix["initialisations"] if item["initialisation"] == "source_pretrained"), {"selected_routes": {}})
    source_routes = source_init.get("selected_routes", {})
    for scenario in _scenario_items(matrix, smoke=smoke):
        for initialisation in matrix["initialisations"]:
            route = source_routes.get(scenario["scenario_id"], {}) if initialisation["initialisation"] == "source_pretrained" else {}
            for variant in _variant_items(matrix, smoke=smoke):
                key = (scenario["scenario_id"], initialisation["initialisation"], variant["variant"])
                params = locked.get(key, {"objective": variant["tuning_grid"][0]})["objective"]
                for seed in seeds:
                    specs.append(
                            RunSpec(
                                phase="final",
                                scenario_id=scenario["scenario_id"],
                                dataset_id=scenario.get("dataset_id", "UNKNOWN_DATASET"),
                                split_id=scenario.get("split_id", "UNKNOWN_SPLIT"),
                                deployment_condition=scenario.get("deployment_condition", "UNKNOWN_CONDITION"),
                            initialisation=initialisation["initialisation"],
                            source_dataset=route.get("source_dataset"),
                            source_route=route.get("source_route"),
                            seed=int(seed),
                            variant=variant["variant"],
                            objective=dict(params),
                        )
                    )
    return specs


def _run_dir(output_root: Path, spec: RunSpec) -> Path:
    return Path(output_root) / "runs" / spec.phase / spec.run_id


def _status_counts(output_root: Path, specs: Iterable[RunSpec]) -> dict[str, int]:
    counts = {"pending": 0, "completed": 0, "failed": 0, "blocked": 0}
    for spec in specs:
        status_path = _run_dir(output_root, spec) / "status.json"
        if not status_path.is_file():
            counts["pending"] += 1
            continue
        status = read_json(status_path).get("status", "pending")
        if status in counts:
            counts[status] += 1
        else:
            counts["failed"] += 1
    return counts


def _prepare_arrays(
    *,
    bridge: FormalCodeBridge,
    config: dict[str, Any],
    scenario: RunSpec,
    smoke: bool,
) -> tuple[dict[str, Any], list[str], pd.DataFrame, dict[str, Any], Any]:
    execution = bridge.execution_module()
    training = bridge.training_module()
    contract = config["datasets"][scenario.dataset_id]
    frame, path = execution.load_frame(contract, bridge.formal_worktree)
    split = next(
        item
        for item in execution.build_contract_splits(frame, contract, seed=int(scenario.seed))
        if item.split_id == scenario.split_id
    )
    cell = {
        "candidate_id": f"explanation_consistency_{scenario.run_id}",
        "dataset_id": scenario.dataset_id,
        "split_id": scenario.split_id,
        "deployment_condition": scenario.deployment_condition,
        "seed": int(scenario.seed),
        "adaptation_mode": "full_adaptation",
        "transfer_strategy": "target_scratch" if scenario.initialisation == "target_scratch" else "ordinary_transfer",
        "missing_modality_method": "missing_aware",
        "source_dataset": scenario.source_dataset or "PRIMARY_CYBENCH_MAIZE_US",
        "source_route": scenario.source_route or "supervised",
    }
    train, validation, test = execution._frames(frame, split, contract, cell)
    if smoke:
        sample = contract["sample_id_column"]
        train = train.sort_values(sample).head(48).copy()
        validation = validation.sort_values(sample).head(24).copy()
        test = test.sort_values(sample).head(24).copy()
    dataset = execution._dataset(frame, path, scenario.dataset_id, contract)
    arrays = training.prepare_arrays(
        dataset=dataset,
        train_frame=train,
        validation_frame=validation,
        test_frame=test,
    )
    execution._apply_weather_deployment_condition(
        arrays,
        test,
        list(dataset.weather_features),
        scenario.deployment_condition,
        int(scenario.seed),
    )
    metadata = {
        "job": asdict(scenario),
        "formal_dataset_path": str(path),
        "split_fingerprint": split.id_hash,
        "target_test_used_for_training": False,
        "target_test_used_for_selection": False,
        "smoke_only": smoke,
    }
    return arrays, list(dataset.weather_features), test, metadata, cell


def _initial_state(
    *,
    bridge: FormalCodeBridge,
    config: dict[str, Any],
    formal_campaign: Path,
    output_root: Path,
    spec: RunSpec,
    cell: dict[str, Any],
) -> dict[str, torch.Tensor] | None:
    if spec.initialisation == "target_scratch":
        return None
    execution = bridge.execution_module()
    ensure_source_dependency_alias(formal_campaign=formal_campaign, output_root=output_root, job=cell)
    checkpoint = execution.ensure_source_checkpoint(
        config,
        bridge.formal_worktree,
        Path(output_root),
        cell,
        smoke=False,
    )
    return execution._load_state(checkpoint)


def run_spec(
    *,
    spec: RunSpec,
    matrix: dict[str, Any],
    formal_campaign: Path,
    output_root: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, Any]:
    run_dir = _run_dir(output_root, spec)
    status_path = run_dir / "status.json"
    if resume and status_path.is_file() and read_json(status_path).get("status") == "completed":
        return {"status": "completed", "run_id": spec.run_id, "skipped_existing": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    write_json_atomic(status_path, {"status": "running", "run_id": spec.run_id, "started_unix": started_unix})
    write_json_atomic(run_dir / "config.json", asdict(spec))
    (run_dir / "config.sha256").write_text(stable_json_hash(asdict(spec)) + "\n")
    bridge = FormalCodeBridge()
    config = bridge.load_config()
    training = bridge.training_module()
    arrays, feature_names, test_frame, metadata, cell = _prepare_arrays(
        bridge=bridge,
        config=config,
        scenario=spec,
        smoke=smoke,
    )
    initial = _initial_state(
        bridge=bridge,
        config=config,
        formal_campaign=formal_campaign,
        output_root=output_root,
        spec=spec,
        cell=cell,
    )
    budget = matrix["training_budget"]
    epochs = 1 if smoke else int(budget["epochs"])
    patience = 1 if smoke else int(budget["patience"])
    batch_size = 16 if smoke else int(budget["batch_size"])
    learning_rate = float(
        budget["target_scratch_learning_rate"]
        if spec.initialisation == "target_scratch"
        else budget["source_pretrained_learning_rate"]
    )
    objective = _objective_config(spec.variant, spec.objective)
    try:
        result = train_student_with_explanation_objective(
            training_module=training,
            arrays=arrays,
            feature_names=feature_names,
            seed=int(spec.seed),
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=torch.device("cpu"),
            checkpoint_path=run_dir / "checkpoint.pt",
            metadata=metadata,
            initial_state_dict=initial,
            objective_config=objective,
            evaluate_test=spec.phase == "final" or smoke,
            max_train_batches=1 if smoke else None,
        )
        write_json_atomic(run_dir / "feature_schema.json", {"weather_features": feature_names})
        write_json_atomic(run_dir / "training_reference.json", result["training_reference"])
        write_json_atomic(run_dir / "history.json", {"history": result["history"]})
        metrics = {
            "status": "completed",
            "phase": spec.phase,
            "validation": result["validation_metrics"],
            "validation_attribution": result["validation_attribution"],
            "objective": result["objective"],
            "cost": result["cost"],
            "test_used_for_selection": False,
        }
        if "test_metrics" in result:
            metrics["test"] = result["test_metrics"]
            metrics["attribution"] = result["test_attribution"]
            predictions = pd.DataFrame(
                {
                    "sample_id": test_frame[config["datasets"][spec.dataset_id]["sample_id_column"]].astype(str).tolist(),
                    "y_true": np.asarray(arrays["test"]["target"], dtype=float),
                    "y_pred": np.asarray(result["test_prediction"], dtype=float),
                }
            )
            predictions.to_csv(run_dir / "predictions.csv", index=False)
        write_json_atomic(run_dir / "metrics.json", metrics)
        finished_unix = time.time()
        write_json_atomic(
            status_path,
            {
                "status": "completed",
                "run_id": spec.run_id,
                "started_unix": started_unix,
                "finished_unix": finished_unix,
                "elapsed_seconds": finished_unix - started_unix,
            },
        )
        return {"status": "completed", "run_id": spec.run_id}
    except Exception as error:
        write_json_atomic(
            run_dir / "metrics.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        write_json_atomic(
            status_path,
            {
                "status": "failed",
                "run_id": spec.run_id,
                "started_unix": started_unix,
                "finished_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        failure_path = Path(output_root) / "failure_ledger" / f"{spec.run_id}.json"
        write_json_atomic(failure_path, {"run": asdict(spec), "error_type": type(error).__name__, "error": str(error)})
        raise


def _run_rows(output_root: Path, *, phase: str = "final") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_dir = Path(output_root) / "runs" / phase
    if not phase_dir.is_dir():
        return rows
    for run_dir in sorted(phase_dir.iterdir()):
        config = read_json(run_dir / "config.json", {})
        metrics = read_json(run_dir / "metrics.json", {})
        if metrics.get("status") != "completed":
            continue
        attribution = metrics.get("attribution", {})
        validation_attr = metrics.get("validation_attribution", {})
        rows.append(
            {
                "run_id": run_dir.name,
                "scenario_id": config.get("scenario_id"),
                "dataset_id": config.get("dataset_id"),
                "split_id": config.get("split_id"),
                "deployment_condition": config.get("deployment_condition"),
                "initialisation": config.get("initialisation"),
                "variant": config.get("variant"),
                "seed": config.get("seed"),
                "view": config.get("objective", {}).get("view", "gaussian"),
                "test_mae": metrics.get("test", {}).get("mae"),
                "test_rmse": metrics.get("test", {}).get("rmse"),
                "test_r2": metrics.get("test", {}).get("r2"),
                "validation_rmse": metrics.get("validation", {}).get("rmse"),
                "cosine_similarity": attribution.get("cosine_similarity", validation_attr.get("cosine_similarity")),
                "spearman": attribution.get("spearman", validation_attr.get("spearman")),
                "top5_overlap": attribution.get("top5_overlap", validation_attr.get("top5_overlap")),
                "attribution_norm": attribution.get("attribution_norm", validation_attr.get("attribution_norm")),
                "local_taylor_fidelity_r2": attribution.get("local_taylor_fidelity_r2", validation_attr.get("local_taylor_fidelity_r2")),
                "training_wall_clock_seconds": metrics.get("cost", {}).get("training_wall_clock_seconds"),
                "test_inference_seconds_per_sample": metrics.get("cost", {}).get("test_inference_seconds_per_sample"),
                "parameter_count": metrics.get("cost", {}).get("parameter_count"),
                "checkpoint_size_bytes": metrics.get("cost", {}).get("checkpoint_size_bytes"),
            }
        )
    return rows


def compute_pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            numeric_rows.append(
                {
                    **row,
                    "test_mae": float(row["test_mae"]),
                    "explanation_instability": float(row.get("explanation_instability", 1.0 - float(row.get("cosine_similarity", float("nan"))))),
                    "sparsity": float(row.get("sparsity", row.get("attribution_norm", float("nan")))),
                }
            )
        except (TypeError, ValueError):
            continue
    frontier: list[dict[str, Any]] = []
    for row in numeric_rows:
        dominated = False
        for other in numeric_rows:
            if other is row:
                continue
            no_worse = (
                other["test_mae"] <= row["test_mae"]
                and other["explanation_instability"] <= row["explanation_instability"]
                and other["sparsity"] <= row["sparsity"]
            )
            strictly_better = (
                other["test_mae"] < row["test_mae"]
                or other["explanation_instability"] < row["explanation_instability"]
                or other["sparsity"] < row["sparsity"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def claim_grade_verdict(checks: dict[str, Any]) -> dict[str, Any]:
    if not checks.get("headline_reconstructable", False):
        return {"verdict": "NO-GO", "reason": "headline_not_reconstructable"}
    if int(checks.get("complete_seed_count", 0)) < 3:
        return {"verdict": "NO-GO", "reason": "incomplete_seed_coverage"}
    full = all(
        bool(checks.get(name, False))
        for name in [
            "new_variant_beats_gradient",
            "new_variant_beats_ordinary",
            "prediction_not_degraded",
            "explanation_improved",
            "inference_overhead_near_zero",
        ]
    ) and int(checks.get("scenarios_direction_consistent", 0)) >= 2
    if full:
        return {"verdict": "FULL TECHNICAL GO", "reason": "all_full_go_checks_passed"}
    if checks.get("explanation_improved") and checks.get("new_variant_beats_ordinary"):
        return {"verdict": "CONDITIONAL GO", "reason": "explanation_strong_prediction_context_dependent"}
    if checks.get("explanation_improved") or checks.get("matched_support_insight", False):
        return {"verdict": "DIAGNOSTIC-ONLY", "reason": "diagnostic_signal_without_optimisation_advantage"}
    return {"verdict": "NO-GO", "reason": "no_stable_prediction_or_explanation_gain"}


def write_analysis_outputs(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    rows = _run_rows(output_root, phase="final")
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(analysis_root / "final_run_level_metrics.csv", index=False)
    pareto = compute_pareto_front(
        [
            {
                **row,
                "explanation_instability": 1.0 - float(row["cosine_similarity"]) if row.get("cosine_similarity") is not None else float("nan"),
                "sparsity": row.get("attribution_norm"),
            }
            for row in rows
        ]
    )
    if pareto:
        pd.DataFrame(pareto).to_csv(analysis_root / "pareto_front.csv", index=False)
    completed_seeds = len({int(row["seed"]) for row in rows if row.get("seed") is not None})
    checks = {
        "headline_reconstructable": bool(rows),
        "complete_seed_count": completed_seeds,
        "scenarios_direction_consistent": 0,
        "new_variant_beats_gradient": False,
        "new_variant_beats_ordinary": False,
        "prediction_not_degraded": False,
        "explanation_improved": False,
        "inference_overhead_near_zero": True,
    }
    frame = pd.DataFrame(rows)
    if not frame.empty and {"variant", "test_mae", "cosine_similarity"}.issubset(frame.columns):
        means = frame.groupby("variant", dropna=False).agg(mae=("test_mae", "mean"), cosine=("cosine_similarity", "mean")).reset_index()
        best_new = means[means["variant"].isin(["fidelity_selective_consistency", "deployment_fidelity_consistency"])]
        ordinary = means[means["variant"].eq("explanation_consistency")]
        gradient = means[means["variant"].eq("gradient_regularisation")]
        prediction = means[means["variant"].eq("prediction_only")]
        if not best_new.empty:
            best = best_new.sort_values(["mae", "cosine"], ascending=[True, False]).iloc[0]
            if not gradient.empty:
                checks["new_variant_beats_gradient"] = float(best["mae"]) <= float(gradient["mae"].mean())
            if not ordinary.empty:
                checks["new_variant_beats_ordinary"] = float(best["cosine"]) >= float(ordinary["cosine"].mean())
            if not prediction.empty:
                checks["prediction_not_degraded"] = float(best["mae"]) <= float(prediction["mae"].mean()) * 1.02
                checks["explanation_improved"] = float(best["cosine"]) >= float(prediction["cosine"].mean())
        scenario_dirs = 0
        for _scenario, sub in frame.groupby("scenario_id", dropna=False):
            pred = sub[sub["variant"].eq("prediction_only")]
            new = sub[sub["variant"].isin(["fidelity_selective_consistency", "deployment_fidelity_consistency"])]
            if not pred.empty and not new.empty and float(new["cosine_similarity"].mean()) >= float(pred["cosine_similarity"].mean()):
                scenario_dirs += 1
        checks["scenarios_direction_consistent"] = scenario_dirs
    verdict = claim_grade_verdict(checks)
    payload = {"checks": checks, "verdict": verdict, "run_count": len(rows), "pareto_count": len(pareto)}
    write_json_atomic(analysis_root / "claim_grade_verdict.json", payload)
    return payload


def rebuild_summary(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    summary: dict[str, Any] = {
        "tuning_runs": {"completed": 0, "failed": 0, "pending": 0},
        "final_runs": {"completed": 0, "failed": 0, "pending": 0, "mean_test_mae": None},
    }
    final_mae: list[float] = []
    for phase in ["tuning", "final"]:
        phase_dir = output_root / "runs" / phase
        if not phase_dir.is_dir():
            continue
        for run_dir in phase_dir.iterdir():
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                summary[f"{phase}_runs"]["pending"] += 1
                continue
            metrics = read_json(metrics_path)
            status = str(metrics.get("status"))
            if status == "completed":
                summary[f"{phase}_runs"]["completed"] += 1
                if phase == "final" and "test" in metrics and "mae" in metrics["test"]:
                    final_mae.append(float(metrics["test"]["mae"]))
            else:
                summary[f"{phase}_runs"]["failed"] += 1
    if final_mae:
        summary["final_runs"]["mean_test_mae"] = float(np.mean(final_mae))
        summary["final_runs"]["std_test_mae"] = float(np.std(final_mae))
    analysis_path = output_root / "analysis" / "claim_grade_verdict.json"
    if analysis_path.is_file():
        summary["analysis"] = read_json(analysis_path)
    return summary


def write_summary(output_root: Path) -> dict[str, Any]:
    summary = rebuild_summary(output_root)
    write_json_atomic(Path(output_root) / "summary" / "summary.json", summary)
    return summary


def _write_status(output_root: Path, matrix: dict[str, Any], *, smoke: bool) -> None:
    tuning_specs = _tuning_specs(matrix, smoke=smoke)
    final_specs = _final_specs(matrix, output_root, smoke=smoke)
    write_json_atomic(
        Path(output_root) / "status" / "optimisation_status.json",
        {
            "tuning": _status_counts(output_root, tuning_specs),
            "final": _status_counts(output_root, final_specs),
            "summary": rebuild_summary(output_root),
            "heartbeat_unix": time.time(),
        },
    )


def execute(
    *,
    output_root: Path,
    formal_campaign: Path,
    phase: str,
    smoke: bool,
    resume: bool,
    confirm_matrix_hash: str | None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    matrix = load_or_build_matrix(output_root)
    actual_hash = matrix.get("matrix_hash")
    if confirm_matrix_hash and confirm_matrix_hash != actual_hash:
        raise ValueError(f"MATRIX_HASH_MISMATCH:{confirm_matrix_hash}:{actual_hash}")
    validation = validate_matrix(
        matrix,
        formal_campaign=formal_campaign,
        output_root=output_root,
        writable_root=writable_root_for_output(output_root, allow_tmp=smoke),
    )
    write_json_atomic(output_root / "status" / "preflight_status.json", validation)
    if validation["status"] != "PASS":
        raise ValueError(f"EXPLANATION_CONSISTENCY_PREFLIGHT_FAILED:{validation}")
    specs: list[RunSpec] = []
    if phase in {"tuning", "all"}:
        specs.extend(_tuning_specs(matrix, smoke=smoke))
    if phase in {"final", "all"}:
        if phase == "all":
            # Final specs are rebuilt after tuning below, because locked params
            # are selected from validation metrics.
            pass
        else:
            specs.extend(_final_specs(matrix, output_root, smoke=smoke))
    completed = 0
    failed = 0
    for spec in list(specs):
        if stop_requested(output_root):
            break
        try:
            run_spec(
                spec=spec,
                matrix=matrix,
                formal_campaign=formal_campaign,
                output_root=output_root,
                smoke=smoke,
                resume=resume,
            )
            completed += 1
        except Exception:
            failed += 1
        _write_status(output_root, matrix, smoke=smoke)
    if phase == "all":
        for spec in _final_specs(matrix, output_root, smoke=smoke):
            if stop_requested(output_root):
                break
            try:
                run_spec(
                    spec=spec,
                    matrix=matrix,
                    formal_campaign=formal_campaign,
                    output_root=output_root,
                    smoke=smoke,
                    resume=resume,
                )
                completed += 1
            except Exception:
                failed += 1
            _write_status(output_root, matrix, smoke=smoke)
    analysis = write_analysis_outputs(output_root)
    summary = write_summary(output_root)
    return {"completed": completed, "failed": failed, "summary": summary, "analysis": analysis}


def write_run_table(output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for phase in ["tuning", "final"]:
        phase_dir = Path(output_root) / "runs" / phase
        if not phase_dir.is_dir():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            config = read_json(run_dir / "config.json", {})
            metrics = read_json(run_dir / "metrics.json", {})
            rows.append(
                {
                    "phase": phase,
                    "run_id": run_dir.name,
                    "status": metrics.get("status", "pending"),
                    "scenario_id": config.get("scenario_id"),
                    "initialisation": config.get("initialisation"),
                    "variant": config.get("variant"),
                    "view": config.get("objective", {}).get("view"),
                    "seed": config.get("seed"),
                    "validation_rmse": metrics.get("validation", {}).get("rmse"),
                    "test_mae": metrics.get("test", {}).get("mae"),
                    "attribution_cosine": metrics.get("attribution", {}).get("cosine_similarity"),
                    "training_wall_clock_seconds": metrics.get("cost", {}).get("training_wall_clock_seconds"),
                }
            )
    path = Path(output_root) / "summary" / "run_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["phase", "run_id", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return path
