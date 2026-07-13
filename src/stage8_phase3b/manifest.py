from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .guards import stable_json_hash, validate_output_root


KEY_COLUMNS = [
    "dataset_id",
    "split_id",
    "deployment_condition",
    "transfer_strategy",
    "missing_modality_method",
    "source_dataset",
    "source_route",
]

BASELINE_COLUMNS = [
    "dataset_id",
    "split_id",
    "deployment_condition",
]

PROHIBITED_HIGH_LIMIT_CLAIMS = [
    "absolute_predictive_success",
    "soil_contribution",
    "universal_missing_modality_robustness",
    "universal_transfer_robustness",
    "cross_crop_generalisation",
    "cross_region_generalisation",
]


@dataclass(frozen=True)
class MatrixDefinition:
    name: str
    group_ids: tuple[str, ...]
    baseline_keys: tuple[tuple[str, str, str], ...]
    expected_target_jobs: int


MINIMAL_GROUPS = (
    "e70126f11798",
    "40fc841a0f90",
    "bcf719eef5be",
    "d5ad39cfcb8a",
    "e2dac23184ab",
    "9813870f1d28",
)

RECOMMENDED_GROUPS = (
    "e70126f11798",
    "bb496cc2156d",
    "e50d859fea65",
    "370830075bde",
    "121fc78e55d7",
    "a73058510de1",
    "9965bffc1b51",
    "40fc841a0f90",
    "19cdbfb65fad",
    "bcf719eef5be",
    "db772ae6816f",
    "f49a5165ace2",
    "c4db346b6be2",
    "ea8cc847efcc",
    "20b349e2482d",
    "31d709bf2081",
    "f4ecddc650fe",
    "4ab0b90d8d6c",
    "50e175ab1ecf",
    "6c22e036c70a",
    "1c6227ef68b8",
    "d5ad39cfcb8a",
    "a9cfb6f7aaf6",
    "f20c3cbb65dd",
    "1ca3bbda03c6",
    "79336c5100e2",
    "f5456a5ede89",
    "b8efe86b971e",
    "b0951e7c4d22",
    "02169d83a564",
    "7a543d7d7115",
    "9ee291094016",
    "4ee20ed9ae04",
    "1ed2560bdd10",
    "dd500ff2502a",
    "e2dac23184ab",
    "80b6964c009e",
    "9813870f1d28",
    "063069229114",
    "98f47e555315",
    "b62a36b83c59",
    "14ea780c1f6d",
    "34bba1887ab0",
)

FULL_EXTRA_GROUPS = (
    "4113b29d8f80",
    "50c962a03c0d",
    "58e859bbcfcb",
    "3b2cc498effb",
    "2d6be080c329",
    "f3e8d07c5b57",
)

MATRICES = {
    "minimal": MatrixDefinition(
        name="minimal",
        group_ids=MINIMAL_GROUPS,
        baseline_keys=(
            ("CY-Bench_wheat_AU", "SPATIAL", "complete"),
            ("waite", "ROLLING_1991", "synthetic_no_weather"),
        ),
        expected_target_jobs=24,
    ),
    "recommended": MatrixDefinition(
        name="recommended",
        group_ids=RECOMMENDED_GROUPS,
        baseline_keys=(
            ("CY-Bench_wheat_AU", "SPATIAL", "complete"),
            ("CY-Bench_wheat_AU", "SPATIAL", "synthetic_no_soil"),
            ("CY-Bench_wheat_AU", "SPATIAL", "synthetic_random_0_15"),
            ("CY-Bench_wheat_AU", "GROUP", "synthetic_no_soil"),
            ("waite", "ROLLING_1991", "synthetic_no_weather"),
            ("waite", "ROLLING_1991", "complete"),
            ("CY-Bench_wheat_AU", "RANDOM", "synthetic_no_weather"),
            ("ROSEWORTHY_E5_POINT", "SPATIAL", "synthetic_no_weather"),
        ),
        expected_target_jobs=153,
    ),
    "full": MatrixDefinition(
        name="full",
        group_ids=(*RECOMMENDED_GROUPS, *FULL_EXTRA_GROUPS),
        baseline_keys=(
            ("CY-Bench_wheat_AU", "SPATIAL", "complete"),
            ("CY-Bench_wheat_AU", "SPATIAL", "synthetic_no_soil"),
            ("CY-Bench_wheat_AU", "SPATIAL", "synthetic_random_0_15"),
            ("CY-Bench_wheat_AU", "GROUP", "synthetic_no_soil"),
            ("ROSEWORTHY_E5_POINT", "SPATIAL", "synthetic_no_soil"),
            ("waite", "ROLLING_1991", "synthetic_no_weather"),
            ("waite", "ROLLING_1991", "complete"),
            ("CY-Bench_wheat_AU", "RANDOM", "synthetic_no_weather"),
            ("ROSEWORTHY_E5_POINT", "SPATIAL", "synthetic_no_weather"),
        ),
        expected_target_jobs=174,
    ),
}


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _row_matches(frame: pd.DataFrame, row: pd.Series, columns: Iterable[str]) -> pd.DataFrame:
    matched = frame
    for column in columns:
        expected = "" if pd.isna(row[column]) else str(row[column])
        matched = matched[matched[column].fillna("").astype(str) == expected]
    return matched


def _baseline_matches(run_inventory: pd.DataFrame, key: tuple[str, str, str]) -> pd.DataFrame:
    dataset_id, split_id, condition = key
    return run_inventory[
        (run_inventory["dataset_id"].astype(str) == dataset_id)
        & (run_inventory["split_id"].astype(str) == split_id)
        & (run_inventory["deployment_condition"].astype(str) == condition)
        & (run_inventory["transfer_strategy"].astype(str) == "target_scratch")
        & (run_inventory["missing_modality_method"].astype(str) == "imputation")
    ].sort_values("seed")


def _baseline_key_for_row(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["dataset_id"]),
        str(row["split_id"]),
        str(row["deployment_condition"]),
    )


def _cell_use(row: pd.Series) -> str:
    if _bool(row.get("high_limit_flag")):
        return "boundary_evidence_only"
    tier = str(row.get("positivity_tier", ""))
    usage = str(row.get("paper_usage", ""))
    if tier == "no_improvement" or "not_for_positive_claim" in usage:
        return "negative_or_boundary_control"
    if "boundary" in usage or "supplement" in usage:
        return "component_or_boundary_evidence"
    return "core_or_component_evidence"


def _cell_from_row(row: pd.Series, run_inventory: pd.DataFrame) -> dict:
    formal_jobs = _row_matches(run_inventory, row, KEY_COLUMNS).sort_values("seed")
    if formal_jobs.empty:
        raise ValueError(f"NO_FORMAL_JOBS_FOR_GROUP:{row['candidate_group_id']}")
    baseline_jobs = _baseline_matches(run_inventory, _baseline_key_for_row(row))
    if baseline_jobs.empty:
        raise ValueError(f"NO_MATCHED_BASELINE_FOR_GROUP:{row['candidate_group_id']}")

    high_limit = _bool(row.get("high_limit_flag"))
    return {
        "cell_type": "candidate",
        "candidate_group_id": str(row["candidate_group_id"]),
        "all_seed_candidate_ids": formal_jobs["candidate_id"].astype(str).tolist(),
        "seeds": [int(seed) for seed in formal_jobs["seed"].tolist()],
        "dataset_id": str(row["dataset_id"]),
        "split_id": str(row["split_id"]),
        "deployment_condition": str(row["deployment_condition"]),
        "transfer_strategy": str(row["transfer_strategy"]),
        "missing_modality_method": str(row["missing_modality_method"]),
        "source_dataset": str(row["source_dataset"]),
        "source_route": str(row["source_route"]),
        "positivity_tier": str(row["positivity_tier"]),
        "paper_usage": str(row["paper_usage"]),
        "high_limit_flag": high_limit,
        "high_limit_reasons": "" if pd.isna(row.get("high_limit_reasons")) else str(row.get("high_limit_reasons")),
        "original_metrics": {
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "base_mae": float(row["base_mae"]),
            "base_rmse": float(row["base_rmse"]),
            "base_r2": float(row["base_r2"]),
            "delta_mae_vs_baseline": float(row["delta_mae_vs_baseline"]),
            "delta_rmse_vs_baseline": float(row["delta_rmse_vs_baseline"]),
            "delta_r2_vs_baseline": float(row["delta_r2_vs_baseline"]),
            "seed_consistency_mae": float(row["seed_consistency_mae"]),
        },
        "matched_baseline_ids": baseline_jobs["candidate_id"].astype(str).tolist(),
        "matched_baseline_metrics": {
            "mae": float(baseline_jobs["mae"].mean()),
            "rmse": float(baseline_jobs["rmse"].mean()),
            "r2": float(baseline_jobs["r2"].mean()),
        },
        "replay_scientific_use": _cell_use(row),
        "boundary_evidence_only": high_limit,
        "prohibited_claims": PROHIBITED_HIGH_LIMIT_CLAIMS if high_limit else [],
        "formal_jobs": formal_jobs.to_dict(orient="records"),
    }


def _baseline_cell(key: tuple[str, str, str], run_inventory: pd.DataFrame) -> dict:
    jobs = _baseline_matches(run_inventory, key)
    if jobs.empty:
        raise ValueError(f"NO_BASELINE_JOBS:{key}")
    return {
        "cell_type": "matched_baseline",
        "candidate_group_id": f"baseline::{key[0]}::{key[1]}::{key[2]}",
        "all_seed_candidate_ids": jobs["candidate_id"].astype(str).tolist(),
        "seeds": [int(seed) for seed in jobs["seed"].tolist()],
        "dataset_id": key[0],
        "split_id": key[1],
        "deployment_condition": key[2],
        "transfer_strategy": "target_scratch",
        "missing_modality_method": "imputation",
        "source_dataset": "",
        "source_route": "",
        "positivity_tier": "matched_baseline",
        "paper_usage": "matched_baseline",
        "high_limit_flag": False,
        "high_limit_reasons": "",
        "original_metrics": {
            "mae": float(jobs["mae"].mean()),
            "rmse": float(jobs["rmse"].mean()),
            "r2": float(jobs["r2"].mean()),
        },
        "matched_baseline_ids": jobs["candidate_id"].astype(str).tolist(),
        "matched_baseline_metrics": {
            "mae": float(jobs["mae"].mean()),
            "rmse": float(jobs["rmse"].mean()),
            "r2": float(jobs["r2"].mean()),
        },
        "replay_scientific_use": "matched_baseline",
        "boundary_evidence_only": False,
        "prohibited_claims": [],
        "formal_jobs": jobs.to_dict(orient="records"),
    }


def build_manifest_from_sources(
    *,
    candidate_csv: Path,
    run_inventory_csv: Path,
    matrix: str,
    formal_campaign: Path,
    output_root: Path,
    writable_root: Path | None = None,
) -> dict:
    if matrix not in MATRICES:
        raise ValueError(f"UNKNOWN_REPLAY_MATRIX:{matrix}")

    writable = Path(writable_root) if writable_root is not None else Path(__file__).resolve().parents[2]
    output = validate_output_root(output_root=Path(output_root), formal_campaign=Path(formal_campaign), writable_root=writable)
    definition = MATRICES[matrix]
    candidates = pd.read_csv(candidate_csv)
    run_inventory = pd.read_csv(run_inventory_csv)

    by_group = candidates.set_index("candidate_group_id", drop=False)
    missing = [group for group in definition.group_ids if group not in by_group.index]
    if missing:
        raise ValueError(f"REPLAY_GROUPS_NOT_FOUND:{missing}")

    cells = [_cell_from_row(by_group.loc[group], run_inventory) for group in definition.group_ids]
    baseline_cells = [_baseline_cell(key, run_inventory) for key in definition.baseline_keys]

    high_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in cells if cell["high_limit_flag"])
    non_high_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in cells if not cell["high_limit_flag"])
    baseline_jobs = sum(len(cell["all_seed_candidate_ids"]) for cell in baseline_cells)
    expected_target_jobs = high_jobs + non_high_jobs + baseline_jobs
    if expected_target_jobs != definition.expected_target_jobs:
        raise ValueError(
            f"REPLAY_MATRIX_JOB_COUNT_MISMATCH:{matrix}:"
            f"{expected_target_jobs}!={definition.expected_target_jobs}"
        )

    manifest = {
        "schema_version": "stage8_phase3b_replay_manifest_v1",
        "matrix": matrix,
        "formal_campaign": str(Path(formal_campaign).resolve()),
        "output_root": str(output),
        "candidate_csv": str(Path(candidate_csv).resolve()),
        "run_inventory_csv": str(Path(run_inventory_csv).resolve()),
        "expected_target_jobs": expected_target_jobs,
        "job_counts": {
            "candidate_cells": len(cells),
            "baseline_cells": len(baseline_cells),
            "high_limit_jobs": high_jobs,
            "non_high_limit_jobs": non_high_jobs,
            "baseline_jobs": baseline_jobs,
        },
        "cells": cells,
        "baseline_cells": baseline_cells,
        "all_formal_candidate_ids": [
            job_id
            for cell in [*cells, *baseline_cells]
            for job_id in cell["all_seed_candidate_ids"]
        ],
        "implementation_fidelity_requirements": {
            "must_call_original_stage8_v4_handlers": True,
            "algorithm_reimplementation_allowed": False,
            "persistence_hooks_must_not_change_numeric_execution": True,
        },
    }
    manifest["manifest_hash"] = stable_json_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def write_manifest(manifest: dict, output_root: Path) -> tuple[Path, Path]:
    output = Path(output_root)
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    manifest_path = control / "replay_manifest.json"
    hash_path = control / "replay_manifest.sha256"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    hash_path.write_text(str(manifest["manifest_hash"]) + "\n")
    return manifest_path, hash_path

