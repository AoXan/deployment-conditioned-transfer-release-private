from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jsonschema
import numpy as np
import yaml


@dataclass(frozen=True)
class BudgetLegality:
    status: str
    requested_budget: float
    effective_budget: float | None
    reasons: tuple[str, ...]


def load_program(config_path: Path, schema_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(config, schema)
    return config


def evaluate_budget_legality(
    *,
    budget: float,
    train_targets: Sequence[float],
    validation_targets: Sequence[float],
    complete_pair_count: int,
    pattern_count: int,
    calibration_count: int,
    pattern_calibration_count: int,
    quartile_edges: Sequence[float],
) -> BudgetLegality:
    reasons: list[str] = []
    if len(train_targets) < 32:
        reasons.append("target_adaptation_train_lt_32")
    if len(validation_targets) < 30:
        reasons.append("inner_validation_lt_30")
    if complete_pair_count < 64:
        reasons.append("teacher_complete_pairs_lt_64")
    if pattern_count < 20:
        reasons.append("evaluation_pattern_lt_20")
    if calibration_count < 50:
        reasons.append("calibration_lt_50")
    if pattern_calibration_count < 30:
        reasons.append("pattern_calibration_requires_declared_fallback")
    if reasons:
        return BudgetLegality("BLOCKED_INSUFFICIENT_SAMPLE", budget, None, tuple(reasons))

    train_hist, _ = np.histogram(np.asarray(train_targets, float), bins=np.asarray(quartile_edges, float))
    val_hist, _ = np.histogram(np.asarray(validation_targets, float), bins=np.asarray(quartile_edges, float))
    if np.any(train_hist < 4) or int((val_hist >= 3).sum()) < 3:
        return BudgetLegality(
            "BLOCKED_INSUFFICIENT_TARGET_COVERAGE",
            budget,
            None,
            ("train_or_validation_target_quartile_coverage",),
        )
    return BudgetLegality("LEGAL", budget, budget, ())


def classify_pair(
    *,
    target_compatible: bool,
    unit_compatible: bool,
    sample_unit_compatible: bool,
    cutoff_compatible: bool,
    modalities_compatible: bool,
    split_locked: bool,
    same_crop: bool,
    granularity_mapping: bool,
) -> str:
    essentials = target_compatible and unit_compatible and cutoff_compatible and modalities_compatible and split_locked
    if essentials and same_crop and (sample_unit_compatible or granularity_mapping):
        return "CONDITIONAL_MATRIX"
    if essentials and same_crop:
        return "REPRESENTATION_TRANSFER_ONLY"
    if essentials and not same_crop:
        return "EXTERNAL_TRANSPORTABILITY_STRESS_TEST"
    return "BLOCKED_DATA_CONTRACT"
