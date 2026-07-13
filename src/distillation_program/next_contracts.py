from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class InputClass(str, Enum):
    DEPLOYABLE_INPUT = "DEPLOYABLE_INPUT"
    LEGAL_PRIVILEGED_TRAINING_INPUT = "LEGAL_PRIVILEGED_TRAINING_INPUT"
    RETROSPECTIVE_DIAGNOSTIC_ONLY = "RETROSPECTIVE_DIAGNOSTIC_ONLY"
    POST_CUTOFF = "POST_CUTOFF"
    TARGET_DIRECT_FUNCTION_RISK = "TARGET_DIRECT_FUNCTION_RISK"
    FORBIDDEN_LEAKAGE = "FORBIDDEN_LEAKAGE"


def classify_teacher_column(column: str, evidence: dict[str, Any]) -> InputClass:
    """Classify one teacher input using a default-deny timing/target policy."""
    if evidence.get("target_duplicate") or evidence.get("forbidden_leakage"):
        return InputClass.FORBIDDEN_LEAKAGE
    if evidence.get("target_direct_function"):
        return InputClass.TARGET_DIRECT_FUNCTION_RISK
    if evidence.get("observed_after_cutoff"):
        return InputClass.POST_CUTOFF
    if evidence.get("retrospective_only"):
        return InputClass.RETROSPECTIVE_DIAGNOSTIC_ONLY
    if evidence.get("cutoff_safe") and evidence.get("deployment_available"):
        return InputClass.DEPLOYABLE_INPUT
    if evidence.get("cutoff_safe") and evidence.get("target_independent") and evidence.get("training_only"):
        return InputClass.LEGAL_PRIVILEGED_TRAINING_INPUT
    return InputClass.FORBIDDEN_LEAKAGE


def _cluster_bootstrap(values: pd.DataFrame, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = values["cluster"].astype(str).unique()
    if len(clusters) < 2:
        return float("nan"), float("nan")
    samples = []
    by_cluster = {cluster: values.loc[values.cluster.astype(str).eq(cluster), "effect"].mean() for cluster in clusters}
    for _ in range(draws):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        samples.append(float(np.mean([by_cluster[item] for item in selected])))
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def derive_outer_train_decision_rule(
    rows: pd.DataFrame,
    *,
    unit: str,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    required = {"fold", "seed", "cluster", "baseline_error", "candidate_error"}
    if not required <= set(rows):
        raise ValueError(f"decision_rule_columns_missing:{sorted(required - set(rows))}")
    work = rows.copy()
    work["effect"] = work["baseline_error"] - work["candidate_error"]
    low, high = _cluster_bootstrap(work, bootstrap_draws, seed)
    fold_seed = work.groupby(["fold", "seed"], observed=True).effect.mean()
    baseline_variation = work.groupby(["fold", "seed"], observed=True).baseline_error.mean().std(ddof=1)
    payload = {
        "schema_version": "outer_train_decision_rule_v1",
        "unit": unit,
        "criteria": {
            "effect_size": {"estimate": float(work.effect.mean()), "scale_source": "inner_validation_paired_error"},
            "clustered_interval": {"lower": low, "upper": high, "cluster": "environment_or_environment_year", "draws": bootstrap_draws},
            "directional_consistency": {"positive": int((fold_seed > 0).sum()), "total": int(len(fold_seed))},
            "worst_group": {"minimum_group_effect": float(work.groupby("cluster").effect.mean().min())},
            "practical_significance": {
                "baseline_seed_fold_sd": None if pd.isna(baseline_variation) else float(baseline_variation),
                "derivation": "outer_train_noise_and_domain_measurement_context",
            },
            "negative_control_separation": {"required": True},
        },
        "provenance": {
            "data_scope": "OUTER_TRAIN_INNER_VALIDATION_ONLY",
            "outer_test_accessed": False,
            "seed": seed,
            "input_sha256": hashlib.sha256(pd.util.hash_pandas_object(work, index=True).values.tobytes()).hexdigest(),
        },
    }
    payload["decision_rule_fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=True).encode()).hexdigest()
    return payload


def validate_decision_rule_provenance(rule: dict[str, Any]) -> None:
    provenance = rule.get("provenance", {})
    if provenance.get("data_scope") != "OUTER_TRAIN_INNER_VALIDATION_ONLY" or provenance.get("outer_test_accessed") is not False:
        raise ValueError("outer_test_decision_rule_forbidden")
    if "fixed_effect_threshold" in rule:
        raise ValueError("unsupported_fixed_scientific_threshold")
