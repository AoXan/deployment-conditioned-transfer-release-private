#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf


DISJOINT_GROUPS = (
    "observation_support",
    "temperature",
    "water_balance",
    "radiation",
)
FEATURE_NAMES = (
    "weather_observed_days",
    "weather_expected_days",
    "weather_coverage",
    "weather_tmin_mean",
    "weather_tmax_mean",
    "weather_prec_sum",
    "weather_rad_sum",
    "weather_et0_sum",
    "weather_vpd_mean",
    "weather_cwb_sum",
)
FEATURE_GROUPS = {
    "weather_observed_days": "observation_support",
    "weather_expected_days": "observation_support",
    "weather_coverage": "observation_support",
    "weather_tmin_mean": "temperature",
    "weather_tmax_mean": "temperature",
    "weather_prec_sum": "water_balance",
    "weather_et0_sum": "water_balance",
    "weather_vpd_mean": "water_balance",
    "weather_cwb_sum": "water_balance",
    "weather_rad_sum": "radiation",
}
GROUP_INDICES = {
    group: [
        index
        for index, feature in enumerate(FEATURE_NAMES)
        if FEATURE_GROUPS[feature] == group
    ]
    for group in DISJOINT_GROUPS
}
ROUTES = (
    "supervised",
    "prediction_kd",
    "combined_kd",
    "representation_kd",
    "missing_aware",
)

ROUTE_LOSS_COEFFICIENTS = {
    "supervised": (0.0, 0.0, 0.0),
    "prediction_kd": (0.5, 0.0, 0.0),
    "representation_kd": (0.0, 0.25, 0.0),
    "combined_kd": (0.5, 0.25, 0.0),
    "missing_aware": (0.25, 0.0, 0.25),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def collapse_repeated_samples(
    rows: pd.DataFrame,
    *,
    value_columns: list[str],
) -> pd.DataFrame:
    required = {"contract", "route", "sample_id", *value_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"MISSING_COLLAPSE_COLUMNS:{sorted(missing)}")
    return (
        rows.groupby(
            ["contract", "route", "sample_id"],
            as_index=False,
            observed=True,
        )[value_columns]
        .mean()
        .sort_values(["contract", "route", "sample_id"])
        .reset_index(drop=True)
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x[np.isfinite(x)]).size < 2:
        return math.nan
    if np.unique(y[np.isfinite(y)]).size < 2:
        return math.nan
    result = stats.spearmanr(x, y, nan_policy="omit")
    return float(result.statistic)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 4999,
    seed: int = 20260709,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [
            np.mean(finite[rng.integers(0, finite.size, finite.size)])
            for _ in range(n_bootstrap)
        ],
        dtype=float,
    )
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def support_reliance_v2(
    support: pd.DataFrame,
    *,
    n_bootstrap: int = 4999,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"contract", "method", "seed", "sample_id", "P_support"}
    missing = required.difference(support.columns)
    if missing:
        raise ValueError(f"MISSING_SUPPORT_COLUMNS:{sorted(missing)}")
    baseline = support[support["method"] == "supervised"][
        ["contract", "seed", "sample_id", "P_support"]
    ].rename(columns={"P_support": "P_support_supervised"})
    compared = support[support["method"] != "supervised"].merge(
        baseline,
        on=["contract", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    compared["delta_P_support"] = (
        compared["P_support"] - compared["P_support_supervised"]
    )
    points = (
        compared.groupby(
            ["contract", "method", "sample_id"],
            as_index=False,
            observed=True,
        )["delta_P_support"]
        .mean()
        .rename(columns={"method": "route"})
        .sort_values(["contract", "route", "sample_id"])
        .reset_index(drop=True)
    )
    records = []
    for index, ((contract, route), group) in enumerate(
        points.groupby(["contract", "route"], observed=True)
    ):
        values = group["delta_P_support"].to_numpy(dtype=float)
        ci_lo, ci_hi = _bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=20260709 + index,
        )
        records.append(
            {
                "contract": contract,
                "route": route,
                "n_clusters": int(values.size),
                "mean_delta_support_share": float(np.mean(values)),
                "median_delta_support_share": float(np.median(values)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
            }
        )
    return points, pd.DataFrame(records)


def signed_feature_redistribution_v2(
    shap: pd.DataFrame,
    *,
    n_bootstrap: int = 4999,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "contract",
        "method",
        "seed",
        "sample_id",
        "feature_name",
        "permutation_shap",
    }
    missing = required.difference(shap.columns)
    if missing:
        raise ValueError(f"MISSING_SHAP_COLUMNS:{sorted(missing)}")
    baseline = shap[shap["method"] == "supervised"][
        [
            "contract",
            "seed",
            "sample_id",
            "feature_name",
            "permutation_shap",
        ]
    ].rename(columns={"permutation_shap": "shap_supervised"})
    compared = shap[shap["method"] != "supervised"].merge(
        baseline,
        on=["contract", "seed", "sample_id", "feature_name"],
        how="inner",
        validate="many_to_one",
    )
    compared["signed_delta"] = (
        compared["permutation_shap"] - compared["shap_supervised"]
    )
    compared["absolute_delta"] = compared["signed_delta"].abs()
    points = (
        compared.groupby(
            ["contract", "method", "sample_id", "feature_name"],
            as_index=False,
            observed=True,
        )[["signed_delta", "absolute_delta"]]
        .mean()
        .rename(columns={"method": "route"})
        .sort_values(["contract", "route", "sample_id", "feature_name"])
        .reset_index(drop=True)
    )
    records = []
    grouped = points.groupby(
        ["contract", "route", "feature_name"],
        observed=True,
    )
    for index, ((contract, route, feature), group) in enumerate(grouped):
        values = group["signed_delta"].to_numpy(dtype=float)
        ci_lo, ci_hi = _bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=20260709 + index,
        )
        records.append(
            {
                "contract": contract,
                "route": route,
                "feature_name": feature,
                "feature_group": FEATURE_GROUPS.get(feature, "unknown"),
                "n_clusters": int(values.size),
                "mean_signed_delta": float(np.mean(values)),
                "median_signed_delta": float(np.median(values)),
                "mean_absolute_delta": float(
                    group["absolute_delta"].mean()
                ),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return points, pd.DataFrame(records)


def linkage_v2(
    linkage: pd.DataFrame,
    *,
    n_permutations: int = 4999,
    n_bootstrap: int = 4999,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = collapse_repeated_samples(
        linkage,
        value_columns=["D_L1", "delta_AE"],
    )
    summary_records = []
    stability_records = []
    trim_records = []
    for index, ((contract, route), raw_group) in enumerate(
        linkage.groupby(["contract", "route"], observed=True)
    ):
        result = cluster_linkage(
            raw_group,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            seed=20260709 + index,
        )
        summary_records.append(
            {"contract": contract, "route": route, **result}
        )
        for seed in sorted(raw_group["seed"].unique()):
            seed_rows = raw_group[raw_group["seed"] == seed]
            stability_records.append(
                {
                    "contract": contract,
                    "route": route,
                    "stability_type": "single_seed",
                    "left_out_seed": int(seed),
                    "rho": _spearman(
                        seed_rows["D_L1"].to_numpy(dtype=float),
                        seed_rows["delta_AE"].to_numpy(dtype=float),
                    ),
                }
            )
            loo = collapse_repeated_samples(
                raw_group[raw_group["seed"] != seed],
                value_columns=["D_L1", "delta_AE"],
            )
            stability_records.append(
                {
                    "contract": contract,
                    "route": route,
                    "stability_type": "leave_one_seed_out",
                    "left_out_seed": int(seed),
                    "rho": _spearman(
                        loo["D_L1"].to_numpy(dtype=float),
                        loo["delta_AE"].to_numpy(dtype=float),
                    ),
                }
            )
        collapsed = points[
            (points["contract"] == contract) & (points["route"] == route)
        ]
        x = collapsed["D_L1"].to_numpy(dtype=float)
        y = collapsed["delta_AE"].to_numpy(dtype=float)
        x_lo, x_hi = np.quantile(x, [0.025, 0.975])
        y_lo, y_hi = np.quantile(y, [0.025, 0.975])
        keep = (
            (x >= x_lo)
            & (x <= x_hi)
            & (y >= y_lo)
            & (y <= y_hi)
        )
        trim_records.append(
            {
                "contract": contract,
                "route": route,
                "n_clusters_untrimmed": int(len(collapsed)),
                "n_clusters_trimmed": int(np.sum(keep)),
                "rho_untrimmed": _spearman(x, y),
                "rho_trimmed": _spearman(x[keep], y[keep]),
            }
        )
    summary = pd.DataFrame(summary_records)
    summary["p_holm"] = holm_adjust(summary["p_value"])
    stability = pd.DataFrame(stability_records)
    return points, summary, stability, pd.DataFrame(trim_records)


def shap_ig_agreement_v2(
    shap: pd.DataFrame,
    ig: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["contract", "method", "seed", "sample_id", "feature_name"]
    merged = shap[
        keys + ["permutation_shap"]
    ].merge(
        ig[keys + ["ig_value", "completeness_pass"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged["completeness_pass"] = (
        merged["completeness_pass"].astype(str).str.lower() == "true"
    )
    eligible = merged.groupby(keys[:-1], observed=True)[
        "completeness_pass"
    ].transform("all")
    merged = merged[eligible].copy()
    records = []
    for key, group in merged.groupby(keys[:-1], observed=True):
        shap_values = group["permutation_shap"].to_numpy(dtype=float)
        ig_values = group["ig_value"].to_numpy(dtype=float)
        top_k = min(3, len(group))
        shap_top = set(np.argsort(np.abs(shap_values))[-top_k:])
        ig_top = set(np.argsort(np.abs(ig_values))[-top_k:])
        records.append(
            {
                "contract": key[0],
                "method": key[1],
                "seed": int(key[2]),
                "sample_id": key[3],
                "rank_rho": _spearman(
                    np.abs(shap_values),
                    np.abs(ig_values),
                ),
                "sign_agreement": float(
                    np.mean(np.sign(shap_values) == np.sign(ig_values))
                ),
                "top3_overlap": float(len(shap_top & ig_top) / top_k),
            }
        )
    run_points = pd.DataFrame(records)
    points = (
        run_points.groupby(
            ["contract", "method", "sample_id"],
            as_index=False,
            observed=True,
        )[["rank_rho", "sign_agreement", "top3_overlap"]]
        .mean()
    )
    summary = (
        points.groupby(["contract", "method"], as_index=False, observed=True)
        .agg(
            n_clusters=("sample_id", "nunique"),
            mean_rank_rho=("rank_rho", "mean"),
            median_rank_rho=("rank_rho", "median"),
            mean_sign_agreement=("sign_agreement", "mean"),
            mean_top3_overlap=("top3_overlap", "mean"),
        )
    )
    return points, summary


def performance_landscape_v2(
    transfer_seed: pd.DataFrame,
    claim_resolution: pd.DataFrame,
) -> pd.DataFrame:
    required_transfer = {
        "contract",
        "method",
        "seed",
        "n",
        "mae",
        "rmse",
        "r2",
    }
    missing = required_transfer.difference(transfer_seed.columns)
    if missing:
        raise ValueError(f"MISSING_PERFORMANCE_COLUMNS:{sorted(missing)}")
    required_claim = {
        "split_id",
        "condition",
        "seed_index",
        "baseline_mae",
        "baseline_rmse",
        "baseline_r2",
    }
    missing = required_claim.difference(claim_resolution.columns)
    if missing:
        raise ValueError(f"MISSING_CLAIM_COLUMNS:{sorted(missing)}")
    scratch = claim_resolution[
        claim_resolution["condition"].astype(str) == "complete"
    ][
        [
            "split_id",
            "seed_index",
            "baseline_mae",
            "baseline_rmse",
            "baseline_r2",
        ]
    ].drop_duplicates()
    duplicate_metrics = (
        scratch.groupby(["split_id", "seed_index"], observed=True)
        .size()
        .gt(1)
    )
    if duplicate_metrics.any():
        raise ValueError("CONFLICTING_COMPLETE_SCRATCH_METRICS")
    scratch["contract"] = scratch["split_id"].astype(str) + "_complete"
    scratch["seed"] = scratch["seed_index"].map(
        {1: 101, 2: 202, 3: 303}
    )
    scratch["method"] = "local_scratch"
    scratch = scratch.rename(
        columns={
            "baseline_mae": "mae",
            "baseline_rmse": "rmse",
            "baseline_r2": "r2",
        }
    )
    sample_counts = transfer_seed[
        transfer_seed["method"] == "supervised"
    ][["contract", "seed", "n"]].drop_duplicates()
    scratch = scratch.merge(
        sample_counts,
        on=["contract", "seed"],
        how="left",
        validate="one_to_one",
    )
    combined = pd.concat(
        [
            transfer_seed[
                ["contract", "method", "seed", "n", "mae", "rmse", "r2"]
            ],
            scratch[
                ["contract", "method", "seed", "n", "mae", "rmse", "r2"]
            ],
        ],
        ignore_index=True,
    )
    scratch_baseline = combined[combined["method"] == "local_scratch"][
        ["contract", "seed", "mae"]
    ].rename(columns={"mae": "scratch_mae"})
    supervised_baseline = combined[combined["method"] == "supervised"][
        ["contract", "seed", "mae"]
    ].rename(columns={"mae": "supervised_mae"})
    combined = combined.merge(
        scratch_baseline,
        on=["contract", "seed"],
        how="left",
        validate="many_to_one",
    ).merge(
        supervised_baseline,
        on=["contract", "seed"],
        how="left",
        validate="many_to_one",
    )
    combined["delta_mae_vs_scratch"] = (
        combined["mae"] - combined["scratch_mae"]
    )
    combined["delta_mae_vs_supervised"] = (
        combined["mae"] - combined["supervised_mae"]
    )
    return combined.sort_values(
        ["contract", "method", "seed"]
    ).reset_index(drop=True)


def entropy_change_v2(
    concentration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "contract",
        "method",
        "seed",
        "sample_id",
        "normalized_entropy",
        "effective_feature_count",
    }
    missing = required.difference(concentration.columns)
    if missing:
        raise ValueError(f"MISSING_ENTROPY_COLUMNS:{sorted(missing)}")
    baseline = concentration[concentration["method"] == "supervised"][
        [
            "contract",
            "seed",
            "sample_id",
            "normalized_entropy",
            "effective_feature_count",
        ]
    ].rename(
        columns={
            "normalized_entropy": "entropy_supervised",
            "effective_feature_count": "effective_features_supervised",
        }
    )
    compared = concentration[
        concentration["method"] != "supervised"
    ].merge(
        baseline,
        on=["contract", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    compared["delta_entropy"] = (
        compared["normalized_entropy"] - compared["entropy_supervised"]
    )
    compared["delta_effective_feature_count"] = (
        compared["effective_feature_count"]
        - compared["effective_features_supervised"]
    )
    points = (
        compared.groupby(
            ["contract", "method", "sample_id"],
            as_index=False,
            observed=True,
        )[["delta_entropy", "delta_effective_feature_count"]]
        .mean()
        .rename(columns={"method": "route"})
    )
    summary = (
        points.groupby(["contract", "route"], as_index=False, observed=True)
        .agg(
            n_clusters=("sample_id", "nunique"),
            mean_delta_entropy=("delta_entropy", "mean"),
            median_delta_entropy=("delta_entropy", "median"),
            mean_delta_effective_features=(
                "delta_effective_feature_count",
                "mean",
            ),
        )
    )
    return points, summary


def cluster_linkage(
    rows: pd.DataFrame,
    *,
    n_permutations: int = 4999,
    n_bootstrap: int = 4999,
    seed: int = 20260709,
) -> dict[str, float | int]:
    collapsed = collapse_repeated_samples(
        rows,
        value_columns=["D_L1", "delta_AE"],
    )
    x = collapsed["D_L1"].to_numpy(dtype=float)
    y = collapsed["delta_AE"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    observed = _spearman(x, y)
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [_spearman(x, rng.permutation(y)) for _ in range(n_permutations)],
        dtype=float,
    )
    p_value = float(
        (1 + np.sum(np.abs(null) >= abs(observed)))
        / (n_permutations + 1)
    )
    bootstrap = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(x), len(x))
        value = _spearman(x[selected], y[selected])
        if np.isfinite(value):
            bootstrap.append(value)
    ci_lo, ci_hi = (
        np.quantile(bootstrap, [0.025, 0.975])
        if bootstrap
        else (np.nan, np.nan)
    )
    return {
        "n_rows": int(len(rows)),
        "n_clusters": int(len(x)),
        "rho": observed,
        "p_value": p_value,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
    }


def common_sample_counts(registry: pd.DataFrame) -> dict[str, int]:
    required = {"contract", "seed", "sample_id"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"MISSING_REGISTRY_COLUMNS:{sorted(missing)}")
    unique = registry[["contract", "sample_id"]].drop_duplicates()
    contract_sets = {
        contract: set(rows["sample_id"].astype(str))
        for contract, rows in unique.groupby("contract", observed=True)
    }
    overlap = set.intersection(*contract_sets.values()) if contract_sets else set()
    result = {
        "contract_seed_rows": int(len(registry)),
        "unique_samples": int(registry["sample_id"].nunique()),
        "cross_contract_overlap": int(len(overlap)),
    }
    for contract, sample_ids in contract_sets.items():
        result[f"{contract}_unique_samples"] = int(len(sample_ids))
    return result


def ig_baseline_semantics(
    *,
    baseline_scaled: np.ndarray,
    scaler_mean: np.ndarray,
) -> str:
    baseline = np.asarray(baseline_scaled, dtype=float)
    mean = np.asarray(scaler_mean, dtype=float)
    if np.allclose(baseline, 0.0) and np.isfinite(mean).all():
        return "TRAIN_SCALER_MEAN"
    return "OTHER_REFERENCE"


def taylor_fidelity_gate(
    quality: pd.DataFrame,
    *,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    values = pd.to_numeric(
        quality["taylor_relative_residual"],
        errors="coerce",
    )
    eligible = values[np.isfinite(values)]
    passed = eligible <= threshold
    return {
        "threshold": float(threshold),
        "n_eligible": int(len(eligible)),
        "n_pass": int(passed.sum()),
        "pass_rate": float(passed.mean()) if len(eligible) else math.nan,
    }


def stable_ood_distances(
    reference_raw: np.ndarray,
    query_raw: np.ndarray,
) -> dict[str, object]:
    reference = np.asarray(reference_raw, dtype=float).copy()
    query = np.asarray(query_raw, dtype=float).copy()
    if reference.ndim != 2 or query.ndim != 2:
        raise ValueError("OOD_INPUTS_MUST_BE_2D")
    if reference.shape[1] != query.shape[1]:
        raise ValueError("OOD_FEATURE_WIDTH_MISMATCH")
    medians = np.nanmedian(reference, axis=0)
    for matrix in (reference, query):
        missing = ~np.isfinite(matrix)
        if missing.any():
            row_index, column_index = np.where(missing)
            matrix[row_index, column_index] = medians[column_index]
    means = reference.mean(axis=0)
    scales = reference.std(axis=0, ddof=0)
    usable = np.isfinite(scales) & (scales > 1e-12)
    if usable.sum() < 2:
        raise ValueError("INSUFFICIENT_NONCONSTANT_OOD_FEATURES")
    reference_std = (reference[:, usable] - means[usable]) / scales[usable]
    query_std = (query[:, usable] - means[usable]) / scales[usable]
    estimator = LedoitWolf().fit(reference_std)
    centered = query_std - estimator.location_
    mahalanobis = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            centered,
            estimator.precision_,
            centered,
        ).clip(min=0.0)
    )
    nearest = cdist(query_std, reference_std).min(axis=1)
    return {
        "n_dimensions": int(usable.sum()),
        "usable_mask": usable,
        "mahalanobis": mahalanobis,
        "nearest_neighbor": nearest,
    }


def ig_completeness_rate(quality: pd.DataFrame) -> float:
    if "completeness_pass" not in quality.columns:
        raise ValueError("MISSING_IG_COMPLETENESS_PASS")
    values = quality["completeness_pass"]
    if values.dtype != bool:
        values = values.astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    if values.isna().any():
        raise ValueError("INVALID_IG_COMPLETENESS_PASS")
    return float(values.mean())


def early_split_ranking_v3(root: Path) -> pd.DataFrame:
    tables = root / "outputs/tables"
    regret = pd.read_csv(tables / "stage7_6_random_selected_model_regret.csv")
    trusted = pd.read_csv(tables / "stage7_trusted_result_index.csv")
    trusted = trusted[
        trusted["final_status"].eq("trusted_valid")
        & ~trusted["excluded_from_primary_analysis"].fillna(False).astype(bool)
    ].copy()
    trusted_keys = set(
        zip(
            trusted["experiment_id"].astype(str),
            trusted["view_id"].astype(str),
            trusted["model"].astype(str),
            trusted["validation_axis"].astype(str),
        )
    )
    records: list[dict[str, object]] = []
    for row in regret.itertuples(index=False):
        strict_key = (
            str(row.experiment_id),
            str(row.view_id),
            str(row.strict_best_model),
            str(row.validation_axis),
        )
        random_key = (
            str(row.experiment_id),
            str(row.view_id),
            str(row.random_selected_model),
            "random",
        )
        names = f"{row.strict_best_model} {row.random_selected_model}".lower()
        diagnostic = any(
            token in names
            for token in ("oracle", "gate", "diagnostic")
        )
        records.append(
            {
                "experiment_id": row.experiment_id,
                "dataset_track": row.track,
                "view_id": row.view_id,
                "deployment_axis": row.validation_axis,
                "axis_class": row.axis_class,
                "random_selected_model": row.random_selected_model,
                "deployment_best_model": row.strict_best_model,
                "random_selected_random_mae": row.random_selected_random_MAE,
                "random_selected_deployment_mae": row.random_selected_strict_MAE,
                "deployment_best_mae": row.strict_best_MAE,
                "deployment_regret_mae": row.deployment_regret_MAE,
                "ranking_reversal": bool(row.random_selection_reversal),
                "random_source_trusted": random_key in trusted_keys,
                "deployment_source_trusted": strict_key in trusted_keys,
                "diagnostic_model_involved": diagnostic,
                "evidence_role": (
                    "heterogeneous_diagnostic_motivation"
                    if diagnostic
                    else "heterogeneous_empirical_motivation"
                ),
                "source_table": "outputs/tables/stage7_6_random_selected_model_regret.csv",
            }
        )
    result = pd.DataFrame(records)
    if len(result) != 14 or int(result["ranking_reversal"].sum()) != 9:
        raise ValueError("EARLY_SPLIT_RANKING_SUMMARY_MISMATCH")
    if not result[
        ["random_source_trusted", "deployment_source_trusted"]
    ].all(axis=None):
        raise ValueError("EARLY_SPLIT_UNTRUSTED_LINEAGE")
    return result


def supervised_shap_beeswarm_v3(shap: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contract",
        "method",
        "seed",
        "sample_id",
        "feature_index",
        "feature_name",
        "feature_value_raw",
        "permutation_shap",
    }
    missing = required.difference(shap.columns)
    if missing:
        raise ValueError(f"MISSING_BEESWARM_COLUMNS:{sorted(missing)}")
    supervised = shap[shap["method"].eq("supervised")].copy()
    grouped = supervised.groupby(
        ["contract", "method", "sample_id", "feature_index", "feature_name"],
        observed=True,
    )
    spread = grouped["feature_value_raw"].agg(lambda values: values.max() - values.min())
    if (spread.abs() > 1e-9).any():
        raise ValueError("BEESWARM_RAW_VALUE_SEED_MISMATCH")
    collapsed = grouped.agg(
        feature_value_raw=("feature_value_raw", "mean"),
        permutation_shap=("permutation_shap", "mean"),
        seed_rows=("seed", "nunique"),
    ).reset_index()
    collapsed["feature_group"] = collapsed["feature_name"].map(FEATURE_GROUPS)
    collapsed["constant_within_contract"] = collapsed.groupby(
        ["contract", "feature_name"], observed=True
    )["feature_value_raw"].transform("nunique").eq(1)
    if collapsed["feature_group"].isna().any():
        raise ValueError("BEESWARM_UNMAPPED_FEATURE")
    return collapsed.sort_values(
        ["contract", "feature_index", "sample_id"]
    ).reset_index(drop=True)


def supervised_shap_beeswarm_v4(shap: pd.DataFrame) -> pd.DataFrame:
    """Collapse supervised rows and record honest feature-specific colour scales."""
    collapsed = supervised_shap_beeswarm_v3(shap).copy()
    collapsed["feature_value_colour"] = np.nan
    collapsed["feature_value_scale"] = "pooled_feature_p02_p98"
    for feature, rows in collapsed.groupby("feature_name", observed=True):
        values = rows["feature_value_raw"].to_numpy(dtype=float)
        lower, upper = np.nanpercentile(values, [2, 98])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            collapsed.loc[rows.index, "feature_value_scale"] = "constant"
            continue
        scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
        collapsed.loc[rows.index, "feature_value_colour"] = scaled
    constant = collapsed["constant_within_contract"].astype(bool)
    if collapsed.loc[~constant, "feature_value_colour"].isna().any():
        raise ValueError("BEESWARM_COLOUR_SCALE_MISSING")
    return collapsed.sort_values(
        ["contract", "feature_index", "sample_id"]
    ).reset_index(drop=True)


def _rank_agreement_rows(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
) -> pd.DataFrame:
    records = []
    keys = ["contract", "method", "seed", "sample_id"]
    for key, group in frame.groupby(keys, observed=True):
        if group["feature_name"].nunique() != len(FEATURE_NAMES):
            continue
        records.append(
            {
                "contract": key[0],
                "method": key[1],
                "seed": int(key[2]),
                "sample_id": key[3],
                "rank_rho": _spearman(
                    group[left].to_numpy(dtype=float),
                    group[right].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(records)


def explanation_agreement_v3(
    shap: pd.DataFrame,
    ig: pd.DataFrame,
    ig_quality: pd.DataFrame,
    taylor: pd.DataFrame,
    taylor_quality: pd.DataFrame,
) -> pd.DataFrame:
    quality = ig_quality[
        ig_quality["completeness_pass"].astype(str).str.lower().eq("true")
    ][["contract", "method", "seed", "sample_id"]]
    shap_columns = [
        "contract", "method", "seed", "sample_id", "feature_name",
        "permutation_shap",
    ]
    ig_columns = [
        "contract", "method", "seed", "sample_id", "feature_name", "ig_value",
    ]
    shap_ig = shap[shap_columns].merge(
        ig[ig_columns],
        on=["contract", "method", "seed", "sample_id", "feature_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        quality,
        on=["contract", "method", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    ig_rows = _rank_agreement_rows(
        shap_ig,
        left="permutation_shap",
        right="ig_value",
    )
    ig_rows["comparison"] = "SHAP--IG"
    ig_rows["quality_status"] = "completeness_qualified"

    qualified = taylor_quality[
        taylor_quality["taylor_relative_residual"].le(0.05)
    ][["contract", "method", "seed", "sample_id"]]
    taylor_rows = taylor[
        [
            "contract", "method", "seed", "sample_id", "feature_name",
            "permutation_shap", "taylor_shapley_contribution",
        ]
    ].merge(
        qualified,
        on=["contract", "method", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    taylor_rank = _rank_agreement_rows(
        taylor_rows,
        left="permutation_shap",
        right="taylor_shapley_contribution",
    )
    taylor_rank["comparison"] = "SHAP--Taylor"
    taylor_rank["quality_status"] = "local_fidelity_qualified_diagnostic"
    samples = pd.concat([ig_rows, taylor_rank], ignore_index=True)
    records = []
    for key, group in samples.groupby(
        ["comparison", "quality_status", "contract", "method", "seed"],
        observed=True,
    ):
        values = group["rank_rho"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "comparison": key[0],
                "quality_status": key[1],
                "contract": key[2],
                "method": key[3],
                "seed": int(key[4]),
                "n_qualified_samples": int(values.size),
                "mean_rank_rho": float(np.mean(values)) if values.size else math.nan,
                "median_rank_rho": float(np.median(values)) if values.size else math.nan,
                "q1_rank_rho": float(np.quantile(values, 0.25)) if values.size else math.nan,
                "q3_rank_rho": float(np.quantile(values, 0.75)) if values.size else math.nan,
                "display_estimate": bool(values.size >= 3),
            }
        )
    result = pd.DataFrame(records)
    expected_ig = len(ROUTES) * 2 * 3
    if len(result[result["comparison"].eq("SHAP--IG")]) != expected_ig:
        raise ValueError("INCOMPLETE_SHAP_IG_CELL_AGREEMENT")
    return result.sort_values(
        ["comparison", "contract", "method", "seed"]
    ).reset_index(drop=True)


def route_loss_coefficients_v3() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route": route,
                "lambda_prediction": weights[0],
                "lambda_representation": weights[1],
                "lambda_consistency": weights[2],
                "teacher_required": route != "supervised",
                "target_adaptation": "supervised_fine_tuning_only",
            }
            for route, weights in ROUTE_LOSS_COEFFICIENTS.items()
        ]
    )


def experiment_map_v4(root: Path) -> pd.DataFrame:
    """Describe the retained evidence without pooling incompatible studies."""
    records = [
        {
            "experiment_family": "Early split-selection inventory",
            "dataset_or_case": "Heterogeneous Stage 7 tracks",
            "sample_unit": "Track-specific; not pooled",
            "target_and_modalities": "Track-specific yield interfaces",
            "split_or_contract": "Random versus deployment-oriented axes",
            "methods_or_routes": "HGB, RF, Gate, Oracle, Best single",
            "seeds": "Track-specific",
            "metrics": "MAE; within-row only",
            "scientific_role": "Motivation",
            "evidence_tier": "Tier 2",
            "provenance_status": "trusted lineage; empirical and diagnostic rows separated",
            "source_artifact": "outputs/tables/stage7_6_random_selected_model_regret.csv",
            "manuscript_location": "Supplement: Early Split-Selection Motivation",
        },
        {
            "experiment_family": "US-maize to Australian-wheat transfer",
            "dataset_or_case": "CY-Bench regional crop-year target",
            "sample_unit": "Australian wheat region-year",
            "target_and_modalities": "Yield; ten weather features",
            "split_or_contract": "GROUP complete; SPATIAL complete",
            "methods_or_routes": "Scratch, supervised, prediction, representation, combined, missing-aware",
            "seeds": "101, 202, 303",
            "metrics": "MAE, RMSE, R2",
            "scientific_role": "Primary transfer evidence",
            "evidence_tier": "Tier 1",
            "provenance_status": "run-level matched and rebuilt",
            "source_artifact": "outputs/common_sample_attribution_v1/run_20260708_212657/sample_level_common_shap.csv",
            "manuscript_location": "Main: Experimental Setup and Results",
        },
        {
            "experiment_family": "Common-sample attribution and linkage",
            "dataset_or_case": "CY-Bench Australian wheat transfer",
            "sample_unit": "sample_id cluster after seed collapse",
            "target_and_modalities": "Yield; four disjoint weather groups",
            "split_or_contract": "GROUP complete; SPATIAL complete",
            "methods_or_routes": "Supervised reference and four non-supervised routes",
            "seeds": "101, 202, 303",
            "metrics": "Permutation-SHAP shares, d_phi, Delta e",
            "scientific_role": "Behavioural redistribution",
            "evidence_tier": "Tier 1",
            "provenance_status": "cluster-aware rebuilt",
            "source_artifact": "outputs/common_sample_attribution_v1/run_20260708_212657/attribution_error_linkage.csv",
            "manuscript_location": "Main: RQ2--RQ3",
        },
        {
            "experiment_family": "Perturbation, OOD, IG, and Taylor checks",
            "dataset_or_case": "CY-Bench Australian wheat transfer",
            "sample_unit": "route-contract-seed cell or qualified sample cluster",
            "target_and_modalities": "Yield; four disjoint weather groups",
            "split_or_contract": "GROUP complete; SPATIAL complete",
            "methods_or_routes": "All retained transfer routes",
            "seeds": "101, 202, 303",
            "metrics": "normalised stress, OOD distance, IG completeness, Taylor residual",
            "scientific_role": "Validity boundary",
            "evidence_tier": "Tier 1 diagnostic",
            "provenance_status": "9D Ledoit--Wolf; Taylor diagnostic only",
            "source_artifact": "outputs/common_sample_attribution_validation_v1/run_20260709_090144/perturbation_normalized_sensitivity.csv",
            "manuscript_location": "Main: RQ4; Supplement: explanation quality",
        },
        {
            "experiment_family": "Local missing-modality random forests",
            "dataset_or_case": "CY-Bench Australian wheat",
            "sample_unit": "target-test observation",
            "target_and_modalities": "Yield; weather and soil",
            "split_or_contract": "Locked complete, 15% random-missing, no-weather settings",
            "methods_or_routes": "Joint-feature RF, branch average, single-modality controls",
            "seeds": "101, 202, 303",
            "metrics": "MAE",
            "scientific_role": "Strong local configuration",
            "evidence_tier": "Tier 3",
            "provenance_status": "secondary; no algorithmic fusion claim",
            "source_artifact": "outputs/tables/stage7_modality_subview_metrics.csv",
            "manuscript_location": "Supplement: Secondary Missing-Modality Evidence",
        },
        {
            "experiment_family": "Privileged-information teacher--student",
            "dataset_or_case": "CY-Bench Australian wheat",
            "sample_unit": "target-test observation",
            "target_and_modalities": "Yield; complete teacher and weather-only student",
            "split_or_contract": "GROUP no-soil and supplementary complete setting",
            "methods_or_routes": "Observed target, pseudo target, fixed blend",
            "seeds": "101, 202, 303",
            "metrics": "MAE",
            "scientific_role": "Boundary evidence of potential",
            "evidence_tier": "Tier 3",
            "provenance_status": "all seeds retained; no stable teacher-effect claim",
            "source_artifact": "outputs/stage8_claim_resolution_v1/analysis/mechanism_compliance_audit.csv",
            "manuscript_location": "Supplement: Privileged-Information Teacher--Student",
        },
        {
            "experiment_family": "Roseworthy E5",
            "dataset_or_case": "Roseworthy within-paddock and historical views",
            "sample_unit": "point-level or rolling historical observation",
            "target_and_modalities": "Yield; weather and soil views",
            "split_or_contract": "View-specific; not regional transfer",
            "methods_or_routes": "Secondary local models",
            "seeds": "Protocol-specific",
            "metrics": "View-specific prediction metrics",
            "scientific_role": "Australian boundary case",
            "evidence_tier": "Tier 3",
            "provenance_status": "CC BY-NC academic use confirmed; exact cleaned-file provenance pending",
            "source_artifact": "outputs/tables/stage7_metric_index.csv",
            "manuscript_location": "Supplement: Supporting Agricultural Cases",
        },
        {
            "experiment_family": "Waite historical case",
            "dataset_or_case": "Waite trial long-history view",
            "sample_unit": "single-station historical trial observation",
            "target_and_modalities": "Yield; near-constant soil descriptors in analysed view",
            "split_or_contract": "Temporal forward and diagnostic plot-group",
            "methods_or_routes": "Naive, ridge, RF, HGB variants",
            "seeds": "101, 202, 303 where recorded",
            "metrics": "MAE, RMSE, R2",
            "scientific_role": "Boundary case",
            "evidence_tier": "Tier 3",
            "provenance_status": "trusted supporting rows; no soil-contribution claim",
            "source_artifact": "outputs/tables/stage7_metric_index.csv",
            "manuscript_location": "Supplement: Supporting Agricultural Cases",
        },
        {
            "experiment_family": "Brazilian CY-Bench cases",
            "dataset_or_case": "Brazilian crop-yield target cases",
            "sample_unit": "CY-Bench regional crop-year",
            "target_and_modalities": "Yield; harmonised weather interface",
            "split_or_contract": "Historical case-specific protocol",
            "methods_or_routes": "Not retained as a primary comparison",
            "seeds": "Not carried into final inference",
            "metrics": "No final headline metric retained",
            "scientific_role": "Historical cross-country context",
            "evidence_tier": "Tier 3",
            "provenance_status": "no row-level final metric in retained index; not pooled",
            "source_artifact": "outputs/tables/stage7_metric_index.csv",
            "manuscript_location": "Supplement: Supporting Agricultural Cases",
        },
    ]
    frame = pd.DataFrame(records)
    for relative in frame["source_artifact"]:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"EXPERIMENT_MAP_SOURCE_MISSING:{relative}")
    return frame


def experiment_map_public_v4_1(root: Path) -> pd.DataFrame:
    """Return only reader-facing fields; internal provenance stays out of tables."""
    frame = experiment_map_v4(root)
    role = {
        "Tier 1": "Primary evidence",
        "Tier 1 diagnostic": "Primary diagnostic evidence",
        "Tier 2": "Motivating evidence",
        "Tier 3": "Secondary or boundary evidence",
    }
    dataset = frame["dataset_or_case"].replace(
        {"Heterogeneous Stage 7 tracks": "Heterogeneous earlier study set"}
    )
    return pd.DataFrame(
        {
            "study_component": frame["experiment_family"],
            "dataset_and_sample_unit": dataset + "; " + frame["sample_unit"],
            "task_and_deployment_setting": frame["split_or_contract"] + "; " + frame["target_and_modalities"],
            "methods": frame["methods_or_routes"],
            "retained_result": frame["metrics"] + "; " + frame["scientific_role"],
            "role": frame["evidence_tier"].map(role).fillna("Secondary or boundary evidence"),
            "reproducibility_note": "Protocol and sample unit are described in this supplement; no rows are pooled across study components.",
        }
    )


def experiment_map_public_v4_4(root: Path) -> pd.DataFrame:
    """Return a reader-facing map without internal tracking fields."""
    frame = experiment_map_v4(root)
    roles = {
        "Tier 1": "Primary evidence",
        "Tier 1 diagnostic": "Secondary validity analysis",
        "Tier 2": "Motivating evidence",
        "Tier 3": "Secondary or boundary evidence",
    }
    observations = {
        "Early split-selection inventory": "Deployment-oriented partitions can change the model selected by an interpolation-oriented comparison.",
        "US-maize to Australian-wheat transfer": "Prediction KD improves over the fixed scratch reference under SPATIAL and shows negative transfer under GROUP.",
        "Common-sample attribution and linkage": "Information-use changes depend on route and contract; one pooled attribution--error association survives family correction.",
        "Perturbation, OOD, IG, and Taylor checks": "Attribution and finite stress rankings align weakly; perturbations move inputs away from well-represented training regions.",
        "Local missing-modality random forests": "A joint-feature RF is the strongest local configuration in the retained missing-modality comparison.",
        "Privileged-information teacher--student": "Teacher supervision is run-sensitive and does not support a stable component-level claim.",
        "Roseworthy E5": "Within-paddock results provide an Australian secondary case.",
        "Waite historical case": "The long-history view provides a boundary case with weakly varying soil descriptors.",
        "Brazilian CY-Bench cases": "The historical cases motivate cross-country questions but do not enter the primary quantitative comparison.",
    }
    limitations = {
        "Early split-selection inventory": "Datasets, model sets, split axes, and native metrics differ; diagnostic and oracle comparisons are reported separately.",
        "US-maize to Australian-wheat transfer": "Three transferred runs use one fixed SPATIAL scratch reference; absolute SPATIAL R2 remains negative.",
        "Common-sample attribution and linkage": "Attribution is background-dependent and does not identify causal feature effects.",
        "Perturbation, OOD, IG, and Taylor checks": "Finite replacement is distribution-shifting, and Taylor is usable only on the small fidelity-qualified subset.",
        "Local missing-modality random forests": "This supports a local configuration comparison, not a new fusion algorithm.",
        "Privileged-information teacher--student": "All runs are retained and the teacher effect is not stable.",
        "Roseworthy E5": "The sample unit differs from the regional target, and exact cleaned-file correspondence remains unresolved.",
        "Waite historical case": "The single-station setting and near-constant soil descriptors limit generalisation.",
        "Brazilian CY-Bench cases": "No compatible row-level headline result is used in the final comparison.",
    }
    dataset = frame["dataset_or_case"].replace(
        {"Heterogeneous Stage 7 tracks": "Heterogeneous earlier study set"}
    )
    return pd.DataFrame(
        {
            "study_component": frame["experiment_family"],
            "dataset_and_unit": dataset + "; " + frame["sample_unit"],
            "deployment_setting": frame["split_or_contract"],
            "methods": frame["methods_or_routes"],
            "main_observation": frame["experiment_family"].map(observations),
            "role_in_paper": frame["evidence_tier"].map(roles),
            "limitation": frame["experiment_family"].map(limitations),
        }
    )


def scratch_baseline_audit(root: Path, performance_landscape: pd.DataFrame) -> pd.DataFrame:
    """Classify the repeated SPATIAL scratch value without exposing identifiers."""
    spatial = performance_landscape[
        (performance_landscape["contract"] == "SPATIAL_complete")
        & (performance_landscape["method"] == "local_scratch")
    ]
    metric_equal = len(spatial) == 3 and all(
        bool(np.isclose(spatial[col].astype(float), spatial[col].iloc[0]).all())
        for col in ("mae", "rmse", "r2")
    )
    lineage_path = root / "outputs/stage8_representation_support_v1/checkpoint_lineage.csv"
    replay_path = root / "outputs/stage8_representation_support_v1/prediction_replay_validation.csv"
    classification = "unresolved"
    replay_pass = 0
    identical_prediction_files = False
    if lineage_path.is_file() and replay_path.is_file():
        lineage = pd.read_csv(lineage_path)
        replay = pd.read_csv(replay_path)
        rows = lineage[
            (lineage["split_id"] == "SPATIAL")
            & (lineage["transfer_strategy"] == "target_scratch")
            & (lineage["missing_modality_method"] == "imputation")
        ].merge(replay, on=["candidate_id", "split_id", "seed", "transfer_strategy"])
        replay_pass = int((rows["reproduction_status"] == "PASS").sum())
        identical_prediction_files = (
            len(rows) == 3
            and rows["official_predictions_sha256"].nunique() == 1
            and rows["row_count"].astype(int).eq(31).all()
        )
        if metric_equal and replay_pass == 3 and identical_prediction_files:
            classification = "fixed_reference_reused"
    return pd.DataFrame(
        [{
            "contract": "SPATIAL",
            "route": "Scratch",
            "classification": classification,
            "metric_rows": int(len(spatial)),
            "replay_pass_rows": replay_pass,
            "prediction_files_identical": bool(identical_prediction_files),
            "public_display": "1.204 (fixed baseline)" if classification == "fixed_reference_reused" else "unresolved",
        }]
    )


def source_protocol_v4_4(root: Path, formal_root: Path) -> pd.DataFrame:
    """Reconcile source training, target adaptation, and SHAP background facts."""
    primary_root = Path(os.environ.get("AGRITECH_PRIMARY_RESULT_ROOT", root))
    source_data = (
        formal_root
        / "outputs/distillation_program/stage8_v4_australian_narrative_v1"
        / "universal_weather_full_campaign_v2/adapted_views"
        / "PRIMARY_CYBENCH_MAIZE_US.csv.gz"
    )
    frame = pd.read_csv(source_data, usecols=["year", "sample_id"])
    years = sorted(frame["year"].dropna().astype(int).unique())
    if years != list(range(2001, 2024)):
        raise ValueError(f"SOURCE_YEAR_RANGE_MISMATCH:{years}")
    source_counts = {
        "source_train_rows": int((frame["year"] < 2022).sum()),
        "source_validation_rows": int((frame["year"] == 2022).sum()),
        "source_test_rows": int((frame["year"] == 2023).sum()),
    }
    if source_counts != {
        "source_train_rows": 35282,
        "source_validation_rows": 1512,
        "source_test_rows": 1430,
    }:
        raise ValueError(f"SOURCE_SPLIT_COUNT_MISMATCH:{source_counts}")

    training_code = (
        formal_root
        / "src/distillation_v4/universal_weather_v2/training.py"
    ).read_text(encoding="utf-8")
    execution_code = (
        formal_root
        / "src/distillation_v4/universal_weather_v2_remediation/execution.py"
    ).read_text(encoding="utf-8")
    required_training_markers = (
        "freeze_teacher(teacher)",
        "student = build_student_v2()",
        "mask = torch.rand_like",
        "< 0.2",
        "best_validation - 1e-8",
    )
    required_execution_markers = (
        'route="supervised"',
        '"ROUTE_SPECIFIC_SOURCE_CHECKPOINT"',
        'target_test_used_for_training": False',
        'target_test_used_for_selection": False',
    )
    if any(marker not in training_code for marker in required_training_markers):
        raise ValueError("SOURCE_TRAINING_PROTOCOL_MARKER_MISSING")
    if any(marker not in execution_code for marker in required_execution_markers):
        raise ValueError("TARGET_ADAPTATION_PROTOCOL_MARKER_MISSING")

    background = pd.read_csv(
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "background_registry.csv"
    )
    embedding = pd.read_csv(
        root / "outputs/stage8_representation_support_v1/embedding_manifest.csv"
    )
    checkpoint_lineage = pd.read_csv(
        root / "outputs/stage8_representation_support_v1/checkpoint_lineage.csv"
    )
    cells = (
        embedding[["contract", "method", "seed", "candidate_id", "file_path"]]
        .drop_duplicates()
        .merge(checkpoint_lineage, on=["candidate_id", "seed"], how="left")
    )
    replay_root = (
        primary_root
        / "outputs/stage8_phase3b_confirmatory_replay_v3_expanded"
        / "jobs/replay_artifacts"
    )
    background_rows: list[dict[str, object]] = []
    for (contract, seed), group in cells.groupby(
        ["contract", "seed"], observed=True
    ):
        if contract not in {"GROUP_complete", "SPATIAL_complete"}:
            continue
        supervised = group[group["method"] == "supervised"]
        if len(supervised) != 1:
            raise ValueError(f"SUPERVISED_CELL_UNRESOLVED:{contract}:{seed}")
        candidate_id = str(supervised.iloc[0]["candidate_id"])
        folds = pd.read_csv(replay_root / candidate_id / "fold_assignments.csv")
        selected = set(
            background[
                (background["contract"] == contract)
                & (background["seed"].astype(int) == int(seed))
            ]["sample_id"].astype(str)
        )
        train_ids = set(folds.loc[folds["split"] == "train", "sample_id"].astype(str))
        validation_ids = set(
            folds.loc[folds["split"] == "validation", "sample_id"].astype(str)
        )
        test_ids = set(folds.loc[folds["split"] == "test", "sample_id"].astype(str))
        counts = {
            "background_rows": len(selected),
            "training_rows": len(selected & train_ids),
            "validation_rows": len(selected & validation_ids),
            "test_rows": len(selected & test_ids),
        }
        if counts["background_rows"] != 32 or counts["test_rows"] != 0:
            raise ValueError(f"SHAP_BACKGROUND_NOT_TEST_DISJOINT:{contract}:{seed}:{counts}")
        background_rows.append(
            {
                "item": "shap_background",
                "contract": contract,
                "run": f"S{[101, 202, 303].index(int(seed)) + 1}",
                **counts,
                "description": "First 32 non-test development rows in dataset order",
            }
        )

    summary_rows = [
        {
            "item": key,
            "contract": "SOURCE",
            "run": "all",
            "background_rows": value,
            "training_rows": "",
            "validation_rows": "",
            "test_rows": "",
            "description": "Fixed temporal source split",
        }
        for key, value in source_counts.items()
    ]
    return pd.DataFrame([*summary_rows, *background_rows])


def disjoint_group_agreement(
    attribution: pd.DataFrame,
    perturbation: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["contract", "method", "seed", "group"]
    left = attribution[attribution["group"].isin(DISJOINT_GROUPS)].copy()
    right = perturbation[perturbation["group"].isin(DISJOINT_GROUPS)].copy()
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for key, group in merged.groupby(keys[:3], observed=True):
        if set(group["group"]) != set(DISJOINT_GROUPS):
            continue
        attr = group["attribution_share"].to_numpy(dtype=float)
        stress = group["standardized_sensitivity"].to_numpy(dtype=float)
        top_attr = str(group.iloc[int(np.argmax(attr))]["group"])
        top_stress = str(group.iloc[int(np.argmax(stress))]["group"])
        rows.append(
            {
                "contract": key[0],
                "method": key[1],
                "seed": int(key[2]),
                "n_groups": int(len(group)),
                "spearman_rho": _spearman(attr, stress),
                "top_attribution_group": top_attr,
                "top_stress_group": top_stress,
                "top_group_match": top_attr == top_stress,
            }
        )
    return pd.DataFrame(rows)


def _load_common_predictions(shap_path: Path) -> pd.DataFrame:
    columns = [
        "contract",
        "method",
        "seed",
        "sample_id",
        "sample_prediction",
        "target_yield",
        "absolute_error",
    ]
    frame = pd.read_csv(shap_path, usecols=columns)
    return frame.drop_duplicates(
        ["contract", "method", "seed", "sample_id"]
    )


def performance_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for key, rows in predictions.groupby(
        ["contract", "method", "seed"],
        observed=True,
    ):
        y = rows["target_yield"].to_numpy(dtype=float)
        pred = rows["sample_prediction"].to_numpy(dtype=float)
        errors = pred - y
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        seed_rows.append(
            {
                "contract": key[0],
                "method": key[1],
                "seed": int(key[2]),
                "n": int(len(rows)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "r2": (
                    float(1 - np.sum(errors**2) / denominator)
                    if denominator > 0
                    else np.nan
                ),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    aggregate = (
        seed_frame.groupby(["contract", "method"], observed=True)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )
    return seed_frame, aggregate


def support_reliance_clustered(support: pd.DataFrame) -> pd.DataFrame:
    baseline = support[support["method"] == "supervised"][
        ["contract", "seed", "sample_id", "P_support"]
    ].rename(columns={"P_support": "P_support_supervised"})
    compared = support[support["method"] != "supervised"].merge(
        baseline,
        on=["contract", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    compared["delta_P_support"] = (
        compared["P_support"] - compared["P_support_supervised"]
    )
    collapsed = (
        compared.groupby(
            ["contract", "method", "sample_id"],
            as_index=False,
            observed=True,
        )["delta_P_support"]
        .mean()
    )
    rows = []
    for key, group in collapsed.groupby(["contract", "method"], observed=True):
        values = group["delta_P_support"].to_numpy(dtype=float)
        if np.allclose(values, 0):
            p_value = 1.0
        else:
            p_value = float(
                stats.wilcoxon(
                    values,
                    alternative="two-sided",
                    zero_method="zsplit",
                ).pvalue
            )
        seed_means = (
            compared[
                (compared["contract"] == key[0])
                & (compared["method"] == key[1])
            ]
            .groupby("seed", observed=True)["delta_P_support"]
            .mean()
        )
        rows.append(
            {
                "contract": key[0],
                "route": key[1],
                "n_rows": int(
                    len(
                        compared[
                            (compared["contract"] == key[0])
                            & (compared["method"] == key[1])
                        ]
                    )
                ),
                "n_clusters": int(len(values)),
                "mean_delta_support_share": float(np.mean(values)),
                "median_delta_support_share": float(np.median(values)),
                "p_value": p_value,
                "seed_direction_consistency": float(
                    np.mean(
                        np.sign(seed_means)
                        == np.sign(np.mean(seed_means))
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"])
    return result


def linkage_clustered(linkage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (key, group) in enumerate(
        linkage.groupby(["contract", "route"], observed=True)
    ):
        result = cluster_linkage(group, seed=20260709 + index)
        seed_rhos = [
            _spearman(
                seed_rows["D_L1"].to_numpy(dtype=float),
                seed_rows["delta_AE"].to_numpy(dtype=float),
            )
            for _, seed_rows in group.groupby("seed", observed=True)
        ]
        rows.append(
            {
                "contract": key[0],
                "route": key[1],
                **result,
                "seed_rho_mean": float(np.nanmean(seed_rhos)),
                "seed_rho_min": float(np.nanmin(seed_rhos)),
                "seed_rho_max": float(np.nanmax(seed_rhos)),
                "seed_direction_consistency": float(
                    np.mean(
                        np.sign(seed_rhos)
                        == np.sign(np.nanmean(seed_rhos))
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_value"])
    return frame


def attribution_group_shares(shap_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        shap_path,
        usecols=[
            "contract",
            "method",
            "seed",
            "sample_id",
            "feature_name",
            "permutation_shap",
        ],
    )
    frame["group"] = frame["feature_name"].map(FEATURE_GROUPS)
    if frame["group"].isna().any():
        missing = sorted(frame.loc[frame["group"].isna(), "feature_name"].unique())
        raise ValueError(f"UNMAPPED_FEATURES:{missing}")
    frame["absolute_attribution"] = frame["permutation_shap"].abs()
    grouped = (
        frame.groupby(
            ["contract", "method", "seed", "group"],
            as_index=False,
            observed=True,
        )["absolute_attribution"]
        .mean()
    )
    totals = grouped.groupby(
        ["contract", "method", "seed"],
        observed=True,
    )["absolute_attribution"].transform("sum")
    grouped["attribution_share"] = grouped["absolute_attribution"] / totals
    return grouped


def rebuild_ood_quality(
    *,
    root: Path,
    formal_root: Path,
) -> pd.DataFrame:
    common_dir = (
        root / "outputs/common_sample_attribution_v1/run_20260708_212657"
    )
    raw_path = (
        formal_root
        / "outputs/distillation_program/stage8_v4_australian_narrative_v1"
        / "universal_weather_full_campaign_v2/adapted_views"
        / "CY-Bench_wheat_AU.csv.gz"
    )
    if not raw_path.is_file():
        raise FileNotFoundError(f"MISSING_FORMAL_RAW_VIEW:{raw_path}")
    raw = pd.read_csv(raw_path)
    shap = pd.read_csv(
        common_dir / "sample_level_common_shap.csv",
        usecols=[
            "contract",
            "method",
            "seed",
            "sample_id",
            "feature_index",
            "feature_name",
            "feature_value_raw",
        ],
    )
    shap = shap[shap["method"] == "supervised"]
    preprocessor = pd.read_csv(
        root
        / "outputs/exact_raw_weather_shap_taylor_v3"
        / "run_20260708_181605/formal_preprocessor_parameters.csv"
    )
    rows = []
    for (contract, seed), cell in shap.groupby(
        ["contract", "seed"],
        observed=True,
    ):
        split_id = str(contract).replace("_complete", "")
        params = preprocessor[
            (preprocessor["split_id"] == split_id)
            & (preprocessor["seed"] == seed)
        ].sort_values("feature_index")
        feature_names = params["feature_name"].astype(str).tolist()
        if feature_names != list(FEATURE_NAMES):
            raise ValueError(
                f"FEATURE_ORDER_MISMATCH:{contract}:{seed}:{feature_names}"
            )
        medians = params["imputer_median"].to_numpy(dtype=float)
        test_ids = set(cell["sample_id"].astype(str))
        reference = raw[~raw["sample_id"].astype(str).isin(test_ids)][
            feature_names
        ].to_numpy(dtype=float)
        missing = ~np.isfinite(reference)
        if missing.any():
            row_index, column_index = np.where(missing)
            reference[row_index, column_index] = medians[column_index]
        ordered = (
            cell.sort_values(["sample_id", "feature_index"])
            .pivot(
                index="sample_id",
                columns="feature_index",
                values="feature_value_raw",
            )
            .sort_index(axis=1)
        )
        original = ordered.to_numpy(dtype=float)
        baseline = stable_ood_distances(reference, original)
        for group, indices in GROUP_INDICES.items():
            perturbed = original.copy()
            perturbed[:, indices] = medians[indices]
            shifted = stable_ood_distances(reference, perturbed)
            if shifted["n_dimensions"] != baseline["n_dimensions"]:
                raise ValueError("OOD_DIMENSION_DRIFT")
            rows.append(
                {
                    "contract": contract,
                    "seed": int(seed),
                    "perturbation_group": group,
                    "n_test_samples": int(len(original)),
                    "n_reference_samples": int(len(reference)),
                    "n_dimensions": int(baseline["n_dimensions"]),
                    "reference_scope": "NON_TEST_POOL",
                    "covariance": "LEDOIT_WOLF",
                    "mean_baseline_mahalanobis": float(
                        np.mean(baseline["mahalanobis"])
                    ),
                    "mean_perturbed_mahalanobis": float(
                        np.mean(shifted["mahalanobis"])
                    ),
                    "mean_baseline_nn_distance": float(
                        np.mean(baseline["nearest_neighbor"])
                    ),
                    "mean_perturbed_nn_distance": float(
                        np.mean(shifted["nearest_neighbor"])
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not (result["n_dimensions"] == 9).all():
        raise ValueError(
            f"UNEXPECTED_OOD_DIMENSIONS:{sorted(result.n_dimensions.unique())}"
        )
    return result


def three_view_route_pairs(
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    normalized_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    baseline_predictions = predictions[predictions["method"] == "supervised"][
        [
            "contract",
            "seed",
            "sample_id",
            "sample_prediction",
            "absolute_error",
        ]
    ].rename(
        columns={
            "sample_prediction": "supervised_prediction",
            "absolute_error": "supervised_absolute_error",
        }
    )
    paired = predictions[predictions["method"] != "supervised"].merge(
        baseline_predictions,
        on=["contract", "seed", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    paired["delta_absolute_error"] = (
        paired["absolute_error"] - paired["supervised_absolute_error"]
    )
    paired["prediction_difference"] = (
        paired["sample_prediction"] - paired["supervised_prediction"]
    ).abs()
    pred_summary = (
        paired.groupby(["contract", "method", "seed"], observed=True)
        .agg(
            mean_delta_absolute_error=("delta_absolute_error", "mean"),
            fraction_error_improved=(
                "delta_absolute_error",
                lambda values: float(np.mean(np.asarray(values) < 0)),
            ),
            mean_absolute_prediction_difference=(
                "prediction_difference",
                "mean",
            ),
        )
        .reset_index()
        .rename(columns={"method": "route"})
    )
    redistribution = (
        linkage.groupby(["contract", "route", "seed"], observed=True)["D_L1"]
        .mean()
        .reset_index(name="mean_attribution_redistribution_l1")
    )
    disjoint = normalized_sensitivity[
        normalized_sensitivity["perturbation_group"].isin(DISJOINT_GROUPS)
    ].copy()
    supervised = disjoint[disjoint["method"] == "supervised"][
        [
            "contract",
            "seed",
            "perturbation_group",
            "standardized_sensitivity",
        ]
    ].rename(
        columns={
            "standardized_sensitivity": "supervised_standardized_sensitivity"
        }
    )
    stress = disjoint[disjoint["method"] != "supervised"].merge(
        supervised,
        on=["contract", "seed", "perturbation_group"],
        how="inner",
        validate="many_to_one",
    )
    stress["absolute_stress_response_difference"] = (
        stress["standardized_sensitivity"]
        - stress["supervised_standardized_sensitivity"]
    ).abs()
    stress_summary = (
        stress.groupby(["contract", "method", "seed"], observed=True)[
            "absolute_stress_response_difference"
        ]
        .mean()
        .reset_index(name="mean_stress_response_difference")
        .rename(columns={"method": "route"})
    )
    return (
        pred_summary.merge(
            redistribution,
            on=["contract", "route", "seed"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            stress_summary,
            on=["contract", "route", "seed"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["contract", "route", "seed"])
        .reset_index(drop=True)
    )


def validate_lineage(root: Path) -> pd.DataFrame:
    path = root / "outputs/stage8_representation_support_v1/checkpoint_lineage.csv"
    lineage = pd.read_csv(path)
    rows = []
    for _, row in lineage.iterrows():
        for kind, path_column, hash_column in (
            ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
            (
                "official_predictions",
                "official_predictions_path",
                "official_predictions_sha256",
            ),
        ):
            value = str(row.get(path_column, ""))
            expected = str(row.get(hash_column, ""))
            if value in {"", "N/A", "nan", "None"}:
                status = "NOT_APPLICABLE"
                actual = ""
            else:
                candidate = Path(value)
                if not candidate.is_file():
                    status = "MISSING"
                    actual = ""
                else:
                    actual = sha256_file(candidate)
                    status = "PASS" if actual == expected else "HASH_MISMATCH"
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "split_id": row["split_id"],
                    "seed": int(row["seed"]),
                    "source_route": row["source_route"],
                    "artifact_kind": kind,
                    "path": value,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def build_registry(
    *,
    root: Path,
    counts: dict[str, int],
    performance: pd.DataFrame,
    support: pd.DataFrame,
    linkage: pd.DataFrame,
    agreement: pd.DataFrame,
    taylor_gate: dict[str, float | int],
    ig_pass_rate: float,
    claim_resolution: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(
        claim_id: str,
        topic: str,
        value: object,
        unit: str,
        source: str,
        method_source: str,
        status: str,
        boundary: str,
    ) -> None:
        records.append(
            {
                "claim_id": claim_id,
                "manuscript_topic": topic,
                "value": value,
                "unit": unit,
                "source_artifact": source,
                "method_or_config_source": method_source,
                "verification_status": status,
                "claim_boundary": boundary,
            }
        )

    registry_source = (
        "outputs/common_sample_attribution_v1/run_20260708_212657/"
        "common_sample_registry.csv"
    )
    for key, value in counts.items():
        add(
            f"sample_count_{key}",
            "common-sample design",
            value,
            "samples_or_rows",
            registry_source,
            "scripts/analysis/common_sample_attribution_v1/run_analysis.py",
            "RAW_REBUILT",
            "Contract-seed rows are not unique physical samples.",
        )
    perf_source = (
        "outputs/common_sample_attribution_v1/run_20260708_212657/"
        "sample_level_common_shap.csv"
    )
    for row in performance.itertuples(index=False):
        add(
            f"performance_{row.contract}_{row.method}",
            "deployment-conditioned predictive performance",
            f"MAE={row.mae_mean:.6f}; RMSE={row.rmse_mean:.6f}; "
            f"R2={row.r2_mean:.6f}",
            "target_yield",
            perf_source,
            "formal checkpoint lineage plus matched predictions",
            "RAW_REBUILT",
            "Mean across three seeds; common-sample subset.",
        )
    for row in support.itertuples(index=False):
        add(
            f"support_{row.contract}_{row.route}",
            "observation-support attribution share",
            f"delta={row.mean_delta_support_share:.6f}; "
            f"p_holm={row.p_holm:.6g}",
            "absolute_attribution_share",
            (
                "outputs/common_sample_attribution_v1/run_20260708_212657/"
                "support_physical_reliance.csv"
            ),
            "clustered by sample_id after seed-matched pairing",
            "RAW_REBUILT",
            "Model attribution, not causal feature effect.",
        )
    for row in linkage.itertuples(index=False):
        add(
            f"linkage_{row.contract}_{row.route}",
            "attribution-error linkage",
            f"rho={row.rho:.6f}; p_holm={row.p_holm:.6g}; "
            f"CI=[{row.ci_lo:.6f},{row.ci_hi:.6f}]",
            "cluster-level Spearman",
            (
                "outputs/common_sample_attribution_v1/run_20260708_212657/"
                "attribution_error_linkage.csv"
            ),
            "sample_id cluster collapse, permutation, bootstrap, Holm family",
            "RAW_REBUILT",
            "Association does not imply attribution causes error change.",
        )
    add(
        "perturbation_agreement",
        "attribution-stress-response agreement",
        (
            f"mean_rho={agreement.spearman_rho.mean():.6f}; "
            f"top_match={agreement.top_group_match.mean():.6f}"
        ),
        "four_disjoint_group_comparison",
        (
            "sample_level_common_shap.csv + "
            "perturbation_normalized_sensitivity.csv"
        ),
        "four mutually exclusive feature groups only",
        "RAW_REBUILT",
        "Finite OOD stress response is not natural feature importance.",
    )
    add(
        "ig_completeness",
        "explanation quality",
        ig_pass_rate,
        "fraction_within_5pct_relative_residual",
        (
            "outputs/common_sample_attribution_v1/run_20260708_212657/"
            "ig_quality.csv"
        ),
        "64-step IG; zero standardized baseline equals train scaler mean",
        "RAW_REBUILT",
        "Completeness is a numerical quality check, not causal validity.",
    )
    add(
        "taylor_fidelity",
        "explanation quality",
        taylor_gate["pass_rate"],
        "fraction_within_5pct_relative_residual",
        (
            "outputs/pathwise_taylor_shapley_v1/run_20260708_183055/"
            "pathwise_quality.csv"
        ),
        "pathwise local Taylor residual gate",
        "RAW_REBUILT",
        "Low pass rate restricts Taylor to local diagnostics.",
    )
    for row in claim_resolution.itertuples(index=False):
        add(
            f"transfer_{row.label}_seed_{row.seed_index}",
            "matched source-transfer performance",
            (
                f"candidate_MAE={row.candidate_mae:.6f}; "
                f"baseline_MAE={row.baseline_mae:.6f}; "
                f"delta={row.delta_mae:.6f}"
            ),
            "target_yield",
            "outputs/stage8_claim_resolution_v1/analysis/source_transfer_matched_error_audit.csv",
            (
                "src/distillation_v4/universal_weather_v2_remediation/"
                "execution.py"
            ),
            "RUN_LEVEL_REBUILT",
            "Matched local baseline; route and contract specific.",
        )
    return pd.DataFrame(records)


def required_paths(root: Path, *, profile: str = "v1") -> list[Path]:
    paths = [
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "common_sample_registry.csv",
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "sample_level_common_shap.csv",
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "support_physical_reliance.csv",
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "attribution_error_linkage.csv",
        root
        / "outputs/common_sample_attribution_v1/run_20260708_212657"
        / "ig_quality.csv",
        root
        / "outputs/common_sample_attribution_validation_v1/run_20260709_090144"
        / "perturbation_normalized_sensitivity.csv",
        root
        / "outputs/pathwise_taylor_shapley_v1/run_20260708_183055"
        / "pathwise_quality.csv",
        root
        / "outputs/stage8_claim_resolution_v1/analysis"
        / "source_transfer_matched_error_audit.csv",
    ]
    if profile in {"v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}:
        paths.extend(
            [
                root
                / "outputs/common_sample_attribution_v1"
                / "run_20260708_212657"
                / "sample_level_common_ig.csv",
                root
                / "outputs/common_sample_attribution_v1"
                / "run_20260708_212657"
                / "attribution_concentration.csv",
            ]
        )
    if profile in {"v3", "v4", "v4_1", "v4_2", "v4_4"}:
        paths.extend(
            [
                root / "outputs/tables/stage7_6_random_selected_model_regret.csv",
                root / "outputs/tables/stage7_trusted_result_index.csv",
                root
                / "outputs/pathwise_taylor_shapley_v1/run_20260708_183055"
                / "sample_level_pathwise_taylor_shapley.csv",
            ]
        )
    return paths


def run(
    root: Path,
    *,
    write: bool,
    output_root: Path | None = None,
    formal_root: Path | None = None,
    profile: str = "v1",
) -> dict[str, object]:
    if profile not in {"v1", "v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}:
        raise ValueError(f"UNKNOWN_PROFILE:{profile}")
    inputs = required_paths(root, profile=profile)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_REQUIRED_INPUTS:{missing}")
    common_dir = (
        root / "outputs/common_sample_attribution_v1/run_20260708_212657"
    )
    validation_dir = (
        root
        / "outputs/common_sample_attribution_validation_v1"
        / "run_20260709_090144"
    )
    registry_raw = pd.read_csv(common_dir / "common_sample_registry.csv")
    counts = common_sample_counts(registry_raw)
    shap_raw = pd.read_csv(common_dir / "sample_level_common_shap.csv")
    predictions = _load_common_predictions(
        common_dir / "sample_level_common_shap.csv"
    )
    performance_seed, performance = performance_summary(predictions)
    support = support_reliance_clustered(
        pd.read_csv(common_dir / "support_physical_reliance.csv")
    )
    linkage_raw = pd.read_csv(common_dir / "attribution_error_linkage.csv")
    linkage = linkage_clustered(linkage_raw)
    attribution = attribution_group_shares(
        common_dir / "sample_level_common_shap.csv"
    )
    normalized = pd.read_csv(
        validation_dir / "perturbation_normalized_sensitivity.csv"
    )
    perturbation = normalized.rename(
        columns={
            "perturbation_group": "group",
            "standardized_sensitivity": "standardized_sensitivity",
        }
    )
    agreement = disjoint_group_agreement(attribution, perturbation)
    if len(agreement) != 30 or not (agreement["n_groups"] == 4).all():
        raise ValueError("INCOMPLETE_DISJOINT_GROUP_AGREEMENT")
    taylor_quality = pd.read_csv(
        root
        / "outputs/pathwise_taylor_shapley_v1/run_20260708_183055"
        / "pathwise_quality.csv"
    )
    taylor_gate = taylor_fidelity_gate(taylor_quality)
    ig_quality = pd.read_csv(common_dir / "ig_quality.csv")
    ig_pass_rate = ig_completeness_rate(ig_quality)
    formal = formal_root or root
    ood_quality = rebuild_ood_quality(root=root, formal_root=formal)
    three_view = three_view_route_pairs(
        predictions,
        linkage_raw,
        normalized,
    )
    claim_resolution = pd.read_csv(
        root
        / "outputs/stage8_claim_resolution_v1/analysis"
        / "source_transfer_matched_error_audit.csv"
    )
    lineage = validate_lineage(root)
    source_protocol = (
        source_protocol_v4_4(root, formal)
        if profile == "v4_4"
        else None
    )
    registry = build_registry(
        root=root,
        counts=counts,
        performance=performance,
        support=support,
        linkage=linkage,
        agreement=agreement,
        taylor_gate=taylor_gate,
        ig_pass_rate=ig_pass_rate,
        claim_resolution=claim_resolution,
    )
    summary = {
        "profile": profile,
        "counts": counts,
        "performance_rows": int(len(performance)),
        "support_rows": int(len(support)),
        "linkage_rows": int(len(linkage)),
        "agreement_rows": int(len(agreement)),
        "taylor_gate": taylor_gate,
        "ig_pass_rate": ig_pass_rate,
        "lineage_status_counts": lineage["status"].value_counts().to_dict(),
        "ood_rows": int(len(ood_quality)),
        "ood_dimensions": sorted(
            int(value) for value in ood_quality["n_dimensions"].unique()
        ),
    }
    if source_protocol is not None:
        summary["source_protocol_rows"] = int(len(source_protocol))
        summary["shap_background_test_rows"] = int(
            pd.to_numeric(
                source_protocol.get("test_rows", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
    extended_outputs: dict[str, pd.DataFrame] = {}
    if profile in {"v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}:
        performance_landscape = performance_landscape_v2(
            performance_seed,
            claim_resolution,
        )
        scratch_audit = scratch_baseline_audit(root, performance_landscape)
        if profile in {"v4_1", "v4_2", "v4_4"} and scratch_audit.iloc[0]["classification"] != "fixed_reference_reused":
            raise ValueError("SPATIAL_SCRATCH_BASELINE_UNRESOLVED")
        support_points, support_v2_summary = support_reliance_v2(
            pd.read_csv(common_dir / "support_physical_reliance.csv")
        )
        signed_points, signed_summary = signed_feature_redistribution_v2(
            shap_raw
        )
        (
            linkage_points,
            linkage_v2_summary,
            linkage_stability,
            linkage_trimmed,
        ) = linkage_v2(linkage_raw)
        shap_ig_points, shap_ig_summary = shap_ig_agreement_v2(
            shap_raw,
            pd.read_csv(common_dir / "sample_level_common_ig.csv"),
        )
        entropy_points, entropy_summary = entropy_change_v2(
            pd.read_csv(common_dir / "attribution_concentration.csv")
        )
        extended_outputs = {
            "source_performance_landscape_seed.csv": performance_landscape,
            "source_scratch_baseline_status.csv": scratch_audit,
            "source_support_delta_samples.csv": support_points,
            "source_support_delta_summary.csv": support_v2_summary,
            "source_signed_feature_delta_samples.csv": signed_points,
            "source_signed_feature_delta_summary.csv": signed_summary,
            "source_linkage_pointcloud.csv": linkage_points,
            "source_linkage_summary.csv": linkage_v2_summary,
            "source_linkage_seed_stability.csv": linkage_stability,
            "source_linkage_trimming.csv": linkage_trimmed,
            "source_shap_ig_agreement_samples.csv": shap_ig_points,
            "source_shap_ig_agreement_summary.csv": shap_ig_summary,
            "source_entropy_change_samples.csv": entropy_points,
            "source_entropy_change_summary.csv": entropy_summary,
            "source_perturbation_normalized_sensitivity.csv": normalized,
            "source_perturbation_agreement_disjoint.csv": agreement,
            "source_three_view_route_pairs.csv": three_view,
            "source_attribution_group_shares.csv": attribution,
            "source_lineage_validation.csv": lineage,
            "source_ood_quality_9d.csv": ood_quality,
        }
        if source_protocol is not None:
            extended_outputs["source_protocol_summary.csv"] = source_protocol
        summary.update(
            {
                "performance_landscape_rows": int(
                    len(performance_landscape)
                ),
                "spatial_scratch_baseline": str(scratch_audit.iloc[0]["classification"]),
                "support_point_rows": int(len(support_points)),
                "signed_feature_summary_rows": int(len(signed_summary)),
                "linkage_point_rows": int(len(linkage_points)),
                "shap_ig_summary_rows": int(len(shap_ig_summary)),
                "entropy_summary_rows": int(len(entropy_summary)),
            }
        )
    if profile in {"v3", "v4", "v4_1", "v4_2", "v4_4"}:
        early_split = early_split_ranking_v3(root)
        beeswarm = (
            supervised_shap_beeswarm_v4(shap_raw)
            if profile in {"v4", "v4_1", "v4_2", "v4_4"}
            else supervised_shap_beeswarm_v3(shap_raw)
        )
        agreement_by_cell = explanation_agreement_v3(
            shap_raw,
            pd.read_csv(common_dir / "sample_level_common_ig.csv"),
            ig_quality,
            pd.read_csv(
                root
                / "outputs/pathwise_taylor_shapley_v1/run_20260708_183055"
                / "sample_level_pathwise_taylor_shapley.csv"
            ),
            taylor_quality,
        )
        route_coefficients = route_loss_coefficients_v3()
        extended_outputs.update(
            {
                "source_early_split_ranking.csv": early_split,
                "source_supervised_shap_beeswarm.csv": beeswarm,
                "source_explanation_agreement_by_cell.csv": agreement_by_cell,
                "source_route_loss_coefficients.csv": route_coefficients,
            }
        )
        if profile in {"v4", "v4_1", "v4_2", "v4_4"}:
            if profile == "v4_4":
                experiment_map = experiment_map_public_v4_4(root)
            elif profile in {"v4_1", "v4_2"}:
                experiment_map = experiment_map_public_v4_1(root)
            else:
                experiment_map = experiment_map_v4(root)
            extended_outputs["source_complete_experiment_map.csv"] = experiment_map
            summary["experiment_map_rows"] = int(len(experiment_map))
        summary.update(
            {
                "early_split_rows": int(len(early_split)),
                "early_split_reversals": int(early_split["ranking_reversal"].sum()),
                "beeswarm_rows": int(len(beeswarm)),
                "agreement_cell_rows": int(len(agreement_by_cell)),
                "route_coefficient_rows": int(len(route_coefficients)),
            }
        )
    if profile in {"v4", "v4_1", "v4_2", "v4_4"}:
        frozen = json.loads(
            (root / "figures/final_publication_v4_3/evidence_build_manifest.json")
            .read_text(encoding="utf-8")
        )["summary"]
        frozen_keys = (
            "performance_landscape_rows",
            "agreement_rows",
            "ood_dimensions",
            "ig_pass_rate",
            "taylor_gate",
            "early_split_rows",
            "early_split_reversals",
            "beeswarm_rows",
            "agreement_cell_rows",
            "route_coefficient_rows",
        )
        for key in frozen_keys:
            if summary.get(key) != frozen.get(key):
                raise ValueError(f"V3_HEADLINE_MISMATCH:{key}")
    if profile == "v4_4":
        previous_dir = root / "figures/final_publication_v4_3"
        frozen_tables = (
            "source_performance_landscape_seed.csv",
            "source_support_delta_summary.csv",
            "source_linkage_summary.csv",
            "source_perturbation_agreement_disjoint.csv",
            "source_ood_quality_9d.csv",
        )
        for filename in frozen_tables:
            previous = pd.read_csv(previous_dir / filename)
            current = extended_outputs[filename].reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    current,
                    previous.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )
            except AssertionError as error:
                raise ValueError(f"V4_3_FROZEN_TABLE_MISMATCH:{filename}") from error
    if write:
        destination = output_root or root
        if profile in {"v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}:
            figures = (
                destination
                if output_root is not None
                else root / f"figures/final_publication_{profile}"
            )
            docs = None
        else:
            docs = destination / "docs/final_codex_synthesis"
            figures = destination / "figures/final_publication"
            docs.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        if docs is not None:
            registry.to_csv(
                docs / "evidence_and_method_registry.csv",
                index=False,
            )
        outputs = (
            extended_outputs
            if profile in {"v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}
            else {
                "source_performance_seed.csv": performance_seed,
                "source_performance_aggregate.csv": performance,
                "source_support_reliance_clustered.csv": support,
                "source_linkage_clustered.csv": linkage,
                "source_perturbation_agreement_disjoint.csv": agreement,
                "source_three_view_route_pairs.csv": three_view,
                "source_attribution_group_shares.csv": attribution,
                "source_lineage_validation.csv": lineage,
                "source_ood_quality_9d.csv": ood_quality,
            }
        )
        for filename, frame in outputs.items():
            frame.to_csv(figures / filename, index=False)
        manifest = {
            "schema_version": (
                f"final_codex_synthesis_evidence_{profile}"
                if profile in {"v2", "v3", "v4", "v4_1", "v4_2", "v4_4"}
                else "final_codex_synthesis_evidence_v1"
            ),
            "summary": summary,
            "inputs": {
                str(path.relative_to(root)): {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in inputs
            },
            "statistical_unit": (
                "sample_id clusters after contract-route-seed matching"
            ),
            "holm_family": "8 route-by-contract hypotheses",
            "perturbation_groups": list(DISJOINT_GROUPS),
        }
        (figures / "evidence_build_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=("v1", "v2", "v3", "v4", "v4_1", "v4_2", "v4_4"),
        default="v1",
    )
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is not None and args.output_dir is not None:
        raise SystemExit("Use only one of --output-root or --output-dir")
    output = args.output_dir or args.output_root
    summary = run(
        args.root.resolve(),
        write=bool(args.write),
        output_root=(
            output.resolve()
            if output is not None
            else None
        ),
        formal_root=args.formal_root.resolve(),
        profile=args.profile,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
