#!/usr/bin/env python3
"""Verify frozen headline and APSIM values against publication source tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_PRIMARY = {
    ("GROUP_complete", "local_scratch", "mae"): 0.729,
    ("GROUP_complete", "prediction_kd", "mae"): 1.157,
    ("SPATIAL_complete", "local_scratch", "mae"): 1.204,
    ("SPATIAL_complete", "supervised", "mae"): 1.024,
    ("SPATIAL_complete", "prediction_kd", "mae"): 0.954,
}

EXPECTED_APSIM = {
    ("GROUP", "data_driven", "validation_selected_weather_transfer", "mae_mean"): 0.837,
    ("GROUP", "constrained_central", "APSIM", "mae_mean"): 0.948,
    ("SPATIAL", "data_driven", "validation_selected_weather_transfer", "mae_mean"): 0.990,
    ("SPATIAL", "constrained_central", "APSIM", "mae_mean"): 1.057,
}

EXPECTED_CASE_MEANS = {
    ("late_fusion", "SPATIAL", "complete", "primary_mae"): 0.944,
    ("late_fusion", "SPATIAL", "synthetic_random_0_15", "primary_mae"): 0.920,
    ("late_fusion", "SPATIAL", "synthetic_no_weather", "primary_mae"): 1.024,
    ("teacher_student", "GROUP", "synthetic_no_soil", "official_mae"): 0.780,
    ("teacher_student", "GROUP", "synthetic_no_soil", "primary_mae"): 0.793,
    ("teacher_student", "GROUP", "synthetic_no_soil", "secondary_mae"): 0.793,
}

EXPECTED_ROSEWORTHY = {
    ("complete", "scratch", "mae"): 0.465,
    ("complete", "scratch", "rmse"): 0.637,
    ("complete", "scratch", "r2"): 0.131,
    ("no_soil", "scratch", "mae"): 0.499,
    ("no_soil", "scratch", "rmse"): 0.667,
    ("no_soil", "scratch", "r2"): 0.049,
    ("no_soil", "supervised", "mae"): 0.465,
    ("no_soil", "prediction_kd", "mae"): 0.470,
    ("no_soil", "combined_kd", "mae"): 0.466,
    ("no_soil", "representation_kd", "mae"): 0.473,
    ("no_soil", "missing_aware", "mae"): 0.468,
}

EXPECTED_WAITE_MAE = {
    ("temporal_forward", 101): 0.691,
    ("temporal_forward", 202): 0.698,
    ("temporal_forward", 303): 0.685,
    ("plot_group", 101): 0.502,
    ("plot_group", 202): 0.477,
    ("plot_group", 303): 0.499,
}

EXPECTED_BEHAVIOUR = {
    "rank_agreement_mean": -0.040,
    "rank_agreement_group": 0.067,
    "rank_agreement_spatial": -0.147,
    "top_group_match": 0.300,
    "ig_completeness": 0.989,
    "taylor_local_fidelity": 0.156,
}


def close_displayed(actual: float, expected: float, tolerance: float = 5.1e-4) -> bool:
    return abs(actual - expected) <= tolerance


def verify(root: Path) -> dict[str, object]:
    primary_path = root / "figures/final_publication_v4_18/source_performance_landscape_seed.csv"
    apsim_path = root / "outputs/apsim_comparison/run_summary.csv"
    case_path = root / "outputs/stage8_claim_resolution_v1/analysis/claim_gate_seed_results.csv"
    roseworthy_path = root / "data_manifest/publication/roseworthy_run_metrics.csv"
    waite_path = root / "data_manifest/publication/waite_run_metrics.csv"
    primary = pd.read_csv(primary_path)
    apsim = pd.read_csv(apsim_path)
    cases = pd.read_csv(case_path)
    roseworthy = pd.read_csv(roseworthy_path)
    waite = pd.read_csv(waite_path)
    checks: list[dict[str, object]] = []

    for (contract, method, metric), expected in EXPECTED_PRIMARY.items():
        values = primary.loc[
            primary["contract"].eq(contract) & primary["method"].eq(method), metric
        ]
        if values.empty:
            raise ValueError(f"missing primary cell: {contract}/{method}/{metric}")
        actual = float(values.iloc[0] if method == "local_scratch" and contract.startswith("SPATIAL") else values.mean())
        checks.append({"cell": f"{contract}/{method}/{metric}", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"primary mismatch for {contract}/{method}/{metric}: {actual} != {expected}")

    spatial_scratch = primary.loc[
        primary["contract"].eq("SPATIAL_complete") & primary["method"].eq("local_scratch"), "mae"
    ]
    checks.append({"cell": "SPATIAL fixed scratch", "unique_values": int(spatial_scratch.nunique())})
    if len(spatial_scratch) != 3 or spatial_scratch.nunique() != 1:
        raise ValueError("SPATIAL scratch is not the required fixed three-row reference")

    for (contract, condition, model, metric), expected in EXPECTED_APSIM.items():
        values = apsim.loc[
            apsim["contract"].eq(contract)
            & apsim["condition"].eq(condition)
            & apsim["model"].eq(model),
            metric,
        ]
        if len(values) != 1:
            raise ValueError(f"missing or duplicate APSIM cell: {contract}/{condition}/{model}/{metric}")
        actual = float(values.iloc[0])
        checks.append({"cell": f"{contract}/{condition}/{model}/{metric}", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"APSIM mismatch for {contract}/{condition}/{model}/{metric}: {actual} != {expected}")

    for (gate, split, condition, metric), expected in EXPECTED_CASE_MEANS.items():
        values = cases.loc[
            cases["gate"].eq(gate)
            & cases["split_id"].eq(split)
            & cases["condition"].eq(condition),
            metric,
        ]
        if len(values) != 3:
            raise ValueError(f"expected three case rows: {gate}/{split}/{condition}/{metric}")
        actual = float(values.mean())
        checks.append({"cell": f"{gate}/{split}/{condition}/{metric}", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"case mismatch for {gate}/{split}/{condition}/{metric}: {actual} != {expected}")

    for (condition, route, metric), expected in EXPECTED_ROSEWORTHY.items():
        values = roseworthy.loc[
            roseworthy["condition"].eq(condition) & roseworthy["route"].eq(route),
            metric,
        ]
        if len(values) != 3:
            raise ValueError(f"expected three Roseworthy rows: {condition}/{route}/{metric}")
        actual = float(values.mean())
        checks.append({"cell": f"Roseworthy/{condition}/{route}/{metric}", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"Roseworthy mismatch for {condition}/{route}/{metric}: {actual} != {expected}")

    for (protocol, seed), expected in EXPECTED_WAITE_MAE.items():
        values = waite.loc[
            waite["protocol"].eq(protocol) & waite["seed"].eq(seed),
            "mae",
        ]
        if len(values) != 1:
            raise ValueError(f"missing Waite row: {protocol}/{seed}")
        actual = float(values.iloc[0])
        checks.append({"cell": f"Waite/{protocol}/{seed}/mae", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"Waite mismatch for {protocol}/{seed}: {actual} != {expected}")

    agreement = pd.read_csv(
        root / "figures/final_publication_v4_18/source_perturbation_agreement_disjoint.csv"
    )
    evidence = json.loads(
        (root / "figures/final_publication_v4_18/evidence_build_manifest.json").read_text()
    )["summary"]
    observed_behaviour = {
        "rank_agreement_mean": float(agreement["spearman_rho"].mean()),
        "rank_agreement_group": float(
            agreement[agreement["contract"].eq("GROUP_complete")]["spearman_rho"].mean()
        ),
        "rank_agreement_spatial": float(
            agreement[agreement["contract"].eq("SPATIAL_complete")]["spearman_rho"].mean()
        ),
        "top_group_match": float(agreement["top_group_match"].mean()),
        "ig_completeness": float(evidence["ig_pass_rate"]),
        "taylor_local_fidelity": float(evidence["taylor_gate"]["pass_rate"]),
    }
    for measure, expected in EXPECTED_BEHAVIOUR.items():
        actual = observed_behaviour[measure]
        checks.append({"cell": f"behaviour/{measure}", "actual": actual, "expected": expected})
        if not close_displayed(actual, expected):
            raise ValueError(f"behaviour mismatch for {measure}: {actual} != {expected}")

    checks.append({"cell": "primary negative R2 preserved", "count": int((primary["r2"] < 0).sum())})
    if not (primary["r2"] < 0).any():
        raise ValueError("negative primary R2 values were lost")
    return {"status": "PASS", "checks": len(checks), "details": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.output and not args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
