from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    output = Path(
        args.output
    ).resolve()

    manifest = json.loads(
        (
            output
            / "analysis_manifest.json"
        ).read_text()
    )

    ig = pd.read_csv(
        output
        / "sample_level_integrated_gradients.csv"
    )
    taylor = pd.read_csv(
        output
        / "sample_level_pathwise_taylor_shapley.csv"
    )
    interactions = pd.read_csv(
        output
        / "sample_level_interactions.csv"
    )
    quality = pd.read_csv(
        output
        / "pathwise_quality.csv"
    )
    agreement = pd.read_csv(
        output
        / "method_agreement_by_seed.csv"
    )

    ig_samples = (
        ig[
            [
                "contract",
                "method",
                "seed",
                "sample_id",
            ]
        ]
        .drop_duplicates()
    )

    taylor_samples = (
        taylor[
            [
                "contract",
                "method",
                "seed",
                "sample_id",
            ]
        ]
        .drop_duplicates()
    )

    ig_quality = (
        ig[
            [
                "contract",
                "method",
                "seed",
                "sample_id",
                "prediction",
                "baseline_prediction",
                "ig_residual",
            ]
        ]
        .drop_duplicates()
    )

    ig_quality[
        "relative_residual"
    ] = (
        ig_quality[
            "ig_residual"
        ].abs()
        / (
            ig_quality[
                "prediction"
            ]
            - ig_quality[
                "baseline_prediction"
            ]
        ).abs().clip(
            lower=1e-8
        )
    )

    ig_gate_rate = float(
        (
            ig_quality[
                "relative_residual"
            ]
            <= 0.05
        ).mean()
    )

    taylor_gate_rate = float(
        (
            quality[
                "taylor_relative_residual"
            ]
            <= 0.05
        ).mean()
    )

    checks = {
        "manifest_completed": (
            manifest["status"]
            == "COMPLETED"
        ),
        "ig_covers_30_cells": (
            ig_samples[
                [
                    "contract",
                    "method",
                    "seed",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            == 30
        ),
        "taylor_covers_30_cells": (
            taylor_samples[
                [
                    "contract",
                    "method",
                    "seed",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            == 30
        ),
        "ten_ig_features": (
            ig[
                "feature_name"
            ].nunique()
            == 10
        ),
        "ten_taylor_features": (
            taylor[
                "feature_name"
            ].nunique()
            == 10
        ),
        "forty_five_interaction_pairs": (
            interactions[
                [
                    "feature_i",
                    "feature_j",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            == 45
        ),
        "all_numeric_finite": all(
            np.isfinite(
                frame.select_dtypes(
                    include=[np.number]
                ).to_numpy(
                    dtype=float
                )
            ).all()
            for frame in [
                ig,
                taylor,
                interactions,
                quality,
                agreement,
            ]
        ),
    }

    hard_failures = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if hard_failures:
        decision = (
            "SCIENTIFICALLY_BLOCKED"
        )
    elif (
        ig_gate_rate >= 0.90
        and taylor_gate_rate >= 0.80
    ):
        decision = (
            "IG_AND_TAYLOR_SHAPLEY_USABLE"
        )
    elif ig_gate_rate >= 0.90:
        decision = (
            "IG_USABLE_TAYLOR_SHAPLEY_DIAGNOSTIC"
        )
    else:
        decision = (
            "SHAP_PRIMARY_SUPPLEMENTARY_METHODS_NOT_VALIDATED"
        )

    report = {
        "decision": decision,
        "checks": checks,
        "hard_failures": (
            hard_failures
        ),
        "counts": {
            "ig_samples": len(
                ig_samples
            ),
            "taylor_samples": len(
                taylor_samples
            ),
            "interaction_rows": len(
                interactions
            ),
            "agreement_rows": len(
                agreement
            ),
        },
        "quality": {
            "ig_five_percent_gate_rate": (
                ig_gate_rate
            ),
            "taylor_shapley_five_percent_gate_rate": (
                taylor_gate_rate
            ),
            "mean_ig_relative_residual": float(
                ig_quality[
                    "relative_residual"
                ].mean()
            ),
            "mean_taylor_relative_residual": float(
                quality[
                    "taylor_relative_residual"
                ].mean()
            ),
            "median_shap_ig_correlation": float(
                agreement[
                    "shap_ig_pearson"
                ].median()
            ),
            "median_shap_taylor_correlation": float(
                agreement[
                    "shap_taylor_pearson"
                ].median()
            ),
        },
        "interpretation": {
            "SHAP_PRIMARY": (
                "Permutation SHAP remains the "
                "primary feature-attribution result."
            ),
            "IG": (
                "Integrated gradients is accepted "
                "only when path completeness is adequate."
            ),
            "TAYLOR_SHAPLEY": (
                "Taylor–Shapley interactions are "
                "accepted only for samples passing "
                "the reconstruction gate."
            ),
        },
    }

    audit_directory = (
        output
        / "posthoc_scientific_audit"
    )
    audit_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        audit_directory
        / "supplement_audit.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print("=" * 88)
    print(
        "SUPPLEMENTARY ATTRIBUTION AUDIT"
    )
    print("=" * 88)
    print("Decision:", decision)
    print(
        "IG samples:",
        len(ig_samples),
    )
    print(
        "Taylor–Shapley samples:",
        len(taylor_samples),
    )
    print(
        "IG ≤5% residual rate:",
        f"{ig_gate_rate:.2%}",
    )
    print(
        "Taylor–Shapley ≤5% residual rate:",
        f"{taylor_gate_rate:.2%}",
    )
    print(
        "Median SHAP–IG correlation:",
        f"{report['quality']['median_shap_ig_correlation']:.4f}",
    )
    print(
        "Median SHAP–Taylor correlation:",
        f"{report['quality']['median_shap_taylor_correlation']:.4f}",
    )
    print(
        "Hard failures:",
        hard_failures,
    )
    print(
        "Report:",
        audit_directory
        / "supplement_audit.json",
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
