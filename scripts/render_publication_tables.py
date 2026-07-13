#!/usr/bin/env python3
"""Rebuild machine-readable publication table sources from frozen result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROUTE_ORDER = ["local_scratch", "supervised", "prediction_kd", "combined_kd", "representation_kd", "missing_aware"]


def render(root: Path, output: Path, validate_only: bool) -> dict[str, object]:
    source = root / "figures/final_publication_v4_18/source_performance_landscape_seed.csv"
    apsim_source = root / "outputs/apsim_comparison/run_summary.csv"
    claim_source = root / "outputs/stage8_claim_resolution_v1/analysis/claim_gate_seed_results.csv"
    performance = pd.read_csv(source)
    required = {"contract", "method", "seed", "n", "mae", "rmse", "r2"}
    if not required.issubset(performance.columns):
        raise ValueError(f"performance source missing columns: {sorted(required - set(performance.columns))}")
    summary_rows = []
    for contract in ("GROUP_complete", "SPATIAL_complete"):
        for route in ROUTE_ORDER:
            cell = performance[performance.contract.eq(contract) & performance.method.eq(route)]
            if len(cell) != 3:
                raise ValueError(f"expected three source rows for {contract}/{route}, found {len(cell)}")
            row = {"contract": contract.replace("_complete", ""), "route": route, "runs": 3}
            for metric in ("mae", "rmse", "r2"):
                row[f"{metric}_mean"] = float(cell[metric].mean())
                row[f"{metric}_sd"] = 0.0 if contract.startswith("SPATIAL") and route == "local_scratch" else float(cell[metric].std(ddof=1))
            row["reference_status"] = "fixed reference" if contract.startswith("SPATIAL") and route == "local_scratch" else "run-level"
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    apsim = pd.read_csv(apsim_source)
    apsim_keep = apsim[
        apsim["condition"].isin(["constrained_central", "available_central", "data_driven"])
    ].copy()
    claims = pd.read_csv(claim_source)
    case_rows = []
    rf = claims[claims["gate"].eq("late_fusion")]
    for row in rf.itertuples(index=False):
        for variant, mae, rmse in (
            ("branch_average", row.official_mae, row.official_rmse),
            ("joint_feature", row.primary_mae, row.primary_rmse),
            ("best_branch", row.secondary_mae, row.secondary_rmse),
        ):
            case_rows.append(
                {
                    "family": "local_rf",
                    "contract": row.split_id,
                    "condition": row.condition,
                    "variant": variant,
                    "seed": row.seed,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
    privileged = claims[
        claims["gate"].eq("teacher_student")
        & claims["split_id"].eq("GROUP")
        & claims["condition"].eq("synthetic_no_soil")
    ]
    for row in privileged.itertuples(index=False):
        for variant, mae, rmse in (
            ("fixed_blend", row.official_mae, row.official_rmse),
            ("observed_target", row.primary_mae, row.primary_rmse),
            ("pseudo_target", row.secondary_mae, row.secondary_rmse),
        ):
            case_rows.append(
                {
                    "family": "privileged_information",
                    "contract": row.split_id,
                    "condition": row.condition,
                    "variant": variant,
                    "seed": row.seed,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
    case_metrics = pd.DataFrame(case_rows)
    products = {
        "table1_primary_performance.csv": summary,
        "table2_rf_pi_run_metrics.csv": case_metrics,
        "table_s_apsim_common_comparison.csv": apsim_keep,
    }
    if not validate_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in products.items():
            frame.to_csv(output / name, index=False)
        manifest = {
            "status": "PASS",
            "products": {name: len(frame) for name, frame in products.items()},
            "sources": [
                str(source.relative_to(root)),
                str(claim_source.relative_to(root)),
                str(apsim_source.relative_to(root)),
            ],
        }
        (output / "table_generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "PASS", "products": {name: len(frame) for name, frame in products.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/publication_repro/tables"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(render(root, output, args.validate_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
