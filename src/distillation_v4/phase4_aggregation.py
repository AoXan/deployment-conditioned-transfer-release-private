from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
from typing import Any

import numpy as np


STUDENT_ROUTES = {
    "supervised",
    "soil_direct",
    "missing_aware",
    "fine_tune",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
}

COMPARISON_ROUTES = {
    "soil_direct",
    "missing_aware",
    "fine_tune",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"PHASE4_REQUIRED_FILE_MISSING:{path.name}"
        )
    return json.loads(path.read_text())


def load_phase4_records(
    output: Path,
) -> list[dict[str, Any]]:
    root = (
        output
        / "development/phase4_repair_matrix"
    )

    records = [
        json.loads(path.read_text())
        for path in sorted(
            root.glob("**/job_record.json")
        )
    ]

    identifiers = [
        str(record["route_id"])
        for record in records
    ]

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "DUPLICATE_PHASE4_ROUTE_RECORD"
        )

    for record in records:
        if record.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "UNACCEPTED_PHASE4_ROUTE_RECORD"
            )

        if record.get("outer_test_used") is not False:
            raise RuntimeError(
                "OUTER_TEST_PHASE4_AGGREGATION_FORBIDDEN"
            )

        value = float(record["validation_mae"])
        if not math.isfinite(value):
            raise RuntimeError(
                "NONFINITE_PHASE4_VALIDATION_MAE"
            )

    return records


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError(
            "PHASE4_EMPTY_AGGREGATION_VALUES"
        )
    return float(np.mean(values))


def aggregate_phase4_development(
    *,
    output: Path,
) -> dict[str, Any]:
    manifest = _load_json(
        output
        / "control/frozen_phase4_route_manifest.json"
    )
    execution = _load_json(
        output
        / "status/phase4_repair_matrix_execution.json"
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE4_ENGINEERING_MATRIX_INCOMPLETE"
        )

    records = load_phase4_records(output)
    planned_routes = manifest.get("routes")

    if not isinstance(planned_routes, list):
        raise RuntimeError(
            "PHASE4_MANIFEST_ROUTES_INVALID"
        )

    planned_ids = {
        str(route["route_id"])
        for route in planned_routes
    }
    observed_ids = {
        str(record["route_id"])
        for record in records
    }

    missing = sorted(
        planned_ids.difference(observed_ids)
    )
    unexpected = sorted(
        observed_ids.difference(planned_ids)
    )

    if missing or unexpected:
        raise RuntimeError(
            "PHASE4_RECORD_MANIFEST_MISMATCH:"
            f"missing={missing}:unexpected={unexpected}"
        )

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    references: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for record in records:
        dataset = str(record["dataset"])
        fold = str(record["fold"])
        route = str(record["route"])

        if route == "reference":
            key = (dataset, fold)
            if key in references:
                raise RuntimeError(
                    "DUPLICATE_PHASE4_REFERENCE:"
                    f"{dataset}:{fold}"
                )
            references[key] = record
        else:
            grouped[(dataset, fold, route)].append(
                record
            )

    datasets = sorted(
        {
            str(route["dataset"])
            for route in planned_routes
        }
    )

    dataset_summaries: dict[str, Any] = {}
    route_survival: dict[str, Any] = {}

    for dataset in datasets:
        planned_dataset = [
            route
            for route in planned_routes
            if str(route["dataset"]) == dataset
        ]

        planned_student_routes = sorted(
            {
                str(route["route"])
                for route in planned_dataset
                if str(route["route"]) in STUDENT_ROUTES
            }
        )

        folds = sorted(
            {
                str(route["fold"])
                for route in planned_dataset
            }
        )

        route_summary: dict[str, Any] = {}

        for route in planned_student_routes:
            route_records = [
                record
                for record in records
                if (
                    str(record["dataset"]) == dataset
                    and str(record["route"]) == route
                )
            ]

            route_summary[route] = {
                "job_count": len(route_records),
                "mean_validation_mae": _mean(
                    [
                        float(record["validation_mae"])
                        for record in route_records
                    ]
                ),
                "mean_optimizer_steps": _mean(
                    [
                        float(record["optimizer_steps"])
                        for record in route_records
                    ]
                ),
                "mean_fit_time_seconds": _mean(
                    [
                        float(record["fit_time_seconds"])
                        for record in route_records
                    ]
                ),
            }

        if "supervised" not in planned_student_routes:
            raise RuntimeError(
                f"PHASE4_S0_NOT_PLANNED:{dataset}"
            )

        paired_effects: dict[str, Any] = {}

        for route in sorted(
            set(planned_student_routes)
            .intersection(COMPARISON_ROUTES)
        ):
            effects: list[float] = []
            pairs: list[dict[str, Any]] = []

            for fold in folds:
                baseline_by_seed = {
                    int(record["seed"]): record
                    for record in grouped.get(
                        (
                            dataset,
                            fold,
                            "supervised",
                        ),
                        [],
                    )
                }
                route_by_seed = {
                    int(record["seed"]): record
                    for record in grouped.get(
                        (
                            dataset,
                            fold,
                            route,
                        ),
                        [],
                    )
                }

                if set(baseline_by_seed) != set(
                    route_by_seed
                ):
                    raise RuntimeError(
                        "PHASE4_PAIRED_SEED_MISMATCH:"
                        f"{dataset}:{fold}:{route}"
                    )

                for seed in sorted(baseline_by_seed):
                    s0_mae = float(
                        baseline_by_seed[
                            seed
                        ]["validation_mae"]
                    )
                    route_mae = float(
                        route_by_seed[
                            seed
                        ]["validation_mae"]
                    )

                    # Positive means improvement over S0.
                    gain = s0_mae - route_mae
                    effects.append(gain)

                    reference = references.get(
                        (dataset, fold)
                    )
                    teacher_gap = None
                    gap_recovered = None
                    gap_reason = None

                    if reference is not None:
                        ref_mae = float(
                            reference[
                                "validation_mae"
                            ]
                        )
                        teacher_gap = (
                            s0_mae - ref_mae
                        )

                        if teacher_gap > 0.0:
                            gap_recovered = (
                                gain / teacher_gap
                            )
                        else:
                            gap_reason = (
                                "REFERENCE_DOES_NOT_"
                                "OUTPERFORM_MATCHED_S0"
                            )

                    pairs.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "s0_validation_mae": (
                                s0_mae
                            ),
                            "route_validation_mae": (
                                route_mae
                            ),
                            "student_gain": gain,
                            "negative_transfer": (
                                gain < 0.0
                            ),
                            "teacher_gap": teacher_gap,
                            "fraction_teacher_gap_recovered": (
                                gap_recovered
                            ),
                            "gap_recovered_unavailable_reason": (
                                gap_reason
                            ),
                        }
                    )

            paired_effects[route] = {
                "pair_count": len(pairs),
                "mean_student_gain": _mean(
                    effects
                ),
                "median_student_gain": float(
                    np.median(effects)
                ),
                "negative_transfer_rate": float(
                    np.mean(
                        [
                            effect < 0.0
                            for effect in effects
                        ]
                    )
                ),
                "positive_gain_fraction": float(
                    np.mean(
                        [
                            effect > 0.0
                            for effect in effects
                        ]
                    )
                ),
                "pairs": pairs,
                "interpretation_role": (
                    "DEVELOPMENT_DESCRIPTIVE_ONLY"
                ),
            }

        reference_records = [
            record
            for record in records
            if (
                str(record["dataset"]) == dataset
                and str(record["route"])
                == "reference"
            )
        ]

        dataset_summaries[dataset] = {
            "routes": route_summary,
            "paired_effects": paired_effects,
            "reference_fold_count": len(
                reference_records
            ),
            "mean_reference_validation_mae": (
                _mean(
                    [
                        float(
                            record[
                                "validation_mae"
                            ]
                        )
                        for record in reference_records
                    ]
                )
                if reference_records
                else None
            ),
            "outer_test_used": False,
            "claim_boundary": (
                "OUTER_TRAIN_DEVELOPMENT_EVIDENCE_ONLY"
            ),
        }

        # Survival is structural. It does not depend on
        # whether the development effect is positive.
        route_survival[dataset] = {
            route: {
                "survived": True,
                "reason": (
                    "ROUTE_PREDECLARED_AND_"
                    "ENGINEERING_MATRIX_COMPLETE"
                ),
            }
            for route in planned_student_routes
        }

    return {
        "phase": "phase4",
        "status": (
            "DEVELOPMENT_AGGREGATION_COMPLETE"
        ),
        "route_manifest_fingerprint": manifest[
            "phase4_route_manifest_fingerprint"
        ],
        "planned_jobs": len(planned_ids),
        "observed_jobs": len(observed_ids),
        "datasets": dataset_summaries,
        "route_survival": route_survival,
        "outer_test_used": False,
        "downstream_generation_uses_effect_sign": False,
        "outer_test_may_enable_downstream_jobs": False,
    }
