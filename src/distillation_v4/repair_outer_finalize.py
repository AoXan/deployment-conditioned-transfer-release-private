from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from .repair_outer_matrix import (
    validate_frozen_repair_outer_job_matrix,
)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _records(output: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text())
        for path in sorted(
            (
                output / "outer_test_repair"
            ).glob("**/job_record.json")
        )
    ]

    for record in records:
        if record.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "REPAIR_OUTER_UNACCEPTED_RECORD"
            )

        value = float(record["mae"])

        if not math.isfinite(value):
            raise RuntimeError(
                "REPAIR_OUTER_NONFINITE_MAE"
            )

    return records


def aggregate_repair_outer_results(
    *,
    output: Path,
) -> dict[str, Any]:
    matrix = (
        validate_frozen_repair_outer_job_matrix(
            output=output,
            path=(
                output
                / "control/"
                "frozen_repair_outer_job_matrix.json"
            ),
        )
    )

    execution_path = (
        output
        / "status/"
        "repair_outer_matrix_execution.json"
    )

    if not execution_path.is_file():
        raise RuntimeError(
            "REPAIR_OUTER_EXECUTION_STATUS_MISSING"
        )

    execution = json.loads(
        execution_path.read_text()
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "REPAIR_OUTER_ENGINEERING_MATRIX_INCOMPLETE"
        )

    records = _records(output)

    planned_ids = {
        str(job["job_id"])
        for job in matrix["jobs"]
    }
    observed_ids = {
        str(record["job_id"])
        for record in records
    }

    if planned_ids != observed_ids:
        raise RuntimeError(
            "REPAIR_OUTER_RECORD_MATRIX_MISMATCH"
        )

    keyed = {
        (
            str(record["dataset"]),
            str(record["fold"]),
            int(record["seed"]),
            str(record["route"]),
        ): record
        for record in records
    }

    if len(keyed) != len(records):
        raise RuntimeError(
            "REPAIR_OUTER_DUPLICATE_PAIR_KEY"
        )

    datasets = sorted(
        {
            str(record["dataset"])
            for record in records
        }
    )

    dataset_summaries: dict[str, Any] = {}
    route_verdicts: list[str] = []

    for dataset in datasets:
        routes = sorted(
            {
                str(record["route"])
                for record in records
                if record["dataset"] == dataset
            }
        )

        if "supervised" not in routes:
            raise RuntimeError(
                "REPAIR_OUTER_S0_MISSING:"
                + dataset
            )

        summaries: dict[str, Any] = {}

        for route in routes:
            if route == "supervised":
                continue

            pairs: list[dict[str, Any]] = []

            route_records = [
                record
                for record in records
                if (
                    record["dataset"] == dataset
                    and record["route"] == route
                )
            ]

            for record in route_records:
                key = (
                    dataset,
                    str(record["fold"]),
                    int(record["seed"]),
                    "supervised",
                )

                if key not in keyed:
                    raise RuntimeError(
                        "REPAIR_OUTER_MATCHED_S0_MISSING:"
                        f"{dataset}:{record['fold']}:"
                        f"{record['seed']}:{route}"
                    )

                baseline = keyed[key]
                route_mae = float(record["mae"])
                s0_mae = float(baseline["mae"])
                gain = s0_mae - route_mae

                pairs.append(
                    {
                        "fold": record["fold"],
                        "seed": record["seed"],
                        "s0_mae": s0_mae,
                        "route_mae": route_mae,
                        "student_gain": gain,
                        "negative_transfer": (
                            gain < 0.0
                        ),
                    }
                )

            gains = [
                float(pair["student_gain"])
                for pair in pairs
            ]

            positive_fraction = float(
                np.mean(
                    [
                        gain > 0.0
                        for gain in gains
                    ]
                )
            )
            negative_rate = float(
                np.mean(
                    [
                        gain < 0.0
                        for gain in gains
                    ]
                )
            )
            mean_gain = float(np.mean(gains))

            if mean_gain > 0.0 and (
                positive_fraction >= 2.0 / 3.0
            ):
                verdict = "GO"
            elif mean_gain > 0.0:
                verdict = "CONDITIONAL_GO"
            elif mean_gain < 0.0:
                verdict = "NO_GO"
            else:
                verdict = "INSUFFICIENT_EVIDENCE"

            route_verdicts.append(verdict)

            summaries[route] = {
                "pair_count": len(pairs),
                "mean_student_gain": mean_gain,
                "median_student_gain": float(
                    np.median(gains)
                ),
                "positive_gain_fraction": (
                    positive_fraction
                ),
                "negative_transfer_rate": (
                    negative_rate
                ),
                "worst_pair_gain": float(
                    np.min(gains)
                ),
                "best_pair_gain": float(
                    np.max(gains)
                ),
                "verdict": verdict,
                "pairs": pairs,
            }

        dataset_summaries[dataset] = {
            "routes": summaries,
        }

    if not route_verdicts:
        campaign_verdict = "INSUFFICIENT_EVIDENCE"
    elif "NO_GO" in route_verdicts:
        campaign_verdict = "CONDITIONAL_GO"
    elif all(
        verdict == "GO"
        for verdict in route_verdicts
    ):
        campaign_verdict = "GO"
    elif any(
        verdict in {"GO", "CONDITIONAL_GO"}
        for verdict in route_verdicts
    ):
        campaign_verdict = "CONDITIONAL_GO"
    else:
        campaign_verdict = "INSUFFICIENT_EVIDENCE"

    return {
        "status": (
            "REPAIR_OUTER_AGGREGATION_COMPLETE"
        ),
        "campaign_verdict": campaign_verdict,
        "datasets": dataset_summaries,
        "planned_jobs": len(planned_ids),
        "observed_jobs": len(observed_ids),
        "route_matrix_mutated": False,
        "control_matrix_mutated": False,
        "outer_test_may_enable_downstream_jobs": False,
        "claim_boundary": (
            "VERDICTS_APPLY_ONLY_TO_FROZEN_DATASETS_"
            "FOLDS_SEEDS_ROUTES_AND_ENDPOINTS"
        ),
    }


def finalize_repair_outer(
    *,
    output: Path,
) -> dict[str, Any]:
    aggregation = aggregate_repair_outer_results(
        output=output,
    )

    aggregation_path = (
        output
        / "reports/"
        "repair_outer_aggregation.json"
    )
    _atomic_json(
        aggregation_path,
        aggregation,
    )

    status = {
        "phase": "repair_outer",
        "status": (
            "SCIENTIFIC_PHASE_COMPLETE"
        ),
        "scientific_verdict": aggregation[
            "campaign_verdict"
        ],
        "outer_test_complete": True,
        "route_matrix_mutated": False,
        "control_matrix_mutated": False,
        "downstream_jobs_created": False,
        "aggregation": str(
            aggregation_path.relative_to(output)
        ),
        "claim_boundary": aggregation[
            "claim_boundary"
        ],
    }

    _atomic_json(
        output
        / "status/repair_outer_final.json",
        status,
    )

    return {
        "status": status,
        "aggregation": aggregation,
    }
