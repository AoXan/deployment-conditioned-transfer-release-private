from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
from typing import Any

from .contracts import CampaignConfig
from .phase5_contract import (
    read_ckd_sensitivity_contract,
)


CONTROL_BY_ROUTE = {
    "soil_direct": (
        "soil_representation_sensitivity_control"
    ),
    "missing_aware": (
        "missing_aware_ablation_suite_control"
    ),
    "fine_tune": "active_compute_steps_control",
    "prediction_kd": (
        "shuffled_teacher_prediction_control"
    ),
    "representation_kd": (
        "random_representation_transfer_control"
    ),
    "combined_kd": (
        "ckd_balancing_sensitivity_control"
    ),
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"PHASE5_REQUIRED_FILE_MISSING:{path.name}"
        )

    return json.loads(path.read_text())


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _record_artifact_fingerprint(
    record: dict[str, Any],
) -> str:
    payload = {
        key: record.get(key)
        for key in (
            "route_id",
            "route",
            "dataset",
            "fold",
            "seed",
            "checkpoint_sha256",
            "preprocessor_sha256",
            "prediction_artifact_sha256",
            "representation_artifact_sha256",
            "loss_ledger_sha256",
            "execution_fingerprint",
        )
    }

    return _fingerprint(payload)


def _canonical_seed(
    *,
    dataset: str,
    fold: str,
    route: str,
    records_by_key: dict[
        tuple[str, str, str, int],
        dict[str, Any],
    ],
) -> int:
    s0_seeds = {
        seed
        for (
            record_dataset,
            record_fold,
            record_route,
            seed,
        ) in records_by_key
        if (
            record_dataset == dataset
            and record_fold == fold
            and record_route == "supervised"
        )
    }

    route_seeds = {
        seed
        for (
            record_dataset,
            record_fold,
            record_route,
            seed,
        ) in records_by_key
        if (
            record_dataset == dataset
            and record_fold == fold
            and record_route == route
        )
    }

    common = sorted(
        s0_seeds.intersection(route_seeds)
    )

    if not common:
        raise RuntimeError(
            "PHASE5_CANONICAL_SEED_UNAVAILABLE:"
            f"{dataset}:{fold}:{route}"
        )

    # This is a predeclared deterministic rule,
    # not performance-based seed selection.
    return int(common[0])


def build_phase5_matrix(
    *,
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    phase4_status = _json(
        output / "status/phase4_repair.json"
    )

    if phase4_status.get("status") != (
        "SCIENTIFIC_PHASE_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE4_NOT_COMPLETE_FOR_PHASE5_PLANNING"
        )

    if phase4_status.get(
        "phase5_release_allowed"
    ) is not True:
        raise RuntimeError(
            "PHASE5_PLANNING_NOT_RELEASED"
        )

    if phase4_status.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE5_PLANNING_FORBIDDEN"
        )

    requirements = _json(
        output
        / "control/frozen_phase5_control_requirements.json"
    )
    phase4_manifest = _json(
        output
        / "control/frozen_phase4_route_manifest.json"
    )

    if requirements.get("execution_started") is not False:
        raise RuntimeError(
            "PHASE5_REQUIREMENTS_ALREADY_EXECUTING"
        )

    records = [
        json.loads(path.read_text())
        for path in sorted(
            (
                output
                / "development/phase4_repair_matrix"
            ).glob("**/job_record.json")
        )
    ]

    records_by_key: dict[
        tuple[str, str, str, int],
        dict[str, Any],
    ] = {}

    for record in records:
        route = str(record["route"])

        if route == "reference":
            continue

        key = (
            str(record["dataset"]),
            str(record["fold"]),
            route,
            int(record["seed"]),
        )

        if key in records_by_key:
            raise RuntimeError(
                "PHASE5_DUPLICATE_PHASE4_SOURCE_RECORD:"
                + ":".join(map(str, key))
            )

        if record.get("status") != (
            "ENGINEERING_ACCEPTED"
        ):
            raise RuntimeError(
                "PHASE5_SOURCE_RECORD_NOT_ACCEPTED"
            )

        if record.get("outer_test_used") is not False:
            raise RuntimeError(
                "PHASE5_SOURCE_OUTER_TEST_FORBIDDEN"
            )

        records_by_key[key] = record

    jobs: list[dict[str, Any]] = []
    blocked_requirements: list[dict[str, Any]] = []
    per_dataset_counts: dict[str, int] = {}

    for dataset in config.primary_datasets:
        dataset_requirements = (
            requirements["datasets"][dataset][
                "requirements"
            ]
        )

        requirement_routes = {
            str(item["required_by_route"])
            for item in dataset_requirements
        }

        folds = sorted(
            {
                str(route["fold"])
                for route in phase4_manifest["routes"]
                if (
                    str(route["dataset"]) == dataset
                    and str(route["route"])
                    != "reference"
                )
            }
        )

        ckd_contract = (
            read_ckd_sensitivity_contract(
                output=output,
                dataset=dataset,
            )
        )

        for route in sorted(requirement_routes):
            control = CONTROL_BY_ROUTE.get(route)

            if control is None:
                raise RuntimeError(
                    "PHASE5_CONTROL_MAPPING_MISSING:"
                    + route
                )

            if (
                route == "combined_kd"
                and ckd_contract is None
            ):
                blocked_requirements.append(
                    {
                        "dataset": dataset,
                        "route": route,
                        "control": control,
                        "materialized": False,
                        "reason": (
                            "CKD_SENSITIVITY_CONTRACT_NOT_OPEN"
                        ),
                    }
                )
                continue

            for fold in folds:
                seed = _canonical_seed(
                    dataset=dataset,
                    fold=fold,
                    route=route,
                    records_by_key=records_by_key,
                )

                source_key = (
                    dataset,
                    fold,
                    route,
                    seed,
                )
                baseline_key = (
                    dataset,
                    fold,
                    "supervised",
                    seed,
                )

                source = records_by_key[source_key]
                baseline = records_by_key[baseline_key]

                job_id = "__".join(
                    (
                        dataset,
                        "phase5",
                        control,
                        fold,
                        f"seed_{seed}",
                    )
                )

                job: dict[str, Any] = {
                    "job_id": job_id,
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "control": control,
                    "required_by_route": route,
                    "source_route_id": source[
                        "route_id"
                    ],
                    "source_route_artifact_fingerprint": (
                        _record_artifact_fingerprint(
                            source
                        )
                    ),
                    "matched_s0_route_id": baseline[
                        "route_id"
                    ],
                    "matched_s0_artifact_fingerprint": (
                        _record_artifact_fingerprint(
                            baseline
                        )
                    ),
                    "canonical_seed_policy": (
                        "LOWEST_COMMON_FROZEN_PHASE4_SEED"
                    ),
                    "seed_selected_by_performance": False,
                    "selection_scope": (
                        "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                    ),
                    "outer_test_used": False,
                }

                if control in {
                    "soil_representation_sensitivity_control",
                    "missing_aware_ablation_suite_control",
                }:
                    job[
                        "soil_extension_contract"
                    ] = dict(
                        config.raw[
                            "soil_missing_aware"
                        ]
                    )

                if control == (
                    "active_compute_steps_control"
                ):
                    job["target_optimizer_steps"] = int(
                        source["optimizer_steps"]
                    )
                    job["target_fit_time_seconds"] = (
                        float(
                            source[
                                "fit_time_seconds"
                            ]
                        )
                    )
                    job["matching_policy"] = (
                        "MATCH_SOURCE_OPTIMIZER_STEPS;"
                        "REPORT_TIME_MISMATCH"
                    )

                elif control == (
                    "shuffled_teacher_prediction_control"
                ):
                    teacher = source.get(
                        "prediction_teacher"
                    )

                    if not isinstance(teacher, dict):
                        raise RuntimeError(
                            "PHASE5_PKD_TEACHER_LINEAGE_MISSING:"
                            + source["route_id"]
                        )

                    job["teacher"] = teacher
                    job["shuffle_seed"] = seed
                    job["shuffle_scope"] = (
                        "WITHIN_TRAIN_PARTITION_ONLY"
                    )

                elif control == (
                    "random_representation_transfer_control"
                ):
                    teacher = source.get(
                        "representation_teacher"
                    )

                    if not isinstance(teacher, dict):
                        raise RuntimeError(
                            "PHASE5_RKD_TEACHER_LINEAGE_MISSING:"
                            + source["route_id"]
                        )

                    job["teacher"] = teacher
                    job["random_representation_seed"] = (
                        seed
                    )
                    job["random_control_scope"] = (
                        "MATCHED_SHAPE_AND_TRAIN_PARTITION"
                    )

                elif control == (
                    "ckd_balancing_sensitivity_control"
                ):
                    job["ckd_sensitivity_contract"] = (
                        ckd_contract
                    )

                jobs.append(job)

        dataset_count = sum(
            1
            for job in jobs
            if job["dataset"] == dataset
        )
        per_dataset_counts[dataset] = dataset_count

        ceiling = int(config.budgets["phase5"])

        if dataset_count > ceiling:
            raise RuntimeError(
                "PHASE5_PER_DATASET_CEILING_EXCEEDED:"
                f"{dataset}:{dataset_count}:{ceiling}"
            )

    job_ids = [job["job_id"] for job in jobs]

    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError(
            "PHASE5_JOB_ID_DUPLICATE"
        )

    payload = {
        "phase": "phase5",
        "status": "JOB_MATRIX_FROZEN",
        "budget_semantics": (
            "PER_DATASET_MAXIMUM_NOT_REQUIRED_COUNT"
        ),
        "per_dataset_job_ceiling": int(
            config.budgets["phase5"]
        ),
        "campaign_job_ceiling": (
            int(config.budgets["phase5"])
            * len(config.primary_datasets)
        ),
        "materialized_jobs_by_dataset": (
            per_dataset_counts
        ),
        "materialized_job_count": len(jobs),
        "jobs": jobs,
        "blocked_requirements": blocked_requirements,
        "source_phase4_route_fingerprint": (
            phase4_manifest[
                "phase4_route_manifest_fingerprint"
            ]
        ),
        "source_phase5_requirements_fingerprint": (
            requirements[
                "phase5_control_requirements_fingerprint"
            ]
        ),
        "canonical_seed_policy": (
            "LOWEST_COMMON_FROZEN_PHASE4_SEED"
        ),
        "effect_sign_used_for_job_generation": False,
        "outer_test_used": False,
        "execution_started": False,
        "execution_allowed": False,
        "outer_release_allowed": False,
    }

    payload["phase5_job_matrix_fingerprint"] = (
        _fingerprint(payload)
    )

    return payload


def freeze_phase5_matrix(
    *,
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    payload = build_phase5_matrix(
        config=config,
        output=output,
    )

    path = (
        output
        / "control/frozen_phase5_job_matrix.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)

    status = {
        "phase": "phase5",
        "status": "JOB_MATRIX_FROZEN",
        "materialized_job_count": (
            payload["materialized_job_count"]
        ),
        "materialized_jobs_by_dataset": (
            payload["materialized_jobs_by_dataset"]
        ),
        "per_dataset_job_ceiling": (
            payload["per_dataset_job_ceiling"]
        ),
        "phase5_job_matrix_fingerprint": (
            payload[
                "phase5_job_matrix_fingerprint"
            ]
        ),
        "execution_started": False,
        "execution_allowed": False,
        "outer_test_used": False,
        "outer_release_allowed": False,
        "scientific_phase_complete": False,
    }

    status_path = (
        output / "status/phase5_repair.json"
    )
    status_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    status_path.write_text(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        "manifest": payload,
        "status": status,
    }
