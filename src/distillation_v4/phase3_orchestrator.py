from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

import pandas as pd

from .contracts import CampaignConfig
from .phase3_execution import (
    HANDLERS,
    execute_phase3_job,
)
from .phase3_matrix import build_phase3_jobs
from .phase3_fingerprint import (
    canonical_mapping_fingerprint,
    phase3_execution_fingerprint,
)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def load_and_validate_phase3_record(
    *,
    record_path: Path,
    output_root: Path,
    expected_job_id: str,
    expected_execution_fingerprint: str,
) -> dict[str, Any]:
    record = json.loads(record_path.read_text())
    if record.get("job_id") != expected_job_id:
        raise RuntimeError("PHASE3_RESUME_JOB_ID_MISMATCH")
    if record.get("status") != "ENGINEERING_ACCEPTED":
        raise RuntimeError("PHASE3_RESUME_RECORD_NOT_ACCEPTED")
    if record.get("outer_test_used") is not False:
        raise RuntimeError("PHASE3_RESUME_OUTER_TEST_FORBIDDEN")
    if record.get("execution_fingerprint") != expected_execution_fingerprint:
        raise RuntimeError("PHASE3_RESUME_FINGERPRINT_MISMATCH")
    marker_path = record_path.parent / "completion_marker.json"
    if not marker_path.is_file():
        raise RuntimeError("PHASE3_RESUME_COMPLETION_MARKER_MISSING")
    marker = json.loads(marker_path.read_text())
    if marker.get("job_id") != expected_job_id or marker.get("status") != "ENGINEERING_ACCEPTED":
        raise RuntimeError("PHASE3_RESUME_COMPLETION_MARKER_INVALID")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("preprocessor", "preprocessor_sha256"),
    ):
        value = record.get(path_key)
        digest = record.get(hash_key)
        if (value is None) != (digest is None):
            raise RuntimeError("PHASE3_RESUME_ARTIFACT_LINEAGE_INCOMPLETE")
        if value is None:
            continue
        artifact = (output_root / str(value)).resolve()
        try:
            artifact.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RuntimeError("PHASE3_RESUME_ARTIFACT_OUTSIDE_OUTPUT") from exc
        if not artifact.is_file():
            raise RuntimeError("PHASE3_RESUME_ARTIFACT_MISSING")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError("PHASE3_RESUME_ARTIFACT_HASH_MISMATCH")
    return record


def _folds(
    config: CampaignConfig,
    dataset: str,
) -> tuple[str, ...]:
    return tuple(
        config.raw
        .get("datasets", {})
        .get(dataset, {})
        .get(
            "outer_folds",
            (
                "test_2021",
                "test_2022",
                "test_2023",
            ),
        )
    )


def run_phase3_repair_matrix(
    config: CampaignConfig,
    output: Path,
    *,
    frame_loader: Callable[
        [CampaignConfig, Path, str],
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    contract = config.neural_training

    planned: dict[str, list[Any]] = {
        dataset: build_phase3_jobs(
            dataset=dataset,
            folds=_folds(config, dataset),
            deterministic_seed=(
                config.deterministic_seed
            ),
            stochastic_seeds=(
                config.stochastic_seeds
            ),
        )
        for dataset in config.primary_datasets
    }

    planned_count = sum(
        len(jobs)
        for jobs in planned.values()
    )
    expected_count = (
        int(config.budgets["phase3"])
        * len(config.primary_datasets)
    )

    if planned_count != expected_count:
        raise RuntimeError(
            "PHASE3_PLANNED_BUDGET_MISMATCH:"
            f"{planned_count}:{expected_count}"
        )

    candidate_set = {
        job.candidate
        for jobs in planned.values()
        for job in jobs
    }
    missing_handlers = sorted(
        candidate_set.difference(HANDLERS)
    )
    if missing_handlers:
        raise RuntimeError(
            "PHASE3_HANDLERS_INCOMPLETE:"
            + ",".join(missing_handlers)
        )

    plan_path = (
        output
        / "control/phase3_repair_execution_matrix.json"
    )
    _atomic_json(
        plan_path,
        {
            "phase": "phase3",
            "status": "EXECUTION_MATRIX_FROZEN",
            "budget_semantics": "EXACT_FROZEN_MATRIX",
            "per_dataset_jobs": int(
                config.budgets["phase3"]
            ),
            "campaign_jobs": expected_count,
            "datasets": {
                dataset: {
                    "count": len(jobs),
                    "jobs": [
                        job.to_mapping()
                        for job in jobs
                    ],
                }
                for dataset, jobs in planned.items()
            },
            "outer_test_used": False,
        },
    )

    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    execution_fingerprints: dict[str, str] = {}

    training_contract_mapping = {
        "batch_size": contract.batch_size,
        "max_epochs": contract.max_epochs,
        "min_epochs": contract.min_epochs,
        "early_stopping_patience": (
            contract.early_stopping_patience
        ),
        "learning_rate": contract.learning_rate,
        "weight_decay": contract.weight_decay,
        "gradient_clip_norm": (
            contract.gradient_clip_norm
        ),
        "target_scaling": contract.target_scaling,
        "shuffle_each_epoch": (
            contract.shuffle_each_epoch
        ),
        "restore_best_checkpoint": (
            contract.restore_best_checkpoint
        ),
        "minimum_optimizer_steps": (
            contract.minimum_optimizer_steps
        ),
        "minimum_parameter_delta": (
            contract.minimum_parameter_delta
        ),
    }

    code_fingerprint = (
        canonical_mapping_fingerprint(
            {
                "module": (
                    "stage8_v4_phase3_execution_v1"
                ),
                "handler_candidates": sorted(
                    HANDLERS
                ),
            }
        )
    )

    for dataset, jobs in planned.items():
        frame = frame_loader(
            config,
            output,
            dataset,
        )

        feature_columns = sorted(
            column
            for column in frame.columns
            if (
                column.startswith("weather_")
                or column.startswith("soil_")
            )
        )
        feature_contract = {
            "columns": feature_columns,
            "target": "target_yield",
            "year": "year",
        }
        data_fingerprint = (
            canonical_mapping_fingerprint(
                {
                    "dataset": dataset,
                    "rows": len(frame),
                    "columns": sorted(
                        frame.columns.tolist()
                    ),
                    "year_min": int(
                        frame.year.min()
                    ),
                    "year_max": int(
                        frame.year.max()
                    ),
                }
            )
        )

        for job in jobs:
            split_fingerprint = (
                canonical_mapping_fingerprint(
                    {
                        "dataset": dataset,
                        "fold": job.fold,
                        "validation_year": (
                            int(
                                job.fold.split("_")[-1]
                            )
                            - 1
                        ),
                    }
                )
            )
            execution_fingerprint = (
                phase3_execution_fingerprint(
                    job=job,
                    training_contract=(
                        training_contract_mapping
                    ),
                    feature_contract=(
                        feature_contract
                    ),
                    split_fingerprint=(
                        split_fingerprint
                    ),
                    data_fingerprint=(
                        data_fingerprint
                    ),
                    code_fingerprint=(
                        code_fingerprint
                    ),
                )
            )

            previous_job = execution_fingerprints.get(
                execution_fingerprint
            )
            if previous_job is not None:
                raise RuntimeError(
                    "PHASE3_DUPLICATE_EXECUTION_FINGERPRINT:"
                    f"{previous_job}:{job.job_id}"
                )

            execution_fingerprints[
                execution_fingerprint
            ] = job.job_id
            job_output = (
                output
                / "development/phase3_repair_matrix"
                / dataset
                / job.job_id
            )

            existing_record = job_output / "job_record.json"
            if existing_record.is_file():
                try:
                    record = load_and_validate_phase3_record(
                        record_path=existing_record,
                        output_root=output,
                        expected_job_id=job.job_id,
                        expected_execution_fingerprint=execution_fingerprint,
                    )
                    record["resume_status"] = "REUSED_VALIDATED_ARTIFACT"
                    accepted.append(record)
                    continue
                except RuntimeError as exc:
                    failure = {
                        "job_id": job.job_id,
                        "dataset": dataset,
                        "fold": job.fold,
                        "seed": job.seed,
                        "candidate": job.candidate,
                        "status": "BLOCKED_EXISTING_ARTIFACT_INVALID",
                        "failure_type": type(exc).__name__,
                        "reason": str(exc),
                        "outer_test_used": False,
                    }
                    failures.append(failure)
                    continue

            try:
                record = execute_phase3_job(
                    job=job,
                    frame=frame,
                    output=job_output,
                    output_root=output,
                    contract=contract,
                )
                record["execution_fingerprint"] = (
                    execution_fingerprint
                )
                record["resume_status"] = "EXECUTED_NEW"

                record_path = (
                    job_output / "job_record.json"
                )
                record_path.write_text(
                    json.dumps(
                        record,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

                accepted.append(record)
            except Exception as exc:
                failure = {
                    "job_id": job.job_id,
                    "dataset": dataset,
                    "fold": job.fold,
                    "seed": job.seed,
                    "candidate": job.candidate,
                    "status": "FAILED_RETRYABLE",
                    "failure_type": type(exc).__name__,
                    "reason": str(exc),
                    "outer_test_used": False,
                }
                failures.append(failure)

                failure_path = (
                    output
                    / "failures/phase3_repair_matrix.jsonl"
                )
                failure_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                with failure_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            failure,
                            sort_keys=True,
                        )
                        + "\n"
                    )

    accepted_count = len(accepted)
    failed_count = len(failures)
    complete = (
        accepted_count == expected_count
        and failed_count == 0
    )

    evidence_files = [
        str(plan_path.relative_to(output)),
    ]
    evidence_files.extend(
        str(path.relative_to(output))
        for path in sorted(
            (
                output
                / "development/phase3_repair_matrix"
            ).glob("**/job_record.json")
        )
    )
    evidence_files.extend(
        str(path.relative_to(output))
        for path in sorted(
            (
                output
                / "development/phase3_repair_matrix"
            ).glob("**/completion_marker.json")
        )
    )

    execution_status = {
        "phase": "phase3",
        "status": (
            "ENGINEERING_MATRIX_COMPLETE"
            if complete
            else "ENGINEERING_MATRIX_INCOMPLETE"
        ),
        "planned_jobs": expected_count,
        "accepted_jobs": accepted_count,
        "failed_jobs": failed_count,
        "unique_execution_fingerprints": len(
            execution_fingerprints
        ),
        "budget_semantics": "EXACT_FROZEN_MATRIX",
        "per_dataset_budget": int(
            config.budgets["phase3"]
        ),
        "campaign_budget": expected_count,
        "candidate_handlers_complete": True,
        "outer_test_used": False,
        "teacher_qualification_complete": False,
        "teacher_registry_formally_released": False,
        "phase4_release_allowed": False,
        "evidence_files": evidence_files,
    }

    _atomic_json(
        output
        / "status/phase3_repair_matrix_execution.json",
        execution_status,
    )

    # This is intentionally not yet a scientific phase completion.
    _atomic_json(
        output
        / "status/phase3_repair.json",
        {
            **execution_status,
            "status": (
                "SCIENTIFIC_PHASE_INCOMPLETE"
            ),
            "reason": (
                "EXECUTION_MATRIX_COMPLETE_BUT_"
                "TEACHER_QUALIFICATION_NOT_FROZEN"
                if complete
                else "EXECUTION_MATRIX_INCOMPLETE"
            ),
        },
    )

    return {
        "execution_status": execution_status,
        "accepted_records": accepted,
        "failures": failures,
    }
