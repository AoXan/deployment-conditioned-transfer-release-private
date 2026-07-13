from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .execution import execute_target_job, ensure_source_checkpoint
from .legacy_regression_guards import audit_job_artifacts
from .remaining_handlers import (
    execute_source_controls,
    execute_source_screening,
    execute_teacher_qualification,
)
from .runtime_control import (
    append_failure,
    exact_resume_decision,
    stop_requested,
    write_heartbeat,
)
from .parallel_runtime import (
    ParallelTask,
    ResourceBudget,
    run_parallel_tasks,
)


IMMUTABLE_CORE_JOB_MANIFEST = (
    "immutable_core_job_manifest.json"
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _walk_job_lists(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {
            "candidate_id",
            "dataset_id",
            "split_id",
            "seed",
            "transfer_strategy",
        } <= set(value):
            yield value

        for child in value.values():
            yield from _walk_job_lists(child)

    elif isinstance(value, list):
        for child in value:
            yield from _walk_job_lists(child)


def load_immutable_core_jobs(
    output_root: Path,
) -> list[dict[str, Any]]:
    path = (
        output_root
        / "control"
        / IMMUTABLE_CORE_JOB_MANIFEST
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"IMMUTABLE_CORE_JOB_MANIFEST_MISSING:{path}"
        )

    payload = json.loads(path.read_text())
    jobs = list(_walk_job_lists(payload))

    unique: dict[str, dict[str, Any]] = {}
    for job in jobs:
        candidate_id = str(job["candidate_id"])

        if candidate_id in unique:
            if unique[candidate_id] != job:
                raise ValueError(
                    "DUPLICATE_CANDIDATE_ID_WITH_DRIFT:"
                    + candidate_id
                )
            continue

        unique[candidate_id] = dict(job)

    result = sorted(
        unique.values(),
        key=lambda item: str(item["candidate_id"]),
    )

    if not result:
        raise ValueError(
            "IMMUTABLE_CORE_JOB_MANIFEST_HAS_NO_JOBS"
        )

    return result


def _requires_source_dependency(
    job: dict[str, Any],
) -> bool:
    if job.get("transfer_strategy") == "target_scratch":
        return False

    method = job.get("missing_modality_method")
    if method in {
        "imputation",
        "missing_indicators",
        "late_fusion",
        "teacher_student_m2",
    }:
        return False

    return bool(
        job.get("source_dataset")
        and job.get("source_route")
    )


def _source_dependency_key(
    job: dict[str, Any],
) -> tuple[str, str, int]:
    return (
        str(job["source_dataset"]),
        str(job["source_route"]),
        int(job["seed"]),
    )


def _target_directory(
    output_root: Path,
    candidate_id: str,
) -> Path:
    return (
        output_root
        / "jobs"
        / "targets"
        / candidate_id
    )


def _target_resume_decision(
    output_root: Path,
    job: dict[str, Any],
    plan_fingerprint: str,
) -> tuple[bool, str]:
    directory = _target_directory(
        output_root,
        str(job["candidate_id"]),
    )

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False, "MANIFEST_MISSING"

    try:
        manifest = _load_object(manifest_path)
    except Exception:
        return False, "MANIFEST_INVALID"

    required = {
        "resolved_config_hash",
        "code_tree_hash",
        "data_hash",
        "split_fingerprint",
        "feature_contract_hash",
    }

    if not required <= set(manifest):
        return False, "RESUME_IDENTITY_FIELDS_MISSING"

    exact = exact_resume_decision(
        directory,
        expected_namespace=str(output_root),
        expected_plan_fingerprint=plan_fingerprint,
        expected_config_hash=str(
            manifest["resolved_config_hash"]
        ),
        expected_code_hash=str(
            manifest["code_tree_hash"]
        ),
        expected_data_hash=str(manifest["data_hash"]),
        expected_split_hash=str(
            manifest["split_fingerprint"]
        ),
        expected_feature_hash=str(
            manifest["feature_contract_hash"]
        ),
    )

    if exact != (True, "EXACT_ACCEPTED_MATCH"):
        return exact

    dependency_hash = manifest.get(
        "source_checkpoint_hash"
    )

    audit = audit_job_artifacts(
        directory,
        expected_namespace=str(output_root),
        expected_plan_fingerprint=plan_fingerprint,
        expected_job_id=str(job["candidate_id"]),
        expected_source_route=(
            str(job["source_route"])
            if job.get("source_route")
            else None
        ),
        expected_split_fingerprint=str(
            manifest["split_fingerprint"]
        ),
        expected_dependency_checkpoint_hash=(
            str(dependency_hash)
            if dependency_hash
            else None
        ),
    )

    if not audit.accepted:
        return (
            False,
            "SCIENTIFIC_ARTIFACT_AUDIT_REJECTED:"
            + "|".join(audit.reasons),
        )

    return True, "EXACT_ACCEPTED_MATCH"


def execute_source_dependency_stage(
    *,
    config: dict[str, Any],
    source_job: dict[str, Any],
    repository_root: str,
    output_root: str,
) -> dict[str, Any]:
    repository = Path(repository_root)
    output = Path(output_root)

    screening = execute_source_screening(
        config,
        source_job,
        repository_root=repository,
        output_root=output,
        smoke=False,
    )
    qualification = execute_teacher_qualification(
        config,
        source_job,
        repository_root=repository,
        output_root=output,
        smoke=False,
    )
    return {
        "screening": screening,
        "qualification": qualification,
    }


def execute_source_controls_stage(
    *,
    config: dict[str, Any],
    source_job: dict[str, Any],
    repository_root: str,
    output_root: str,
) -> dict[str, Any]:
    return execute_source_controls(
        config,
        source_job,
        repository_root=Path(repository_root),
        output_root=Path(output_root),
        smoke=False,
    )


def execute_target_stage(
    *,
    config: dict[str, Any],
    job: dict[str, Any],
    repository_root: str,
    output_root: str,
) -> dict[str, Any]:
    return execute_target_job(
        config,
        job,
        repository_root=Path(repository_root),
        output_root=Path(output_root),
        smoke=False,
    )


def _raise_if_parallel_failed(
    *,
    output_root: Path,
    stage: str,
    summary: Any,
    plan_fingerprint: str,
) -> None:
    if getattr(summary, "status", None) == "STOPPED_BY_OPERATOR":
        return
    if summary.failed == 0:
        return

    for result in summary.results:
        if result.status == "COMPLETED":
            continue
        append_failure(
            output_root,
            {
                "stage": stage,
                "job_id": result.task_id,
                "error_type": result.error_type,
                "reason": result.error_message,
                "traceback": result.traceback,
                "pid": result.pid,
                "plan_fingerprint": plan_fingerprint,
            },
        )

    first = next(
        result
        for result in summary.results
        if result.status != "COMPLETED"
    )
    raise RuntimeError(
        f"PARALLEL_STAGE_FAILED:{stage}:{first.task_id}:"
        f"{first.error_type}:{first.error_message}"
    )


def _source_dependency_resume_decision(
    config: dict[str, Any],
    repository_root: Path,
    output_root: Path,
    source_job: dict[str, Any],
) -> tuple[bool, str]:
    try:
        ensure_source_checkpoint(
            config,
            repository_root,
            output_root,
            {
                "source_dataset": source_job["source_dataset"],
                "source_route": source_job["source_route"],
                "seed": source_job["seed"],
            },
            smoke=False,
        )
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"

    return True, "SOURCE_DEPENDENCY_ACCEPTED"


def _run_parallel_formal_campaign(
    *,
    config: dict[str, Any],
    repository_root: Path,
    output_root: Path,
    plan_fingerprint: str,
    resume: bool,
    jobs: list[dict[str, Any]],
    source_workers: int,
    target_workers: int,
    source_worker_threads: int,
    target_worker_threads: int,
) -> dict[str, Any]:
    for job in jobs:
        job["plan_fingerprint"] = plan_fingerprint

    dependency_jobs = {
        _source_dependency_key(job)
        for job in jobs
        if _requires_source_dependency(job)
    }
    source_datasets = {item[0] for item in dependency_jobs}
    total_work = len(dependency_jobs) * 2 + len(source_datasets) + len(jobs)
    started = time.time()

    write_heartbeat(
        output_root,
        status="FORMAL_DEPENDENCY_STAGE_PARALLEL",
        current_job=None,
        completed=0,
        failed=0,
        remaining=total_work,
        plan_fingerprint=plan_fingerprint,
    )

    if stop_requested(output_root):
        return {
            "status": "STOPPED_BY_OPERATOR",
            "completed": 0,
            "resumed": 0,
            "failed": 0,
            "executed_targets": 0,
            "parallel": True,
        }

    dependency_tasks: list[ParallelTask] = []
    completed = 0
    resumed = 0
    for source_dataset, source_route, seed in sorted(dependency_jobs):
        dependency_id = f"{source_dataset}:{source_route}:{seed}"
        source_job = {
            "source_dataset": source_dataset,
            "source_route": source_route,
            "seed": seed,
        }
        if resume:
            source_accepted, _source_reason = (
                _source_dependency_resume_decision(
                    config,
                    repository_root,
                    output_root,
                    source_job,
                )
            )
            if source_accepted:
                resumed += 1
                completed += 2
                continue

        dependency_tasks.append(
            ParallelTask(
                task_id=dependency_id,
                callable_path=(
                    "src.distillation_v4.universal_weather_v2_remediation."
                    "formal_dispatcher:execute_source_dependency_stage"
                ),
                kwargs={
                    "config": config,
                    "source_job": source_job,
                    "repository_root": str(repository_root),
                    "output_root": str(output_root),
                },
                output_dir=str(
                    output_root
                    / "jobs"
                    / "source"
                    / source_dataset
                    / source_route
                    / f"seed_{seed}"
                ),
            )
        )

    accepted_dependencies = 0

    def _on_dependency_result(
        result: Any,
        active: tuple[str, ...],
        queued: int,
    ) -> None:
        nonlocal accepted_dependencies
        if result.status == "COMPLETED":
            accepted_dependencies += 1
        write_heartbeat(
            output_root,
            status=(
                "SOURCE_DEPENDENCY_ACCEPTED_PARALLEL"
                if result.status == "COMPLETED"
                else "SOURCE_DEPENDENCY_FAILED_PARALLEL"
            ),
            current_job=result.task_id,
            completed=completed + accepted_dependencies * 2,
            failed=0 if result.status == "COMPLETED" else 1,
            remaining=max(
                0,
                total_work - completed - accepted_dependencies * 2,
            ),
            plan_fingerprint=plan_fingerprint,
        )

    dependency_summary = run_parallel_tasks(
        dependency_tasks,
        budget=ResourceBudget(
            max_workers=source_workers,
            worker_threads=source_worker_threads,
        ),
        stop_requested=lambda: stop_requested(output_root),
        on_result=_on_dependency_result,
    )
    _raise_if_parallel_failed(
        output_root=output_root,
        stage="source_dependency_parallel",
        summary=dependency_summary,
        plan_fingerprint=plan_fingerprint,
    )

    if dependency_summary.status == "STOPPED_BY_OPERATOR":
        return {
            "status": "STOPPED_BY_OPERATOR",
            "completed": completed + dependency_summary.completed * 2,
            "resumed": resumed,
            "failed": dependency_summary.failed,
            "executed_targets": 0,
            "parallel": True,
        }

    completed += dependency_summary.completed * 2
    failed = dependency_summary.failed

    control_tasks = [
        ParallelTask(
            task_id=f"source_controls:{source_dataset}",
            callable_path=(
                "src.distillation_v4.universal_weather_v2_remediation."
                "formal_dispatcher:execute_source_controls_stage"
            ),
            kwargs={
                "config": config,
                "source_job": {"source_dataset": source_dataset},
                "repository_root": str(repository_root),
                "output_root": str(output_root),
            },
            output_dir=str(
                output_root / "jobs" / "source_controls" / source_dataset
            ),
        )
        for source_dataset in sorted(source_datasets)
    ]
    control_summary = run_parallel_tasks(
        control_tasks,
        budget=ResourceBudget(
            max_workers=max(1, min(source_workers, len(control_tasks) or 1)),
            worker_threads=source_worker_threads,
        ),
        stop_requested=lambda: stop_requested(output_root),
    )
    _raise_if_parallel_failed(
        output_root=output_root,
        stage="source_controls_parallel",
        summary=control_summary,
        plan_fingerprint=plan_fingerprint,
    )
    if control_summary.status == "STOPPED_BY_OPERATOR":
        return {
            "status": "STOPPED_BY_OPERATOR",
            "completed": completed + control_summary.completed,
            "resumed": resumed,
            "failed": control_summary.failed,
            "executed_targets": 0,
            "parallel": True,
        }

    completed += control_summary.completed

    target_tasks: list[ParallelTask] = []
    job_by_id = {str(job["candidate_id"]): job for job in jobs}
    accepted_targets = 0

    def _on_target_result(
        result: Any,
        active: tuple[str, ...],
        queued: int,
    ) -> None:
        nonlocal accepted_targets, failed

        if result.status != "COMPLETED":
            failed += 1
            append_failure(
                output_root,
                {
                    "stage": "target_execution_parallel",
                    "job_id": result.task_id,
                    "error_type": result.error_type,
                    "reason": result.error_message,
                    "traceback": result.traceback,
                    "worker_pid": result.pid,
                    "plan_fingerprint": plan_fingerprint,
                },
            )
            write_heartbeat(
                output_root,
                status="FAILED",
                current_job=result.task_id,
                completed=completed + accepted_targets,
                failed=failed,
                remaining=max(0, total_work - completed - accepted_targets),
                plan_fingerprint=plan_fingerprint,
            )
            raise RuntimeError(
                "TARGET_WORKER_FAILED:"
                f"{result.task_id}:{result.error_type}:"
                f"{result.error_message}"
            )

        accepted, reason = _target_resume_decision(
            output_root,
            job_by_id[result.task_id],
            plan_fingerprint,
        )
        if not accepted:
            failed += 1
            append_failure(
                output_root,
                {
                    "stage": "target_acceptance_parallel",
                    "job_id": result.task_id,
                    "error_type": "PostExecutionAcceptanceError",
                    "reason": reason,
                    "traceback": None,
                    "worker_pid": result.pid,
                    "plan_fingerprint": plan_fingerprint,
                },
            )
            write_heartbeat(
                output_root,
                status="FAILED",
                current_job=result.task_id,
                completed=completed + accepted_targets,
                failed=failed,
                remaining=max(0, total_work - completed - accepted_targets),
                plan_fingerprint=plan_fingerprint,
            )
            raise RuntimeError(
                "POST_EXECUTION_ACCEPTANCE_FAILED:"
                + result.task_id
                + ":"
                + reason
            )

        accepted_targets += 1
        write_heartbeat(
            output_root,
            status="TARGET_ACCEPTED_PARALLEL",
            current_job=result.task_id,
            completed=completed + accepted_targets,
            failed=failed,
            remaining=max(0, total_work - completed - accepted_targets),
            plan_fingerprint=plan_fingerprint,
        )

    for job in jobs:
        candidate_id = str(job["candidate_id"])

        if stop_requested(output_root):
            write_heartbeat(
                output_root,
                status="STOPPED_BY_OPERATOR",
                current_job=candidate_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )
            return {
                "status": "STOPPED_BY_OPERATOR",
                "completed": completed,
                "resumed": resumed,
                "failed": failed,
                "executed_targets": 0,
                "parallel": True,
            }

        if resume:
            can_resume, _resume_reason = _target_resume_decision(
                output_root,
                job,
                plan_fingerprint,
            )
            if can_resume:
                resumed += 1
                completed += 1
                continue

        target_tasks.append(
            ParallelTask(
                task_id=candidate_id,
                callable_path=(
                    "src.distillation_v4.universal_weather_v2_remediation."
                    "formal_dispatcher:execute_target_stage"
                ),
                kwargs={
                    "config": config,
                    "job": job,
                    "repository_root": str(repository_root),
                    "output_root": str(output_root),
                },
                output_dir=str(_target_directory(output_root, candidate_id)),
            )
        )

    write_heartbeat(
        output_root,
        status="TARGET_EXECUTION_PARALLEL",
        current_job=None,
        completed=completed,
        failed=failed,
        remaining=total_work - completed,
        plan_fingerprint=plan_fingerprint,
    )
    target_summary = run_parallel_tasks(
        target_tasks,
        budget=ResourceBudget(
            max_workers=target_workers,
            worker_threads=target_worker_threads,
        ),
        stop_requested=lambda: stop_requested(output_root),
        on_result=_on_target_result,
    )
    _raise_if_parallel_failed(
        output_root=output_root,
        stage="target_execution_parallel",
        summary=target_summary,
        plan_fingerprint=plan_fingerprint,
    )

    if target_summary.status == "STOPPED_BY_OPERATOR":
        return {
            "status": "STOPPED_BY_OPERATOR",
            "completed": completed + accepted_targets,
            "resumed": resumed,
            "failed": failed,
            "executed_targets": accepted_targets,
            "parallel": True,
            "not_submitted": target_summary.not_submitted,
        }

    executed_targets = accepted_targets
    completed += accepted_targets

    result = {
        "schema_version": "formal_dispatch_summary_v1",
        "status": "COMPLETED",
        "plan_fingerprint": plan_fingerprint,
        "declared_target_jobs": len(jobs),
        "declared_source_dependencies": len(dependency_jobs),
        "executed_targets": executed_targets,
        "resumed_targets": resumed,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
        "parallel": True,
        "source_workers": source_workers,
        "target_workers": target_workers,
    }

    control = output_root / "control"
    control.mkdir(parents=True, exist_ok=True)
    temporary = control / "formal_dispatch_summary.json.tmp"
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(control / "formal_dispatch_summary.json")

    write_heartbeat(
        output_root,
        status="FORMAL_COMPLETED",
        current_job=None,
        completed=completed,
        failed=failed,
        remaining=0,
        plan_fingerprint=plan_fingerprint,
    )
    return result


def run_formal_campaign(
    *,
    config: dict[str, Any],
    repository_root: Path,
    output_root: Path,
    plan_fingerprint: str,
    resume: bool = True,
    jobs_override: list[dict[str, Any]] | None = None,
    parallel: bool = False,
    source_workers: int = 1,
    target_workers: int = 1,
    source_worker_threads: int = 1,
    target_worker_threads: int = 1,
) -> dict[str, Any]:
    """Execute the immutable formal DAG in dependency order.

    Source dependencies are explicit stages. Target handlers are never
    permitted to train a missing source model.
    """

    jobs = (
        [dict(job) for job in jobs_override]
        if jobs_override is not None
        else load_immutable_core_jobs(output_root)
    )

    if parallel:
        return _run_parallel_formal_campaign(
            config=config,
            repository_root=repository_root,
            output_root=output_root,
            plan_fingerprint=plan_fingerprint,
            resume=resume,
            jobs=jobs,
            source_workers=source_workers,
            target_workers=target_workers,
            source_worker_threads=source_worker_threads,
            target_worker_threads=target_worker_threads,
        )

    for job in jobs:
        job["plan_fingerprint"] = plan_fingerprint

    dependency_jobs = {
        _source_dependency_key(job)
        for job in jobs
        if _requires_source_dependency(job)
    }

    completed = 0
    resumed = 0
    failed = 0
    started = time.time()

    total_work = len(dependency_jobs) * 3 + len(jobs)

    write_heartbeat(
        output_root,
        status="FORMAL_DEPENDENCY_STAGE",
        current_job=None,
        completed=0,
        failed=0,
        remaining=total_work,
        plan_fingerprint=plan_fingerprint,
    )

    source_datasets: set[str] = set()

    for source_dataset, source_route, seed in sorted(
        dependency_jobs
    ):
        if stop_requested(output_root):
            write_heartbeat(
                output_root,
                status="STOPPED_BY_OPERATOR",
                current_job=None,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )
            return {
                "status": "STOPPED_BY_OPERATOR",
                "completed": completed,
                "resumed": resumed,
                "failed": failed,
                "executed_targets": 0,
            }

        source_job = {
            "source_dataset": source_dataset,
            "source_route": source_route,
            "seed": seed,
        }
        dependency_id = (
            f"{source_dataset}:{source_route}:{seed}"
        )

        try:
            write_heartbeat(
                output_root,
                status="SOURCE_SCREENING",
                current_job=dependency_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )

            execute_source_screening(
                config,
                source_job,
                repository_root=repository_root,
                output_root=output_root,
                smoke=False,
            )
            completed += 1

            write_heartbeat(
                output_root,
                status="TEACHER_QUALIFICATION",
                current_job=dependency_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )

            execute_teacher_qualification(
                config,
                source_job,
                repository_root=repository_root,
                output_root=output_root,
                smoke=False,
            )
            completed += 1
            source_datasets.add(source_dataset)

        except Exception as error:
            failed += 1
            append_failure(
                output_root,
                {
                    "stage": "source_dependency",
                    "job_id": dependency_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "plan_fingerprint": plan_fingerprint,
                },
            )
            write_heartbeat(
                output_root,
                status="FAILED",
                current_job=dependency_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )
            raise

    for source_dataset in sorted(source_datasets):
        control_id = f"source_controls:{source_dataset}"

        try:
            write_heartbeat(
                output_root,
                status="SOURCE_CONTROLS",
                current_job=control_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )

            execute_source_controls(
                config,
                {"source_dataset": source_dataset},
                repository_root=repository_root,
                output_root=output_root,
                smoke=False,
            )
            completed += 1

        except Exception as error:
            failed += 1
            append_failure(
                output_root,
                {
                    "stage": "source_controls",
                    "job_id": control_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "plan_fingerprint": plan_fingerprint,
                },
            )
            raise

    executed_targets = 0

    for job in jobs:
        candidate_id = str(job["candidate_id"])

        if stop_requested(output_root):
            write_heartbeat(
                output_root,
                status="STOPPED_BY_OPERATOR",
                current_job=candidate_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )
            return {
                "status": "STOPPED_BY_OPERATOR",
                "completed": completed,
                "resumed": resumed,
                "failed": failed,
                "executed_targets": executed_targets,
            }

        if resume:
            can_resume, resume_reason = (
                _target_resume_decision(
                    output_root,
                    job,
                    plan_fingerprint,
                )
            )
            if can_resume:
                resumed += 1
                completed += 1
                continue

        try:
            write_heartbeat(
                output_root,
                status="TARGET_EXECUTION",
                current_job=candidate_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )

            execute_target_job(
                config,
                job,
                repository_root=repository_root,
                output_root=output_root,
                smoke=False,
            )

            accepted, reason = _target_resume_decision(
                output_root,
                job,
                plan_fingerprint,
            )
            if not accepted:
                raise RuntimeError(
                    "POST_EXECUTION_ACCEPTANCE_FAILED:"
                    + reason
                )

            executed_targets += 1
            completed += 1

        except Exception as error:
            failed += 1
            append_failure(
                output_root,
                {
                    "stage": "target_execution",
                    "job_id": candidate_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "plan_fingerprint": plan_fingerprint,
                },
            )
            write_heartbeat(
                output_root,
                status="FAILED",
                current_job=candidate_id,
                completed=completed,
                failed=failed,
                remaining=total_work - completed,
                plan_fingerprint=plan_fingerprint,
            )
            raise

    result = {
        "schema_version": "formal_dispatch_summary_v1",
        "status": "COMPLETED",
        "plan_fingerprint": plan_fingerprint,
        "declared_target_jobs": len(jobs),
        "declared_source_dependencies": len(
            dependency_jobs
        ),
        "executed_targets": executed_targets,
        "resumed_targets": resumed,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }

    control = output_root / "control"
    control.mkdir(parents=True, exist_ok=True)
    temporary = (
        control / "formal_dispatch_summary.json.tmp"
    )
    temporary.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(
        control / "formal_dispatch_summary.json"
    )

    write_heartbeat(
        output_root,
        status="FORMAL_COMPLETED",
        current_job=None,
        completed=completed,
        failed=failed,
        remaining=0,
        plan_fingerprint=plan_fingerprint,
    )

    return result
