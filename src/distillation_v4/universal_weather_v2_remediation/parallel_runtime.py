from __future__ import annotations

import importlib
import os
import pickle
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ResourceBudget:
    max_workers: int
    worker_threads: int = 1

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS_MUST_BE_POSITIVE")
        if self.worker_threads < 1:
            raise ValueError("WORKER_THREADS_MUST_BE_POSITIVE")


@dataclass(frozen=True)
class ParallelTask:
    task_id: str
    callable_path: str
    kwargs: dict[str, Any]
    output_dir: str | None = None
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParallelTaskResult:
    task_id: str
    status: str
    pid: int
    elapsed_seconds: float
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    submitted: bool = True


@dataclass(frozen=True)
class ParallelRunSummary:
    status: str
    total: int
    completed: int
    failed: int
    cancelled: int
    submitted: int
    not_submitted: int
    elapsed_seconds: float
    results: tuple[ParallelTaskResult, ...]
    active_task_ids: tuple[str, ...] = ()


def configure_worker_threads(worker_threads: int) -> None:
    """Constrain nested CPU parallelism inside a spawned worker."""

    value = str(int(worker_threads))
    for name in THREAD_ENV_VARS:
        os.environ[name] = value

    try:
        import torch

        torch.set_num_threads(int(worker_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        # Torch may be unavailable in lightweight tests. Thread env vars
        # above still protect sklearn/numpy-style workers.
        return


def validate_parallel_tasks(tasks: list[ParallelTask]) -> None:
    task_ids: set[str] = set()
    output_dirs: set[str] = set()

    for task in tasks:
        if not task.task_id:
            raise ValueError("PARALLEL_TASK_ID_REQUIRED")
        if task.task_id in task_ids:
            raise ValueError(f"DUPLICATE_PARALLEL_TASK_ID:{task.task_id}")
        task_ids.add(task.task_id)

        if not task.callable_path or ":" not in task.callable_path:
            raise ValueError(
                f"PARALLEL_CALLABLE_PATH_INVALID:{task.task_id}"
            )

        if task.output_dir:
            resolved = str(Path(task.output_dir).expanduser().resolve())
            if resolved in output_dirs:
                raise ValueError(
                    f"DUPLICATE_PARALLEL_OUTPUT_DIR:{resolved}"
                )
            output_dirs.add(resolved)

        missing = [
            dependency
            for dependency in task.dependency_ids
            if dependency not in task_ids
        ]
        if missing:
            # Tasks are executed in independent batches. Within one batch,
            # dependencies should already be satisfied by completed stages,
            # not by another task submitted simultaneously.
            continue

        try:
            _resolve_callable(task.callable_path)
        except Exception as error:
            raise ValueError(
                f"PARALLEL_CALLABLE_UNRESOLVED:{task.task_id}:"
                f"{type(error).__name__}:{error}"
            ) from error

        try:
            pickle.dumps((task, 1))
        except Exception as error:
            raise ValueError(
                f"PARALLEL_TASK_NOT_PICKLABLE:{task.task_id}:"
                f"{type(error).__name__}:{error}"
            ) from error


def _resolve_callable(callable_path: str) -> Callable[..., Any]:
    module_name, function_name = callable_path.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"PARALLEL_TARGET_NOT_CALLABLE:{callable_path}")
    return function


def _run_one_task(
    task: ParallelTask,
    worker_threads: int,
) -> ParallelTaskResult:
    configure_worker_threads(worker_threads)
    started = time.time()
    pid = os.getpid()

    try:
        function = _resolve_callable(task.callable_path)
        result = function(**task.kwargs)
        return ParallelTaskResult(
            task_id=task.task_id,
            status="COMPLETED",
            pid=pid,
            elapsed_seconds=time.time() - started,
            result=result,
        )
    except Exception as error:
        return ParallelTaskResult(
            task_id=task.task_id,
            status="FAILED",
            pid=pid,
            elapsed_seconds=time.time() - started,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )


def run_parallel_tasks(
    tasks: list[ParallelTask],
    *,
    budget: ResourceBudget,
    stop_requested: Callable[[], bool] | None = None,
    on_result: Callable[[ParallelTaskResult, tuple[str, ...], int], None]
    | None = None,
) -> ParallelRunSummary:
    """Run independent tasks with spawn semantics and ordered results."""

    validate_parallel_tasks(tasks)
    if not tasks:
        return ParallelRunSummary(
            status="COMPLETED",
            total=0,
            completed=0,
            failed=0,
            cancelled=0,
            submitted=0,
            not_submitted=0,
            elapsed_seconds=0.0,
            results=(),
        )

    started = time.time()
    ordered: dict[str, ParallelTaskResult] = {}
    context = get_context("spawn")
    pending_index = 0
    submitted = 0
    cancelled = 0
    stopped = False
    active: dict[Future[ParallelTaskResult], ParallelTask] = {}

    def should_stop() -> bool:
        return bool(stop_requested and stop_requested())

    try:
        with ProcessPoolExecutor(
            max_workers=budget.max_workers,
            mp_context=context,
        ) as executor:
            while pending_index < len(tasks) or active:
                while (
                    pending_index < len(tasks)
                    and len(active) < budget.max_workers
                    and not should_stop()
                ):
                    task = tasks[pending_index]
                    pending_index += 1
                    future = executor.submit(
                        _run_one_task,
                        task,
                        budget.worker_threads,
                    )
                    active[future] = task
                    submitted += 1

                if should_stop():
                    stopped = True

                if not active:
                    break

                done, _not_done = wait(
                    active,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    task = active.pop(future)
                    active_ids = tuple(
                        sorted(item.task_id for item in active.values())
                    )
                    try:
                        result = future.result()
                    except BrokenProcessPool as error:
                        result = ParallelTaskResult(
                            task_id=task.task_id,
                            status="FAILED",
                            pid=-1,
                            elapsed_seconds=0.0,
                            error_type=type(error).__name__,
                            error_message=str(error),
                            traceback=traceback.format_exc(),
                        )
                    except Exception as error:
                        result = ParallelTaskResult(
                            task_id=task.task_id,
                            status="FAILED",
                            pid=-1,
                            elapsed_seconds=0.0,
                            error_type=type(error).__name__,
                            error_message=str(error),
                            traceback=traceback.format_exc(),
                        )

                    ordered[task.task_id] = result
                    if on_result is not None:
                        try:
                            on_result(
                                result,
                                active_ids,
                                len(tasks) - pending_index,
                            )
                        except Exception as error:
                            stopped = False
                            for remaining in active:
                                if remaining.cancel():
                                    cancelled += 1
                            raise RuntimeError(
                                "PARALLEL_ON_RESULT_FAILED:"
                                f"{result.task_id}:"
                                f"{type(error).__name__}:{error}"
                            ) from error

            if stopped:
                for future in active:
                    if future.cancel():
                        cancelled += 1
                # Running futures are allowed to finish by leaving the
                # executor context; no SIGKILL or terminate is used.
    except BrokenProcessPool as error:
        missing = [
            task
            for task in tasks
            if task.task_id not in ordered
        ]
        if missing:
            ordered[missing[0].task_id] = ParallelTaskResult(
                task_id=missing[0].task_id,
                status="FAILED",
                pid=-1,
                elapsed_seconds=0.0,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )

    for task in tasks:
        if task.task_id not in ordered:
            ordered[task.task_id] = ParallelTaskResult(
                task_id=task.task_id,
                status="NOT_SUBMITTED",
                pid=-1,
                elapsed_seconds=0.0,
                submitted=False,
            )

    results = tuple(ordered[task.task_id] for task in tasks)
    failed = sum(1 for result in results if result.status != "COMPLETED")
    not_submitted = sum(
        1 for result in results if result.status == "NOT_SUBMITTED"
    )
    completed = len(results) - failed
    if stopped:
        status = "STOPPED_BY_OPERATOR"
    elif failed:
        status = "FAILED"
    else:
        status = "COMPLETED"

    return ParallelRunSummary(
        status=status,
        total=len(results),
        completed=completed,
        failed=failed,
        cancelled=cancelled,
        submitted=submitted,
        not_submitted=not_submitted,
        elapsed_seconds=time.time() - started,
        results=results,
    )
