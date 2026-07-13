from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .contracts import CampaignConfig
from .phase3_qualification import (
    qualify_phase3_teachers,
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


def finalize_phase3_repair(
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    execution_path = (
        output
        / "status/phase3_repair_matrix_execution.json"
    )

    if not execution_path.is_file():
        raise RuntimeError(
            "PHASE3_ENGINEERING_MATRIX_STATUS_MISSING"
        )

    execution = json.loads(
        execution_path.read_text()
    )

    expected_jobs = (
        int(config.budgets["phase3"])
        * len(config.primary_datasets)
    )

    if execution.get("status") != (
        "ENGINEERING_MATRIX_COMPLETE"
    ):
        raise RuntimeError(
            "PHASE3_ENGINEERING_MATRIX_INCOMPLETE"
        )

    if int(execution.get("planned_jobs", -1)) != (
        expected_jobs
    ):
        raise RuntimeError(
            "PHASE3_ENGINEERING_PLAN_COUNT_MISMATCH"
        )

    if int(execution.get("accepted_jobs", -1)) != (
        expected_jobs
    ):
        raise RuntimeError(
            "PHASE3_ENGINEERING_ACCEPTED_COUNT_MISMATCH"
        )

    if int(execution.get("failed_jobs", -1)) != 0:
        raise RuntimeError(
            "PHASE3_ENGINEERING_FAILURES_PRESENT"
        )

    if int(
        execution.get(
            "unique_execution_fingerprints",
            -1,
        )
    ) != expected_jobs:
        raise RuntimeError(
            "PHASE3_EXECUTION_FINGERPRINT_COUNT_MISMATCH"
        )

    folds_by_dataset = {
        dataset: tuple(
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
        for dataset in config.primary_datasets
    }

    qualification = qualify_phase3_teachers(
        output=output,
        datasets=config.primary_datasets,
        folds_by_dataset=folds_by_dataset,
        expected_jobs=expected_jobs,
    )

    gates = qualification["gates"]

    qualification_complete = all(
        gate[
            "prediction_teacher_complete_across_folds"
        ]
        or gate[
            "representation_teacher_complete_across_folds"
        ]
        for gate in gates.values()
    )

    status = {
        "phase": "phase3",
        "status": "SCIENTIFIC_PHASE_COMPLETE",
        "planned_jobs": expected_jobs,
        "accepted_jobs": expected_jobs,
        "failed_jobs": 0,
        "engineering_matrix_complete": True,
        "qualification_complete": True,
        "at_least_one_teacher_role_complete_per_dataset": (
            qualification_complete
        ),
        "teacher_registry_formally_released": True,
        "frozen_teacher_count": len(
            qualification["registry"]["teachers"]
        ),
        "frozen_teacher_registry": (
            qualification["registry_path"]
        ),
        "frozen_teacher_registry_sha256": (
            qualification["registry_sha256"]
        ),
        "outer_refit_allowed": False,
        "outer_test_used": False,
        "phase4_release_allowed": True,
        "evidence_files": (
            execution["evidence_files"]
            + [
                qualification["registry_path"],
            ]
            + [
                (
                    "control/gates/"
                    f"{dataset}__teachers_repair.json"
                )
                for dataset in config.primary_datasets
            ]
        ),
    }

    _atomic_json(
        output / "status/phase3_repair.json",
        status,
    )

    return {
        "status": status,
        "qualification": qualification,
    }
