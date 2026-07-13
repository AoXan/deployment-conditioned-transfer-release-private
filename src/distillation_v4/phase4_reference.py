from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from .frozen_teacher import (
    load_frozen_teacher,
)


def materialize_frozen_reference(
    *,
    output_root: Path,
    teacher_record: dict[str, Any],
    output: Path,
    route_id: str,
) -> dict[str, Any]:
    teacher = load_frozen_teacher(
        output_root=output_root,
        record=teacher_record,
    )

    output.mkdir(parents=True, exist_ok=True)

    record = {
        "route_id": route_id,
        "route": "reference",
        "status": "ENGINEERING_ACCEPTED",
        "dataset": teacher.dataset,
        "fold": teacher.fold,
        "seed": teacher.seed,
        "candidate": teacher.candidate,
        "checkpoint": str(
            teacher.checkpoint_path.relative_to(
                output_root.resolve()
            )
        ),
        "checkpoint_sha256": (
            teacher.checkpoint_sha256
        ),
        "teacher_refit_performed": False,
        "student_training_performed": False,
        "outer_test_used": False,
        "scientific_role": (
            "PRIVILEGED_REFERENCE_ONLY"
        ),
    }

    (
        output / "reference_record.json"
    ).write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    (
        output / "completion_marker.json"
    ).write_text(
        json.dumps(
            {
                "route_id": record.get(
                    "route_id"
                ),
                "status": "ENGINEERING_ACCEPTED",
                "outer_test_used": False,
                "teacher_refit_performed": False,
                "student_training_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return record
