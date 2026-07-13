from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json


QUALIFIED_DECISIONS = {
    "DEVELOPMENT_OPEN",
    "DEVELOPMENT_CONDITIONAL_OPEN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _validate_optional_preprocessor_lineage(
    *,
    teacher: dict[str, Any],
    output_root: Path,
) -> None:
    preprocessor_value = teacher.get("preprocessor")
    preprocessor_hash = teacher.get(
        "preprocessor_sha256"
    )

    if (preprocessor_value is None) != (
        preprocessor_hash is None
    ):
        raise RuntimeError(
            "TEACHER_PREPROCESSOR_LINEAGE_INCOMPLETE"
        )

    if preprocessor_value is None:
        return

    preprocessor_path = (
        output_root / str(preprocessor_value)
    ).resolve()

    try:
        preprocessor_path.relative_to(
            output_root.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "TEACHER_PREPROCESSOR_OUTSIDE_OUTPUT_ROOT"
        ) from exc

    if not preprocessor_path.is_file():
        raise RuntimeError(
            "TEACHER_PREPROCESSOR_MISSING"
        )

    if sha256_file(preprocessor_path) != str(
        preprocessor_hash
    ):
        raise RuntimeError(
            "TEACHER_PREPROCESSOR_HASH_MISMATCH"
        )


def validate_frozen_teacher_registry(
    path: Path,
    *,
    output_root: Path,
    require_formal_release: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("FROZEN_TEACHER_REGISTRY_MISSING")

    registry: dict[str, Any] = json.loads(path.read_text())

    if registry.get("status") != "FROZEN_TEACHER_REGISTRY":
        raise RuntimeError("TEACHER_REGISTRY_NOT_FROZEN")

    if registry.get("outer_refit_allowed") is not False:
        raise RuntimeError("OUTER_TEACHER_REFIT_NOT_FORBIDDEN")

    if require_formal_release and registry.get(
        "release_status"
    ) != "FORMALLY_RELEASED":
        raise RuntimeError(
            "TEACHER_REGISTRY_NOT_FORMALLY_RELEASED"
        )

    teachers = registry.get("teachers")
    if not isinstance(teachers, list):
        raise RuntimeError("TEACHER_REGISTRY_TEACHERS_INVALID")

    seen: set[tuple[str, str, str]] = set()

    for teacher in teachers:

        _validate_optional_preprocessor_lineage(

            teacher=teacher,

            output_root=output_root,

        )
        dataset = str(teacher.get("dataset"))
        role = str(teacher.get("role"))
        fold = str(teacher.get("fold"))

        if not fold.startswith("test_"):
            raise RuntimeError(
                f"FROZEN_TEACHER_FOLD_INVALID:"
                f"{dataset}:{role}:{fold}"
            )

        key = (dataset, role, fold)

        if key in seen:
            raise RuntimeError(
                f"DUPLICATE_FROZEN_TEACHER:"
                f"{dataset}:{role}:{fold}"
            )
        seen.add(key)

        decision = str(teacher.get("decision"))
        checkpoint = teacher.get("checkpoint")
        checkpoint_hash = teacher.get("checkpoint_sha256")

        if decision in QUALIFIED_DECISIONS:
            if not checkpoint or not checkpoint_hash:
                raise RuntimeError(
                    f"QUALIFIED_TEACHER_CHECKPOINT_MISSING:"
                    f"{dataset}:{role}:{fold}"
                )

            checkpoint_path = (
                output_root / str(checkpoint)
            ).resolve()

            try:
                checkpoint_path.relative_to(
                    output_root.resolve()
                )
            except ValueError as exc:
                raise RuntimeError(
                    "TEACHER_CHECKPOINT_OUTSIDE_OUTPUT_ROOT"
                ) from exc

            if not checkpoint_path.is_file():
                raise RuntimeError(
                    f"TEACHER_CHECKPOINT_FILE_MISSING:"
                    f"{dataset}:{role}:{fold}"
                )

            actual = sha256_file(checkpoint_path)
            if actual != checkpoint_hash:
                raise RuntimeError(
                    f"TEACHER_CHECKPOINT_HASH_MISMATCH:"
                    f"{dataset}:{role}:{fold}"
                )

        elif checkpoint or checkpoint_hash:
            raise RuntimeError(
                f"UNQUALIFIED_TEACHER_HAS_CHECKPOINT:"
                f"{dataset}:{role}"
            )

    return registry
