from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from src.distillation_program.next_models import (
    ModalitySpecificRegressor,
    SharedDenseRegressor,
)

from .teacher_registry import sha256_file


@dataclass(frozen=True)
class FrozenTeacher:
    dataset: str
    fold: str
    role: str
    candidate: str
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    payload: Any


def _load_torch_teacher(
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "state_dict",
        "candidate",
        "dims",
        "target_mean",
        "target_scale",
    }
    missing = sorted(
        required.difference(payload)
    )
    if missing:
        raise RuntimeError(
            "FROZEN_NEURAL_TEACHER_FIELDS_MISSING:"
            + ",".join(missing)
        )

    candidate = str(payload["candidate"])
    dims = {
        str(name): int(width)
        for name, width in payload["dims"].items()
    }

    if candidate == "shared_early_fusion":
        model = SharedDenseRegressor(
            dims,
            hidden=32,
        )
    elif candidate == (
        "modality_specific_late_fusion"
    ):
        model = ModalitySpecificRegressor(
            dims,
            hidden=32,
        )
    else:
        raise RuntimeError(
            f"FROZEN_NEURAL_TEACHER_UNKNOWN:"
            f"{candidate}"
        )

    model.load_state_dict(payload["state_dict"])
    model.eval()

    if any(
        parameter.requires_grad
        for parameter in model.parameters()
    ):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return {
        **payload,
        "model": model,
        "frozen": True,
    }


def load_frozen_teacher(
    *,
    output_root: Path,
    record: dict[str, Any],
) -> FrozenTeacher:
    checkpoint = (
        output_root / str(record["checkpoint"])
    ).resolve()

    try:
        checkpoint.relative_to(
            output_root.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "FROZEN_TEACHER_OUTSIDE_OUTPUT_ROOT"
        ) from exc

    if not checkpoint.is_file():
        raise RuntimeError(
            "FROZEN_TEACHER_CHECKPOINT_MISSING"
        )

    expected_hash = str(
        record["checkpoint_sha256"]
    )
    actual_hash = sha256_file(checkpoint)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "FROZEN_TEACHER_CHECKPOINT_HASH_MISMATCH"
        )

    suffix = checkpoint.suffix.lower()

    if suffix == ".pt":
        payload = _load_torch_teacher(checkpoint)
    elif suffix in {
        ".joblib",
        ".pkl",
        ".pickle",
        ".bin",
    }:
        payload = joblib.load(checkpoint)
    else:
        raise RuntimeError(
            f"FROZEN_TEACHER_FORMAT_UNSUPPORTED:"
            f"{suffix}"
        )

    return FrozenTeacher(
        dataset=str(record["dataset"]),
        fold=str(record["fold"]),
        role=str(record["role"]),
        candidate=str(record["candidate"]),
        seed=int(record["seed"]),
        checkpoint_path=checkpoint,
        checkpoint_sha256=expected_hash,
        payload=payload,
    )
