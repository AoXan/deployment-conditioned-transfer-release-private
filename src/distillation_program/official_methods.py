from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import yaml


class FidelityStatus(str, Enum):
    CANDIDATE_PENDING_VERIFICATION = "CANDIDATE_PENDING_VERIFICATION"
    OFFICIAL_FIDELITY_VERIFIED = "OFFICIAL_FIDELITY_VERIFIED"
    OFFICIAL_FIDELITY_PARTIAL = "OFFICIAL_FIDELITY_PARTIAL"
    BLOCKED_OFFICIAL_FIDELITY = "BLOCKED_OFFICIAL_FIDELITY"
    REJECTED_INCOMPATIBLE_ASSUMPTIONS = "REJECTED_INCOMPATIBLE_ASSUMPTIONS"


@dataclass(frozen=True)
class MethodCandidate:
    method_id: str
    paper_url: str
    official_repository: str
    official_commit: str
    status: FidelityStatus
    required_verification: tuple[str, ...]
    source_files: tuple[str, ...]


def load_method_registry(path: Path) -> dict[str, MethodCandidate]:
    payload = yaml.safe_load(path.read_text())
    result = {}
    for method_id, row in payload["methods"].items():
        result[method_id] = MethodCandidate(
            method_id=method_id,
            paper_url=row["paper_url"],
            official_repository=row["official_repository"],
            official_commit=row["official_commit"],
            status=FidelityStatus(row["status"]),
            required_verification=tuple(row["required_verification"]),
            source_files=tuple(row["source_files"]),
        )
    return result


def regression_distillation_loss(
    *,
    student_prediction,
    teacher_prediction,
    target,
    supervised_weight: float,
    distillation_weight: float,
):
    supervised = torch.nn.functional.huber_loss(student_prediction, target)
    prediction = torch.nn.functional.huber_loss(student_prediction, teacher_prediction.detach())
    total = supervised_weight * supervised + distillation_weight * prediction
    return total, {"supervised_huber": supervised, "prediction_huber": prediction}


def gmd_project_conflicting(gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    originals = [gradient.detach().clone() for gradient in gradients]
    projected: list[torch.Tensor] = []
    for index, gradient in enumerate(originals):
        adjusted = gradient.clone()
        for other_index, other in enumerate(originals):
            if index == other_index:
                continue
            dot = torch.dot(adjusted.flatten(), other.flatten())
            if dot < 0:
                adjusted = adjusted - dot / other.square().sum().clamp_min(1e-12) * other
        projected.append(adjusted)
    return projected
