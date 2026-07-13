from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .official_methods import gmd_project_conflicting


def load_adaptation_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text())
    records = payload["methods"]
    for method, row in records.items():
        missing = {"preserved", "changed", "unsupported"} - set(row)
        if missing:
            raise ValueError(f"adaptation_delta_incomplete:{method}:{sorted(missing)}")
    return records


class GarciaRegressionMechanism(torch.nn.Module):
    """Agricultural regression adaptation of staged frozen-teacher modality distillation.

    This is explicitly not an official reproduction: it retains teacher freezing,
    soft-target imitation and feature rectification while replacing video
    classification streams with tabular modality blocks.
    """

    def __init__(self, teacher: torch.nn.Module, *, deployable_width: int, representation_width: int):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.student_encoder = torch.nn.Sequential(torch.nn.Linear(deployable_width, representation_width), torch.nn.ReLU())
        self.student_head = torch.nn.Linear(representation_width, 1)
        self.teacher_rectifier = torch.nn.LazyLinear(representation_width)

    def _teacher_representation(self, rich: torch.Tensor) -> torch.Tensor:
        if isinstance(self.teacher, torch.nn.Sequential) and len(self.teacher) >= 2:
            value = rich
            for layer in list(self.teacher)[:-1]:
                value = layer(value)
            return value
        return self.teacher(rich).reshape(len(rich), -1)

    def forward(self, deployable: torch.Tensor, rich: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        student_representation = self.student_encoder(deployable)
        student_prediction = self.student_head(student_representation).squeeze(1)
        with torch.no_grad():
            teacher_prediction = self.teacher(rich).reshape(-1)
            teacher_representation = self._teacher_representation(rich)
        rectified = self.teacher_rectifier(teacher_representation.detach())
        return {
            "prediction": student_prediction,
            "losses": {
                "supervised": torch.nn.functional.huber_loss(student_prediction, target),
                "soft_target": torch.nn.functional.huber_loss(student_prediction, teacher_prediction),
                "feature_rectification": torch.nn.functional.mse_loss(student_representation, rectified),
            },
            "teacher_frozen": all(not parameter.requires_grad for parameter in self.teacher.parameters()),
        }


class SMILRegressionMechanism(torch.nn.Module):
    """Regression adaptation retaining incomplete sampling, reconstruction and alignment."""

    def __init__(self, *, width: int, hidden: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(width * 2, hidden), torch.nn.ReLU())
        self.reconstructor = torch.nn.Linear(hidden, width)
        self.head = torch.nn.Linear(hidden, 1)
        self.missing_prior = torch.nn.Parameter(torch.zeros(width))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        if values.shape != mask.shape:
            raise ValueError("smil_values_mask_shape_mismatch")
        completed = mask * values + (1 - mask) * self.missing_prior
        representation = self.encoder(torch.cat([completed, mask], dim=1))
        prediction = self.head(representation).squeeze(1)
        reconstruction = self.reconstructor(representation)
        complete = mask.all(1)
        incomplete = ~complete
        complete_center = representation[complete].mean(0) if complete.any() else representation.detach().mean(0)
        incomplete_center = representation[incomplete].mean(0) if incomplete.any() else representation.detach().mean(0)
        return {
            "prediction": prediction,
            "losses": {
                "supervised": torch.nn.functional.huber_loss(prediction, target),
                "reconstruction": (((reconstruction - values) ** 2) * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1),
                "complete_incomplete_alignment": torch.nn.functional.mse_loss(incomplete_center, complete_center.detach()),
            },
        }


class GMDRegressionMechanism:
    """Regression adaptation retaining combination sampling and conflict projection/merge."""

    @staticmethod
    def sample_missing_combinations(*, modalities: int, count: int, seed: int) -> torch.Tensor:
        if modalities < 1 or count < 1:
            raise ValueError("gmd_invalid_sampling_contract")
        rng = np.random.default_rng(seed)
        masks = rng.integers(0, 2, size=(count, modalities), endpoint=False)
        empty = masks.sum(1) == 0
        masks[empty, rng.integers(0, modalities, size=int(empty.sum()))] = 1
        return torch.tensor(masks, dtype=torch.float32)

    @staticmethod
    def merge(gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(gradients) < 2:
            raise ValueError("gmd_requires_multiple_modality_objectives")
        return gmd_project_conflicting(gradients)
