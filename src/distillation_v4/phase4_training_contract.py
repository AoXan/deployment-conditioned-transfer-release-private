from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase4LossWeights:
    supervised: float
    prediction: float
    representation: float

    def normalized(self) -> "Phase4LossWeights":
        values = (
            float(self.supervised),
            float(self.prediction),
            float(self.representation),
        )

        if any(value < 0.0 for value in values):
            raise ValueError(
                "PHASE4_LOSS_WEIGHT_NEGATIVE"
            )

        total = sum(values)
        if total <= 0.0:
            raise ValueError(
                "PHASE4_LOSS_WEIGHT_TOTAL_INVALID"
            )

        return Phase4LossWeights(
            supervised=values[0] / total,
            prediction=values[1] / total,
            representation=values[2] / total,
        )

    def to_mapping(self) -> dict[str, float]:
        return {
            "supervised": self.supervised,
            "prediction": self.prediction,
            "representation": self.representation,
        }


@dataclass(frozen=True)
class Phase4TrainingResult:
    route: str
    prediction: Any
    best_state_dict: dict[str, Any]
    target_mean: float
    target_scale: float
    optimizer_steps: int
    parameter_delta_l2: float
    selected_epoch: int
    fit_time_seconds: float
    inference_time_seconds: float
    loss_ledger: list[dict[str, float]]
    representation: Any
