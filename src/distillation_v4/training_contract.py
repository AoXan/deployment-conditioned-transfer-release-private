from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NeuralTrainingContract:
    batch_size: int
    max_epochs: int
    min_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    target_scaling: str
    shuffle_each_epoch: bool
    restore_best_checkpoint: bool
    minimum_optimizer_steps: int
    minimum_parameter_delta: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "NeuralTrainingContract":
        required = {
            "batch_size",
            "max_epochs",
            "min_epochs",
            "early_stopping_patience",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "target_scaling",
            "shuffle_each_epoch",
            "restore_best_checkpoint",
            "minimum_optimizer_steps",
            "minimum_parameter_delta",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"TRAINING_CONTRACT_FIELDS_MISSING:{','.join(missing)}")

        contract = cls(
            batch_size=int(value["batch_size"]),
            max_epochs=int(value["max_epochs"]),
            min_epochs=int(value["min_epochs"]),
            early_stopping_patience=int(value["early_stopping_patience"]),
            learning_rate=float(value["learning_rate"]),
            weight_decay=float(value["weight_decay"]),
            gradient_clip_norm=float(value["gradient_clip_norm"]),
            target_scaling=str(value["target_scaling"]),
            shuffle_each_epoch=bool(value["shuffle_each_epoch"]),
            restore_best_checkpoint=bool(value["restore_best_checkpoint"]),
            minimum_optimizer_steps=int(value["minimum_optimizer_steps"]),
            minimum_parameter_delta=float(value["minimum_parameter_delta"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.batch_size < 2:
            raise ValueError("MINIBATCH_REQUIRED")
        if self.max_epochs < 2:
            raise ValueError("MAX_EPOCHS_TOO_SMALL")
        if self.min_epochs < 1 or self.min_epochs >= self.max_epochs:
            raise ValueError("INVALID_MIN_EPOCHS")
        if self.early_stopping_patience < 1:
            raise ValueError("INVALID_EARLY_STOPPING_PATIENCE")
        if self.learning_rate <= 0:
            raise ValueError("INVALID_LEARNING_RATE")
        if self.weight_decay < 0:
            raise ValueError("INVALID_WEIGHT_DECAY")
        if self.gradient_clip_norm <= 0:
            raise ValueError("INVALID_GRADIENT_CLIP_NORM")
        if self.target_scaling != "train_standard":
            raise ValueError("TRAIN_ONLY_TARGET_STANDARDISATION_REQUIRED")
        if not self.shuffle_each_epoch:
            raise ValueError("TRAIN_SHUFFLE_REQUIRED")
        if not self.restore_best_checkpoint:
            raise ValueError("BEST_CHECKPOINT_RESTORE_REQUIRED")
        if self.minimum_optimizer_steps < 10:
            raise ValueError("MINIMUM_OPTIMIZER_STEPS_TOO_SMALL")
        if self.minimum_parameter_delta <= 0:
            raise ValueError("INVALID_MINIMUM_PARAMETER_DELTA")
