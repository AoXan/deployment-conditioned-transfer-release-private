from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any

import numpy as np
import torch

from .phase4_training_contract import (
    Phase4LossWeights,
    Phase4TrainingResult,
)
from .training_contract import NeuralTrainingContract


class DeployableStudent(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 32,
        representation_dim: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden,
                representation_dim,
            ),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(
            representation_dim,
            1,
        )

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(values)
        prediction = self.head(
            representation
        ).squeeze(1)
        return prediction, representation


def _parameter_vector(
    model: torch.nn.Module,
) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().cpu().reshape(-1)
            for parameter in model.parameters()
        ]
    )


def _validate_teacher_tensors(
    *,
    route: str,
    train_rows: int,
    validation_rows: int,
    teacher_prediction_train: torch.Tensor | None,
    teacher_prediction_validation: torch.Tensor | None,
    teacher_representation_train: torch.Tensor | None,
    teacher_representation_validation: torch.Tensor | None,
) -> None:
    if route in {
        "prediction_kd",
        "combined_kd",
    }:
        if (
            teacher_prediction_train is None
            or teacher_prediction_validation is None
        ):
            raise ValueError(
                "PHASE4_PREDICTION_TEACHER_REQUIRED"
            )
        if len(teacher_prediction_train) != train_rows:
            raise ValueError(
                "PHASE4_TRAIN_TEACHER_PREDICTION_MISALIGNED"
            )
        if (
            len(teacher_prediction_validation)
            != validation_rows
        ):
            raise ValueError(
                "PHASE4_VALIDATION_TEACHER_PREDICTION_MISALIGNED"
            )

    if route in {
        "representation_kd",
        "combined_kd",
    }:
        if (
            teacher_representation_train is None
            or teacher_representation_validation is None
        ):
            raise ValueError(
                "PHASE4_REPRESENTATION_TEACHER_REQUIRED"
            )
        if (
            len(teacher_representation_train)
            != train_rows
        ):
            raise ValueError(
                "PHASE4_TRAIN_TEACHER_REPRESENTATION_MISALIGNED"
            )
        if (
            len(teacher_representation_validation)
            != validation_rows
        ):
            raise ValueError(
                "PHASE4_VALIDATION_TEACHER_REPRESENTATION_MISALIGNED"
            )


def fit_phase4_student(
    *,
    route: str,
    train_values: torch.Tensor,
    train_target: torch.Tensor,
    validation_values: torch.Tensor,
    validation_target: torch.Tensor,
    contract: NeuralTrainingContract,
    seed: int,
    loss_weights: Phase4LossWeights,
    teacher_prediction_train: torch.Tensor | None = None,
    teacher_prediction_validation: torch.Tensor | None = None,
    teacher_representation_train: torch.Tensor | None = None,
    teacher_representation_validation: torch.Tensor | None = None,
    pretrained_encoder_state: dict[str, Any] | None = None,
) -> Phase4TrainingResult:
    allowed = {
        "supervised",
        "fine_tune",
        "prediction_kd",
        "representation_kd",
        "combined_kd",
    }
    if route not in allowed:
        raise ValueError(
            f"PHASE4_STUDENT_ROUTE_UNKNOWN:{route}"
        )

    if train_values.ndim != 2:
        raise ValueError(
            "PHASE4_TRAIN_VALUES_NOT_MATRIX"
        )
    if validation_values.ndim != 2:
        raise ValueError(
            "PHASE4_VALIDATION_VALUES_NOT_MATRIX"
        )
    if len(train_values) != len(train_target):
        raise ValueError(
            "PHASE4_TRAIN_TARGET_MISALIGNED"
        )
    if len(validation_values) != len(
        validation_target
    ):
        raise ValueError(
            "PHASE4_VALIDATION_TARGET_MISALIGNED"
        )

    _validate_teacher_tensors(
        route=route,
        train_rows=len(train_values),
        validation_rows=len(validation_values),
        teacher_prediction_train=(
            teacher_prediction_train
        ),
        teacher_prediction_validation=(
            teacher_prediction_validation
        ),
        teacher_representation_train=(
            teacher_representation_train
        ),
        teacher_representation_validation=(
            teacher_representation_validation
        ),
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    representation_dim = 32
    if teacher_representation_train is not None:
        representation_dim = int(
            teacher_representation_train.shape[1]
        )

    model = DeployableStudent(
        int(train_values.shape[1]),
        hidden=32,
        representation_dim=representation_dim,
    )

    if pretrained_encoder_state is not None:
        model.encoder.load_state_dict(
            pretrained_encoder_state,
            strict=True,
        )

    initial_parameters = _parameter_vector(model)

    target_mean_tensor = train_target.mean()
    target_scale_tensor = train_target.std(
        unbiased=False
    )
    if float(target_scale_tensor) < 1e-8:
        target_scale_tensor = torch.tensor(
            1.0,
            dtype=train_target.dtype,
        )

    target_mean = float(target_mean_tensor)
    target_scale = float(target_scale_tensor)

    scaled_train_target = (
        train_target - target_mean_tensor
    ) / target_scale_tensor
    scaled_validation_target = (
        validation_target - target_mean_tensor
    ) / target_scale_tensor

    normalized = loss_weights.normalized()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    best_state = deepcopy(model.state_dict())
    best_score = math.inf
    best_epoch = -1
    stale = 0
    optimizer_steps = 0
    ledger: list[dict[str, float]] = []

    started = time.perf_counter()

    for epoch in range(contract.max_epochs):
        model.train()

        if contract.shuffle_each_epoch:
            order = torch.randperm(
                len(train_values),
                generator=generator,
            )
        else:
            order = torch.arange(len(train_values))

        epoch_supervised = 0.0
        epoch_prediction = 0.0
        epoch_representation = 0.0
        epoch_total = 0.0
        batches = 0

        for start in range(
            0,
            len(order),
            contract.batch_size,
        ):
            indices = order[
                start:start + contract.batch_size
            ]
            optimizer.zero_grad()

            prediction_scaled, representation = model(
                train_values[indices]
            )
            prediction_original = (
                prediction_scaled * target_scale
                + target_mean
            )

            supervised = torch.nn.functional.huber_loss(
                prediction_scaled,
                scaled_train_target[indices],
            )

            prediction_loss = torch.tensor(
                0.0,
                dtype=supervised.dtype,
            )
            if route in {
                "prediction_kd",
                "combined_kd",
            }:
                prediction_loss = (
                    torch.nn.functional.mse_loss(
                        prediction_original,
                        teacher_prediction_train[indices],
                    )
                )

            representation_loss = torch.tensor(
                0.0,
                dtype=supervised.dtype,
            )
            if route in {
                "representation_kd",
                "combined_kd",
            }:
                representation_loss = (
                    torch.nn.functional.mse_loss(
                        representation,
                        teacher_representation_train[
                            indices
                        ],
                    )
                )

            total = (
                normalized.supervised * supervised
                + normalized.prediction
                * prediction_loss
                + normalized.representation
                * representation_loss
            )

            total.backward()

            if contract.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    contract.gradient_clip_norm,
                )

            optimizer.step()
            optimizer_steps += 1
            batches += 1

            epoch_supervised += float(
                supervised.detach()
            )
            epoch_prediction += float(
                prediction_loss.detach()
            )
            epoch_representation += float(
                representation_loss.detach()
            )
            epoch_total += float(total.detach())

        model.eval()
        with torch.no_grad():
            validation_scaled, _ = model(
                validation_values
            )
            validation_mae = float(
                torch.mean(
                    torch.abs(
                        validation_scaled
                        - scaled_validation_target
                    )
                )
            )

        ledger.append(
            {
                "epoch": float(epoch),
                "supervised": (
                    epoch_supervised / batches
                ),
                "prediction_imitation": (
                    epoch_prediction / batches
                ),
                "representation_alignment": (
                    epoch_representation / batches
                ),
                "total": epoch_total / batches,
                "validation_scaled_mae": (
                    validation_mae
                ),
            }
        )

        if validation_mae < best_score:
            best_score = validation_mae
            best_state = deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if (
            epoch + 1 >= contract.min_epochs
            and stale
            >= contract.early_stopping_patience
        ):
            break

    fit_time = time.perf_counter() - started

    model.load_state_dict(best_state)
    model.eval()

    inference_started = time.perf_counter()
    with torch.no_grad():
        validation_scaled, validation_representation = (
            model(validation_values)
        )
        prediction = (
            validation_scaled * target_scale
            + target_mean
        ).cpu().numpy()
    inference_time = (
        time.perf_counter() - inference_started
    )

    final_parameters = _parameter_vector(model)
    parameter_delta = float(
        torch.linalg.vector_norm(
            final_parameters - initial_parameters
        )
    )

    if optimizer_steps < (
        contract.minimum_optimizer_steps
    ):
        raise RuntimeError(
            "PHASE4_MINIMUM_OPTIMIZER_STEPS_NOT_MET"
        )

    if parameter_delta < (
        contract.minimum_parameter_delta
    ):
        raise RuntimeError(
            "PHASE4_PARAMETER_DELTA_TOO_SMALL"
        )

    return Phase4TrainingResult(
        route=route,
        prediction=prediction,
        best_state_dict=best_state,
        target_mean=target_mean,
        target_scale=target_scale,
        optimizer_steps=optimizer_steps,
        parameter_delta_l2=parameter_delta,
        selected_epoch=best_epoch,
        fit_time_seconds=fit_time,
        inference_time_seconds=inference_time,
        loss_ledger=ledger,
        representation=(
            validation_representation.cpu().numpy()
        ),
    )
