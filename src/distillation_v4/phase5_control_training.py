from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import torch

from .phase4_student import DeployableStudent


@dataclass(frozen=True)
class ExactStepTrainingResult:
    prediction: np.ndarray
    representation: np.ndarray
    state_dict: dict[str, torch.Tensor]
    target_mean: float
    target_scale: float
    optimizer_steps: int
    parameter_delta_l2: float
    fit_time_seconds: float
    inference_time_seconds: float
    loss_ledger: list[dict[str, float]]


def _parameter_vector(
    model: torch.nn.Module,
) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().cpu().reshape(-1)
            for parameter in model.parameters()
        ]
    )


def fit_exact_step_supervised_control(
    *,
    train_values: torch.Tensor,
    train_target: torch.Tensor,
    validation_values: torch.Tensor,
    target_optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float | None,
    seed: int,
) -> ExactStepTrainingResult:
    if target_optimizer_steps <= 0:
        raise ValueError(
            "PHASE5_TARGET_OPTIMIZER_STEPS_INVALID"
        )

    if len(train_values) != len(train_target):
        raise ValueError(
            "PHASE5_TRAIN_TARGET_MISALIGNED"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeployableStudent(
        int(train_values.shape[1]),
        hidden=32,
        representation_dim=32,
    )

    initial = _parameter_vector(model)

    target_mean_tensor = train_target.mean()
    target_scale_tensor = train_target.std(
        unbiased=False
    )

    if float(target_scale_tensor) < 1e-8:
        target_scale_tensor = torch.tensor(
            1.0,
            dtype=train_target.dtype,
        )

    scaled_target = (
        train_target - target_mean_tensor
    ) / target_scale_tensor

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    generator = torch.Generator().manual_seed(seed)
    ledger: list[dict[str, float]] = []
    optimizer_steps = 0
    epoch = 0

    fit_started = time.perf_counter()

    while optimizer_steps < target_optimizer_steps:
        order = torch.randperm(
            len(train_values),
            generator=generator,
        )

        epoch_losses: list[float] = []

        for start in range(
            0,
            len(order),
            batch_size,
        ):
            if optimizer_steps >= target_optimizer_steps:
                break

            indices = order[
                start:start + batch_size
            ]

            optimizer.zero_grad()
            prediction, _ = model(
                train_values[indices]
            )

            loss = torch.nn.functional.huber_loss(
                prediction,
                scaled_target[indices],
            )

            loss.backward()

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            optimizer.step()

            optimizer_steps += 1
            epoch_losses.append(
                float(loss.detach())
            )

        ledger.append(
            {
                "epoch": float(epoch),
                "optimizer_steps_total": float(
                    optimizer_steps
                ),
                "supervised": float(
                    np.mean(epoch_losses)
                ),
                "prediction_imitation": 0.0,
                "representation_alignment": 0.0,
                "total": float(
                    np.mean(epoch_losses)
                ),
            }
        )
        epoch += 1

    fit_time = time.perf_counter() - fit_started

    model.eval()
    inference_started = time.perf_counter()

    with torch.no_grad():
        prediction_scaled, representation = model(
            validation_values
        )
        prediction = (
            prediction_scaled
            * float(target_scale_tensor)
            + float(target_mean_tensor)
        ).cpu().numpy()

    inference_time = (
        time.perf_counter() - inference_started
    )

    final = _parameter_vector(model)
    delta = float(
        torch.linalg.vector_norm(final - initial)
    )

    if optimizer_steps != target_optimizer_steps:
        raise RuntimeError(
            "PHASE5_COMPUTE_CONTROL_STEP_MISMATCH"
        )

    if delta <= 0.0:
        raise RuntimeError(
            "PHASE5_COMPUTE_CONTROL_PARAMETER_DELTA_ZERO"
        )

    return ExactStepTrainingResult(
        prediction=prediction,
        representation=(
            representation.cpu().numpy()
        ),
        state_dict=deepcopy(model.state_dict()),
        target_mean=float(target_mean_tensor),
        target_scale=float(target_scale_tensor),
        optimizer_steps=optimizer_steps,
        parameter_delta_l2=delta,
        fit_time_seconds=fit_time,
        inference_time_seconds=inference_time,
        loss_ledger=ledger,
    )
