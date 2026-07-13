from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
import torch

from .training_contract import NeuralTrainingContract


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, target: torch.Tensor) -> "TargetStandardizer":
        if target.ndim != 1:
            raise ValueError("TARGET_MUST_BE_ONE_DIMENSIONAL")
        if target.numel() < 2:
            raise ValueError("TARGET_REQUIRES_AT_LEAST_TWO_ROWS")
        mean = float(target.mean())
        scale = float(target.std(unbiased=False))
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise ValueError("NONFINITE_TARGET_STATISTICS")
        if scale <= 1e-12:
            raise ValueError("CONSTANT_TARGET_NOT_SUPPORTED")
        return cls(mean=mean, scale=scale)

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.scale

    def inverse_transform(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.scale + self.mean


@dataclass
class RepairTrainingResult:
    prediction: np.ndarray
    validation_target: np.ndarray
    history: list[dict[str, float | int]]
    selected_epoch: int
    optimizer_steps: int
    parameter_delta_l2: float
    target_mean: float
    target_scale: float
    fit_time_seconds: float
    inference_time_seconds: float
    best_state_dict: dict[str, torch.Tensor]
    audit: dict[str, Any]


def _validate_modalities(
    values: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> None:
    if not values:
        raise ValueError(f"{name.upper()}_MODALITIES_EMPTY")

    lengths = {tensor.shape[0] for tensor in values.values()}
    lengths.add(mask.shape[0])
    lengths.add(target.shape[0])

    if len(lengths) != 1:
        raise ValueError(f"{name.upper()}_ROW_COUNT_MISMATCH")

    if target.ndim != 1:
        raise ValueError(f"{name.upper()}_TARGET_MUST_BE_ONE_DIMENSIONAL")

    if mask.ndim != 2 or mask.shape[1] != len(values):
        raise ValueError(f"{name.upper()}_MASK_SHAPE_INVALID")

    for modality, tensor in values.items():
        if tensor.ndim != 2:
            raise ValueError(
                f"{name.upper()}_MODALITY_NOT_MATRIX:{modality}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name.upper()}_NONFINITE_MODALITY:{modality}"
            )

    if not torch.isfinite(mask).all():
        raise ValueError(f"{name.upper()}_NONFINITE_MASK")
    if not torch.isfinite(target).all():
        raise ValueError(f"{name.upper()}_NONFINITE_TARGET")


def _state_vector(model: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.detach().reshape(-1).cpu()
        for parameter in model.parameters()
    ]
    if not values:
        raise ValueError("MODEL_HAS_NO_PARAMETERS")
    return torch.cat(values)


def _batch_modalities(
    values: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.index_select(0, indices)
        for name, tensor in values.items()
    }


def fit_repair_regressor(
    *,
    model: torch.nn.Module,
    train_values: Mapping[str, torch.Tensor],
    train_mask: torch.Tensor,
    train_target: torch.Tensor,
    validation_values: Mapping[str, torch.Tensor],
    validation_mask: torch.Tensor,
    validation_target: torch.Tensor,
    contract: NeuralTrainingContract,
    seed: int,
) -> RepairTrainingResult:
    contract.validate()

    _validate_modalities(
        train_values,
        train_mask,
        train_target,
        name="train",
    )
    _validate_modalities(
        validation_values,
        validation_mask,
        validation_target,
        name="validation",
    )

    if tuple(train_values) != tuple(validation_values):
        raise ValueError("TRAIN_VALIDATION_MODALITY_ORDER_MISMATCH")

    torch.manual_seed(seed)

    scaler = TargetStandardizer.fit(train_target)
    train_target_scaled = scaler.transform(train_target)

    initial_vector = _state_vector(model).clone()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    optimizer_steps = 0
    history: list[dict[str, float | int]] = []

    row_count = int(train_target.shape[0])
    effective_batch_size = min(contract.batch_size, row_count)

    fit_started = time.perf_counter()

    for epoch in range(contract.max_epochs):
        model.train()

        order = torch.randperm(row_count, generator=generator)
        epoch_loss_sum = 0.0
        epoch_rows = 0

        for start in range(0, row_count, effective_batch_size):
            indices = order[start : start + effective_batch_size]
            batch_values = _batch_modalities(train_values, indices)
            batch_mask = train_mask.index_select(0, indices)
            batch_target = train_target_scaled.index_select(0, indices)

            optimizer.zero_grad(set_to_none=True)
            prediction_scaled, _ = model(batch_values, batch_mask)
            loss = torch.nn.functional.huber_loss(
                prediction_scaled,
                batch_target,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                contract.gradient_clip_norm,
            )

            if not torch.isfinite(gradient_norm):
                raise RuntimeError("NONFINITE_GRADIENT_NORM")

            optimizer.step()
            optimizer_steps += 1

            batch_rows = int(indices.shape[0])
            epoch_loss_sum += float(loss.detach()) * batch_rows
            epoch_rows += batch_rows

        model.eval()
        with torch.no_grad():
            validation_scaled, _ = model(
                validation_values,
                validation_mask,
            )
            validation_prediction = scaler.inverse_transform(
                validation_scaled
            )
            validation_mae = float(
                torch.mean(
                    torch.abs(
                        validation_prediction - validation_target
                    )
                )
            )

        if not math.isfinite(validation_mae):
            raise RuntimeError("NONFINITE_VALIDATION_SCORE")

        improved = validation_mae < best_score - 1e-12
        if improved:
            best_score = validation_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "train_huber_scaled": epoch_loss_sum / epoch_rows,
                "validation_mae_original_scale": validation_mae,
                "best_validation_mae_original_scale": best_score,
                "improved": int(improved),
            }
        )

        minimum_training_complete = (
            epoch + 1 >= contract.min_epochs
            and optimizer_steps >= contract.minimum_optimizer_steps
        )
        if (
            minimum_training_complete
            and stale_epochs >= contract.early_stopping_patience
        ):
            break

    fit_time = time.perf_counter() - fit_started

    if best_state is None or best_epoch < 0:
        raise RuntimeError("BEST_CHECKPOINT_NOT_SELECTED")

    model.load_state_dict(best_state)

    final_vector = _state_vector(model)
    parameter_delta = float(torch.linalg.vector_norm(
        final_vector - initial_vector
    ))

    if parameter_delta < contract.minimum_parameter_delta:
        raise RuntimeError("PARAMETER_DELTA_BELOW_MINIMUM")

    inference_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        validation_scaled, _ = model(
            validation_values,
            validation_mask,
        )
        validation_prediction = scaler.inverse_transform(
            validation_scaled
        )
    inference_time = time.perf_counter() - inference_started

    prediction = validation_prediction.detach().cpu().numpy()
    truth = validation_target.detach().cpu().numpy()

    if not np.isfinite(prediction).all():
        raise RuntimeError("NONFINITE_PREDICTION")
    if float(np.std(prediction)) <= 1e-12:
        raise RuntimeError("DEGENERATE_CONSTANT_PREDICTION")

    audit = {
        "trainer": "stage8_v4_repair_minibatch_v1",
        "target_scaling": "train_standard",
        "target_scaler_fit_scope": "TRAIN_ONLY",
        "prediction_inverse_transformed": True,
        "batch_size_requested": contract.batch_size,
        "batch_size_effective": effective_batch_size,
        "training_rows": row_count,
        "validation_rows": int(validation_target.shape[0]),
        "optimizer_steps": optimizer_steps,
        "selected_epoch": best_epoch,
        "best_checkpoint_restored": True,
        "parameter_delta_l2": parameter_delta,
        "shuffle_each_epoch": True,
        "validation_selection_scope": "INNER_VALIDATION_ONLY",
    }

    return RepairTrainingResult(
        prediction=prediction,
        validation_target=truth,
        history=history,
        selected_epoch=best_epoch,
        optimizer_steps=optimizer_steps,
        parameter_delta_l2=parameter_delta,
        target_mean=scaler.mean,
        target_scale=scaler.scale,
        fit_time_seconds=fit_time,
        inference_time_seconds=inference_time,
        best_state_dict=deepcopy(best_state),
        audit=audit,
    )
