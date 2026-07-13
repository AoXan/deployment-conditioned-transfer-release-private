from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


@dataclass(frozen=True)
class TrainingReference:
    mean: np.ndarray
    std: np.ndarray
    fit_scope: str = "train_only"

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "fit_scope": self.fit_scope,
        }


@dataclass(frozen=True)
class ExplanationObjectiveConfig:
    variant: str
    lambda_g: float = 0.0
    lambda_c: float = 0.0
    lambda_f: float = 0.0
    lambda_p: float = 0.0
    lambda_s: float = 0.0
    perturbation_scale: float = 0.02
    view: str = "gaussian"
    sparsity: str = "entropy"
    fidelity_temperature: float = 1.0
    prediction_temperature: float = 1.0
    eps: float = 1e-8

    def enabled_lambdas(self) -> dict[str, float]:
        if self.variant == "prediction_only":
            return {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0}
        if self.variant == "gradient_regularisation":
            return {"lambda_g": self.lambda_g, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0}
        if self.variant == "explanation_consistency":
            return {"lambda_g": 0.0, "lambda_c": self.lambda_c, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0}
        if self.variant == "sparsity_only":
            return {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": self.lambda_s}
        if self.variant == "consistency_sparsity":
            return {"lambda_g": 0.0, "lambda_c": self.lambda_c, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": self.lambda_s}
        if self.variant == "fidelity_weighted_consistency":
            return {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": self.lambda_f or self.lambda_c, "lambda_p": 0.0, "lambda_s": 0.0}
        if self.variant == "prediction_aware_selective_consistency":
            return {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": 0.0, "lambda_p": self.lambda_p or self.lambda_c, "lambda_s": 0.0}
        if self.variant == "deployment_contract_aware_consistency":
            return {"lambda_g": 0.0, "lambda_c": self.lambda_c, "lambda_f": 0.0, "lambda_p": 0.0, "lambda_s": 0.0}
        if self.variant == "fidelity_selective_consistency":
            return {
                "lambda_g": 0.0,
                "lambda_c": 0.0,
                "lambda_f": self.lambda_f or self.lambda_c,
                "lambda_p": self.lambda_p or self.lambda_c,
                "lambda_s": 0.0,
            }
        if self.variant == "deployment_fidelity_consistency":
            return {"lambda_g": 0.0, "lambda_c": 0.0, "lambda_f": self.lambda_f or self.lambda_c, "lambda_p": 0.0, "lambda_s": 0.0}
        raise ValueError(f"UNKNOWN_EXPLANATION_OBJECTIVE_VARIANT:{self.variant}")


@dataclass
class TaylorAttributionResult:
    prediction: torch.Tensor
    representation: torch.Tensor | None
    gradient: torch.Tensor
    attribution: torch.Tensor
    availability: torch.Tensor


@dataclass
class ObjectiveTerms:
    total: torch.Tensor
    parts: dict[str, float]
    loss_tensors: dict[str, torch.Tensor]


def build_training_reference(arrays: dict[str, Any]) -> TrainingReference:
    train_weather = np.asarray(arrays["train"]["weather"], dtype=np.float32)
    mean = np.nanmean(train_weather, axis=0).astype(np.float32)
    std = np.nanstd(train_weather, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return TrainingReference(mean=mean, std=std)


def _as_feature_vector(value: torch.Tensor | np.ndarray | list[float], *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 1:
        raise ValueError("FEATURE_VECTOR_MUST_BE_1D")
    return tensor


def _default_availability(weather: torch.Tensor, deployment_mask: torch.Tensor | None) -> torch.Tensor:
    finite = torch.isfinite(weather)
    if deployment_mask is None:
        return finite
    supplied = torch.as_tensor(deployment_mask, dtype=torch.bool, device=weather.device)
    if supplied.shape != weather.shape:
        raise ValueError("DEPLOYMENT_MASK_SHAPE_MISMATCH")
    return finite & supplied


def _forward_model(
    model: nn.Module,
    weather: torch.Tensor,
    feature_names: list[str],
    availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        output = model(weather)
    except Exception:
        from src.distillation_v4.universal_weather_v2.tokens import build_weather_tokens_v2

        token_batch = build_weather_tokens_v2(
            weather,
            feature_names,
            availability=availability.to(weather.dtype),
            device=weather.device,
        )
        output = model(token_batch)
    if isinstance(output, tuple):
        prediction = output[0].reshape(-1)
        representation = output[1] if len(output) > 1 else None
    else:
        prediction = output.reshape(-1)
        representation = None
    return prediction, representation


def taylor_attribution(
    *,
    model: nn.Module,
    weather: torch.Tensor,
    feature_names: list[str],
    reference: torch.Tensor | np.ndarray | list[float],
    deployment_mask: torch.Tensor | None = None,
    create_graph: bool = True,
) -> TaylorAttributionResult:
    if weather.ndim != 2:
        raise ValueError("WEATHER_BATCH_MUST_BE_2D")
    reference_tensor = _as_feature_vector(reference, device=weather.device)
    if reference_tensor.shape[0] != weather.shape[1]:
        raise ValueError("REFERENCE_FEATURE_COUNT_MISMATCH")
    availability = _default_availability(weather, deployment_mask)
    clean = torch.where(
        availability,
        weather,
        reference_tensor.unsqueeze(0).expand_as(weather),
    ).detach().clone().requires_grad_(True)
    prediction, representation = _forward_model(model, clean, feature_names, availability)
    gradient = torch.autograd.grad(
        prediction.sum(),
        clean,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    attribution = (clean - reference_tensor.unsqueeze(0)) * gradient
    attribution = torch.where(availability, attribution, torch.zeros_like(attribution))
    gradient = torch.where(availability, gradient, torch.zeros_like(gradient))
    return TaylorAttributionResult(
        prediction=prediction,
        representation=representation,
        gradient=gradient,
        attribution=attribution,
        availability=availability,
    )


def cosine_consistency_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    availability: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = availability.to(first.dtype)
    first_masked = first * mask
    second_masked = second * mask
    first_norm = first_masked.norm(dim=1)
    second_norm = second_masked.norm(dim=1)
    both_zero = (first_norm <= eps) & (second_norm <= eps)
    one_zero = ((first_norm <= eps) ^ (second_norm <= eps)).to(first.dtype)
    denom = (first_norm * second_norm).clamp_min(eps)
    cosine = (first_masked * second_masked).sum(dim=1) / denom
    distance = 1.0 - cosine.clamp(-1.0, 1.0)
    distance = torch.where(both_zero, torch.zeros_like(distance), distance)
    distance = torch.where(one_zero > 0, torch.ones_like(distance), distance)
    return distance.mean()


def cosine_distance_per_sample(
    first: torch.Tensor,
    second: torch.Tensor,
    availability: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = availability.to(first.dtype)
    first_masked = first * mask
    second_masked = second * mask
    first_norm = first_masked.norm(dim=1)
    second_norm = second_masked.norm(dim=1)
    both_zero = (first_norm <= eps) & (second_norm <= eps)
    one_zero = ((first_norm <= eps) ^ (second_norm <= eps)).to(first.dtype)
    denom = (first_norm * second_norm).clamp_min(eps)
    cosine = (first_masked * second_masked).sum(dim=1) / denom
    distance = 1.0 - cosine.clamp(-1.0, 1.0)
    distance = torch.where(both_zero, torch.zeros_like(distance), distance)
    distance = torch.where(one_zero > 0, torch.ones_like(distance), distance)
    return distance


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights)).clamp_min(0.0)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def attribution_entropy_loss(attribution: torch.Tensor, availability: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    masked = attribution.abs() * availability.to(attribution.dtype)
    active = availability.sum(dim=1).clamp_min(1).to(attribution.dtype)
    total = masked.sum(dim=1, keepdim=True).clamp_min(eps)
    probability = masked / total
    entropy = -(probability * (probability + eps).log()).sum(dim=1)
    normalizer = active.log().clamp_min(1.0)
    return (entropy / normalizer).mean()


def make_gaussian_view(
    weather: torch.Tensor,
    *,
    availability: torch.Tensor,
    train_std: torch.Tensor | np.ndarray | list[float],
    scale: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    std = _as_feature_vector(train_std, device=weather.device).clamp_min(1e-6)
    noise = torch.randn(weather.shape, generator=generator, device=weather.device, dtype=weather.dtype)
    perturbed = weather + noise * std.unsqueeze(0) * float(scale)
    return torch.where(availability, perturbed, weather)


def make_dropout_view(
    weather: torch.Tensor,
    *,
    availability: torch.Tensor,
    drop_probability: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    random = torch.rand(weather.shape, generator=generator, device=weather.device, dtype=weather.dtype)
    keep = (random >= float(drop_probability)) | (~availability)
    return torch.where(keep, weather, torch.full_like(weather, float("nan")))


def make_modality_respecting_view(
    weather: torch.Tensor,
    *,
    availability: torch.Tensor,
    train_std: torch.Tensor | np.ndarray | list[float],
    feature_names: list[str],
    scale: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    std = _as_feature_vector(train_std, device=weather.device).clamp_min(1e-6)
    weather_mask = torch.tensor(
        [str(name).lower().startswith("weather") or str(name).lower().startswith("climate") for name in feature_names],
        dtype=torch.bool,
        device=weather.device,
    ).unsqueeze(0).expand_as(weather)
    active = availability & weather_mask
    noise = torch.randn(weather.shape, generator=generator, device=weather.device, dtype=weather.dtype)
    perturbed = weather + noise * std.unsqueeze(0) * float(scale)
    return torch.where(active, perturbed, weather)


def _construct_view(
    weather: torch.Tensor,
    *,
    availability: torch.Tensor,
    train_std: torch.Tensor | np.ndarray | list[float],
    feature_names: list[str],
    config: ExplanationObjectiveConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if config.view == "gaussian":
        return make_gaussian_view(
            weather,
            availability=availability,
            train_std=train_std,
            scale=config.perturbation_scale,
            generator=generator,
        )
    if config.view == "dropout":
        return make_dropout_view(
            weather,
            availability=availability,
            drop_probability=config.perturbation_scale,
            generator=generator,
        )
    if config.view == "modality_respecting":
        return make_modality_respecting_view(
            weather,
            availability=availability,
            train_std=train_std,
            feature_names=feature_names,
            scale=config.perturbation_scale,
            generator=generator,
        )
    raise ValueError(f"UNKNOWN_EXPLANATION_VIEW:{config.view}")


def _local_taylor_residual(
    *,
    original: TaylorAttributionResult,
    viewed: TaylorAttributionResult,
    weather: torch.Tensor,
    view: torch.Tensor,
    common: torch.Tensor,
) -> torch.Tensor:
    delta = torch.where(common, view - weather, torch.zeros_like(weather))
    approx = original.prediction + (delta * original.gradient).sum(dim=1)
    return (viewed.prediction - approx).abs()


def gradient_conflict_metrics(model: nn.Module, loss_tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    parameters = [param for param in model.parameters() if param.requires_grad]

    def flat_grad(loss: torch.Tensor | None) -> torch.Tensor | None:
        if loss is None:
            return None
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        chunks = [torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1) for param, grad in zip(parameters, grads)]
        if not chunks:
            return None
        return torch.cat(chunks)

    pred = flat_grad(loss_tensors.get("prediction"))
    consistency = flat_grad(loss_tensors.get("consistency"))
    sparsity = flat_grad(loss_tensors.get("sparsity"))

    def cosine(first: torch.Tensor | None, second: torch.Tensor | None) -> float:
        if first is None or second is None:
            return float("nan")
        denom = float(first.norm().detach().cpu() * second.norm().detach().cpu())
        if denom <= 1e-12:
            return float("nan")
        value = torch.dot(first, second) / first.norm().clamp_min(1e-12) / second.norm().clamp_min(1e-12)
        return float(value.detach().cpu())

    def norm(value: torch.Tensor | None) -> float:
        return float("nan") if value is None else float(value.norm().detach().cpu())

    return {
        "pred_consistency_grad_cosine": cosine(pred, consistency),
        "pred_sparsity_grad_cosine": cosine(pred, sparsity),
        "prediction_grad_norm": norm(pred),
        "consistency_grad_norm": norm(consistency),
        "sparsity_grad_norm": norm(sparsity),
    }


def compute_objective_terms(
    *,
    model: nn.Module,
    weather: torch.Tensor,
    target: torch.Tensor,
    feature_names: list[str],
    reference: torch.Tensor | np.ndarray | list[float],
    train_std: torch.Tensor | np.ndarray | list[float],
    deployment_mask: torch.Tensor | None,
    config: ExplanationObjectiveConfig,
    generator: torch.Generator | None,
) -> ObjectiveTerms:
    lambdas = config.enabled_lambdas()
    original = taylor_attribution(
        model=model,
        weather=weather,
        feature_names=feature_names,
        reference=reference,
        deployment_mask=deployment_mask,
        create_graph=True,
    )
    prediction_loss = nn.functional.mse_loss(original.prediction, target)
    grad_loss = (original.gradient.pow(2) * original.availability.to(original.gradient.dtype)).sum(dim=1).mean()
    view = _construct_view(
        weather,
        availability=original.availability,
        train_std=train_std,
        feature_names=feature_names,
        config=config,
        generator=generator,
    )
    viewed = taylor_attribution(
        model=model,
        weather=view,
        feature_names=feature_names,
        reference=reference,
        deployment_mask=original.availability,
        create_graph=True,
    )
    common = original.availability & viewed.availability
    per_sample_distance = cosine_distance_per_sample(original.attribution, viewed.attribution, common, eps=config.eps)
    consistency_loss = per_sample_distance.mean()
    residual = _local_taylor_residual(original=original, viewed=viewed, weather=weather, view=view, common=common)
    fidelity_weight = torch.exp(-float(config.fidelity_temperature) * residual.detach()).clamp_min(config.eps)
    prediction_delta = (original.prediction - viewed.prediction).abs().detach()
    prediction_weight = torch.exp(-float(config.prediction_temperature) * prediction_delta).clamp_min(config.eps)
    fidelity_consistency_loss = weighted_mean(per_sample_distance, fidelity_weight, eps=config.eps)
    selective_consistency_loss = weighted_mean(per_sample_distance, prediction_weight, eps=config.eps)
    sparse_loss = attribution_entropy_loss(original.attribution, original.availability, eps=config.eps)
    total = (
        prediction_loss
        + float(lambdas["lambda_g"]) * grad_loss
        + float(lambdas["lambda_c"]) * consistency_loss
        + float(lambdas["lambda_f"]) * fidelity_consistency_loss
        + float(lambdas["lambda_p"]) * selective_consistency_loss
        + float(lambdas["lambda_s"]) * sparse_loss
    )
    parts = {
        "prediction": float(prediction_loss.detach().cpu()),
        "gradient": float(grad_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
        "fidelity_consistency": float(fidelity_consistency_loss.detach().cpu()),
        "selective_consistency": float(selective_consistency_loss.detach().cpu()),
        "sparsity": float(sparse_loss.detach().cpu()),
        "fidelity_weight_mean": float(fidelity_weight.mean().detach().cpu()),
        "prediction_weight_mean": float(prediction_weight.mean().detach().cpu()),
        "active_features_mean": float(common.sum(dim=1).float().mean().detach().cpu()),
        "local_taylor_residual": float(residual.mean().detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return ObjectiveTerms(
        total=total,
        parts=parts,
        loss_tensors={
            "prediction": prediction_loss,
            "gradient": grad_loss,
            "consistency": consistency_loss,
            "fidelity_consistency": fidelity_consistency_loss,
            "selective_consistency": selective_consistency_loss,
            "sparsity": sparse_loss,
        },
    )


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or second.size < 2:
        return float("nan")
    if np.nanstd(first) <= 1e-12 or np.nanstd(second) <= 1e-12:
        return float("nan")
    value = spearmanr(first, second, nan_policy="omit").correlation
    return float(value) if np.isfinite(value) else float("nan")


def _topk_overlap(first: np.ndarray, second: np.ndarray, k: int) -> float:
    if first.size == 0 or second.size == 0:
        return float("nan")
    k = int(min(k, first.size, second.size))
    if k <= 0:
        return float("nan")
    a = set(np.argsort(-np.abs(first))[:k].tolist())
    b = set(np.argsort(-np.abs(second))[:k].tolist())
    return float(len(a & b) / k)


def attribution_metrics(
    *,
    model: nn.Module,
    arrays: dict[str, Any],
    split: str,
    feature_names: list[str],
    reference: TrainingReference,
    config: ExplanationObjectiveConfig,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    weather_array = np.asarray(arrays[split]["weather"], dtype=np.float32)
    if weather_array.size == 0:
        return {
            "cosine_similarity": float("nan"),
            "spearman": float("nan"),
            "top5_overlap": float("nan"),
            "attribution_norm": float("nan"),
            "local_taylor_fidelity_r2": float("nan"),
            "samples": 0,
        }
    model.eval()
    reference_tensor = torch.as_tensor(reference.mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(reference.std, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 991)
    cosines: list[float] = []
    spearmans: list[float] = []
    overlaps: list[float] = []
    norms: list[float] = []
    fidelity_true: list[float] = []
    fidelity_pred: list[float] = []
    sample_count = 0
    for batch_index, start in enumerate(range(0, weather_array.shape[0], batch_size)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = torch.as_tensor(weather_array[start : start + batch_size], dtype=torch.float32, device=device)
        original = taylor_attribution(
            model=model,
            weather=batch,
            feature_names=feature_names,
            reference=reference_tensor,
            deployment_mask=None,
            create_graph=False,
        )
        view = _construct_view(
            batch,
            availability=original.availability,
            train_std=std_tensor,
            feature_names=feature_names,
            config=config,
            generator=generator,
        )
        viewed = taylor_attribution(
            model=model,
            weather=view,
            feature_names=feature_names,
            reference=reference_tensor,
            deployment_mask=original.availability,
            create_graph=False,
        )
        common = original.availability & viewed.availability
        original_np = original.attribution.detach().cpu().numpy()
        viewed_np = viewed.attribution.detach().cpu().numpy()
        common_np = common.detach().cpu().numpy()
        for row_a, row_b, row_mask in zip(original_np, viewed_np, common_np):
            row_a = row_a[row_mask]
            row_b = row_b[row_mask]
            norm_product = np.linalg.norm(row_a) * np.linalg.norm(row_b)
            if norm_product <= 1e-12:
                cosines.append(1.0 if np.linalg.norm(row_a) <= 1e-12 and np.linalg.norm(row_b) <= 1e-12 else 0.0)
            else:
                cosines.append(float(np.dot(row_a, row_b) / norm_product))
            spearmans.append(_safe_spearman(row_a, row_b))
            overlaps.append(_topk_overlap(row_a, row_b, k=5))
            norms.append(float(np.linalg.norm(row_a, ord=1)))
        reference_batch = reference_tensor.unsqueeze(0).expand_as(batch)
        ref_result = taylor_attribution(
            model=model,
            weather=reference_batch,
            feature_names=feature_names,
            reference=reference_tensor,
            deployment_mask=original.availability,
            create_graph=False,
        )
        approx = ref_result.prediction.detach().cpu().numpy() + original_np.sum(axis=1)
        fidelity_true.extend(original.prediction.detach().cpu().numpy().tolist())
        fidelity_pred.extend(approx.tolist())
        sample_count += int(batch.shape[0])
    fidelity_r2 = float("nan")
    if len(fidelity_true) >= 2 and np.var(fidelity_true) > 1e-12:
        residual = np.asarray(fidelity_true) - np.asarray(fidelity_pred)
        fidelity_r2 = float(1.0 - np.sum(residual**2) / np.sum((np.asarray(fidelity_true) - np.mean(fidelity_true)) ** 2))
    return {
        "cosine_similarity": float(np.nanmean(cosines)),
        "spearman": float(np.nanmean(spearmans)),
        "top5_overlap": float(np.nanmean(overlaps)),
        "attribution_norm": float(np.nanmean(norms)),
        "local_taylor_fidelity_r2": fidelity_r2,
        "samples": sample_count,
    }


def train_student_with_explanation_objective(
    *,
    training_module: Any,
    arrays: dict[str, Any],
    feature_names: list[str],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    initial_state_dict: dict[str, torch.Tensor] | None,
    objective_config: ExplanationObjectiveConfig,
    evaluate_test: bool,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    training_module.set_seed(seed)
    student = training_module.build_student_v2().to(device)
    if initial_state_dict is not None:
        student.load_state_dict(initial_state_dict, strict=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = training_module.make_loader(arrays["train"], batch_size=batch_size, shuffle=True)
    reference = build_training_reference(arrays)
    reference_tensor = torch.as_tensor(reference.mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(reference.std, dtype=torch.float32, device=device)
    best_validation = math.inf
    best_state = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        student.train()
        epoch_parts: list[dict[str, float]] = []
        epoch_conflicts: list[dict[str, float]] = []
        generator = torch.Generator(device=device).manual_seed(int(seed) * 1009 + epoch)
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            weather = batch["weather"].to(device)
            target = batch["target"].to(device)
            terms = compute_objective_terms(
                model=student,
                weather=weather,
                target=target,
                feature_names=feature_names,
                reference=reference_tensor,
                train_std=std_tensor,
                deployment_mask=None,
                config=objective_config,
                generator=generator,
            )
            if batch_index == 0:
                epoch_conflicts.append(gradient_conflict_metrics(student, terms.loss_tensors))
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            optimizer.step()
            epoch_parts.append(terms.parts)
        validation_metrics, _ = training_module.evaluate_student(
            model=student,
            arrays=arrays["validation"],
            feature_names=feature_names,
            device=device,
            batch_size=batch_size,
        )
        validation_loss = validation_metrics["rmse"] ** 2
        part_keys = sorted({key for item in epoch_parts for key in item})
        mean_parts = {key: float(np.mean([item[key] for item in epoch_parts if key in item])) if epoch_parts else float("nan") for key in part_keys}
        conflict_keys = sorted({key for item in epoch_conflicts for key in item})
        mean_conflicts = {
            key: float(np.nanmean([item[key] for item in epoch_conflicts if key in item])) if epoch_conflicts else float("nan")
            for key in conflict_keys
        }
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss_parts": mean_parts,
                "gradient_conflict": mean_conflicts,
                "validation_metrics": validation_metrics,
                "wall_clock_seconds": time.perf_counter() - epoch_started,
            }
        )
        if validation_loss < best_validation - 1e-8:
            best_validation = float(validation_loss)
            best_state = copy.deepcopy(student.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("EXPLANATION_CONSISTENCY_NO_BEST_STATE")
    student.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = training_module.save_checkpoint(
        path=checkpoint_path,
        model=student,
        metadata={
            **metadata,
            "model_role": "UNIVERSAL_WEATHER_STUDENT_V2_EXPLANATION_CONSISTENCY",
            "objective": asdict(objective_config),
            "best_validation_loss": best_validation,
            "history": history,
            "training_reference": reference.to_json(),
            "target_test_used_for_selection": False,
            "outer_test_used_for_selection": False,
        },
        optimizer=None,
    )
    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.is_file() else None
    validation_attribution = attribution_metrics(
        model=student,
        arrays=arrays,
        split="validation",
        feature_names=feature_names,
        reference=reference,
        config=objective_config,
        batch_size=batch_size,
        device=device,
        max_batches=4,
        seed=seed,
    )
    result: dict[str, Any] = {
        "model": student,
        "manifest": manifest,
        "history": history,
        "best_validation_loss": best_validation,
        "validation_metrics": history[-1]["validation_metrics"],
        "validation_attribution": validation_attribution,
        "training_reference": reference.to_json(),
        "objective": asdict(objective_config),
        "cost": {
            "training_wall_clock_seconds": time.perf_counter() - training_started,
            "mean_epoch_wall_clock_seconds": float(np.mean([item.get("wall_clock_seconds", float("nan")) for item in history])),
            "parameter_count": int(sum(param.numel() for param in student.parameters())),
            "checkpoint_size_bytes": checkpoint_size_bytes,
        },
    }
    if evaluate_test:
        inference_started = time.perf_counter()
        test_metrics, test_prediction = training_module.evaluate_student(
            model=student,
            arrays=arrays["test"],
            feature_names=feature_names,
            device=device,
            batch_size=batch_size,
        )
        inference_seconds = time.perf_counter() - inference_started
        test_attribution = attribution_metrics(
            model=student,
            arrays=arrays,
            split="test",
            feature_names=feature_names,
            reference=reference,
            config=objective_config,
            batch_size=batch_size,
            device=device,
            max_batches=4,
            seed=seed + 17,
        )
        result.update(
            {
                "test_metrics": test_metrics,
                "test_prediction": test_prediction,
                "test_attribution": test_attribution,
            }
        )
        result["cost"]["test_inference_wall_clock_seconds"] = inference_seconds
        result["cost"]["test_inference_seconds_per_sample"] = inference_seconds / max(1, int(np.asarray(arrays["test"]["target"]).shape[0]))
    return result


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
