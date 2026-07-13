from __future__ import annotations

from typing import Iterable

import torch


class DistillationObjective(torch.nn.Module):
    """Multi-objective loss with transparent and automatic balancing strategies."""

    def __init__(self, names: Iterable[str], *, balancing: str):
        super().__init__()
        self.names = tuple(names)
        self.balancing = balancing
        if balancing not in {"equal", "loss_scale_normalisation", "log_variance", "gradnorm"}:
            raise ValueError(f"unknown_loss_balancing:{balancing}")
        self.log_variances = torch.nn.Parameter(torch.zeros(len(self.names))) if balancing == "log_variance" else None
        self.task_logits = torch.nn.Parameter(torch.zeros(len(self.names))) if balancing == "gradnorm" else None
        self.register_buffer("initial_losses", torch.full((len(self.names),), float("nan")))

    def forward(self, losses: dict[str, torch.Tensor], *, shared_parameters: list[torch.nn.Parameter]):
        if tuple(losses) != self.names:
            raise ValueError("loss_names_or_order_mismatch")
        vector = torch.stack([losses[name] for name in self.names])
        if not torch.isfinite(vector).all():
            raise FloatingPointError("non_finite_distillation_loss")
        with torch.no_grad():
            missing = torch.isnan(self.initial_losses)
            self.initial_losses[missing] = vector.detach()[missing].clamp_min(torch.finfo(vector.dtype).eps)

        gradient_norms = []
        for loss in vector:
            gradients = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
            flat = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
            gradient_norms.append(torch.cat(flat).norm() if flat else torch.zeros((), device=loss.device))
        gradient_norms_tensor = torch.stack(gradient_norms)
        relative_rates = (vector.detach() / self.initial_losses.clamp_min(torch.finfo(vector.dtype).eps)).detach()

        if self.balancing == "equal":
            weights = torch.ones_like(vector)
            total = vector.sum()
        elif self.balancing == "loss_scale_normalisation":
            weights = self.initial_losses.reciprocal()
            weights = weights / weights.mean()
            total = (weights * vector).sum()
        elif self.balancing == "log_variance":
            weights = torch.exp(-self.log_variances)
            total = (weights * vector + self.log_variances).sum()
        else:
            learned = torch.softmax(self.task_logits, dim=0) * len(self.names)
            target = gradient_norms_tensor.detach().mean() * (relative_rates / relative_rates.mean().clamp_min(1e-12))
            balancing_penalty = torch.nn.functional.l1_loss(learned * gradient_norms_tensor, target)
            weights = learned
            total = (weights * vector).sum() + balancing_penalty

        ledger = {
            "balancing": self.balancing,
            "raw_losses": {name: float(losses[name].detach()) for name in self.names},
            "effective_weights": {name: float(weights[index].detach()) for index, name in enumerate(self.names)},
            "gradient_norms": {name: float(gradient_norms_tensor[index].detach()) for index, name in enumerate(self.names)},
            "relative_training_rates": {name: float(relative_rates[index]) for index, name in enumerate(self.names)},
            "total_loss": float(total.detach()),
        }
        return total, ledger
