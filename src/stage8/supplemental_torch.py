"""Real PyTorch modality encoders and privileged teacher/student objectives.

The module is import-safe without PyTorch; construction fails explicitly rather
than substituting a mock model.
"""
from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("BLOCKED_PENDING_ENVIRONMENT_APPROVAL") from exc
    return torch, nn


def build_modality_model(modality_dims: dict[str, int], hidden: int = 32):
    torch, nn = require_torch()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)) for name, dim in modality_dims.items()})
            self.missing_tokens = nn.ParameterDict({name: nn.Parameter(torch.zeros(hidden)) for name in modality_dims})
            self.gate = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, modalities, masks):
            encoded, logits = [], []
            for name, encoder in self.encoders.items():
                present = masks[name].float().view(-1, 1)
                z = encoder(modalities[name]) * present + self.missing_tokens[name].view(1, -1) * (1 - present)
                encoded.append(z)
                logits.append(self.gate(torch.cat([z, present], dim=1)))
            weights = torch.softmax(torch.cat(logits, dim=1), dim=1)
            shared = sum(z * weights[:, i:i+1] for i, z in enumerate(encoded))
            return self.head(shared).squeeze(1), shared, weights
    return Model()


def privileged_student_loss(*, student_prediction, target, student_latent, teacher_prediction, teacher_latent, reconstructed, reconstruction_target, reconstruction_mask, weights):
    torch, _ = require_torch()
    supervised = torch.nn.functional.huber_loss(student_prediction, target)
    prediction_kd = torch.nn.functional.huber_loss(student_prediction, teacher_prediction.detach())
    latent_alignment = 1 - torch.nn.functional.cosine_similarity(student_latent, teacher_latent.detach(), dim=1).mean()
    mask = reconstruction_mask.float()
    reconstruction = (((reconstructed - reconstruction_target) ** 2) * mask).sum() / mask.sum().clamp_min(1)
    total = weights["supervised"] * supervised + weights["prediction_kd"] * prediction_kd + weights["latent"] * latent_alignment + weights["reconstruction"] * reconstruction
    return total, {"supervised_huber": supervised, "prediction_distillation": prediction_kd, "latent_alignment": latent_alignment, "masked_reconstruction": reconstruction}
