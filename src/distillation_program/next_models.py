from __future__ import annotations

from typing import Mapping

import torch


class SharedDenseRegressor(torch.nn.Module):
    def __init__(self, modality_dims: Mapping[str, int], *, hidden: int):
        super().__init__()
        self.modalities = tuple(modality_dims)
        width = sum(modality_dims.values()) + len(modality_dims)
        self.backbone = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, values: Mapping[str, torch.Tensor], mask: torch.Tensor):
        joined = torch.cat([values[name] for name in self.modalities] + [mask], dim=1)
        representation = self.backbone(joined)
        return self.head(representation).squeeze(1), {"representation": representation}


class ModalitySpecificRegressor(torch.nn.Module):
    def __init__(self, modality_dims: Mapping[str, int], *, hidden: int):
        super().__init__()
        self.modalities = tuple(modality_dims)
        self.encoders = torch.nn.ModuleDict({
            name: torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
            for name, width in modality_dims.items()
        })
        self.missing_tokens = torch.nn.ParameterDict({name: torch.nn.Parameter(torch.zeros(hidden)) for name in modality_dims})
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden * len(self.modalities), hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def encode(self, values: Mapping[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        encoded = []
        for index, name in enumerate(self.modalities):
            present = mask[:, index:index + 1]
            representation = self.encoders[name](values[name])
            token = self.missing_tokens[name].expand_as(representation)
            encoded.append(present * representation + (1 - present) * token)
        return torch.cat(encoded, dim=1)

    def forward(self, values: Mapping[str, torch.Tensor], mask: torch.Tensor):
        representation = self.encode(values, mask)
        return self.head(representation).squeeze(1), {"representation": representation}


class RoutedModalityRegressor(ModalitySpecificRegressor):
    def __init__(self, modality_dims: Mapping[str, int], *, hidden: int, experts: int):
        super().__init__(modality_dims, hidden=hidden)
        representation_width = hidden * len(self.modalities)
        self.router = torch.nn.Linear(representation_width + len(self.modalities), experts)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(representation_width, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
            for _ in range(experts)
        ])

    def forward(self, values: Mapping[str, torch.Tensor], mask: torch.Tensor):
        representation = self.encode(values, mask)
        routing = torch.softmax(self.router(torch.cat([representation, mask], dim=1)), dim=1)
        expert_predictions = torch.cat([expert(representation) for expert in self.experts], dim=1)
        prediction = (routing * expert_predictions).sum(1)
        load = routing.mean(0)
        load_balancing = ((load - load.mean()) ** 2).mean()
        return prediction, {
            "representation": representation,
            "routing_weights": routing,
            "expert_predictions": expert_predictions,
            "load_balancing_loss": load_balancing,
        }


def count_capacity(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "active_parameters": int(trainable), "trainable_parameters": int(trainable)}
