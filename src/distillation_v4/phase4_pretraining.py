from __future__ import annotations

from copy import deepcopy

import torch

from .training_contract import NeuralTrainingContract


class MultimodalPretrainer(torch.nn.Module):
    def __init__(
        self,
        weather_dim: int,
        soil_dim: int,
        *,
        hidden: int = 32,
        representation_dim: int = 32,
    ) -> None:
        super().__init__()
        self.weather_encoder = torch.nn.Sequential(
            torch.nn.Linear(weather_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden,
                representation_dim,
            ),
            torch.nn.ReLU(),
        )
        self.soil_encoder = torch.nn.Sequential(
            torch.nn.Linear(soil_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden,
                representation_dim,
            ),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(
            representation_dim * 2,
            1,
        )

    def forward(
        self,
        weather: torch.Tensor,
        soil: torch.Tensor,
    ) -> torch.Tensor:
        weather_representation = (
            self.weather_encoder(weather)
        )
        soil_representation = (
            self.soil_encoder(soil)
        )
        return self.head(
            torch.cat(
                (
                    weather_representation,
                    soil_representation,
                ),
                dim=1,
            )
        ).squeeze(1)


def pretrain_weather_encoder(
    *,
    train_weather: torch.Tensor,
    train_soil: torch.Tensor,
    train_target: torch.Tensor,
    validation_weather: torch.Tensor,
    validation_soil: torch.Tensor,
    validation_target: torch.Tensor,
    contract: NeuralTrainingContract,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    model = MultimodalPretrainer(
        int(train_weather.shape[1]),
        int(train_soil.shape[1]),
    )

    target_mean = train_target.mean()
    target_scale = train_target.std(
        unbiased=False
    )
    if float(target_scale) < 1e-8:
        target_scale = torch.tensor(
            1.0,
            dtype=train_target.dtype,
        )

    train_scaled = (
        train_target - target_mean
    ) / target_scale
    validation_scaled = (
        validation_target - target_mean
    ) / target_scale

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(contract.max_epochs):
        model.train()

        order = torch.randperm(
            len(train_weather),
            generator=generator,
        )

        for start in range(
            0,
            len(order),
            contract.batch_size,
        ):
            indices = order[
                start:start + contract.batch_size
            ]
            optimizer.zero_grad()

            prediction = model(
                train_weather[indices],
                train_soil[indices],
            )
            loss = torch.nn.functional.huber_loss(
                prediction,
                train_scaled[indices],
            )
            loss.backward()

            if contract.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    contract.gradient_clip_norm,
                )

            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                validation_weather,
                validation_soil,
            )
            score = float(
                torch.mean(
                    torch.abs(
                        validation_prediction
                        - validation_scaled
                    )
                )
            )

        if score < best_score:
            best_score = score
            best_state = deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if (
            epoch + 1 >= contract.min_epochs
            and stale
            >= contract.early_stopping_patience
        ):
            break

    model.load_state_dict(best_state)

    # Only deployable weather encoder is transferred.
    return deepcopy(
        model.weather_encoder.state_dict()
    )
