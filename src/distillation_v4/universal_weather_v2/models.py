from __future__ import annotations

import torch
from torch import nn

from .tokens import WeatherTokenBatchV2


class UniversalWeatherEncoderV2(nn.Module):
    def __init__(
        self,
        *,
        feature_vocabulary_size: int = 8192,
        maximum_time_positions: int = 1024,
        embedding_dim: int = 64,
        feedforward_dim: int = 192,
        representation_dim: int = 128,
        attention_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim % attention_heads:
            raise ValueError(
                "EMBEDDING_DIM_NOT_DIVISIBLE_BY_HEADS"
            )

        self.feature_embedding = nn.Embedding(
            feature_vocabulary_size,
            embedding_dim,
        )

        self.time_embedding = nn.Embedding(
            maximum_time_positions,
            embedding_dim,
        )

        self.value_embedding = nn.Sequential(
            nn.Linear(2, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dim,
                embedding_dim,
            ),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
                enable_nested_tensor=False,
)

        self.output = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(
                embedding_dim,
                representation_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        batch: WeatherTokenBatchV2,
    ) -> torch.Tensor:
        numeric = torch.stack(
            [
                batch.values,
                batch.availability,
            ],
            dim=-1,
        )

        tokens = (
            self.value_embedding(numeric)
            + self.feature_embedding(
                batch.feature_ids
            )
            + self.time_embedding(
                batch.time_ids
            )
        )

        encoded = self.transformer(
            tokens,
            src_key_padding_mask=(
                batch.padding_mask
            ),
        )

        usable = (
            (~batch.padding_mask)
            & (batch.availability > 0)
        ).to(encoded.dtype).unsqueeze(-1)

        denominator = usable.sum(
            dim=1
        ).clamp_min(1.0)

        pooled = (
            encoded * usable
        ).sum(dim=1) / denominator

        return self.output(pooled)


class UniversalWeatherRegressorV2(nn.Module):
    def __init__(
        self,
        *,
        encoder: UniversalWeatherEncoderV2,
        representation_dim: int = 128,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.head = nn.Sequential(
            nn.Linear(
                representation_dim,
                head_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                head_hidden_dim,
                1,
            ),
        )

    def encode(
        self,
        batch: WeatherTokenBatchV2,
    ) -> torch.Tensor:
        return self.encoder(batch)

    def forward(
        self,
        batch: WeatherTokenBatchV2,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        representation = self.encode(
            batch
        )

        prediction = self.head(
            representation
        ).squeeze(-1)

        return prediction, representation


class PrivilegedWeatherTeacherV2(nn.Module):
    def __init__(
        self,
        *,
        weather_encoder: UniversalWeatherEncoderV2,
        privileged_width: int,
        representation_dim: int = 128,
        privileged_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.weather_encoder = weather_encoder

        self.privileged_encoder = nn.Sequential(
            nn.Linear(
                privileged_width,
                privileged_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                privileged_hidden_dim,
                representation_dim,
            ),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                representation_dim * 2,
                representation_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(
            representation_dim,
            1,
        )

    def forward(
        self,
        weather: WeatherTokenBatchV2,
        privileged: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        weather_representation = (
            self.weather_encoder(weather)
        )

        privileged_representation = (
            self.privileged_encoder(
                privileged
            )
        )

        fused = self.fusion(
            torch.cat(
                [
                    weather_representation,
                    privileged_representation,
                ],
                dim=1,
            )
        )

        return (
            self.head(fused).squeeze(-1),
            fused,
        )
