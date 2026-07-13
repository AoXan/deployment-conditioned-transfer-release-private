from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from sklearn.cross_decomposition import (
    PLSRegression,
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .training_contract import (
    NeuralTrainingContract,
)


SOIL_REPRESENTATIONS = {
    "raw",
    "pca",
    "pls",
}


class SoilRepresentationTransformer:
    def __init__(
        self,
        *,
        method: str,
        max_components: int = 8,
    ) -> None:
        if method not in SOIL_REPRESENTATIONS:
            raise ValueError(
                "SOIL_REPRESENTATION_INVALID:"
                + method
            )

        if max_components < 1:
            raise ValueError(
                "SOIL_COMPONENT_CAP_INVALID"
            )

        self.method = method
        self.max_components = int(
            max_components
        )

        self.imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
        )
        self.scaler = StandardScaler()
        self.reducer: (
            PCA
            | PLSRegression
            | None
        ) = None
        self.output_dim_: int | None = None

    def fit(
        self,
        values: np.ndarray,
        target: np.ndarray,
    ) -> "SoilRepresentationTransformer":
        values = np.asarray(
            values,
            dtype=float,
        )
        target = np.asarray(
            target,
            dtype=float,
        )

        if values.ndim != 2:
            raise ValueError(
                "SOIL_VALUES_NOT_MATRIX"
            )

        if len(values) != len(target):
            raise ValueError(
                "SOIL_TARGET_MISALIGNED"
            )

        transformed = (
            self.imputer.fit_transform(values)
        )
        transformed = (
            self.scaler.fit_transform(
                transformed
            )
        )

        component_count = min(
            self.max_components,
            int(transformed.shape[1]),
            max(1, len(transformed) - 1),
        )

        if self.method == "pca":
            self.reducer = PCA(
                n_components=component_count,
                svd_solver="full",
            )
            transformed = (
                self.reducer.fit_transform(
                    transformed
                )
            )

        elif self.method == "pls":
            self.reducer = PLSRegression(
                n_components=component_count,
                scale=False,
                max_iter=500,
                tol=1e-6,
            )
            self.reducer.fit(
                transformed,
                target,
            )
            transformed = (
                self.reducer.transform(
                    transformed
                )
            )

        self.output_dim_ = int(
            transformed.shape[1]
        )
        return self

    def transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        if self.output_dim_ is None:
            raise RuntimeError(
                "SOIL_TRANSFORMER_NOT_FITTED"
            )

        transformed = self.imputer.transform(
            np.asarray(
                values,
                dtype=float,
            )
        )
        transformed = self.scaler.transform(
            transformed
        )

        if self.reducer is not None:
            transformed = (
                self.reducer.transform(
                    transformed
                )
            )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )

    def fit_transform(
        self,
        values: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        self.fit(values, target)
        return self.transform(values)


def soil_availability_mask(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 2:
        raise ValueError(
            "SOIL_AVAILABILITY_NOT_MATRIX"
        )

    return np.isfinite(values).any(
        axis=1,
    ).astype(np.float32)


class FlexibleSoilRegressor(
    torch.nn.Module
):
    def __init__(
        self,
        *,
        weather_dim: int,
        soil_dim: int,
        hidden: int = 32,
        representation_dim: int = 32,
    ) -> None:
        super().__init__()

        self.weather_encoder = (
            torch.nn.Sequential(
                torch.nn.Linear(
                    weather_dim,
                    hidden,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    hidden,
                    representation_dim,
                ),
                torch.nn.ReLU(),
            )
        )

        self.soil_encoder = (
            torch.nn.Sequential(
                torch.nn.Linear(
                    soil_dim,
                    hidden,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    hidden,
                    representation_dim,
                ),
                torch.nn.ReLU(),
            )
        )

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(
                representation_dim * 2 + 1,
                hidden,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        weather: torch.Tensor,
        soil: torch.Tensor,
        soil_available: torch.Tensor,
        *,
        explicit_mask: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if soil_available.ndim == 1:
            soil_available = (
                soil_available.unsqueeze(1)
            )

        weather_representation = (
            self.weather_encoder(weather)
        )
        soil_representation = (
            self.soil_encoder(soil)
        )
        soil_representation = (
            soil_representation
            * soil_available
        )

        mask_feature = (
            soil_available
            if explicit_mask
            else torch.ones_like(
                soil_available
            )
        )

        representation = torch.cat(
            (
                weather_representation,
                soil_representation,
                mask_feature,
            ),
            dim=1,
        )

        prediction = self.fusion(
            representation
        ).squeeze(1)

        return (
            prediction,
            representation,
        )


@dataclass(frozen=True)
class SoilTrainingResult:
    prediction: np.ndarray
    representation: np.ndarray
    best_state_dict: dict[
        str,
        torch.Tensor,
    ]
    target_mean: float
    target_scale: float
    optimizer_steps: int
    parameter_delta_l2: float
    selected_epoch: int
    fit_time_seconds: float
    inference_time_seconds: float
    loss_ledger: list[
        dict[str, float]
    ]
    parameter_count: int
    observed_soil_fraction_train: float
    observed_soil_fraction_validation: float


def parameter_count(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def _parameter_vector(
    model: torch.nn.Module,
) -> torch.Tensor:
    return torch.cat(
        [
            parameter
            .detach()
            .cpu()
            .reshape(-1)
            for parameter
            in model.parameters()
        ]
    )


def effective_soil_mask(
    observed_mask: torch.Tensor,
    *,
    dropout_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not (
        0.0
        <= dropout_probability
        < 1.0
    ):
        raise ValueError(
            "SOIL_MODALITY_DROPOUT_INVALID"
        )

    if dropout_probability == 0.0:
        return observed_mask

    keep = (
        torch.rand(
            observed_mask.shape,
            generator=generator,
        )
        >= dropout_probability
    ).to(observed_mask.dtype)

    return observed_mask * keep


def fit_flexible_soil_model(
    *,
    route: str,
    train_weather: torch.Tensor,
    train_soil: torch.Tensor,
    train_soil_available: torch.Tensor,
    train_target: torch.Tensor,
    validation_weather: torch.Tensor,
    validation_soil: torch.Tensor,
    validation_soil_available: torch.Tensor,
    validation_target: torch.Tensor,
    contract: NeuralTrainingContract,
    seed: int,
    hidden: int = 32,
    representation_dim: int = 32,
    soil_dropout_probability: float,
    explicit_availability_mask: bool,
) -> SoilTrainingResult:
    if route not in {
        "soil_direct",
        "missing_aware",
        "soil_control",
    }:
        raise ValueError(
            "SOIL_ROUTE_INVALID:"
            + route
        )

    if (
        route == "soil_direct"
        and soil_dropout_probability
        != 0.0
    ):
        raise ValueError(
            "SOIL_DIRECT_DROPOUT_FORBIDDEN"
        )

    train_rows = len(train_weather)
    validation_rows = len(
        validation_weather
    )

    alignments = (
        (
            "train_soil",
            train_soil,
            train_rows,
        ),
        (
            "train_soil_available",
            train_soil_available,
            train_rows,
        ),
        (
            "train_target",
            train_target,
            train_rows,
        ),
        (
            "validation_soil",
            validation_soil,
            validation_rows,
        ),
        (
            "validation_soil_available",
            validation_soil_available,
            validation_rows,
        ),
        (
            "validation_target",
            validation_target,
            validation_rows,
        ),
    )

    for name, value, expected in alignments:
        if len(value) != expected:
            raise ValueError(
                "SOIL_ROW_ALIGNMENT_INVALID:"
                + name
            )

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlexibleSoilRegressor(
        weather_dim=int(
            train_weather.shape[1]
        ),
        soil_dim=int(
            train_soil.shape[1]
        ),
        hidden=hidden,
        representation_dim=(
            representation_dim
        ),
    )

    initial_parameters = (
        _parameter_vector(model)
    )

    target_mean_tensor = (
        train_target.mean()
    )
    target_scale_tensor = (
        train_target.std(
            unbiased=False
        )
    )

    if float(target_scale_tensor) < 1e-8:
        target_scale_tensor = (
            torch.tensor(
                1.0,
                dtype=train_target.dtype,
            )
        )

    scaled_train_target = (
        train_target
        - target_mean_tensor
    ) / target_scale_tensor

    scaled_validation_target = (
        validation_target
        - target_mean_tensor
    ) / target_scale_tensor

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=(
            contract.weight_decay
        ),
    )

    order_generator = (
        torch.Generator()
        .manual_seed(seed)
    )
    dropout_generator = (
        torch.Generator()
        .manual_seed(seed + 104729)
    )

    best_state = deepcopy(
        model.state_dict()
    )
    best_validation_mae = math.inf
    best_epoch = -1
    stale_epochs = 0
    optimizer_steps = 0
    loss_ledger: list[
        dict[str, float]
    ] = []

    fit_started = time.perf_counter()

    for epoch in range(
        contract.max_epochs
    ):
        model.train()

        order = torch.randperm(
            train_rows,
            generator=order_generator,
        )

        epoch_losses: list[float] = []
        epoch_effective_masks: list[
            float
        ] = []

        for start in range(
            0,
            train_rows,
            contract.batch_size,
        ):
            indices = order[
                start:
                start + contract.batch_size
            ]

            optimizer.zero_grad()

            batch_mask = (
                effective_soil_mask(
                    train_soil_available[
                        indices
                    ],
                    dropout_probability=(
                        soil_dropout_probability
                    ),
                    generator=(
                        dropout_generator
                    ),
                )
            )

            prediction_scaled, _ = model(
                train_weather[indices],
                train_soil[indices],
                batch_mask,
                explicit_mask=(
                    explicit_availability_mask
                ),
            )

            loss = (
                torch.nn.functional
                .huber_loss(
                    prediction_scaled,
                    scaled_train_target[
                        indices
                    ],
                )
            )

            loss.backward()

            if (
                contract
                .gradient_clip_norm
                is not None
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    contract
                    .gradient_clip_norm,
                )

            optimizer.step()
            optimizer_steps += 1

            epoch_losses.append(
                float(loss.detach())
            )
            epoch_effective_masks.append(
                float(batch_mask.mean())
            )

        model.eval()

        with torch.no_grad():
            (
                validation_prediction_scaled,
                _,
            ) = model(
                validation_weather,
                validation_soil,
                validation_soil_available,
                explicit_mask=(
                    explicit_availability_mask
                ),
            )

            validation_mae = float(
                torch.mean(
                    torch.abs(
                        validation_prediction_scaled
                        - scaled_validation_target
                    )
                )
            )

        loss_ledger.append(
            {
                "epoch": float(epoch),
                "supervised": float(
                    np.mean(epoch_losses)
                ),
                "total": float(
                    np.mean(epoch_losses)
                ),
                "validation_scaled_mae": (
                    validation_mae
                ),
                "effective_soil_fraction": (
                    float(
                        np.mean(
                            epoch_effective_masks
                        )
                    )
                ),
                "soil_dropout_probability": (
                    float(
                        soil_dropout_probability
                    )
                ),
            }
        )

        if (
            validation_mae
            < best_validation_mae
        ):
            best_validation_mae = (
                validation_mae
            )
            best_state = deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (
            epoch + 1
            >= contract.min_epochs
            and stale_epochs
            >= contract
            .early_stopping_patience
        ):
            break

    fit_time_seconds = (
        time.perf_counter()
        - fit_started
    )

    model.load_state_dict(
        best_state
    )
    model.eval()

    inference_started = (
        time.perf_counter()
    )

    with torch.no_grad():
        (
            validation_prediction_scaled,
            validation_representation,
        ) = model(
            validation_weather,
            validation_soil,
            validation_soil_available,
            explicit_mask=(
                explicit_availability_mask
            ),
        )

        prediction = (
            validation_prediction_scaled
            * float(target_scale_tensor)
            + float(target_mean_tensor)
        ).cpu().numpy()

    inference_time_seconds = (
        time.perf_counter()
        - inference_started
    )

    final_parameters = (
        _parameter_vector(model)
    )
    parameter_delta_l2 = float(
        torch.linalg.vector_norm(
            final_parameters
            - initial_parameters
        )
    )

    if (
        optimizer_steps
        < contract
        .minimum_optimizer_steps
    ):
        raise RuntimeError(
            "SOIL_MINIMUM_OPTIMIZER_STEPS_NOT_MET"
        )

    if (
        parameter_delta_l2
        < contract
        .minimum_parameter_delta
    ):
        raise RuntimeError(
            "SOIL_PARAMETER_DELTA_TOO_SMALL"
        )

    return SoilTrainingResult(
        prediction=prediction,
        representation=(
            validation_representation
            .cpu()
            .numpy()
        ),
        best_state_dict=best_state,
        target_mean=float(
            target_mean_tensor
        ),
        target_scale=float(
            target_scale_tensor
        ),
        optimizer_steps=optimizer_steps,
        parameter_delta_l2=(
            parameter_delta_l2
        ),
        selected_epoch=best_epoch,
        fit_time_seconds=(
            fit_time_seconds
        ),
        inference_time_seconds=(
            inference_time_seconds
        ),
        loss_ledger=loss_ledger,
        parameter_count=parameter_count(
            model
        ),
        observed_soil_fraction_train=(
            float(
                train_soil_available.mean()
            )
        ),
        observed_soil_fraction_validation=(
            float(
                validation_soil_available
                .mean()
            )
        ),
    )


def build_soil_checkpoint(
    *,
    result: SoilTrainingResult,
    route: str,
    weather_dim: int,
    soil_dim: int,
    hidden: int,
    representation_dim: int,
    soil_representation: str,
    soil_dropout_probability: float,
    explicit_availability_mask: bool,
    weather_features: list[str],
    soil_features: list[str],
) -> dict[str, Any]:
    return {
        "model_family": (
            "flexible_soil_regressor"
        ),
        "state_dict": (
            result.best_state_dict
        ),
        "route": route,
        "weather_dim": weather_dim,
        "soil_dim": soil_dim,
        "hidden": hidden,
        "representation_dim": (
            representation_dim
        ),
        "target_mean": (
            result.target_mean
        ),
        "target_scale": (
            result.target_scale
        ),
        "selected_epoch": (
            result.selected_epoch
        ),
        "parameter_count": (
            result.parameter_count
        ),
        "parameter_matching_group": (
            "SOIL_DIRECT_MISSING_AWARE_V1"
        ),
        "soil_representation": (
            soil_representation
        ),
        "soil_dropout_probability": (
            soil_dropout_probability
        ),
        "explicit_availability_mask": (
            explicit_availability_mask
        ),
        "weather_features": list(
            weather_features
        ),
        "soil_features": list(
            soil_features
        ),
    }


def predict_soil_checkpoint(
    *,
    payload: dict[str, Any],
    weather: torch.Tensor,
    soil: torch.Tensor,
    soil_available: torch.Tensor,
    explicit_mask_override: (
        bool
        | None
    ) = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if payload.get(
        "model_family"
    ) != "flexible_soil_regressor":
        raise RuntimeError(
            "SOIL_CHECKPOINT_FAMILY_INVALID"
        )

    model = FlexibleSoilRegressor(
        weather_dim=int(
            payload["weather_dim"]
        ),
        soil_dim=int(
            payload["soil_dim"]
        ),
        hidden=int(
            payload["hidden"]
        ),
        representation_dim=int(
            payload[
                "representation_dim"
            ]
        ),
    )

    model.load_state_dict(
        payload["state_dict"]
    )
    model.eval()

    explicit_mask = (
        bool(
            payload[
                "explicit_availability_mask"
            ]
        )
        if explicit_mask_override
        is None
        else explicit_mask_override
    )

    with torch.no_grad():
        (
            prediction_scaled,
            representation,
        ) = model(
            weather,
            soil,
            soil_available,
            explicit_mask=explicit_mask,
        )

        prediction = (
            prediction_scaled
            * float(
                payload["target_scale"]
            )
            + float(
                payload["target_mean"]
            )
        )

    return (
        prediction.cpu().numpy(),
        representation.cpu().numpy(),
    )
