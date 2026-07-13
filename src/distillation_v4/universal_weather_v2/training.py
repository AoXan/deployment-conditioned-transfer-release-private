from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from .artifacts import (
    load_checkpoint,
    save_checkpoint,
    stable_hash,
)
from .data import (
    UniversalDatasetV2,
    load_dataset_v2,
)
from .models import (
    UniversalWeatherEncoderV2,
    UniversalWeatherRegressorV2,
)
from .tokens import build_weather_tokens_v2


ROUTES = {
    "supervised",
    "prediction_kd",
    "representation_kd",
    "combined_kd",
    "missing_aware",
}


@dataclass
class ArrayPreprocessorV2:
    imputer: SimpleImputer
    scaler: StandardScaler
    columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: Iterable[str],
    ) -> "ArrayPreprocessorV2":
        columns = tuple(
            map(str, columns)
        )

        matrix = frame[
            list(columns)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(
            dtype=np.float64,
        )

        imputer = SimpleImputer(
            strategy="median"
        )

        transformed = imputer.fit_transform(
            matrix
        )

        scaler = StandardScaler()

        scaler.fit(transformed)

        return cls(
            imputer=imputer,
            scaler=scaler,
            columns=columns,
        )

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        matrix = frame[
            list(self.columns)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(
            dtype=np.float64,
        )

        matrix = self.imputer.transform(
            matrix
        )

        matrix = self.scaler.transform(
            matrix
        )

        return matrix.astype(
            np.float32
        )

    def state(self) -> dict[str, Any]:
        return {
            "columns": list(
                self.columns
            ),
            "imputer_statistics": (
                self.imputer.statistics_
                .astype(float)
                .tolist()
            ),
            "scaler_mean": (
                self.scaler.mean_
                .astype(float)
                .tolist()
            ),
            "scaler_scale": (
                self.scaler.scale_
                .astype(float)
                .tolist()
            ),
        }


class WeatherDatasetTorchV2(Dataset):
    def __init__(
        self,
        *,
        weather: np.ndarray,
        target: np.ndarray,
        privileged: np.ndarray | None,
    ) -> None:
        self.weather = torch.as_tensor(
            weather,
            dtype=torch.float32,
        )

        self.target = torch.as_tensor(
            target,
            dtype=torch.float32,
        )

        self.privileged = (
            None
            if privileged is None
            else torch.as_tensor(
                privileged,
                dtype=torch.float32,
            )
        )

    def __len__(self) -> int:
        return int(
            self.target.shape[0]
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        item = {
            "weather": self.weather[
                index
            ],
            "target": self.target[
                index
            ],
        }

        if self.privileged is not None:
            item["privileged"] = (
                self.privileged[index]
            )

        return item


class PrivilegedTeacherV2(nn.Module):
    def __init__(
        self,
        *,
        weather_model: nn.Module,
        privileged_dim: int,
        representation_dim: int = 128,
    ) -> None:
        super().__init__()

        self.weather_model = (
            weather_model
        )

        self.privileged_dim = int(
            privileged_dim
        )

        if self.privileged_dim > 0:
            self.privileged_branch = (
                nn.Sequential(
                    nn.Linear(
                        self.privileged_dim,
                        representation_dim,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(
                        representation_dim
                    ),
                )
            )

            self.fusion = nn.Sequential(
                nn.Linear(
                    representation_dim * 2,
                    representation_dim,
                ),
                nn.ReLU(),
                nn.LayerNorm(
                    representation_dim
                ),
            )
        else:
            self.privileged_branch = None
            self.fusion = nn.Identity()

        self.head = nn.Linear(
            representation_dim,
            1,
        )

    def forward(
        self,
        token_batch: Any,
        privileged: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _prediction, weather_rep = (
            self.weather_model(
                token_batch
            )
        )

        if (
            self.privileged_branch
            is not None
        ):
            if privileged is None:
                raise ValueError(
                    "PRIVILEGED_INPUT_REQUIRED"
                )

            privileged_rep = (
                self.privileged_branch(
                    privileged
                )
            )

            representation = self.fusion(
                torch.cat(
                    [
                        weather_rep,
                        privileged_rep,
                    ],
                    dim=-1,
                )
            )
        else:
            representation = weather_rep

        prediction = self.head(
            representation
        ).squeeze(-1)

        return prediction, representation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def build_student_v2() -> nn.Module:
    return UniversalWeatherRegressorV2(
        encoder=UniversalWeatherEncoderV2()
    )


def model_forward_v2(
    model: nn.Module,
    *,
    weather: torch.Tensor,
    feature_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    token_batch = build_weather_tokens_v2(
        weather,
        feature_names,
    )

    output = model(token_batch)

    if (
        not isinstance(output, tuple)
        or len(output) != 2
    ):
        raise RuntimeError(
            "V2_MODEL_MUST_RETURN_PREDICTION_AND_REPRESENTATION"
        )

    prediction, representation = (
        output
    )

    prediction = prediction.reshape(-1)

    return prediction, representation


def metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(
            r2_score(
                target,
                prediction,
            )
        ),
        "rmse": float(
            mean_squared_error(
                target,
                prediction,
            )
            ** 0.5
        ),
        "mae": float(
            mean_absolute_error(
                target,
                prediction,
            )
        ),
        "prediction_std": float(
            np.std(prediction)
        ),
    }


def split_frame_by_ids(
    dataset: UniversalDatasetV2,
    split_contract: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    frame = dataset.frame

    id_column = (
        dataset.sample_id_column
    )

    indexed = frame.set_index(
        id_column,
        drop=False,
    )

    def select(
        key: str,
    ) -> pd.DataFrame:
        ids = list(
            map(
                str,
                split_contract[key],
            )
        )

        missing = [
            sample_id
            for sample_id in ids
            if sample_id
            not in indexed.index
        ]

        if missing:
            raise ValueError(
                f"SPLIT_IDS_MISSING:{key}:"
                f"{len(missing)}"
            )

        return indexed.loc[
            ids
        ].reset_index(
            drop=True
        )

    return (
        select("train_ids"),
        select("validation_ids"),
        select("test_ids"),
    )


def prepare_arrays(
    *,
    dataset: UniversalDatasetV2,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    weather_preprocessor = (
        ArrayPreprocessorV2.fit(
            train_frame,
            dataset.weather_features,
        )
    )

    privileged_preprocessor = None

    if dataset.privileged_features:
        privileged_preprocessor = (
            ArrayPreprocessorV2.fit(
                train_frame,
                dataset.privileged_features,
            )
        )

    def transform(
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        target = pd.to_numeric(
            frame[
                dataset.target_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float32,
        )

        if not np.isfinite(
            target
        ).all():
            raise ValueError(
                "NONFINITE_TARGET"
            )

        privileged = None

        if (
            privileged_preprocessor
            is not None
        ):
            privileged = (
                privileged_preprocessor
                .transform(frame)
            )

        return {
            "weather": (
                weather_preprocessor
                .transform(frame)
            ),
            "target": target,
            "privileged": privileged,
        }

    return {
        "train": transform(
            train_frame
        ),
        "validation": transform(
            validation_frame
        ),
        "test": transform(
            test_frame
        ),
        "weather_preprocessor": (
            weather_preprocessor
        ),
        "privileged_preprocessor": (
            privileged_preprocessor
        ),
    }


def make_loader(
    arrays: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = WeatherDatasetTorchV2(
        weather=arrays["weather"],
        target=arrays["target"],
        privileged=arrays[
            "privileged"
        ],
    )

    return DataLoader(
        dataset,
        batch_size=min(
            batch_size,
            len(dataset),
        ),
        shuffle=shuffle,
        drop_last=False,
    )


def evaluate_student(
    *,
    model: nn.Module,
    arrays: dict[str, Any],
    feature_names: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[
    dict[str, float],
    np.ndarray,
]:
    model.eval()

    loader = make_loader(
        arrays,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            weather = batch[
                "weather"
            ].to(device)

            prediction, _representation = (
                model_forward_v2(
                    model,
                    weather=weather,
                    feature_names=(
                        feature_names
                    ),
                )
            )

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                batch["target"].numpy()
            )

    prediction_array = np.concatenate(
        predictions
    )

    target_array = np.concatenate(
        targets
    )

    return (
        metric_bundle(
            target_array,
            prediction_array,
        ),
        prediction_array,
    )


def train_teacher_v2(
    *,
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
) -> dict[str, Any]:
    set_seed(seed)

    weather_model = build_student_v2()

    privileged_dim = (
        0
        if arrays["train"][
            "privileged"
        ] is None
        else int(
            arrays["train"][
                "privileged"
            ].shape[1]
        )
    )

    model = PrivilegedTeacherV2(
        weather_model=weather_model,
        privileged_dim=(
            privileged_dim
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    train_loader = make_loader(
        arrays["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = make_loader(
        arrays["validation"],
        batch_size=batch_size,
        shuffle=False,
    )

    best_loss = math.inf
    best_state = None
    no_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            weather = batch[
                "weather"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            privileged = batch.get(
                "privileged"
            )

            if privileged is not None:
                privileged = (
                    privileged.to(device)
                )

            token_batch = (
                build_weather_tokens_v2(
                    weather,
                    feature_names,
                )
            )

            prediction, _representation = (
                model(
                    token_batch,
                    privileged,
                )
            )

            loss = criterion(
                prediction,
                target,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        model.eval()
        validation_losses = []

        with torch.no_grad():
            for batch in validation_loader:
                weather = batch[
                    "weather"
                ].to(device)

                target = batch[
                    "target"
                ].to(device)

                privileged = batch.get(
                    "privileged"
                )

                if privileged is not None:
                    privileged = (
                        privileged.to(device)
                    )

                token_batch = (
                    build_weather_tokens_v2(
                        weather,
                        feature_names,
                    )
                )

                prediction, _representation = (
                    model(
                        token_batch,
                        privileged,
                    )
                )

                validation_losses.append(
                    float(
                        criterion(
                            prediction,
                            target,
                        ).cpu()
                    )
                )

        validation_loss = float(
            np.mean(
                validation_losses
            )
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    np.mean(epoch_losses)
                ),
                "validation_loss": (
                    validation_loss
                ),
            }
        )

        if validation_loss < (
            best_loss - 1e-8
        ):
            best_loss = validation_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "TEACHER_NO_BEST_STATE"
        )

    model.load_state_dict(
        best_state
    )

    manifest = save_checkpoint(
        path=checkpoint_path,
        model=model,
        metadata={
            **metadata,
            "model_role": (
                "PRIVILEGED_TEACHER_V2"
            ),
            "best_validation_loss": (
                best_loss
            ),
            "history": history,
            "privileged_dim": (
                privileged_dim
            ),
        },
        optimizer=None,
    )

    return {
        "model": model,
        "manifest": manifest,
        "best_validation_loss": (
            best_loss
        ),
        "history": history,
        "privileged_dim": (
            privileged_dim
        ),
    }


def freeze_teacher(
    teacher: nn.Module,
) -> None:
    teacher.eval()

    for parameter in (
        teacher.parameters()
    ):
        parameter.requires_grad = False


def teacher_is_frozen(
    teacher: nn.Module,
) -> bool:
    return (
        not teacher.training
        and all(
            not parameter.requires_grad
            for parameter in (
                teacher.parameters()
            )
        )
    )


def route_loss_v2(
    *,
    route: str,
    student_prediction: torch.Tensor,
    student_representation: torch.Tensor,
    target: torch.Tensor,
    teacher_prediction: torch.Tensor | None,
    teacher_representation: torch.Tensor | None,
    consistency_prediction: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:
    supervised = nn.functional.mse_loss(
        student_prediction,
        target,
    )

    prediction_kd = torch.zeros(
        (),
        device=target.device,
    )

    representation_kd = (
        torch.zeros(
            (),
            device=target.device,
        )
    )

    consistency = torch.zeros(
        (),
        device=target.device,
    )

    if teacher_prediction is not None:
        prediction_kd = (
            nn.functional.mse_loss(
                student_prediction,
                teacher_prediction,
            )
        )

    if teacher_representation is not None:
        if (
            student_representation.shape
            != teacher_representation.shape
        ):
            raise ValueError(
                "REPRESENTATION_SHAPE_MISMATCH:"
                f"{student_representation.shape}:"
                f"{teacher_representation.shape}"
            )

        representation_kd = (
            nn.functional.mse_loss(
                student_representation,
                teacher_representation,
            )
        )

    if consistency_prediction is not None:
        consistency = (
            nn.functional.mse_loss(
                student_prediction,
                consistency_prediction,
            )
        )

    if route == "supervised":
        total = supervised

    elif route == "prediction_kd":
        total = (
            supervised
            + 0.5 * prediction_kd
        )

    elif route == "representation_kd":
        total = (
            supervised
            + 0.25
            * representation_kd
        )

    elif route == "combined_kd":
        total = (
            supervised
            + 0.5 * prediction_kd
            + 0.25
            * representation_kd
        )

    elif route == "missing_aware":
        total = (
            supervised
            + 0.25 * prediction_kd
            + 0.25 * consistency
        )

    else:
        raise ValueError(
            f"UNKNOWN_ROUTE:{route}"
        )

    parts = {
        "supervised": float(
            supervised.detach().cpu()
        ),
        "prediction_kd": float(
            prediction_kd.detach().cpu()
        ),
        "representation_kd": float(
            representation_kd.detach().cpu()
        ),
        "consistency": float(
            consistency.detach().cpu()
        ),
        "total": float(
            total.detach().cpu()
        ),
    }

    return total, parts


def train_student_route_v2(
    *,
    route: str,
    arrays: dict[str, Any],
    feature_names: list[str],
    teacher: nn.Module | None,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(
            f"UNKNOWN_ROUTE:{route}"
        )

    if (
        route != "supervised"
        and teacher is None
    ):
        raise ValueError(
            f"TEACHER_REQUIRED:{route}"
        )

    if teacher is not None:
        freeze_teacher(teacher)

        if not teacher_is_frozen(
            teacher
        ):
            raise RuntimeError(
                "TEACHER_NOT_FROZEN"
            )

    set_seed(seed)

    student = build_student_v2().to(
        device
    )

    if initial_state_dict is not None:
        student.load_state_dict(
            initial_state_dict,
            strict=True,
        )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    train_loader = make_loader(
        arrays["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    best_validation = math.inf
    best_state = None
    no_improvement = 0
    history = []

    for epoch in range(epochs):
        student.train()

        epoch_parts = []

        for batch in train_loader:
            weather = batch[
                "weather"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            privileged = batch.get(
                "privileged"
            )

            if privileged is not None:
                privileged = (
                    privileged.to(device)
                )

            student_prediction, student_rep = (
                model_forward_v2(
                    student,
                    weather=weather,
                    feature_names=(
                        feature_names
                    ),
                )
            )

            teacher_prediction = None
            teacher_representation = None

            if teacher is not None:
                with torch.no_grad():
                    token_batch = (
                        build_weather_tokens_v2(
                            weather,
                            feature_names,
                        )
                    )

                    (
                        teacher_prediction,
                        teacher_representation,
                    ) = teacher(
                        token_batch,
                        privileged,
                    )

            consistency_prediction = None

            if route == "missing_aware":
                missing_weather = (
                    weather.clone()
                )

                mask = torch.rand_like(
                    missing_weather
                ) < 0.2

                missing_weather[mask] = (
                    float("nan")
                )

                consistency_prediction, _ = (
                    model_forward_v2(
                        student,
                        weather=missing_weather,
                        feature_names=(
                            feature_names
                        ),
                    )
                )

            loss, parts = route_loss_v2(
                route=route,
                student_prediction=(
                    student_prediction
                ),
                student_representation=(
                    student_rep
                ),
                target=target,
                teacher_prediction=(
                    teacher_prediction
                ),
                teacher_representation=(
                    teacher_representation
                ),
                consistency_prediction=(
                    consistency_prediction
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            epoch_parts.append(parts)

        validation_metrics, _ = (
            evaluate_student(
                model=student,
                arrays=arrays[
                    "validation"
                ],
                feature_names=(
                    feature_names
                ),
                device=device,
                batch_size=batch_size,
            )
        )

        validation_loss = (
            validation_metrics["rmse"]
            ** 2
        )

        mean_parts = {
            key: float(
                np.mean(
                    [
                        item[key]
                        for item
                        in epoch_parts
                    ]
                )
            )
            for key in [
                "supervised",
                "prediction_kd",
                "representation_kd",
                "consistency",
                "total",
            ]
        }

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss_parts": (
                    mean_parts
                ),
                "validation_metrics": (
                    validation_metrics
                ),
            }
        )

        if validation_loss < (
            best_validation - 1e-8
        ):
            best_validation = (
                validation_loss
            )

            best_state = copy.deepcopy(
                student.state_dict()
            )

            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "STUDENT_NO_BEST_STATE"
        )

    student.load_state_dict(
        best_state
    )

    test_metrics, test_prediction = (
        evaluate_student(
            model=student,
            arrays=arrays["test"],
            feature_names=feature_names,
            device=device,
            batch_size=batch_size,
        )
    )

    manifest = save_checkpoint(
        path=checkpoint_path,
        model=student,
        metadata={
            **metadata,
            "model_role": (
                "UNIVERSAL_WEATHER_STUDENT_V2"
            ),
            "route": route,
            "best_validation_loss": (
                best_validation
            ),
            "history": history,
            "test_metrics": (
                test_metrics
            ),
        },
        optimizer=None,
    )

    return {
        "model": student,
        "manifest": manifest,
        "route": route,
        "history": history,
        "validation_metrics": history[-1][
            "validation_metrics"
        ],
        "test_metrics": test_metrics,
        "test_prediction": (
            test_prediction
        ),
        "teacher_frozen": (
            teacher is None
            or teacher_is_frozen(
                teacher
            )
        ),
    }


def load_dataset_and_arrays(
    *,
    repository_root: Path,
    dataset_id: str,
    registry_entry: dict[str, Any],
    split_contract: dict[str, Any],
) -> tuple[
    UniversalDatasetV2,
    dict[str, Any],
]:
    dataset = load_dataset_v2(
        dataset_id=dataset_id,
        registry_entry=registry_entry,
        repository_root=(
            repository_root
        ),
    )

    train_frame, validation_frame, test_frame = (
        split_frame_by_ids(
            dataset,
            split_contract,
        )
    )

    arrays = prepare_arrays(
        dataset=dataset,
        train_frame=train_frame,
        validation_frame=(
            validation_frame
        ),
        test_frame=test_frame,
    )

    return dataset, arrays


def training_fingerprint(
    *,
    dataset_id: str,
    route: str,
    seed: int,
    feature_contract: dict[str, Any],
    split_contract: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "schema": (
                "universal_weather_training_fingerprint_v2"
            ),
            "dataset_id": dataset_id,
            "route": route,
            "seed": seed,
            "feature_contract": (
                feature_contract
            ),
            "split_contract": (
                split_contract
            ),
            "hyperparameters": (
                hyperparameters
            ),
        }
    )
