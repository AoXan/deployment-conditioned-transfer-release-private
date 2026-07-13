from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.distillation_program.next_models import (
    ModalitySpecificRegressor,
    SharedDenseRegressor,
)

from .phase3_matrix import Phase3Job
from .repair_trainer import fit_repair_regressor
from .training_contract import NeuralTrainingContract


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_features(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    weather = sorted(
        column
        for column in frame.columns
        if column.startswith("weather_")
    )
    soil = sorted(
        column
        for column in frame.columns
        if (
            column.startswith("soil_")
            and not column.endswith("__missing")
        )
    )

    if not weather:
        raise ValueError("PHASE3_WEATHER_FEATURES_EMPTY")
    if not soil:
        raise ValueError("PHASE3_SOIL_FEATURES_EMPTY")

    return weather, soil


def _split(
    frame: pd.DataFrame,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_year = int(fold.split("_")[-1])
    validation_year = outer_year - 1

    train = frame.loc[
        frame.year < validation_year
    ].copy()
    validation = frame.loc[
        frame.year.eq(validation_year)
    ].copy()

    if train.empty:
        raise ValueError(f"PHASE3_TRAIN_EMPTY:{fold}")
    if validation.empty:
        raise ValueError(
            f"PHASE3_VALIDATION_EMPTY:{fold}"
        )

    return train, validation


def _tabular_pipeline(
    features: list[str],
    *,
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    (
                                        "impute",
                                        SimpleImputer(
                                            strategy="median",
                                            add_indicator=True,
                                        ),
                                    ),
                                    (
                                        "scale",
                                        StandardScaler(),
                                    ),
                                ]
                            ),
                            features,
                        )
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _execute_hgb(
    *,
    job: Phase3Job,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    weather_features: list[str],
    soil_features: list[str],
    output: Path,
) -> dict[str, Any]:
    features = (
        weather_features
        if job.input_profile == "deployable"
        else weather_features + soil_features
    )

    started = time.perf_counter()
    model = _tabular_pipeline(
        features,
        seed=job.seed,
    )
    model.fit(
        train[features],
        train.target_yield,
    )
    prediction = model.predict(
        validation[features]
    )
    fit_time = time.perf_counter() - started

    mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            prediction,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "job": job.to_mapping(),
        },
        checkpoint,
    )

    replay_bundle = joblib.load(checkpoint)
    replay = replay_bundle["model"].predict(
        validation[replay_bundle["features"]]
    )
    replay_error = float(
        np.max(np.abs(replay - prediction))
    )

    if replay_error > 1e-12:
        raise RuntimeError(
            "PHASE3_HGB_CHECKPOINT_REPLAY_MISMATCH"
        )

    np.save(
        output / "validation_prediction.npy",
        prediction,
    )

    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "fold": job.fold,
        "seed": job.seed,
        "component": job.component,
        "candidate": job.candidate,
        "model_family": job.model_family,
        "scientific_roles": list(
            job.scientific_roles
        ),
        "status": "ENGINEERING_ACCEPTED",
        "validation_mae": mae,
        "prediction_std": float(np.std(prediction)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "fit_time_seconds": fit_time,
        "optimizer_steps": None,
        "parameter_delta_l2": None,
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_used": False,
    }


def _fit_modality_preprocessors(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    weather_features: list[str],
    soil_features: list[str],
    include_soil: bool,
    shuffle_soil: bool,
    seed: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    values_train: dict[str, torch.Tensor] = {}
    values_validation: dict[str, torch.Tensor] = {}
    preprocessors: dict[str, Any] = {}

    weather_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    weather_train = weather_pipeline.fit_transform(
        train[weather_features]
    )
    weather_validation = weather_pipeline.transform(
        validation[weather_features]
    )

    values_train["weather"] = torch.tensor(
        weather_train,
        dtype=torch.float32,
    )
    values_validation["weather"] = torch.tensor(
        weather_validation,
        dtype=torch.float32,
    )
    preprocessors["weather"] = weather_pipeline

    if include_soil:
        soil_pipeline = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )

        soil_train = soil_pipeline.fit_transform(
            train[soil_features]
        )
        soil_validation = soil_pipeline.transform(
            validation[soil_features]
        )

        if shuffle_soil:
            generator = np.random.default_rng(seed)
            permutation = generator.permutation(
                soil_train.shape[0]
            )
            soil_train = soil_train[permutation]

        values_train["soil"] = torch.tensor(
            soil_train,
            dtype=torch.float32,
        )
        values_validation["soil"] = torch.tensor(
            soil_validation,
            dtype=torch.float32,
        )
        preprocessors["soil"] = soil_pipeline

    return (
        values_train,
        values_validation,
        preprocessors,
    )


def _build_neural_model(
    *,
    candidate: str,
    modality_dims: Mapping[str, int],
) -> torch.nn.Module:
    if candidate in {
        "shared_early_fusion",
        "shared_neural_receiver",
    }:
        return SharedDenseRegressor(
            modality_dims,
            hidden=32,
        )

    if candidate in {
        "modality_specific_late_fusion",
        "shuffled_soil_neural",
    }:
        return ModalitySpecificRegressor(
            modality_dims,
            hidden=32,
        )

    raise ValueError(
        f"PHASE3_NEURAL_CANDIDATE_UNKNOWN:{candidate}"
    )


def _representation(
    model: torch.nn.Module,
    values: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, auxiliary = model(values, mask)

    representation = auxiliary.get("representation")
    if representation is None:
        raise RuntimeError(
            "PHASE3_REPRESENTATION_NOT_EXPOSED"
        )

    return (
        representation.detach().cpu().numpy()
    )


def _execute_neural(
    *,
    job: Phase3Job,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    weather_features: list[str],
    soil_features: list[str],
    output: Path,
    contract: NeuralTrainingContract,
) -> dict[str, Any]:
    include_soil = job.input_profile != "deployable"
    shuffle_soil = (
        job.candidate == "shuffled_soil_neural"
    )

    (
        train_values,
        validation_values,
        preprocessors,
    ) = _fit_modality_preprocessors(
        train=train,
        validation=validation,
        weather_features=weather_features,
        soil_features=soil_features,
        include_soil=include_soil,
        shuffle_soil=shuffle_soil,
        seed=job.seed,
    )

    dims = {
        name: int(tensor.shape[1])
        for name, tensor in train_values.items()
    }
    model = _build_neural_model(
        candidate=job.candidate,
        modality_dims=dims,
    )

    train_mask = torch.ones(
        len(train),
        len(dims),
        dtype=torch.float32,
    )
    validation_mask = torch.ones(
        len(validation),
        len(dims),
        dtype=torch.float32,
    )

    result = fit_repair_regressor(
        model=model,
        train_values=train_values,
        train_mask=train_mask,
        train_target=torch.tensor(
            train.target_yield.to_numpy(),
            dtype=torch.float32,
        ),
        validation_values=validation_values,
        validation_mask=validation_mask,
        validation_target=torch.tensor(
            validation.target_yield.to_numpy(),
            dtype=torch.float32,
        ),
        contract=contract,
        seed=job.seed,
    )

    train_representation = _representation(
        model,
        train_values,
        train_mask,
    )
    validation_representation = _representation(
        model,
        validation_values,
        validation_mask,
    )

    probe = Ridge(alpha=1.0)
    probe.fit(
        train_representation,
        train.target_yield.to_numpy(float),
    )
    probe_prediction = probe.predict(
        validation_representation
    )
    probe_mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            probe_prediction,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    preprocessor_path = output / "preprocessors.joblib"

    torch.save(
        {
            "state_dict": result.best_state_dict,
            "candidate": job.candidate,
            "dims": dims,
            "target_mean": result.target_mean,
            "target_scale": result.target_scale,
            "job": job.to_mapping(),
        },
        checkpoint,
    )
    joblib.dump(
        preprocessors,
        preprocessor_path,
    )

    replay_bundle = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    replay_model = _build_neural_model(
        candidate=job.candidate,
        modality_dims=replay_bundle["dims"],
    )
    replay_model.load_state_dict(
        replay_bundle["state_dict"]
    )
    replay_model.eval()

    with torch.no_grad():
        replay_scaled, _ = replay_model(
            validation_values,
            validation_mask,
        )
        replay_prediction = (
            replay_scaled
            * replay_bundle["target_scale"]
            + replay_bundle["target_mean"]
        ).cpu().numpy()

    replay_error = float(
        np.max(
            np.abs(
                replay_prediction
                - result.prediction
            )
        )
    )
    if replay_error > 1e-6:
        raise RuntimeError(
            "PHASE3_NEURAL_CHECKPOINT_REPLAY_MISMATCH"
        )

    np.save(
        output / "validation_prediction.npy",
        result.prediction,
    )
    np.save(
        output / "validation_representation.npy",
        validation_representation,
    )

    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "fold": job.fold,
        "seed": job.seed,
        "component": job.component,
        "candidate": job.candidate,
        "model_family": job.model_family,
        "scientific_roles": list(
            job.scientific_roles
        ),
        "status": "ENGINEERING_ACCEPTED",
        "validation_mae": float(
            mean_absolute_error(
                validation.target_yield.to_numpy(float),
                result.prediction,
            )
        ),
        "representation_probe_mae": probe_mae,
        "prediction_std": float(
            np.std(result.prediction)
        ),
        "representation_std": float(
            np.std(validation_representation)
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "preprocessor": str(preprocessor_path),
        "preprocessor_sha256": _sha256(
            preprocessor_path
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "optimizer_steps": result.optimizer_steps,
        "parameter_delta_l2": (
            result.parameter_delta_l2
        ),
        "selected_epoch": result.selected_epoch,
        "fit_time_seconds": (
            result.fit_time_seconds
        ),
        "inference_time_seconds": (
            result.inference_time_seconds
        ),
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_used": False,
        "soil_shuffled_in_train_only": shuffle_soil,
    }


def _execute_random_representation(
    *,
    job: Phase3Job,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    generator = np.random.default_rng(job.seed)

    width = 32
    train_representation = generator.normal(
        size=(len(train), width)
    )
    validation_representation = generator.normal(
        size=(len(validation), width)
    )

    probe = Ridge(alpha=1.0)
    probe.fit(
        train_representation,
        train.target_yield.to_numpy(float),
    )
    prediction = probe.predict(
        validation_representation
    )

    mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            prediction,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "random_representation.npz"
    np.savez_compressed(
        artifact,
        train_representation=train_representation,
        validation_representation=(
            validation_representation
        ),
        prediction=prediction,
        seed=np.array([job.seed], dtype=np.int64),
    )

    replay = np.load(artifact)
    replay_error = float(
        np.max(
            np.abs(
                replay["prediction"]
                - prediction
            )
        )
    )

    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "fold": job.fold,
        "seed": job.seed,
        "component": job.component,
        "candidate": job.candidate,
        "model_family": job.model_family,
        "scientific_roles": list(
            job.scientific_roles
        ),
        "status": "ENGINEERING_ACCEPTED",
        "validation_mae": mae,
        "representation_probe_mae": mae,
        "prediction_std": float(np.std(prediction)),
        "representation_std": float(
            np.std(validation_representation)
        ),
        "control_artifact": str(artifact),
        "control_artifact_sha256": _sha256(
            artifact
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "optimizer_steps": 0,
        "parameter_delta_l2": 0.0,
        "fit_time_seconds": (
            time.perf_counter() - started
        ),
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_used": False,
    }


HANDLERS = {
    "hgb_privileged": "hgb",
    "hgb_deployable": "hgb",
    "shared_early_fusion": "neural",
    "modality_specific_late_fusion": "neural",
    "shared_neural_receiver": "neural",
    "shuffled_soil_neural": "neural",
    "random_representation": "random_representation",
}


def execute_phase3_job(
    *,
    job: Phase3Job,
    frame: pd.DataFrame,
    output: Path,
    output_root: Path,
    contract: NeuralTrainingContract,
) -> dict[str, Any]:
    if job.candidate not in HANDLERS:
        raise ValueError(
            f"PHASE3_HANDLER_MISSING:{job.candidate}"
        )

    train, validation = _split(
        frame,
        job.fold,
    )
    weather_features, soil_features = (
        _numeric_features(frame)
    )

    handler = HANDLERS[job.candidate]

    if handler == "hgb":
        record = _execute_hgb(
            job=job,
            train=train,
            validation=validation,
            weather_features=weather_features,
            soil_features=soil_features,
            output=output,
        )
    elif handler == "neural":
        record = _execute_neural(
            job=job,
            train=train,
            validation=validation,
            weather_features=weather_features,
            soil_features=soil_features,
            output=output,
            contract=contract,
        )
    elif handler == "random_representation":
        record = _execute_random_representation(
            job=job,
            train=train,
            validation=validation,
            output=output,
        )
    else:
        raise AssertionError(handler)

    relative_record = deepcopy(record)
    for field in (
        "checkpoint",
        "preprocessor",
        "control_artifact",
    ):
        value = relative_record.get(field)
        if value:
            relative_record[field] = str(
                Path(value).relative_to(
                    output_root
                )
            )

    _atomic_json(
        output / "job_record.json",
        relative_record,
    )
    _atomic_json(
        output / "completion_marker.json",
        {
            "job_id": job.job_id,
            "status": "ENGINEERING_ACCEPTED",
            "candidate": job.candidate,
            "outer_test_used": False,
        },
    )

    return relative_record
