from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


# EXACT_FORMAL_CODE_ROOT_INJECTION_V1
_FORMAL_CODE_ROOT = Path(
    os.environ.get(
        "FORMAL_CODE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()

if not (
    _FORMAL_CODE_ROOT
    / "src"
    / "distillation_v4"
    / "universal_weather_v2"
).is_dir():
    raise RuntimeError(
        "FORMAL_UNIVERSAL_WEATHER_SOURCE_NOT_FOUND:"
        f"{_FORMAL_CODE_ROOT}"
    )

_formal_root_text = str(
    _FORMAL_CODE_ROOT
)

if _formal_root_text in sys.path:
    sys.path.remove(
        _formal_root_text
    )

sys.path.insert(
    0,
    _formal_root_text,
)


from src.distillation_v4.universal_weather_v2.models import (
    UniversalWeatherEncoderV2,
    UniversalWeatherRegressorV2,
)
from src.distillation_v4.universal_weather_v2.tokens import (
    build_weather_tokens_v2,
)
from src.distillation_v4.universal_weather_v2.artifacts import (
    load_checkpoint,
    sha256_file,
)


ROUTES = [
    "supervised",
    "prediction_kd",
    "combined_kd",
    "representation_kd",
    "missing_aware",
]

CONTRACT_ALIASES = {
    "GROUP_complete": "GROUP",
    "SPATIAL_complete": "SPATIAL",
    "GROUP": "GROUP",
    "SPATIAL": "SPATIAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--worktree", required=True)
    parser.add_argument("--formal-code", required=True)
    parser.add_argument("--extract-script", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--max-test-rows", type=int, default=24)
    parser.add_argument("--background-rows", type=int, default=32)
    parser.add_argument("--shap-permutations", type=int, default=64)

    parser.add_argument(
        "--taylor-relative-sd-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--taylor-relative-range-fallback",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--minimum-physical-step",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--replay-atol",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--scaled-array-atol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260708,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    temporary.replace(path)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_extractor_module(path: Path):
    module_name = "accepted_stage8_embedding_extractor_runtime"

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import extraction script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    required = [
        "load_config",
        "load_frame",
        "build_contract_splits",
        "_dataset",
        "_frames",
        "prepare_arrays",
        "build_manifest_job",
        "validate_predictions",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Extraction module missing interfaces: {missing}"
        )

    return module


class Progress:
    def __init__(self, path: Path, total_cells: int) -> None:
        self.path = path
        self.started = time.time()
        self.state = {
            "status": "INITIALISING",
            "phase": "INITIALISING",
            "total_cells": total_cells,
            "replayed_cells": 0,
            "attributed_cells": 0,
            "failed_cells": 0,
            "total_samples": 0,
            "completed_samples": 0,
            "current_cell": None,
            "current_sample": None,
            "last_message": "",
            "max_replay_error": None,
            "max_scaled_array_error": None,
            "eta_seconds": None,
            "started_unix": self.started,
            "updated_unix": self.started,
        }
        self.write()

    def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        now = time.time()
        self.state["updated_unix"] = now

        completed = int(
            self.state.get("completed_samples", 0) or 0
        )
        total = int(
            self.state.get("total_samples", 0) or 0
        )

        if completed > 0 and total > completed:
            elapsed = now - self.started
            self.state["eta_seconds"] = (
                elapsed / completed
            ) * (total - completed)

        self.write()

    def write(self) -> None:
        atomic_json(self.path, self.state)


def load_cells(
    manifest_path: Path,
    lineage_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    lineage = pd.read_csv(lineage_path)

    cells = (
        manifest[
            [
                "contract",
                "method",
                "seed",
                "candidate_id",
                "file_path",
            ]
        ]
        .drop_duplicates()
        .merge(
            lineage,
            on=["candidate_id", "seed"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["contract", "method", "seed"]
        )
        .reset_index(drop=True)
    )

    if len(cells) != 30:
        raise RuntimeError(
            f"Expected 30 formal cells, found {len(cells)}"
        )

    expected = {
        (contract, route, seed)
        for contract in ["GROUP_complete", "SPATIAL_complete"]
        for route in ROUTES
        for seed in [101, 202, 303]
    }

    observed = {
        (
            str(row.contract),
            str(row.method),
            int(row.seed),
        )
        for row in cells.itertuples(index=False)
    }

    if observed != expected:
        raise RuntimeError(
            "Formal 30-cell matrix does not match "
            "GROUP/SPATIAL × five routes × three seeds"
        )

    if cells["file_path"].astype(str).str.contains(
        "/smoke/"
    ).any():
        raise RuntimeError(
            "Formal manifest references smoke artifacts"
        )

    return cells


def build_model(
    checkpoint_path: Path,
) -> UniversalWeatherRegressorV2:
    model = UniversalWeatherRegressorV2(
        encoder=UniversalWeatherEncoderV2()
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location="cpu",
        strict=True,
    )

    model.eval()
    return model


def predict_scaled(
    model: UniversalWeatherRegressorV2,
    scaled_values: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    values = np.asarray(
        scaled_values,
        dtype=np.float32,
    )

    if values.ndim == 1:
        values = values[None, :]

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D weather values, got {values.shape}"
        )

    if values.shape[1] != len(feature_names):
        raise ValueError(
            "Weather feature width does not match "
            "formal feature list"
        )

    token_batch = build_weather_tokens_v2(
        values,
        feature_names,
        device="cpu",
    )

    with torch.no_grad():
        prediction, _ = model(token_batch)

    return (
        prediction.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(-1)
    )


def raw_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    return (
        frame[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )


def impute_raw(
    matrix: np.ndarray,
    imputer_statistics: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()

    if values.ndim == 1:
        values = values[None, :]

    missing = ~np.isfinite(values)

    if missing.any():
        rows, columns = np.where(missing)
        values[rows, columns] = imputer_statistics[columns]

    return values


def transform_raw(
    raw_values: np.ndarray,
    imputer_statistics: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    imputed = impute_raw(
        raw_values,
        imputer_statistics,
    )

    if np.any(scaler_scale <= 0):
        raise RuntimeError(
            "Formal scaler contains non-positive scale"
        )

    scaled = (
        imputed - scaler_mean[None, :]
    ) / scaler_scale[None, :]

    return scaled.astype(np.float32)


def predict_raw(
    model: UniversalWeatherRegressorV2,
    raw_values: np.ndarray,
    feature_names: list[str],
    imputer_statistics: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    scaled = transform_raw(
        raw_values,
        imputer_statistics,
        scaler_mean,
        scaler_scale,
    )

    return predict_scaled(
        model,
        scaled,
        feature_names,
    )


def select_rows(
    metadata: np.ndarray,
    maximum: int,
) -> np.ndarray:
    n = len(metadata)

    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=int)

    errors = np.asarray(
        metadata["abs_error"],
        dtype=np.float64,
    )

    order = np.argsort(errors, kind="stable")

    positions = np.linspace(
        0,
        n - 1,
        num=maximum,
    )

    selected = order[
        np.rint(positions).astype(int)
    ]

    if len(np.unique(selected)) != maximum:
        raise RuntimeError(
            "Deterministic error-stratified sampling "
            "did not produce unique rows"
        )

    return selected.astype(int)


def physical_steps(
    train_raw_imputed: np.ndarray,
    *,
    relative_sd: float,
    relative_range_fallback: float,
    minimum_step: float,
) -> np.ndarray:
    standard_deviation = np.std(
        train_raw_imputed,
        axis=0,
        ddof=0,
    )

    feature_range = (
        np.max(train_raw_imputed, axis=0)
        - np.min(train_raw_imputed, axis=0)
    )

    steps = standard_deviation * relative_sd

    fallback = (
        feature_range * relative_range_fallback
    )

    steps = np.where(
        steps > minimum_step,
        steps,
        fallback,
    )

    steps = np.where(
        steps > minimum_step,
        steps,
        minimum_step,
    )

    if not np.isfinite(steps).all():
        raise RuntimeError(
            "Non-finite Taylor finite-difference steps"
        )

    return steps


def taylor_attribution(
    *,
    predictor: Callable[[np.ndarray], np.ndarray],
    sample_raw: np.ndarray,
    baseline_raw: np.ndarray,
    steps: np.ndarray,
) -> dict[str, Any]:
    sample = np.asarray(
        sample_raw,
        dtype=np.float64,
    )
    baseline = np.asarray(
        baseline_raw,
        dtype=np.float64,
    )

    displacement = sample - baseline

    sample_prediction = float(
        predictor(sample[None, :])[0]
    )
    baseline_prediction = float(
        predictor(baseline[None, :])[0]
    )

    perturbations = []

    for index, step in enumerate(steps):
        plus = sample.copy()
        minus = sample.copy()

        plus[index] += step
        minus[index] -= step

        perturbations.extend([plus, minus])

    predictions = predictor(
        np.stack(perturbations, axis=0)
    )

    gradient = np.zeros_like(sample)
    diagonal_hessian = np.zeros_like(sample)

    for index, step in enumerate(steps):
        plus_prediction = predictions[2 * index]
        minus_prediction = predictions[2 * index + 1]

        gradient[index] = (
            plus_prediction - minus_prediction
        ) / (2.0 * step)

        diagonal_hessian[index] = (
            plus_prediction
            - 2.0 * sample_prediction
            + minus_prediction
        ) / (step ** 2)

    first = gradient * displacement

    second_diagonal = (
        0.5
        * diagonal_hessian
        * displacement ** 2
    )

    first_reconstruction = (
        baseline_prediction + first.sum()
    )

    second_reconstruction = (
        baseline_prediction
        + first.sum()
        + second_diagonal.sum()
    )

    return {
        "sample_prediction": sample_prediction,
        "baseline_prediction": baseline_prediction,
        "gradient": gradient,
        "diagonal_hessian": diagonal_hessian,
        "first_contribution": first,
        "second_diagonal_contribution": second_diagonal,
        "first_reconstruction": first_reconstruction,
        "second_reconstruction": second_reconstruction,
        "first_residual": (
            sample_prediction
            - first_reconstruction
        ),
        "second_residual": (
            sample_prediction
            - second_reconstruction
        ),
    }


def permutation_shap(
    *,
    predictor: Callable[[np.ndarray], np.ndarray],
    sample_raw: np.ndarray,
    background_raw: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)

    sample = np.asarray(
        sample_raw,
        dtype=np.float64,
    )

    background = np.asarray(
        background_raw,
        dtype=np.float64,
    )

    n_features = sample.shape[0]
    values = np.zeros(n_features, dtype=np.float64)
    base_predictions = []

    for _ in range(permutations):
        background_index = int(
            generator.integers(0, len(background))
        )

        current = background[
            background_index
        ].copy()

        current_prediction = float(
            predictor(current[None, :])[0]
        )

        base_predictions.append(current_prediction)

        order = generator.permutation(n_features)

        for feature_index in order:
            updated = current.copy()
            updated[feature_index] = sample[feature_index]

            next_prediction = float(
                predictor(updated[None, :])[0]
            )

            values[feature_index] += (
                next_prediction
                - current_prediction
            )

            current = updated
            current_prediction = next_prediction

    values /= float(permutations)

    sample_prediction = float(
        predictor(sample[None, :])[0]
    )

    expected_value = float(
        np.mean(base_predictions)
    )

    residual = (
        sample_prediction
        - expected_value
        - values.sum()
    )

    return {
        "values": values,
        "sample_prediction": sample_prediction,
        "expected_value": expected_value,
        "efficiency_residual": residual,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    generator = np.random.default_rng(seed)

    boot = np.empty(
        repetitions,
        dtype=np.float64,
    )

    for index in range(repetitions):
        boot[index] = np.mean(
            generator.choice(
                values,
                size=len(values),
                replace=True,
            )
        )

    return (
        float(np.mean(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def main() -> None:
    args = parse_args()

    worktree = Path(args.worktree).resolve()
    formal_code = Path(args.formal_code).resolve()
    extract_path = Path(args.extract_script).resolve()
    manifest_path = Path(args.manifest).resolve()
    lineage_path = Path(args.lineage).resolve()
    output = Path(args.output).resolve()

    output.mkdir(parents=True, exist_ok=True)

    cells = load_cells(
        manifest_path,
        lineage_path,
    )

    progress = Progress(
        output / "progress.json",
        total_cells=len(cells),
    )

    failures: list[dict[str, Any]] = []

    try:
        extractor = load_extractor_module(
            extract_path
        )

        config_path = (
            extractor.FORMAL_WORKTREE
            / "configs/universal_weather_full_campaign_v2_remediation_v1.yaml"
        )
        schema_path = (
            extractor.FORMAL_WORKTREE
            / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json"
        )

        config = extractor.load_config(
            config_path,
            schema_path,
        )

        contract_cfg = config[
            "datasets"
        ][extractor.DATASET_ID]

        frame, data_path = extractor.load_frame(
            contract_cfg,
            extractor.FORMAL_WORKTREE,
        )

        dataset = extractor._dataset(
            frame,
            data_path,
            extractor.DATASET_ID,
            contract_cfg,
        )

        feature_names = list(
            dataset.weather_features
        )

        if len(feature_names) != 10:
            raise RuntimeError(
                f"Expected 10 formal weather features, "
                f"found {len(feature_names)}"
            )

        sample_id_column = contract_cfg[
            "sample_id_column"
        ]

        lineage = pd.read_csv(lineage_path)

        split_cache: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}

        replay_rows = []
        loaded_cells = []

        progress.update(
            status="RUNNING",
            phase="REPLAY_GATE",
            last_message=(
                "Rebuilding formal preprocessing and "
                "validating all 30 checkpoint replays"
            ),
        )

        for cell_index, row in cells.iterrows():
            contract_full = str(row["contract"])
            split_id = CONTRACT_ALIASES[
                contract_full
            ]
            method = str(row["method"])
            seed = int(row["seed"])
            candidate_id = str(
                row["candidate_id"]
            )

            cell_key = (
                f"{contract_full}/{method}/"
                f"{seed}/{candidate_id}"
            )

            progress.update(
                current_cell=cell_key,
                last_message=(
                    f"Replay gate "
                    f"{cell_index + 1}/30"
                ),
            )

            try:
                cache_key = (split_id, seed)

                if cache_key not in split_cache:
                    split_obj = next(
                        split_item
                        for split_item
                        in extractor.build_contract_splits(
                            frame,
                            contract_cfg,
                            seed=seed,
                        )
                        if split_item.split_id
                        == split_id
                    )

                    representative = lineage[
                        (
                            lineage["split_id"]
                            == split_id
                        )
                        & (
                            lineage["seed"]
                            == seed
                        )
                        & (
                            lineage[
                                "transfer_strategy"
                            ]
                            == "ordinary_transfer"
                        )
                    ].iloc[0]

                    representative_job = (
                        extractor.build_manifest_job(
                            representative
                        )
                    )

                    (
                        train_frame,
                        validation_frame,
                        test_frame,
                    ) = extractor._frames(
                        frame,
                        split_obj,
                        contract_cfg,
                        representative_job,
                    )

                    arrays = extractor.prepare_arrays(
                        dataset=dataset,
                        train_frame=train_frame,
                        validation_frame=(
                            validation_frame
                        ),
                        test_frame=test_frame,
                    )

                    preprocessor = arrays[
                        "weather_preprocessor"
                    ]

                    state = preprocessor.state()

                    state_columns = list(
                        state["columns"]
                    )

                    if state_columns != feature_names:
                        raise RuntimeError(
                            "Formal preprocessor feature "
                            "order mismatch"
                        )

                    imputer_statistics = np.asarray(
                        state[
                            "imputer_statistics"
                        ],
                        dtype=np.float64,
                    )
                    scaler_mean = np.asarray(
                        state["scaler_mean"],
                        dtype=np.float64,
                    )
                    scaler_scale = np.asarray(
                        state["scaler_scale"],
                        dtype=np.float64,
                    )

                    train_raw = raw_matrix(
                        train_frame,
                        feature_names,
                    )
                    test_raw = raw_matrix(
                        test_frame,
                        feature_names,
                    )

                    train_raw_imputed = impute_raw(
                        train_raw,
                        imputer_statistics,
                    )
                    test_raw_imputed = impute_raw(
                        test_raw,
                        imputer_statistics,
                    )

                    independently_scaled_train = (
                        transform_raw(
                            train_raw,
                            imputer_statistics,
                            scaler_mean,
                            scaler_scale,
                        )
                    )
                    independently_scaled_test = (
                        transform_raw(
                            test_raw,
                            imputer_statistics,
                            scaler_mean,
                            scaler_scale,
                        )
                    )

                    formal_scaled_train = np.asarray(
                        arrays["train"]["weather"],
                        dtype=np.float32,
                    )
                    formal_scaled_test = np.asarray(
                        arrays["test"]["weather"],
                        dtype=np.float32,
                    )

                    train_transform_error = float(
                        np.max(
                            np.abs(
                                independently_scaled_train
                                - formal_scaled_train
                            )
                        )
                    )
                    test_transform_error = float(
                        np.max(
                            np.abs(
                                independently_scaled_test
                                - formal_scaled_test
                            )
                        )
                    )

                    if max(
                        train_transform_error,
                        test_transform_error,
                    ) > args.scaled_array_atol:
                        raise RuntimeError(
                            "Independent raw→scaled transform "
                            "does not reproduce prepare_arrays"
                        )

                    train_ids = (
                        train_frame[sample_id_column]
                        .astype(str)
                        .tolist()
                    )
                    test_ids = (
                        test_frame[sample_id_column]
                        .astype(str)
                        .tolist()
                    )

                    overlap = (
                        set(train_ids)
                        & set(test_ids)
                    )

                    if overlap:
                        raise RuntimeError(
                            "Formal split contains "
                            "train/test sample overlap"
                        )

                    baseline_raw = np.median(
                        train_raw_imputed,
                        axis=0,
                    )

                    steps = physical_steps(
                        train_raw_imputed,
                        relative_sd=(
                            args
                            .taylor_relative_sd_step
                        ),
                        relative_range_fallback=(
                            args
                            .taylor_relative_range_fallback
                        ),
                        minimum_step=(
                            args.minimum_physical_step
                        ),
                    )

                    split_cache[cache_key] = {
                        "train_frame": train_frame,
                        "test_frame": test_frame,
                        "train_ids": train_ids,
                        "test_ids": test_ids,
                        "train_raw": train_raw_imputed,
                        "test_raw": test_raw_imputed,
                        "scaled_train": formal_scaled_train,
                        "scaled_test": formal_scaled_test,
                        "imputer_statistics": (
                            imputer_statistics
                        ),
                        "scaler_mean": scaler_mean,
                        "scaler_scale": scaler_scale,
                        "preprocessor_state": state,
                        "baseline_raw": baseline_raw,
                        "steps": steps,
                        "train_transform_error": (
                            train_transform_error
                        ),
                        "test_transform_error": (
                            test_transform_error
                        ),
                    }

                split_data = split_cache[
                    cache_key
                ]

                npz_path = Path(
                    str(row["file_path"])
                ).resolve()

                checkpoint_path = Path(
                    str(row["checkpoint_path"])
                ).resolve()

                official_prediction_path = Path(
                    str(
                        row[
                            "official_predictions_path"
                        ]
                    )
                ).resolve()

                for required_path in [
                    npz_path,
                    checkpoint_path,
                    official_prediction_path,
                ]:
                    if not required_path.is_file():
                        raise FileNotFoundError(
                            str(required_path)
                        )

                expected_checkpoint_hash = str(
                    row["checkpoint_sha256"]
                )

                actual_checkpoint_hash = (
                    sha256_file(checkpoint_path)
                )

                if not actual_checkpoint_hash.startswith(
                    expected_checkpoint_hash
                ):
                    raise RuntimeError(
                        "Checkpoint SHA mismatch"
                    )

                with np.load(
                    npz_path,
                    allow_pickle=True,
                ) as archive:
                    npz_train = np.asarray(
                        archive[
                            "raw_input_train"
                        ],
                        dtype=np.float32,
                    )
                    npz_test = np.asarray(
                        archive[
                            "raw_input_test"
                        ],
                        dtype=np.float32,
                    )
                    metadata_train = archive[
                        "metadata_train"
                    ]
                    metadata_test = archive[
                        "metadata_test"
                    ]

                if (
                    npz_train.shape
                    != split_data[
                        "scaled_train"
                    ].shape
                ):
                    raise RuntimeError(
                        "Formal NPZ train shape mismatch"
                    )

                if (
                    npz_test.shape
                    != split_data[
                        "scaled_test"
                    ].shape
                ):
                    raise RuntimeError(
                        "Formal NPZ test shape mismatch"
                    )

                npz_train_error = float(
                    np.max(
                        np.abs(
                            npz_train
                            - split_data[
                                "scaled_train"
                            ]
                        )
                    )
                )

                npz_test_error = float(
                    np.max(
                        np.abs(
                            npz_test
                            - split_data[
                                "scaled_test"
                            ]
                        )
                    )
                )

                if max(
                    npz_train_error,
                    npz_test_error,
                ) > args.scaled_array_atol:
                    raise RuntimeError(
                        "Formal embeddings NPZ raw_input "
                        "does not match rebuilt preprocessing"
                    )

                npz_train_ids = (
                    metadata_train[
                        "sample_id"
                    ].astype(str).tolist()
                )
                npz_test_ids = (
                    metadata_test[
                        "sample_id"
                    ].astype(str).tolist()
                )

                if (
                    npz_train_ids
                    != split_data["train_ids"]
                ):
                    raise RuntimeError(
                        "Train sample ID order mismatch"
                    )

                if (
                    npz_test_ids
                    != split_data["test_ids"]
                ):
                    raise RuntimeError(
                        "Test sample ID order mismatch"
                    )

                model = build_model(
                    checkpoint_path
                )

                replay_scaled = predict_scaled(
                    model,
                    split_data["scaled_test"],
                    feature_names,
                )

                predictor = lambda raw: predict_raw(
                    model,
                    raw,
                    feature_names,
                    split_data[
                        "imputer_statistics"
                    ],
                    split_data["scaler_mean"],
                    split_data["scaler_scale"],
                )

                replay_raw = predictor(
                    split_data["test_raw"]
                )

                raw_scaled_prediction_error = float(
                    np.max(
                        np.abs(
                            replay_raw
                            - replay_scaled
                        )
                    )
                )

                metadata_prediction = np.asarray(
                    metadata_test["prediction"],
                    dtype=np.float64,
                )

                metadata_replay_error = float(
                    np.max(
                        np.abs(
                            replay_raw
                            - metadata_prediction
                        )
                    )
                )

                official_frame = pd.read_csv(
                    official_prediction_path
                )

                prediction_candidates = [
                    "prediction",
                    "y_pred",
                    "predicted_yield",
                    "target_prediction",
                ]

                prediction_column = next(
                    (
                        column
                        for column
                        in prediction_candidates
                        if column
                        in official_frame.columns
                    ),
                    None,
                )

                if prediction_column is None:
                    raise RuntimeError(
                        "Cannot identify prediction "
                        "column in official predictions"
                    )

                if (
                    "sample_id"
                    not in official_frame.columns
                ):
                    raise RuntimeError(
                        "Official predictions lack sample_id"
                    )

                official_lookup = dict(
                    zip(
                        official_frame[
                            "sample_id"
                        ].astype(str),
                        pd.to_numeric(
                            official_frame[
                                prediction_column
                            ],
                            errors="raise",
                        ).astype(float),
                    )
                )

                aligned_official = np.asarray(
                    [
                        official_lookup[sample_id]
                        for sample_id
                        in split_data["test_ids"]
                    ],
                    dtype=np.float64,
                )

                official_replay_error = float(
                    np.max(
                        np.abs(
                            replay_raw
                            - aligned_official
                        )
                    )
                )

                max_replay_error = max(
                    raw_scaled_prediction_error,
                    metadata_replay_error,
                    official_replay_error,
                )

                if max_replay_error > args.replay_atol:
                    raise RuntimeError(
                        "Prediction replay exceeded "
                        f"tolerance: {max_replay_error}"
                    )

                selected_indices = select_rows(
                    metadata_test,
                    args.max_test_rows,
                )

                replay_rows.append({
                    "contract": contract_full,
                    "split_id": split_id,
                    "method": method,
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "feature_names_json": json.dumps(
                        feature_names
                    ),
                    "npz_path": str(npz_path),
                    "npz_sha256": sha256(npz_path),
                    "checkpoint_path": str(
                        checkpoint_path
                    ),
                    "checkpoint_sha256": (
                        actual_checkpoint_hash
                    ),
                    "official_predictions_path": str(
                        official_prediction_path
                    ),
                    "official_predictions_sha256": (
                        sha256(
                            official_prediction_path
                        )
                    ),
                    "n_train": len(
                        split_data["train_ids"]
                    ),
                    "n_test": len(
                        split_data["test_ids"]
                    ),
                    "selected_test_rows": len(
                        selected_indices
                    ),
                    "prepare_arrays_train_reproduction_error": (
                        split_data[
                            "train_transform_error"
                        ]
                    ),
                    "prepare_arrays_test_reproduction_error": (
                        split_data[
                            "test_transform_error"
                        ]
                    ),
                    "npz_train_scaled_array_error": (
                        npz_train_error
                    ),
                    "npz_test_scaled_array_error": (
                        npz_test_error
                    ),
                    "raw_vs_scaled_prediction_error": (
                        raw_scaled_prediction_error
                    ),
                    "metadata_replay_error": (
                        metadata_replay_error
                    ),
                    "official_replay_error": (
                        official_replay_error
                    ),
                    "max_replay_error": (
                        max_replay_error
                    ),
                    "status": "REPLAY_PASS",
                })

                loaded_cells.append({
                    "contract": contract_full,
                    "split_id": split_id,
                    "method": method,
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "model": model,
                    "predictor": predictor,
                    "split_data": split_data,
                    "metadata_test": metadata_test,
                    "selected_indices": (
                        selected_indices
                    ),
                    "npz_path": npz_path,
                    "checkpoint_path": (
                        checkpoint_path
                    ),
                })

                pd.DataFrame(
                    replay_rows
                ).to_csv(
                    output
                    / "checkpoint_forward_replay.csv",
                    index=False,
                )

                progress.update(
                    replayed_cells=len(
                        replay_rows
                    ),
                    max_replay_error=max(
                        item["max_replay_error"]
                        for item in replay_rows
                    ),
                    max_scaled_array_error=max(
                        max(
                            item[
                                "npz_train_scaled_array_error"
                            ],
                            item[
                                "npz_test_scaled_array_error"
                            ],
                        )
                        for item in replay_rows
                    ),
                )

            except Exception as exc:
                failure = {
                    "phase": "REPLAY_GATE",
                    "contract": contract_full,
                    "method": method,
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }

                failures.append(failure)

                pd.DataFrame(failures).to_csv(
                    output / "failure_ledger.csv",
                    index=False,
                )

                progress.update(
                    status="BLOCKED",
                    phase="REPLAY_GATE_FAILED",
                    failed_cells=len(failures),
                    last_message=str(exc),
                )

                raise

        if len(replay_rows) != 30:
            raise RuntimeError(
                "30/30 replay gate was not satisfied"
            )

        preprocessor_rows = []

        for (
            split_id,
            seed,
        ), split_data in sorted(
            split_cache.items()
        ):
            state = split_data[
                "preprocessor_state"
            ]

            for index, feature_name in enumerate(
                feature_names
            ):
                preprocessor_rows.append({
                    "split_id": split_id,
                    "contract": (
                        f"{split_id}_complete"
                    ),
                    "seed": seed,
                    "feature_index": index,
                    "feature_name": feature_name,
                    "imputer_median": float(
                        state[
                            "imputer_statistics"
                        ][index]
                    ),
                    "scaler_mean": float(
                        state[
                            "scaler_mean"
                        ][index]
                    ),
                    "scaler_scale": float(
                        state[
                            "scaler_scale"
                        ][index]
                    ),
                    "taylor_physical_step": float(
                        split_data[
                            "steps"
                        ][index]
                    ),
                    "train_raw_median": float(
                        split_data[
                            "baseline_raw"
                        ][index]
                    ),
                    "train_raw_min": float(
                        np.min(
                            split_data[
                                "train_raw"
                            ][:, index]
                        )
                    ),
                    "train_raw_max": float(
                        np.max(
                            split_data[
                                "train_raw"
                            ][:, index]
                        )
                    ),
                    "train_raw_std": float(
                        np.std(
                            split_data[
                                "train_raw"
                            ][:, index]
                        )
                    ),
                })

        pd.DataFrame(
            preprocessor_rows
        ).to_csv(
            output
            / "formal_preprocessor_parameters.csv",
            index=False,
        )

        total_samples = sum(
            len(cell["selected_indices"])
            for cell in loaded_cells
        )

        progress.update(
            phase="ATTRIBUTION",
            total_samples=total_samples,
            completed_samples=0,
            last_message=(
                "30/30 replay passed; starting "
                "raw-weather attribution"
            ),
        )

        attribution_rows = []
        completed_samples = 0

        for cell_index, cell in enumerate(
            loaded_cells
        ):
            split_data = cell["split_data"]
            predictor = cell["predictor"]
            metadata = cell["metadata_test"]

            generator = np.random.default_rng(
                args.random_seed
                + cell["seed"]
                + cell_index * 1009
            )

            background_count = min(
                args.background_rows,
                len(split_data["train_raw"]),
            )

            background_indices = (
                generator.choice(
                    len(split_data["train_raw"]),
                    size=background_count,
                    replace=False,
                )
            )

            background = split_data[
                "train_raw"
            ][background_indices]

            baseline = split_data[
                "baseline_raw"
            ]

            steps = split_data["steps"]

            for sample_position, test_index in enumerate(
                cell["selected_indices"]
            ):
                sample_id = str(
                    metadata["sample_id"][
                        test_index
                    ]
                )

                sample_raw = split_data[
                    "test_raw"
                ][test_index]

                progress.update(
                    current_cell=(
                        f"{cell['contract']}/"
                        f"{cell['method']}/"
                        f"{cell['seed']}"
                    ),
                    current_sample=sample_id,
                    last_message=(
                        f"Attribution cell "
                        f"{cell_index + 1}/30, "
                        f"sample "
                        f"{sample_position + 1}/"
                        f"{len(cell['selected_indices'])}"
                    ),
                )

                taylor = taylor_attribution(
                    predictor=predictor,
                    sample_raw=sample_raw,
                    baseline_raw=baseline,
                    steps=steps,
                )

                shap = permutation_shap(
                    predictor=predictor,
                    sample_raw=sample_raw,
                    background_raw=background,
                    permutations=(
                        args.shap_permutations
                    ),
                    seed=(
                        args.random_seed
                        + cell["seed"] * 100003
                        + cell_index * 7919
                        + int(test_index) * 31
                    ),
                )

                prediction_difference = abs(
                    taylor["sample_prediction"]
                    - shap["sample_prediction"]
                )

                if prediction_difference > args.replay_atol:
                    raise RuntimeError(
                        "SHAP and Taylor predictor "
                        "outputs disagree"
                    )

                for feature_index, feature_name in enumerate(
                    feature_names
                ):
                    attribution_rows.append({
                        "contract": cell["contract"],
                        "split_id": cell["split_id"],
                        "method": cell["method"],
                        "seed": cell["seed"],
                        "candidate_id": (
                            cell["candidate_id"]
                        ),
                        "sample_id": sample_id,
                        "sample_index": int(
                            test_index
                        ),
                        "feature_index": (
                            feature_index
                        ),
                        "feature_name": (
                            feature_name
                        ),
                        "feature_value_raw": float(
                            sample_raw[
                                feature_index
                            ]
                        ),
                        "train_median_baseline_raw": float(
                            baseline[
                                feature_index
                            ]
                        ),
                        "raw_displacement": float(
                            sample_raw[
                                feature_index
                            ]
                            - baseline[
                                feature_index
                            ]
                        ),
                        "physical_finite_difference_step": float(
                            steps[
                                feature_index
                            ]
                        ),
                        "first_derivative_per_raw_unit": float(
                            taylor[
                                "gradient"
                            ][feature_index]
                        ),
                        "diagonal_second_derivative_per_raw_unit_squared": float(
                            taylor[
                                "diagonal_hessian"
                            ][feature_index]
                        ),
                        "taylor_first_contribution": float(
                            taylor[
                                "first_contribution"
                            ][feature_index]
                        ),
                        "taylor_second_diagonal_contribution": float(
                            taylor[
                                "second_diagonal_contribution"
                            ][feature_index]
                        ),
                        "permutation_shap": float(
                            shap["values"][
                                feature_index
                            ]
                        ),
                        "prediction": float(
                            shap["sample_prediction"]
                        ),
                        "target": float(
                            metadata["target"][
                                test_index
                            ]
                        ),
                        "absolute_error": float(
                            metadata["abs_error"][
                                test_index
                            ]
                        ),
                        "shap_expected_value": float(
                            shap["expected_value"]
                        ),
                        "shap_efficiency_residual": float(
                            shap[
                                "efficiency_residual"
                            ]
                        ),
                        "taylor_first_reconstruction": float(
                            taylor[
                                "first_reconstruction"
                            ]
                        ),
                        "taylor_second_reconstruction": float(
                            taylor[
                                "second_reconstruction"
                            ]
                        ),
                        "taylor_first_residual": float(
                            taylor[
                                "first_residual"
                            ]
                        ),
                        "taylor_second_residual": float(
                            taylor[
                                "second_residual"
                            ]
                        ),
                    })

                completed_samples += 1

                if (
                    completed_samples % 5 == 0
                    or completed_samples
                    == total_samples
                ):
                    pd.DataFrame(
                        attribution_rows
                    ).to_csv(
                        output
                        / "sample_level_raw_shap_taylor.csv",
                        index=False,
                    )

                progress.update(
                    completed_samples=completed_samples,
                )

            progress.update(
                attributed_cells=cell_index + 1,
            )

        attribution = pd.DataFrame(
            attribution_rows
        )

        feature_seed = (
            attribution.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                    "feature_index",
                    "feature_name",
                ],
                dropna=False,
            )
            .agg(
                n_samples=(
                    "sample_id",
                    "nunique",
                ),
                mean_shap=(
                    "permutation_shap",
                    "mean",
                ),
                mean_abs_shap=(
                    "permutation_shap",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_taylor_first=(
                    "taylor_first_contribution",
                    "mean",
                ),
                mean_abs_taylor_first=(
                    "taylor_first_contribution",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_taylor_second_diagonal=(
                    "taylor_second_diagonal_contribution",
                    "mean",
                ),
                mean_abs_taylor_second_diagonal=(
                    "taylor_second_diagonal_contribution",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_derivative_per_raw_unit=(
                    "first_derivative_per_raw_unit",
                    "mean",
                ),
                mean_abs_derivative_per_raw_unit=(
                    "first_derivative_per_raw_unit",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_absolute_error=(
                    "absolute_error",
                    "mean",
                ),
                mean_abs_shap_efficiency_residual=(
                    "shap_efficiency_residual",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_abs_taylor_first_residual=(
                    "taylor_first_residual",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
                mean_abs_taylor_second_residual=(
                    "taylor_second_residual",
                    lambda values: float(
                        np.mean(np.abs(values))
                    ),
                ),
            )
            .reset_index()
        )

        feature_seed.to_csv(
            output / "feature_level_by_seed.csv",
            index=False,
        )

        supervised = feature_seed[
            feature_seed["method"] == "supervised"
        ].copy()

        paired = feature_seed[
            feature_seed["method"] != "supervised"
        ].merge(
            supervised[
                [
                    "contract",
                    "seed",
                    "feature_index",
                    "feature_name",
                    "mean_shap",
                    "mean_abs_shap",
                    "mean_taylor_first",
                    "mean_abs_taylor_first",
                    "mean_taylor_second_diagonal",
                    "mean_abs_taylor_second_diagonal",
                    "mean_derivative_per_raw_unit",
                    "mean_abs_derivative_per_raw_unit",
                    "mean_absolute_error",
                ]
            ],
            on=[
                "contract",
                "seed",
                "feature_index",
                "feature_name",
            ],
            how="left",
            validate="many_to_one",
            suffixes=("_method", "_supervised"),
        )

        metrics = [
            "mean_shap",
            "mean_abs_shap",
            "mean_taylor_first",
            "mean_abs_taylor_first",
            "mean_taylor_second_diagonal",
            "mean_abs_taylor_second_diagonal",
            "mean_derivative_per_raw_unit",
            "mean_abs_derivative_per_raw_unit",
            "mean_absolute_error",
        ]

        for metric in metrics:
            paired[f"delta_{metric}"] = (
                paired[f"{metric}_method"]
                - paired[
                    f"{metric}_supervised"
                ]
            )

        paired.to_csv(
            output
            / "kd_vs_supervised_by_seed.csv",
            index=False,
        )

        summary_rows = []

        for keys, group in paired.groupby(
            [
                "contract",
                "method",
                "feature_index",
                "feature_name",
            ],
            dropna=False,
        ):
            (
                contract,
                method,
                feature_index,
                feature_name,
            ) = keys

            result = {
                "contract": contract,
                "method": method,
                "feature_index": int(
                    feature_index
                ),
                "feature_name": feature_name,
                "seed_count": int(
                    group["seed"].nunique()
                ),
            }

            for metric in metrics:
                column = f"delta_{metric}"

                values = group[column].to_numpy(
                    dtype=np.float64
                )

                mean, low, high = (
                    bootstrap_mean_ci(
                        values,
                        repetitions=(
                            args.bootstrap_repetitions
                        ),
                        seed=(
                            args.random_seed
                            + int(feature_index)
                            + ROUTES.index(method)
                            * 1009
                            + sum(map(ord, metric))
                        ),
                    )
                )

                result[
                    f"{column}_mean"
                ] = mean
                result[
                    f"{column}_ci_low"
                ] = low
                result[
                    f"{column}_ci_high"
                ] = high
                result[
                    f"{column}_positive_seed_count"
                ] = int(
                    np.sum(values > 0)
                )
                result[
                    f"{column}_negative_seed_count"
                ] = int(
                    np.sum(values < 0)
                )

            summary_rows.append(result)

        summary = pd.DataFrame(
            summary_rows
        )

        summary.to_csv(
            output
            / "kd_vs_supervised_summary.csv",
            index=False,
        )

        quality = (
            attribution.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                    "candidate_id",
                    "sample_id",
                ],
                dropna=False,
            )
            .agg(
                prediction=(
                    "prediction",
                    "first",
                ),
                target=(
                    "target",
                    "first",
                ),
                absolute_error=(
                    "absolute_error",
                    "first",
                ),
                shap_efficiency_residual=(
                    "shap_efficiency_residual",
                    "first",
                ),
                taylor_first_residual=(
                    "taylor_first_residual",
                    "first",
                ),
                taylor_second_residual=(
                    "taylor_second_residual",
                    "first",
                ),
            )
            .reset_index()
        )

        quality.to_csv(
            output
            / "attribution_quality.csv",
            index=False,
        )

        provenance = {
            "status": "COMPLETED",
            "analysis_scope": (
                "RAW_WEATHER_PHYSICAL_VALUE_SPACE"
            ),
            "formal_cell_count": 30,
            "replay_pass_count": len(
                replay_rows
            ),
            "attributed_cell_count": 30,
            "selected_sample_count": (
                total_samples
            ),
            "sample_feature_row_count": (
                len(attribution)
            ),
            "feature_names": feature_names,
            "preprocessing": {
                "imputation": (
                    "training-split median"
                ),
                "scaling": (
                    "training-split StandardScaler"
                ),
                "raw_to_scaled_formula": (
                    "(median-imputed raw value "
                    "- scaler mean) / scaler scale"
                ),
                "preprocessor_rebuilt_per": (
                    "contract and seed"
                ),
            },
            "shap": {
                "method": (
                    "permutation Monte Carlo SHAP estimate"
                ),
                "background": (
                    "training-split raw weather rows, "
                    "median-imputed using formal training "
                    "preprocessor"
                ),
                "permutations": (
                    args.shap_permutations
                ),
                "background_rows": (
                    args.background_rows
                ),
            },
            "taylor": {
                "first_order": (
                    "central finite difference in raw "
                    "weather value space"
                ),
                "second_order": (
                    "diagonal Hessian only; no cross-feature "
                    "interaction terms"
                ),
                "step_rule": {
                    "primary": (
                        args.taylor_relative_sd_step
                    ),
                    "description": (
                        "relative fraction of training-split "
                        "raw feature standard deviation"
                    ),
                    "range_fallback": (
                        args
                        .taylor_relative_range_fallback
                    ),
                    "minimum": (
                        args.minimum_physical_step
                    ),
                },
            },
            "scientific_boundaries": [
                "Attributions describe model behaviour, not causal effects.",
                "GROUP and SPATIAL are analysed separately.",
                "Taylor second order contains diagonal curvature only.",
                "Permutation SHAP is a Monte Carlo estimate.",
                "Feature units follow the units stored in the formal dataset.",
            ],
            "inputs": {
                "embedding_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256(
                        manifest_path
                    ),
                },
                "checkpoint_lineage": {
                    "path": str(lineage_path),
                    "sha256": sha256(
                        lineage_path
                    ),
                },
                "extract_script": {
                    "path": str(extract_path),
                    "sha256": sha256(
                        extract_path
                    ),
                },
                "config": {
                    "path": str(config_path),
                    "sha256": sha256(
                        config_path
                    ),
                },
                "schema": {
                    "path": str(schema_path),
                    "sha256": sha256(
                        schema_path
                    ),
                },
            },
            "software": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "audit_worktree_commit": (
                    git_commit(worktree)
                ),
                "formal_code_commit": (
                    git_commit(formal_code)
                ),
            },
            "parameters": vars(args),
        }

        atomic_json(
            output / "analysis_manifest.json",
            provenance,
        )

        if not (
            output / "failure_ledger.csv"
        ).exists():
            pd.DataFrame(
                columns=[
                    "phase",
                    "contract",
                    "method",
                    "seed",
                    "candidate_id",
                    "exception_type",
                    "message",
                    "traceback",
                ]
            ).to_csv(
                output / "failure_ledger.csv",
                index=False,
            )

        interpretation = """# Raw-weather SHAP + Taylor analysis

## Execution status

- All 30 formal checkpoints passed exact prediction replay.
- Formal preprocessing was rebuilt independently for each contract and seed.
- The rebuilt raw-to-scaled transformation was checked against both
  `prepare_arrays()` and the formal `embeddings.npz` arrays.
- Attribution was performed in the raw weather-value space.

## Interpretation

Permutation SHAP estimates the contribution of replacing training-background
feature values with each sample's observed feature values. Taylor first-order
terms describe local sensitivity around each sample, expressed per raw feature
unit. Taylor second-order values contain diagonal curvature only and do not
include cross-feature Hessian interactions.

These quantities describe model behaviour. They are not causal agronomic
effects. GROUP and SPATIAL results must remain separate.
"""

        (
            output / "scientific_interpretation.md"
        ).write_text(interpretation)

        progress.update(
            status="COMPLETED",
            phase="COMPLETED",
            attributed_cells=30,
            completed_samples=total_samples,
            current_cell=None,
            current_sample=None,
            eta_seconds=0,
            last_message=(
                "Raw-weather SHAP + Taylor "
                "analysis completed"
            ),
        )

        print("=" * 88)
        print("RAW_WEATHER_SHAP_TAYLOR_COMPLETE")
        print("Replay cells: 30/30")
        print("Attributed cells: 30/30")
        print("Selected samples:", total_samples)
        print("Feature rows:", len(attribution))
        print("Output:", output)
        print("=" * 88)

    except Exception as exc:
        failure = {
            "phase": "FATAL",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

        failures.append(failure)

        pd.DataFrame(failures).to_csv(
            output / "failure_ledger.csv",
            index=False,
        )

        atomic_json(
            output / "fatal_error.json",
            failure,
        )

        progress.update(
            status="FAILED",
            phase="FAILED",
            failed_cells=len(failures),
            last_message=str(exc),
        )

        raise


if __name__ == "__main__":
    main()
