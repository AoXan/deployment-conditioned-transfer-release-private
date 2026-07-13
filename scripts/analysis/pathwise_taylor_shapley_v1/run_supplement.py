from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

FORMAL_CODE = Path(
    os.environ.get("FORMAL_CODE_ROOT", Path(__file__).resolve().parents[3])
).resolve()

if str(FORMAL_CODE) in sys.path:
    sys.path.remove(str(FORMAL_CODE))
sys.path.insert(0, str(FORMAL_CODE))

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
}

METHOD_LABELS = {
    "supervised": "Supervised",
    "prediction_kd": "Prediction KD",
    "combined_kd": "Combined KD",
    "representation_kd": "Representation KD",
    "missing_aware": "Missing-aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--worktree", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--base-script", required=True)
    parser.add_argument("--extract-script", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--hessian-path-nodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--hessian-samples-per-cell",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--relative-fd-step",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--minimum-fd-step",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--replay-atol",
        type=float,
        default=5e-6,
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


def atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    temporary.replace(path)


def import_module_from_path(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def build_model(
    checkpoint: Path,
) -> UniversalWeatherRegressorV2:
    model = UniversalWeatherRegressorV2(
        encoder=UniversalWeatherEncoderV2()
    )

    load_checkpoint(
        path=checkpoint,
        model=model,
        map_location="cpu",
        strict=True,
    )

    model.eval()
    return model


def predict_scaled(
    model: UniversalWeatherRegressorV2,
    scaled: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    values = np.asarray(
        scaled,
        dtype=np.float32,
    )

    if values.ndim == 1:
        values = values[None, :]

    batch = build_weather_tokens_v2(
        values,
        feature_names,
        device="cpu",
    )

    with torch.no_grad():
        prediction, _ = model(batch)

    return (
        prediction.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(-1)
    )


def transform_raw(
    raw: np.ndarray,
    imputer: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        raw,
        dtype=np.float64,
    ).copy()

    if values.ndim == 1:
        values = values[None, :]

    missing = ~np.isfinite(values)

    if missing.any():
        rows, columns = np.where(missing)
        values[rows, columns] = (
            imputer[columns]
        )

    return (
        (values - mean[None, :])
        / scale[None, :]
    ).astype(np.float32)


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
    values: np.ndarray,
    imputer: np.ndarray,
) -> np.ndarray:
    result = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    missing = ~np.isfinite(result)

    if missing.any():
        rows, columns = np.where(missing)
        result[rows, columns] = (
            imputer[columns]
        )

    return result


def finite_difference_gradient(
    predictor: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    dimension = len(point)
    perturbations = []

    for index in range(dimension):
        plus = point.copy()
        minus = point.copy()

        plus[index] += steps[index]
        minus[index] -= steps[index]

        perturbations.extend(
            [plus, minus]
        )

    predictions = predictor(
        np.stack(perturbations)
    )

    gradient = np.empty(
        dimension,
        dtype=np.float64,
    )

    for index in range(dimension):
        gradient[index] = (
            predictions[2 * index]
            - predictions[2 * index + 1]
        ) / (2.0 * steps[index])

    return gradient


def finite_difference_hessian(
    predictor: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    dimension = len(point)

    centre_prediction = float(
        predictor(point[None, :])[0]
    )

    hessian = np.zeros(
        (dimension, dimension),
        dtype=np.float64,
    )

    diagonal_points = []

    for index in range(dimension):
        plus = point.copy()
        minus = point.copy()

        plus[index] += steps[index]
        minus[index] -= steps[index]

        diagonal_points.extend(
            [plus, minus]
        )

    diagonal_predictions = predictor(
        np.stack(diagonal_points)
    )

    for index in range(dimension):
        hessian[index, index] = (
            diagonal_predictions[2 * index]
            - 2.0 * centre_prediction
            + diagonal_predictions[
                2 * index + 1
            ]
        ) / (steps[index] ** 2)

    pairs = []
    pair_points = []

    for first in range(dimension):
        for second in range(
            first + 1,
            dimension,
        ):
            pp = point.copy()
            pm = point.copy()
            mp = point.copy()
            mm = point.copy()

            pp[first] += steps[first]
            pp[second] += steps[second]

            pm[first] += steps[first]
            pm[second] -= steps[second]

            mp[first] -= steps[first]
            mp[second] += steps[second]

            mm[first] -= steps[first]
            mm[second] -= steps[second]

            pairs.append((first, second))
            pair_points.extend(
                [pp, pm, mp, mm]
            )

    pair_predictions = predictor(
        np.stack(pair_points)
    )

    for pair_index, (
        first,
        second,
    ) in enumerate(pairs):
        start = 4 * pair_index

        value = (
            pair_predictions[start]
            - pair_predictions[start + 1]
            - pair_predictions[start + 2]
            + pair_predictions[start + 3]
        ) / (
            4.0
            * steps[first]
            * steps[second]
        )

        hessian[first, second] = value
        hessian[second, first] = value

    return hessian


def integrated_gradients(
    predictor: Callable[[np.ndarray], np.ndarray],
    baseline: np.ndarray,
    sample: np.ndarray,
    steps: np.ndarray,
    integration_steps: int,
) -> dict[str, Any]:
    displacement = sample - baseline

    alphas = (
        np.arange(
            integration_steps,
            dtype=np.float64,
        )
        + 0.5
    ) / integration_steps

    gradients = []

    for alpha in alphas:
        point = (
            baseline
            + alpha * displacement
        )

        gradients.append(
            finite_difference_gradient(
                predictor,
                point,
                steps,
            )
        )

    mean_gradient = np.mean(
        np.stack(gradients),
        axis=0,
    )

    contribution = (
        displacement * mean_gradient
    )

    baseline_prediction = float(
        predictor(
            baseline[None, :]
        )[0]
    )

    sample_prediction = float(
        predictor(
            sample[None, :]
        )[0]
    )

    reconstruction = (
        baseline_prediction
        + contribution.sum()
    )

    return {
        "contribution": contribution,
        "mean_gradient": mean_gradient,
        "baseline_prediction": (
            baseline_prediction
        ),
        "sample_prediction": (
            sample_prediction
        ),
        "reconstruction": reconstruction,
        "residual": (
            sample_prediction
            - reconstruction
        ),
    }


def pathwise_taylor_shapley(
    predictor: Callable[[np.ndarray], np.ndarray],
    baseline: np.ndarray,
    sample: np.ndarray,
    steps: np.ndarray,
    path_nodes: int,
) -> dict[str, Any]:
    displacement = sample - baseline
    dimension = len(sample)

    baseline_prediction = float(
        predictor(
            baseline[None, :]
        )[0]
    )

    sample_prediction = float(
        predictor(
            sample[None, :]
        )[0]
    )

    baseline_gradient = (
        finite_difference_gradient(
            predictor,
            baseline,
            steps,
        )
    )

    nodes, weights = (
        np.polynomial.legendre.leggauss(
            path_nodes
        )
    )

    alphas = 0.5 * (
        nodes + 1.0
    )
    weights = 0.5 * weights

    integrated_hessian = np.zeros(
        (dimension, dimension),
        dtype=np.float64,
    )

    for alpha, weight in zip(
        alphas,
        weights,
    ):
        point = (
            baseline
            + alpha * displacement
        )

        hessian = (
            finite_difference_hessian(
                predictor,
                point,
                steps,
            )
        )

        integrated_hessian += (
            weight
            * (1.0 - alpha)
            * hessian
        )

    linear = (
        baseline_gradient
        * displacement
    )

    diagonal = (
        np.diag(
            integrated_hessian
        )
        * displacement ** 2
    )

    pair_matrix = np.zeros(
        (dimension, dimension),
        dtype=np.float64,
    )

    feature_interaction_share = np.zeros(
        dimension,
        dtype=np.float64,
    )

    for first in range(dimension):
        for second in range(
            first + 1,
            dimension,
        ):
            pair_total = (
                2.0
                * integrated_hessian[
                    first,
                    second
                ]
                * displacement[first]
                * displacement[second]
            )

            pair_matrix[
                first,
                second
            ] = pair_total
            pair_matrix[
                second,
                first
            ] = pair_total

            feature_interaction_share[
                first
            ] += 0.5 * pair_total

            feature_interaction_share[
                second
            ] += 0.5 * pair_total

    feature_contribution = (
        linear
        + diagonal
        + feature_interaction_share
    )

    reconstruction = (
        baseline_prediction
        + feature_contribution.sum()
    )

    return {
        "baseline_prediction": (
            baseline_prediction
        ),
        "sample_prediction": (
            sample_prediction
        ),
        "baseline_gradient": (
            baseline_gradient
        ),
        "integrated_hessian": (
            integrated_hessian
        ),
        "linear_contribution": linear,
        "diagonal_curvature": diagonal,
        "interaction_share": (
            feature_interaction_share
        ),
        "feature_contribution": (
            feature_contribution
        ),
        "pair_matrix": pair_matrix,
        "reconstruction": reconstruction,
        "residual": (
            sample_prediction
            - reconstruction
        ),
    }


def choose_hessian_samples(
    shap_cell: pd.DataFrame,
    maximum: int,
) -> list[str]:
    ranked = (
        shap_cell.groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            total_abs_shap=(
                "permutation_shap",
                lambda values: float(
                    np.abs(values).sum()
                ),
            ),
            absolute_error=(
                "absolute_error",
                "first",
            ),
        )
        .sort_values(
            [
                "total_abs_shap",
                "absolute_error",
                "sample_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )

    if len(ranked) <= maximum:
        return ranked[
            "sample_id"
        ].astype(str).tolist()

    positions = np.linspace(
        0,
        len(ranked) - 1,
        maximum,
    )

    return (
        ranked.iloc[
            np.rint(positions).astype(int)
        ]["sample_id"]
        .astype(str)
        .tolist()
    )


def main() -> None:
    args = parse_args()

    worktree = Path(
        args.worktree
    ).resolve()
    base_output = Path(
        args.base_output
    ).resolve()
    base_script = Path(
        args.base_script
    ).resolve()
    extract_script = Path(
        args.extract_script
    ).resolve()
    manifest_path = Path(
        args.manifest
    ).resolve()
    lineage_path = Path(
        args.lineage
    ).resolve()
    output = Path(
        args.output
    ).resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress_path = (
        output / "progress.json"
    )

    progress = {
        "status": "RUNNING",
        "phase": "READINESS",
        "total_cells": 30,
        "completed_cells": 0,
        "completed_ig_samples": 0,
        "completed_hessian_samples": 0,
        "current_cell": None,
        "current_sample": None,
        "last_message": "",
        "started_unix": time.time(),
    }

    atomic_json(
        progress_path,
        progress,
    )

    try:
        base = import_module_from_path(
            "accepted_raw_attribution_base",
            base_script,
        )

        extractor = import_module_from_path(
            "accepted_embedding_extractor_supplement",
            extract_script,
        )

        base_manifest = json.loads(
            (
                base_output
                / "analysis_manifest.json"
            ).read_text()
        )

        if (
            base_manifest.get("status")
            != "COMPLETED"
        ):
            raise RuntimeError(
                "Base SHAP analysis is not completed"
            )

        base_shap = pd.read_csv(
            base_output
            / "sample_level_raw_shap_taylor.csv"
        )

        base_replay = pd.read_csv(
            base_output
            / "checkpoint_forward_replay.csv"
        )

        if len(base_replay) != 30:
            raise RuntimeError(
                "Base replay matrix is not 30 cells"
            )

        cells = base.load_cells(
            manifest_path,
            lineage_path,
        )

        config_path = (
            extractor.FORMAL_WORKTREE
            / "configs/"
            "universal_weather_full_campaign_v2_remediation_v1.yaml"
        )

        schema_path = (
            extractor.FORMAL_WORKTREE
            / "schemas/"
            "universal_weather_full_campaign_v2_remediation_v1.schema.json"
        )

        config = extractor.load_config(
            config_path,
            schema_path,
        )

        contract_cfg = config[
            "datasets"
        ][extractor.DATASET_ID]

        frame, data_path = (
            extractor.load_frame(
                contract_cfg,
                extractor.FORMAL_WORKTREE,
            )
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

        sample_id_column = contract_cfg[
            "sample_id_column"
        ]

        lineage = pd.read_csv(
            lineage_path
        )

        split_cache: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}

        ig_rows = []
        taylor_rows = []
        interaction_rows = []
        quality_rows = []

        for cell_index, cell in cells.iterrows():
            contract = str(
                cell["contract"]
            )
            split_id = (
                CONTRACT_ALIASES[
                    contract
                ]
            )
            method = str(
                cell["method"]
            )
            seed = int(
                cell["seed"]
            )
            candidate_id = str(
                cell["candidate_id"]
            )

            cell_key = (
                f"{contract}/{method}/{seed}"
            )

            progress.update({
                "phase": "ATTRIBUTION",
                "current_cell": cell_key,
                "last_message": (
                    f"Processing cell "
                    f"{cell_index + 1}/30"
                ),
            })

            atomic_json(
                progress_path,
                progress,
            )

            cache_key = (
                split_id,
                seed,
            )

            if cache_key not in split_cache:
                split = next(
                    item
                    for item
                    in extractor.build_contract_splits(
                        frame,
                        contract_cfg,
                        seed=seed,
                    )
                    if item.split_id
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

                job = (
                    extractor
                    .build_manifest_job(
                        representative
                    )
                )

                (
                    train_frame,
                    validation_frame,
                    test_frame,
                ) = extractor._frames(
                    frame,
                    split,
                    contract_cfg,
                    job,
                )

                arrays = (
                    extractor.prepare_arrays(
                        dataset=dataset,
                        train_frame=train_frame,
                        validation_frame=(
                            validation_frame
                        ),
                        test_frame=test_frame,
                    )
                )

                state = (
                    arrays[
                        "weather_preprocessor"
                    ].state()
                )

                imputer = np.asarray(
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

                train_raw = impute_raw(
                    raw_matrix(
                        train_frame,
                        feature_names,
                    ),
                    imputer,
                )

                test_raw = impute_raw(
                    raw_matrix(
                        test_frame,
                        feature_names,
                    ),
                    imputer,
                )

                test_ids = (
                    test_frame[
                        sample_id_column
                    ]
                    .astype(str)
                    .tolist()
                )

                baseline = np.median(
                    train_raw,
                    axis=0,
                )

                standard_deviation = np.std(
                    train_raw,
                    axis=0,
                )

                feature_range = (
                    np.max(
                        train_raw,
                        axis=0,
                    )
                    - np.min(
                        train_raw,
                        axis=0,
                    )
                )

                fd_steps = (
                    standard_deviation
                    * args.relative_fd_step
                )

                fallback = (
                    feature_range
                    * 0.0005
                )

                fd_steps = np.where(
                    fd_steps
                    > args.minimum_fd_step,
                    fd_steps,
                    fallback,
                )

                fd_steps = np.where(
                    fd_steps
                    > args.minimum_fd_step,
                    fd_steps,
                    args.minimum_fd_step,
                )

                split_cache[
                    cache_key
                ] = {
                    "train_raw": train_raw,
                    "test_raw": test_raw,
                    "test_ids": test_ids,
                    "baseline": baseline,
                    "fd_steps": fd_steps,
                    "imputer": imputer,
                    "scaler_mean": (
                        scaler_mean
                    ),
                    "scaler_scale": (
                        scaler_scale
                    ),
                }

            split_data = (
                split_cache[
                    cache_key
                ]
            )

            checkpoint = Path(
                str(
                    cell[
                        "checkpoint_path"
                    ]
                )
            ).resolve()

            expected_hash = str(
                cell[
                    "checkpoint_sha256"
                ]
            )

            actual_hash = (
                sha256_file(checkpoint)
            )

            if not actual_hash.startswith(
                expected_hash
            ):
                raise RuntimeError(
                    "Checkpoint hash mismatch"
                )

            model = build_model(
                checkpoint
            )

            def predictor(
                raw_values: np.ndarray,
            ) -> np.ndarray:
                scaled = transform_raw(
                    raw_values,
                    split_data["imputer"],
                    split_data[
                        "scaler_mean"
                    ],
                    split_data[
                        "scaler_scale"
                    ],
                )

                return predict_scaled(
                    model,
                    scaled,
                    feature_names,
                )

            cell_shap = base_shap[
                (
                    base_shap["contract"]
                    == contract
                )
                & (
                    base_shap["method"]
                    == method
                )
                & (
                    base_shap["seed"]
                    == seed
                )
            ].copy()

            sample_ids = (
                cell_shap[
                    "sample_id"
                ]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            test_lookup = {
                sample_id: index
                for index, sample_id
                in enumerate(
                    split_data["test_ids"]
                )
            }

            hessian_sample_ids = set(
                choose_hessian_samples(
                    cell_shap,
                    args
                    .hessian_samples_per_cell,
                )
            )

            for sample_id in sample_ids:
                if sample_id not in test_lookup:
                    raise RuntimeError(
                        f"Sample ID not found: "
                        f"{sample_id}"
                    )

                test_index = test_lookup[
                    sample_id
                ]

                sample = split_data[
                    "test_raw"
                ][test_index]

                baseline = split_data[
                    "baseline"
                ]

                fd_steps = split_data[
                    "fd_steps"
                ]

                progress.update({
                    "current_sample": (
                        sample_id
                    ),
                    "last_message": (
                        "Integrated gradients"
                    ),
                })

                atomic_json(
                    progress_path,
                    progress,
                )

                ig = integrated_gradients(
                    predictor,
                    baseline,
                    sample,
                    fd_steps,
                    args.ig_steps,
                )

                shap_sample = (
                    cell_shap[
                        cell_shap[
                            "sample_id"
                        ].astype(str)
                        == sample_id
                    ]
                    .set_index(
                        "feature_name"
                    )
                )

                for feature_index, feature_name in enumerate(
                    feature_names
                ):
                    ig_rows.append({
                        "contract": contract,
                        "method": method,
                        "seed": seed,
                        "candidate_id": (
                            candidate_id
                        ),
                        "sample_id": (
                            sample_id
                        ),
                        "feature_index": (
                            feature_index
                        ),
                        "feature_name": (
                            feature_name
                        ),
                        "feature_value_raw": float(
                            sample[
                                feature_index
                            ]
                        ),
                        "baseline_raw": float(
                            baseline[
                                feature_index
                            ]
                        ),
                        "integrated_gradient": float(
                            ig[
                                "contribution"
                            ][feature_index]
                        ),
                        "mean_path_gradient": float(
                            ig[
                                "mean_gradient"
                            ][feature_index]
                        ),
                        "permutation_shap": float(
                            shap_sample.loc[
                                feature_name,
                                "permutation_shap",
                            ]
                        ),
                        "prediction": float(
                            ig[
                                "sample_prediction"
                            ]
                        ),
                        "baseline_prediction": float(
                            ig[
                                "baseline_prediction"
                            ]
                        ),
                        "ig_reconstruction": float(
                            ig[
                                "reconstruction"
                            ]
                        ),
                        "ig_residual": float(
                            ig["residual"]
                        ),
                    })

                progress[
                    "completed_ig_samples"
                ] += 1

                if (
                    sample_id
                    not in hessian_sample_ids
                ):
                    continue

                progress.update({
                    "last_message": (
                        "Full-Hessian pathwise "
                        "Taylor–Shapley"
                    ),
                })

                atomic_json(
                    progress_path,
                    progress,
                )

                taylor = (
                    pathwise_taylor_shapley(
                        predictor,
                        baseline,
                        sample,
                        fd_steps,
                        args
                        .hessian_path_nodes,
                    )
                )

                for feature_index, feature_name in enumerate(
                    feature_names
                ):
                    taylor_rows.append({
                        "contract": contract,
                        "method": method,
                        "seed": seed,
                        "candidate_id": (
                            candidate_id
                        ),
                        "sample_id": (
                            sample_id
                        ),
                        "feature_index": (
                            feature_index
                        ),
                        "feature_name": (
                            feature_name
                        ),
                        "linear_baseline_term": float(
                            taylor[
                                "linear_contribution"
                            ][feature_index]
                        ),
                        "diagonal_curvature_term": float(
                            taylor[
                                "diagonal_curvature"
                            ][feature_index]
                        ),
                        "interaction_share": float(
                            taylor[
                                "interaction_share"
                            ][feature_index]
                        ),
                        "taylor_shapley_contribution": float(
                            taylor[
                                "feature_contribution"
                            ][feature_index]
                        ),
                        "permutation_shap": float(
                            shap_sample.loc[
                                feature_name,
                                "permutation_shap",
                            ]
                        ),
                        "integrated_gradient": float(
                            ig[
                                "contribution"
                            ][feature_index]
                        ),
                        "prediction": float(
                            taylor[
                                "sample_prediction"
                            ]
                        ),
                        "baseline_prediction": float(
                            taylor[
                                "baseline_prediction"
                            ]
                        ),
                        "taylor_reconstruction": float(
                            taylor[
                                "reconstruction"
                            ]
                        ),
                        "taylor_residual": float(
                            taylor["residual"]
                        ),
                    })

                pair_matrix = taylor[
                    "pair_matrix"
                ]

                for first in range(
                    len(feature_names)
                ):
                    for second in range(
                        first + 1,
                        len(feature_names),
                    ):
                        interaction_rows.append({
                            "contract": contract,
                            "method": method,
                            "seed": seed,
                            "candidate_id": (
                                candidate_id
                            ),
                            "sample_id": (
                                sample_id
                            ),
                            "feature_i": (
                                feature_names[
                                    first
                                ]
                            ),
                            "feature_j": (
                                feature_names[
                                    second
                                ]
                            ),
                            "interaction_contribution": float(
                                pair_matrix[
                                    first,
                                    second
                                ]
                            ),
                        })

                quality_rows.append({
                    "contract": contract,
                    "method": method,
                    "seed": seed,
                    "candidate_id": (
                        candidate_id
                    ),
                    "sample_id": (
                        sample_id
                    ),
                    "prediction": float(
                        taylor[
                            "sample_prediction"
                        ]
                    ),
                    "baseline_prediction": float(
                        taylor[
                            "baseline_prediction"
                        ]
                    ),
                    "ig_residual": float(
                        ig["residual"]
                    ),
                    "taylor_residual": float(
                        taylor["residual"]
                    ),
                    "ig_relative_residual": float(
                        abs(
                            ig["residual"]
                        )
                        / max(
                            abs(
                                ig[
                                    "sample_prediction"
                                ]
                                - ig[
                                    "baseline_prediction"
                                ]
                            ),
                            1e-8,
                        )
                    ),
                    "taylor_relative_residual": float(
                        abs(
                            taylor[
                                "residual"
                            ]
                        )
                        / max(
                            abs(
                                taylor[
                                    "sample_prediction"
                                ]
                                - taylor[
                                    "baseline_prediction"
                                ]
                            ),
                            1e-8,
                        )
                    ),
                })

                progress[
                    "completed_hessian_samples"
                ] += 1

                pd.DataFrame(
                    taylor_rows
                ).to_csv(
                    output
                    / "sample_level_pathwise_taylor_shapley.csv",
                    index=False,
                )

                pd.DataFrame(
                    interaction_rows
                ).to_csv(
                    output
                    / "sample_level_interactions.csv",
                    index=False,
                )

                pd.DataFrame(
                    quality_rows
                ).to_csv(
                    output
                    / "pathwise_quality.csv",
                    index=False,
                )

            pd.DataFrame(
                ig_rows
            ).to_csv(
                output
                / "sample_level_integrated_gradients.csv",
                index=False,
            )

            progress[
                "completed_cells"
            ] = cell_index + 1

            atomic_json(
                progress_path,
                progress,
            )

        ig_frame = pd.DataFrame(
            ig_rows
        )
        taylor_frame = pd.DataFrame(
            taylor_rows
        )
        interaction_frame = pd.DataFrame(
            interaction_rows
        )
        quality_frame = pd.DataFrame(
            quality_rows
        )

        ig_summary = (
            ig_frame.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                    "feature_name",
                ],
                as_index=False,
            )
            .agg(
                mean_ig=(
                    "integrated_gradient",
                    "mean",
                ),
                mean_abs_ig=(
                    "integrated_gradient",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
                mean_shap=(
                    "permutation_shap",
                    "mean",
                ),
                mean_abs_shap=(
                    "permutation_shap",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
            )
        )

        ig_summary.to_csv(
            output
            / "feature_level_ig_shap_by_seed.csv",
            index=False,
        )

        taylor_summary = (
            taylor_frame.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                    "feature_name",
                ],
                as_index=False,
            )
            .agg(
                mean_taylor_shapley=(
                    "taylor_shapley_contribution",
                    "mean",
                ),
                mean_abs_taylor_shapley=(
                    "taylor_shapley_contribution",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
                mean_linear_term=(
                    "linear_baseline_term",
                    "mean",
                ),
                mean_abs_diagonal_curvature=(
                    "diagonal_curvature_term",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
                mean_abs_interaction_share=(
                    "interaction_share",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
                mean_abs_shap=(
                    "permutation_shap",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
                mean_abs_ig=(
                    "integrated_gradient",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
            )
        )

        taylor_summary.to_csv(
            output
            / "feature_level_taylor_shapley_by_seed.csv",
            index=False,
        )

        interaction_summary = (
            interaction_frame.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                    "feature_i",
                    "feature_j",
                ],
                as_index=False,
            )
            .agg(
                mean_interaction=(
                    "interaction_contribution",
                    "mean",
                ),
                mean_abs_interaction=(
                    "interaction_contribution",
                    lambda values: float(
                        np.abs(values).mean()
                    ),
                ),
            )
        )

        interaction_summary.to_csv(
            output
            / "interaction_summary_by_seed.csv",
            index=False,
        )

        method_agreement_rows = []

        for keys, group in ig_frame.groupby(
            [
                "contract",
                "method",
                "seed",
            ]
        ):
            contract, method, seed = keys

            method_agreement_rows.append({
                "contract": contract,
                "method": method,
                "seed": int(seed),
                "shap_ig_pearson": float(
                    group[
                        [
                            "permutation_shap",
                            "integrated_gradient",
                        ]
                    ].corr().iloc[0, 1]
                ),
                "mean_abs_ig_residual": float(
                    group[
                        "ig_residual"
                    ].abs().mean()
                ),
            })

        if not taylor_frame.empty:
            for keys, group in taylor_frame.groupby(
                [
                    "contract",
                    "method",
                    "seed",
                ]
            ):
                contract, method, seed = keys

                match = next(
                    row
                    for row
                    in method_agreement_rows
                    if (
                        row["contract"]
                        == contract
                        and row["method"]
                        == method
                        and row["seed"]
                        == int(seed)
                    )
                )

                match[
                    "shap_taylor_pearson"
                ] = float(
                    group[
                        [
                            "permutation_shap",
                            "taylor_shapley_contribution",
                        ]
                    ].corr().iloc[0, 1]
                )

                match[
                    "ig_taylor_pearson"
                ] = float(
                    group[
                        [
                            "integrated_gradient",
                            "taylor_shapley_contribution",
                        ]
                    ].corr().iloc[0, 1]
                )

                match[
                    "mean_abs_taylor_residual"
                ] = float(
                    group[
                        "taylor_residual"
                    ].abs().mean()
                )

        pd.DataFrame(
            method_agreement_rows
        ).to_csv(
            output
            / "method_agreement_by_seed.csv",
            index=False,
        )

        ig_relative = (
            ig_frame[
                [
                    "contract",
                    "method",
                    "seed",
                    "sample_id",
                    "prediction",
                    "baseline_prediction",
                    "ig_residual",
                ]
            ]
            .drop_duplicates()
        )

        ig_relative[
            "relative_residual"
        ] = (
            ig_relative[
                "ig_residual"
            ].abs()
            / (
                ig_relative[
                    "prediction"
                ]
                - ig_relative[
                    "baseline_prediction"
                ]
            ).abs().clip(
                lower=1e-8
            )
        )

        ig_gate_rate = float(
            (
                ig_relative[
                    "relative_residual"
                ]
                <= 0.05
            ).mean()
        )

        taylor_gate_rate = float(
            (
                quality_frame[
                    "taylor_relative_residual"
                ]
                <= 0.05
            ).mean()
        )

        manifest = {
            "status": "COMPLETED",
            "base_shap_output": str(
                base_output
            ),
            "base_shap_manifest_sha256": sha256(
                base_output
                / "analysis_manifest.json"
            ),
            "formal_manifest_sha256": sha256(
                manifest_path
            ),
            "lineage_sha256": sha256(
                lineage_path
            ),
            "formal_cell_count": 30,
            "ig_sample_count": int(
                ig_frame[
                    [
                        "contract",
                        "method",
                        "seed",
                        "sample_id",
                    ]
                ].drop_duplicates().shape[0]
            ),
            "hessian_sample_count": int(
                quality_frame.shape[0]
            ),
            "interaction_pair_count": 45,
            "ig_steps": args.ig_steps,
            "hessian_path_nodes": (
                args.hessian_path_nodes
            ),
            "hessian_samples_per_cell": (
                args.hessian_samples_per_cell
            ),
            "ig_five_percent_gate_rate": (
                ig_gate_rate
            ),
            "taylor_shapley_five_percent_gate_rate": (
                taylor_gate_rate
            ),
            "interpretation": {
                "permutation_shap": (
                    "Primary model-attribution evidence."
                ),
                "integrated_gradients": (
                    "Path-integrated first-order "
                    "attribution and completeness check."
                ),
                "pathwise_taylor_shapley": (
                    "Baseline gradient plus full-Hessian "
                    "integral remainder with pairwise "
                    "interaction terms split equally "
                    "between interacting features."
                ),
            },
            "scientific_boundaries": [
                (
                    "Attributions describe model "
                    "behaviour, not causal effects."
                ),
                (
                    "Pathwise Taylor–Shapley is "
                    "accepted only for samples passing "
                    "the reconstruction-residual gate."
                ),
                (
                    "Permutation SHAP remains a "
                    "Monte Carlo estimator."
                ),
                (
                    "GROUP and SPATIAL results remain "
                    "separate."
                ),
            ],
        }

        atomic_json(
            output / "analysis_manifest.json",
            manifest,
        )

        progress.update({
            "status": "COMPLETED",
            "phase": "COMPLETED",
            "current_cell": None,
            "current_sample": None,
            "last_message": (
                "Supplementary attribution "
                "experiments completed"
            ),
        })

        atomic_json(
            progress_path,
            progress,
        )

        print("=" * 88)
        print(
            "PATHWISE_TAYLOR_SHAPLEY_COMPLETE"
        )
        print("IG samples:", manifest[
            "ig_sample_count"
        ])
        print("Hessian samples:", manifest[
            "hessian_sample_count"
        ])
        print(
            "IG ≤5% residual rate:",
            f"{ig_gate_rate:.2%}",
        )
        print(
            "Taylor–Shapley ≤5% residual rate:",
            f"{taylor_gate_rate:.2%}",
        )
        print("Output:", output)
        print("=" * 88)

    except Exception as exc:
        failure = {
            "status": "FAILED",
            "exception_type": (
                type(exc).__name__
            ),
            "message": str(exc),
            "traceback": (
                traceback.format_exc()
            ),
        }

        atomic_json(
            output / "fatal_error.json",
            failure,
        )

        progress.update({
            "status": "FAILED",
            "phase": "FAILED",
            "last_message": str(exc),
        })

        atomic_json(
            progress_path,
            progress,
        )

        raise


if __name__ == "__main__":
    main()
