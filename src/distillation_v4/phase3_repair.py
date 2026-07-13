from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import time
from typing import Any, Callable

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

from src.distillation_program.next_models import SharedDenseRegressor

from .contracts import CampaignConfig
from .repair_trainer import fit_repair_regressor
from .teacher_registry import sha256_file
from .phase3_matrix import build_phase3_jobs


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _features(frame: pd.DataFrame) -> list[str]:
    selected = [
        name
        for name in frame.columns
        if (
            name.startswith("weather_")
            or (
                name.startswith("soil_")
                and not name.endswith("__missing")
            )
        )
    ]
    if not selected:
        raise ValueError("PHASE3_TEACHER_FEATURES_EMPTY")
    return selected


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

    if train.empty or validation.empty:
        raise ValueError(f"PHASE3_PARTITION_EMPTY:{fold}")

    return train, validation


def _prediction_pipeline(
    features: list[str],
    *,
    seed: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
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
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=seed,
                    early_stopping=False,
                ),
            ),
        ]
    )


def _fit_prediction_teacher(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()

    model = _prediction_pipeline(features, seed=seed)
    model.fit(train[features], train.target_yield)

    prediction = model.predict(validation[features])
    teacher_mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            prediction,
        )
    )

    naive_prediction = np.full(
        len(validation),
        float(train.target_yield.mean()),
    )
    naive_mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            naive_prediction,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "prediction_teacher.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "seed": seed,
            "target": "target_yield",
        },
        checkpoint,
    )

    replay = joblib.load(checkpoint)
    replay_prediction = replay["model"].predict(
        validation[replay["features"]]
    )
    replay_error = float(
        np.max(np.abs(replay_prediction - prediction))
    )

    if replay_error > 1e-12:
        raise RuntimeError(
            "PREDICTION_TEACHER_REPLAY_MISMATCH"
        )

    return {
        "role": "prediction",
        "seed": seed,
        "teacher_mae": teacher_mae,
        "naive_mae": naive_mae,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_replay_max_abs_error": replay_error,
        "fit_time_seconds": time.perf_counter() - started,
        "qualified": bool(
            math.isfinite(teacher_mae)
            and teacher_mae < naive_mae
        ),
    }


def _combined_preprocessor(
    features: list[str],
) -> Pipeline:
    return Pipeline(
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


def _representations(
    model: SharedDenseRegressor,
    values: torch.Tensor,
) -> np.ndarray:
    mask = torch.ones(
        values.shape[0],
        1,
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        _, auxiliary = model(
            {"privileged": values},
            mask,
        )
    return (
        auxiliary["representation"]
        .detach()
        .cpu()
        .numpy()
    )


def _fit_representation_teacher(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
    output: Path,
    contract,
) -> dict[str, Any]:
    started = time.perf_counter()

    preprocess = _combined_preprocessor(features)
    train_array = preprocess.fit_transform(
        train[features]
    )
    validation_array = preprocess.transform(
        validation[features]
    )

    train_tensor = torch.tensor(
        train_array,
        dtype=torch.float32,
    )
    validation_tensor = torch.tensor(
        validation_array,
        dtype=torch.float32,
    )
    train_target = torch.tensor(
        train.target_yield.to_numpy(),
        dtype=torch.float32,
    )
    validation_target = torch.tensor(
        validation.target_yield.to_numpy(),
        dtype=torch.float32,
    )

    model = SharedDenseRegressor(
        {"privileged": train_tensor.shape[1]},
        hidden=32,
    )

    result = fit_repair_regressor(
        model=model,
        train_values={"privileged": train_tensor},
        train_mask=torch.ones(len(train), 1),
        train_target=train_target,
        validation_values={
            "privileged": validation_tensor
        },
        validation_mask=torch.ones(len(validation), 1),
        validation_target=validation_target,
        contract=contract,
        seed=seed,
    )

    train_representation = _representations(
        model,
        train_tensor,
    )
    validation_representation = _representations(
        model,
        validation_tensor,
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

    generator = np.random.default_rng(seed)
    random_train = generator.normal(
        size=train_representation.shape
    )
    random_validation = generator.normal(
        size=validation_representation.shape
    )
    random_probe = Ridge(alpha=1.0)
    random_probe.fit(
        random_train,
        train.target_yield.to_numpy(float),
    )
    random_prediction = random_probe.predict(
        random_validation
    )
    random_probe_mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            random_prediction,
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "representation_teacher.joblib"

    joblib.dump(
        {
            "state_dict": result.best_state_dict,
            "input_width": int(train_tensor.shape[1]),
            "hidden": 32,
            "features": features,
            "preprocessor": preprocess,
            "target_mean": result.target_mean,
            "target_scale": result.target_scale,
            "seed": seed,
        },
        checkpoint,
    )

    bundle = joblib.load(checkpoint)
    replay_model = SharedDenseRegressor(
        {"privileged": bundle["input_width"]},
        hidden=bundle["hidden"],
    )
    replay_model.load_state_dict(bundle["state_dict"])

    replay_representation = _representations(
        replay_model,
        validation_tensor,
    )
    representation_replay_error = float(
        np.max(
            np.abs(
                replay_representation
                - validation_representation
            )
        )
    )

    if representation_replay_error > 1e-7:
        raise RuntimeError(
            "REPRESENTATION_TEACHER_REPLAY_MISMATCH"
        )

    return {
        "role": "representation",
        "seed": seed,
        "probe_mae": probe_mae,
        "random_control_probe_mae": random_probe_mae,
        "probe_improvement": (
            random_probe_mae - probe_mae
        ),
        "teacher_validation_mae": float(
            mean_absolute_error(
                validation.target_yield.to_numpy(float),
                result.prediction,
            )
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_replay_max_abs_error": (
            representation_replay_error
        ),
        "optimizer_steps": result.optimizer_steps,
        "parameter_delta_l2": result.parameter_delta_l2,
        "fit_time_seconds": time.perf_counter() - started,
        "qualified": bool(
            math.isfinite(probe_mae)
            and math.isfinite(random_probe_mae)
            and probe_mae < random_probe_mae
            and result.parameter_delta_l2
            >= contract.minimum_parameter_delta
        ),
    }


def _expected_folds(
    config: CampaignConfig,
    dataset: str,
) -> tuple[str, ...]:
    return tuple(
        config.raw
        .get("datasets", {})
        .get(dataset, {})
        .get(
            "outer_folds",
            ("test_2021", "test_2022", "test_2023"),
        )
    )


def run_phase3_repair(
    config: CampaignConfig,
    output: Path,
    *,
    frame_loader: Callable[
        [CampaignConfig, Path, str],
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    contract = config.neural_training
    seeds = tuple(int(seed) for seed in config.stochastic_seeds)

    records: list[dict[str, Any]] = []
    selected_teachers: list[dict[str, Any]] = []
    evidence_files: list[str] = []

    planned_matrix = {
        dataset: build_phase3_jobs(
            dataset=dataset,
            folds=_expected_folds(config, dataset),
            deterministic_seed=config.deterministic_seed,
            stochastic_seeds=seeds,
        )
        for dataset in config.primary_datasets
    }

    planned_matrix_path = (
        output
        / "control/phase3_repair_planned_matrix.json"
    )
    _atomic_json(
        planned_matrix_path,
        {
            "phase": "phase3",
            "budget_semantics": "PER_DATASET_MAXIMUM",
            "per_dataset_job_ceiling": int(
                config.budgets["phase3"]
            ),
            "campaign_job_ceiling": (
                int(config.budgets["phase3"])
                * len(config.primary_datasets)
            ),
            "datasets": {
                dataset: {
                    "count": len(jobs),
                    "jobs": [
                        job.to_mapping()
                        for job in jobs
                    ],
                }
                for dataset, jobs
                in planned_matrix.items()
            },
        },
    )
    evidence_files.append(
        str(planned_matrix_path.relative_to(output))
    )

    for dataset in config.primary_datasets:
        frame = frame_loader(config, output, dataset)
        features = _features(frame)
        folds = _expected_folds(config, dataset)

        dataset_records: list[dict[str, Any]] = []

        for fold in folds:
            train, validation = _split(frame, fold)

            fold_records: list[dict[str, Any]] = []

            for seed in seeds:
                base = (
                    output
                    / "development/phase3_repair"
                    / dataset
                    / fold
                    / f"seed_{seed}"
                )

                prediction = _fit_prediction_teacher(
                    train=train,
                    validation=validation,
                    features=features,
                    seed=seed,
                    output=base / "prediction",
                )
                prediction.update(
                    {
                        "dataset": dataset,
                        "fold": fold,
                    }
                )

                representation = (
                    _fit_representation_teacher(
                        train=train,
                        validation=validation,
                        features=features,
                        seed=seed,
                        output=base / "representation",
                        contract=contract,
                    )
                )
                representation.update(
                    {
                        "dataset": dataset,
                        "fold": fold,
                    }
                )

                for record in (
                    prediction,
                    representation,
                ):
                    record_path = (
                        base
                        / record["role"]
                        / "teacher_record.json"
                    )
                    serializable = {
                        **record,
                        "checkpoint": str(
                            Path(record["checkpoint"])
                            .relative_to(output)
                        ),
                    }
                    _atomic_json(record_path, serializable)
                    evidence_files.append(
                        str(record_path.relative_to(output))
                    )
                    fold_records.append(serializable)
                    dataset_records.append(serializable)
                    records.append(serializable)

            for role in ("prediction", "representation"):
                candidates = [
                    record
                    for record in fold_records
                    if (
                        record["role"] == role
                        and record["qualified"]
                    )
                ]

                if not candidates:
                    continue

                if role == "prediction":
                    selected = min(
                        candidates,
                        key=lambda record: (
                            record["teacher_mae"],
                            record["seed"],
                        ),
                    )
                else:
                    selected = min(
                        candidates,
                        key=lambda record: (
                            record["probe_mae"],
                            record["seed"],
                        ),
                    )

                selected_teachers.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "role": role,
                        "decision": "DEVELOPMENT_OPEN",
                        "seed": selected["seed"],
                        "checkpoint": selected["checkpoint"],
                        "checkpoint_sha256": (
                            selected["checkpoint_sha256"]
                        ),
                        "selection_scope": (
                            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
                        ),
                        "outer_refit_allowed": False,
                    }
                )

        prediction_records = [
            record
            for record in dataset_records
            if record["role"] == "prediction"
        ]
        representation_records = [
            record
            for record in dataset_records
            if record["role"] == "representation"
        ]

        prediction_open = bool(
            prediction_records
            and all(
                record["qualified"]
                for record in prediction_records
            )
        )
        representation_open = bool(
            representation_records
            and all(
                record["qualified"]
                for record in representation_records
            )
        )

        gate = {
            "dataset": dataset,
            "prediction_teacher": (
                "DEVELOPMENT_OPEN"
                if prediction_open
                else "DEVELOPMENT_BLOCKED"
            ),
            "representation_teacher": (
                "DEVELOPMENT_OPEN"
                if representation_open
                else "DEVELOPMENT_BLOCKED"
            ),
            "prediction_records": len(
                prediction_records
            ),
            "representation_records": len(
                representation_records
            ),
            "selection_scope": (
                "OUTER_TRAIN_INNER_VALIDATION_ONLY"
            ),
            "random_representation_control_required": True,
            "outer_test_may_enable_downstream_jobs": False,
        }

        gate_path = (
            output
            / "control/gates"
            / f"{dataset}__teachers_repair.json"
        )
        _atomic_json(gate_path, gate)
        evidence_files.append(
            str(gate_path.relative_to(output))
        )

    registry = {
        "status": "FROZEN_TEACHER_REGISTRY",
        "release_status": (
            "DEVELOPMENT_ONLY_PENDING_PHASE3_COVERAGE"
        ),
        "budget_semantics": "PER_DATASET_MAXIMUM",
        "formal_phase3_budget_per_dataset": int(
            config.budgets["phase3"]
        ),
        "formal_phase3_campaign_ceiling": (
            int(config.budgets["phase3"])
            * len(config.primary_datasets)
        ),
        "implemented_teacher_jobs": len(records),
        "outer_refit_allowed": False,
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "teachers": selected_teachers,
    }
    registry_path = (
        output
        / "control/frozen_teacher_registry.json"
    )
    _atomic_json(registry_path, registry)
    evidence_files.append(
        str(registry_path.relative_to(output))
    )

    planned_jobs = (
        sum(
            len(_expected_folds(config, dataset))
            for dataset in config.primary_datasets
        )
        * len(seeds)
        * 2
    )
    implemented_submatrix_jobs = planned_jobs
    accepted_submatrix_jobs = len(records)
    submatrix_failed_jobs = (
        implemented_submatrix_jobs
        - accepted_submatrix_jobs
    )

    per_dataset_budget_jobs = int(
        config.budgets["phase3"]
    )
    campaign_budget_jobs = (
        per_dataset_budget_jobs
        * len(config.primary_datasets)
    )

    unimplemented_budget_jobs = max(
        campaign_budget_jobs
        - implemented_submatrix_jobs,
        0,
    )
    excess_implemented_jobs = max(
        implemented_submatrix_jobs
        - campaign_budget_jobs,
        0,
    )

    formal_complete = bool(
        implemented_submatrix_jobs
        == campaign_budget_jobs
        and accepted_submatrix_jobs
        == campaign_budget_jobs
        and submatrix_failed_jobs == 0
    )

    implemented_by_dataset = {
        dataset: sum(
            1
            for record in records
            if record["dataset"] == dataset
        )
        for dataset in config.primary_datasets
    }

    expected_by_dataset = {
        dataset: len(planned_matrix[dataset])
        for dataset in config.primary_datasets
    }

    gap_by_dataset = {
        dataset: (
            expected_by_dataset[dataset]
            - implemented_by_dataset[dataset]
        )
        for dataset in config.primary_datasets
    }

    coverage_gap = {
        "budget_semantics": "PER_DATASET_MAXIMUM",
        "per_dataset_budget_jobs": (
            per_dataset_budget_jobs
        ),
        "campaign_budget_jobs": campaign_budget_jobs,
        "implemented_submatrix_jobs": (
            implemented_submatrix_jobs
        ),
        "accepted_submatrix_jobs": (
            accepted_submatrix_jobs
        ),
        "submatrix_failed_jobs": (
            submatrix_failed_jobs
        ),
        "unimplemented_budget_jobs": (
            unimplemented_budget_jobs
        ),
        "excess_implemented_jobs": (
            excess_implemented_jobs
        ),
        "formal_matrix_coverage_fraction": (
            accepted_submatrix_jobs
            / campaign_budget_jobs
            if campaign_budget_jobs > 0
            else 0.0
        ),
        "expected_jobs_by_dataset": (
            expected_by_dataset
        ),
        "implemented_jobs_by_dataset": (
            implemented_by_dataset
        ),
        "gap_jobs_by_dataset": gap_by_dataset,
        "missing_job_definitions_identified": True,
        "missing_components": {
            "prediction_teacher_seed_rule": (
                "CURRENT_IMPLEMENTATION_USES_"
                "THREE_SEEDS_INSTEAD_OF_ONE"
            ),
            "second_representation_candidate": (
                "NOT_IMPLEMENTED"
            ),
            "matched_deployable_controls": (
                "NOT_IMPLEMENTED_AS_JOBS"
            ),
            "random_representation_control": (
                "EMBEDDED_METRIC_NOT_INDEPENDENT_JOB"
            ),
            "shuffled_soil_neural_control": (
                "NOT_IMPLEMENTED"
            ),
        },
    }

    coverage_path = (
        output
        / "control/phase3_repair_coverage_gap.json"
    )
    _atomic_json(coverage_path, coverage_gap)
    evidence_files.append(
        str(coverage_path.relative_to(output))
    )

    status = {
        "phase": "phase3",
        "status": (
            "SCIENTIFIC_PHASE_COMPLETE"
            if formal_complete
            else "SCIENTIFIC_PHASE_INCOMPLETE"
        ),
        "planned_jobs": campaign_budget_jobs,
        "accepted_jobs": accepted_submatrix_jobs,
        "failed_jobs": (
            campaign_budget_jobs - accepted_submatrix_jobs
        ),
        "implemented_submatrix_jobs": (
            implemented_submatrix_jobs
        ),
        "accepted_submatrix_jobs": accepted_submatrix_jobs,
        "submatrix_failed_jobs": submatrix_failed_jobs,
        "unimplemented_budget_jobs": (
            unimplemented_budget_jobs
        ),
        "formal_matrix_coverage_fraction": (
            coverage_gap[
                "formal_matrix_coverage_fraction"
            ]
        ),
        "evidence_files": sorted(set(evidence_files)),
        "frozen_teacher_count": len(selected_teachers),
        "teacher_registry_status": (
            "DEVELOPMENT_ONLY_NOT_FORMALLY_RELEASED"
            if not formal_complete
            else "FORMALLY_RELEASED"
        ),
        "outer_refit_allowed": False,
        "outer_test_may_enable_downstream_jobs": False,
        "phase4_release_allowed": formal_complete,
    }

    _atomic_json(
        output / "status/phase3_repair.json",
        status,
    )

    return {
        "status": status,
        "registry": registry,
        "records": records,
    }
