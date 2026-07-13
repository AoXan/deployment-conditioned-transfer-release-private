from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch

from .phase4_execution import (
    _features,
    _preprocess,
    _replay_student,
    _split_development,
)
from .phase4_student import fit_phase4_student
from .phase4_teacher_adapter import (
    TeacherSignals,
    infer_frozen_teacher_signals,
)
from .phase4_training_contract import (
    Phase4LossWeights,
)
from .phase5_control_training import (
    fit_exact_step_supervised_control,
)
from .phase5_sources import (
    load_validated_phase4_source,
    resolve_teacher_record,
)
from .training_contract import NeuralTrainingContract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _shuffle_train_signal(
    values: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(values))
    shuffled = np.asarray(values)[order]

    if np.array_equal(shuffled, values) and len(values) > 1:
        shuffled = np.roll(shuffled, 1, axis=0)

    return shuffled


def _random_representation(
    reference: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)

    generator = np.random.default_rng(seed)
    random = generator.normal(
        size=reference.shape
    )

    mean = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)

    scale = np.where(scale < 1e-8, 1.0, scale)

    return random * scale + mean


def execute_phase5_job(
    *,
    job: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    contract: NeuralTrainingContract,
    teacher_signal_provider: Callable[..., TeacherSignals] = (
        infer_frozen_teacher_signals
    ),
) -> dict[str, Any]:
    dataset = str(job["dataset"])
    fold = str(job["fold"])
    seed = int(job["seed"])
    control = str(job["control"])

    source = load_validated_phase4_source(
        output=output_root,
        route_id=str(job["source_route_id"]),
        expected_artifact_fingerprint=str(
            job[
                "source_route_artifact_fingerprint"
            ]
        ),
    )

    baseline = load_validated_phase4_source(
        output=output_root,
        route_id=str(
            job["matched_s0_route_id"]
        ),
        expected_artifact_fingerprint=str(
            job[
                "matched_s0_artifact_fingerprint"
            ]
        ),
    )

    if source["dataset"] != dataset:
        raise RuntimeError(
            "PHASE5_SOURCE_DATASET_MISMATCH"
        )

    if source["fold"] != fold:
        raise RuntimeError(
            "PHASE5_SOURCE_FOLD_MISMATCH"
        )

    if int(source["seed"]) != seed:
        raise RuntimeError(
            "PHASE5_SOURCE_SEED_MISMATCH"
        )

    if baseline["route"] != "supervised":
        raise RuntimeError(
            "PHASE5_MATCHED_BASELINE_NOT_S0"
        )

    train, validation = _split_development(
        frame,
        fold,
    )
    weather_features, _ = _features(frame)

    (
        train_weather,
        validation_weather,
        preprocessor,
    ) = _preprocess(
        train=train,
        validation=validation,
        features=weather_features,
    )

    train_target = torch.tensor(
        train.target_yield.to_numpy(float),
        dtype=torch.float32,
    )
    validation_target = torch.tensor(
        validation.target_yield.to_numpy(float),
        dtype=torch.float32,
    )

    control_metadata: dict[str, Any] = {}

    if control == "active_compute_steps_control":
        target_steps = int(
            job["target_optimizer_steps"]
        )

        exact = fit_exact_step_supervised_control(
            train_values=train_weather,
            train_target=train_target,
            validation_values=validation_weather,
            target_optimizer_steps=target_steps,
            batch_size=contract.batch_size,
            learning_rate=contract.learning_rate,
            weight_decay=contract.weight_decay,
            gradient_clip_norm=(
                contract.gradient_clip_norm
            ),
            seed=seed,
        )

        prediction = exact.prediction
        representation = exact.representation
        state_dict = exact.state_dict
        target_mean = exact.target_mean
        target_scale = exact.target_scale
        optimizer_steps = exact.optimizer_steps
        parameter_delta = (
            exact.parameter_delta_l2
        )
        fit_time = exact.fit_time_seconds
        inference_time = (
            exact.inference_time_seconds
        )
        loss_ledger = exact.loss_ledger
        representation_dim = int(
            representation.shape[1]
        )

        control_metadata = {
            "target_optimizer_steps": target_steps,
            "actual_optimizer_steps": (
                optimizer_steps
            ),
            "optimizer_step_difference": (
                optimizer_steps - target_steps
            ),
            "target_fit_time_seconds": float(
                job["target_fit_time_seconds"]
            ),
            "fit_time_matching_role": (
                "REPORTED_NOT_FORCE_MATCHED"
            ),
        }

    else:
        prediction_teacher_train = None
        prediction_teacher_validation = None
        representation_teacher_train = None
        representation_teacher_validation = None

        route = None
        weights = None

        if control == (
            "shuffled_teacher_prediction_control"
        ):
            teacher_record = resolve_teacher_record(
                teacher_lookup=teacher_lookup,
                dataset=dataset,
                fold=fold,
                role="prediction",
                expected_summary=job["teacher"],
            )

            signals = teacher_signal_provider(
                output_root=output_root,
                teacher_record=teacher_record,
                train=train,
                validation=validation,
            )

            prediction_teacher_train = (
                _shuffle_train_signal(
                    signals.train_prediction,
                    seed=int(job["shuffle_seed"]),
                )
            )
            prediction_teacher_validation = (
                signals.validation_prediction
            )

            route = "prediction_kd"
            weights = Phase4LossWeights(
                supervised=1.0,
                prediction=0.5,
                representation=0.0,
            )

            control_metadata = {
                "shuffle_seed": int(
                    job["shuffle_seed"]
                ),
                "shuffle_scope": (
                    "WITHIN_TRAIN_PARTITION_ONLY"
                ),
                "validation_teacher_signal_shuffled": (
                    False
                ),
            }

        elif control == (
            "random_representation_transfer_control"
        ):
            teacher_record = resolve_teacher_record(
                teacher_lookup=teacher_lookup,
                dataset=dataset,
                fold=fold,
                role="representation",
                expected_summary=job["teacher"],
            )

            signals = teacher_signal_provider(
                output_root=output_root,
                teacher_record=teacher_record,
                train=train,
                validation=validation,
            )

            representation_teacher_train = (
                _random_representation(
                    signals.train_representation,
                    seed=int(
                        job[
                            "random_representation_seed"
                        ]
                    ),
                )
            )
            representation_teacher_validation = (
                _random_representation(
                    signals.validation_representation,
                    seed=(
                        int(
                            job[
                                "random_representation_seed"
                            ]
                        )
                        + 1
                    ),
                )
            )

            route = "representation_kd"
            weights = Phase4LossWeights(
                supervised=1.0,
                prediction=0.0,
                representation=0.5,
            )

            control_metadata = {
                "random_representation_seed": int(
                    job[
                        "random_representation_seed"
                    ]
                ),
                "shape_matched": True,
                "marginal_scale_matched": True,
            }

        elif control == (
            "ckd_balancing_sensitivity_control"
        ):
            prediction_record = resolve_teacher_record(
                teacher_lookup=teacher_lookup,
                dataset=dataset,
                fold=fold,
                role="prediction",
                expected_summary=source[
                    "prediction_teacher"
                ],
            )
            representation_record = (
                resolve_teacher_record(
                    teacher_lookup=teacher_lookup,
                    dataset=dataset,
                    fold=fold,
                    role="representation",
                    expected_summary=source[
                        "representation_teacher"
                    ],
                )
            )

            prediction_signals = (
                teacher_signal_provider(
                    output_root=output_root,
                    teacher_record=prediction_record,
                    train=train,
                    validation=validation,
                )
            )

            if (
                prediction_record
                == representation_record
            ):
                representation_signals = (
                    prediction_signals
                )
            else:
                representation_signals = (
                    teacher_signal_provider(
                        output_root=output_root,
                        teacher_record=(
                            representation_record
                        ),
                        train=train,
                        validation=validation,
                    )
                )

            prediction_teacher_train = (
                prediction_signals.train_prediction
            )
            prediction_teacher_validation = (
                prediction_signals
                .validation_prediction
            )
            representation_teacher_train = (
                representation_signals
                .train_representation
            )
            representation_teacher_validation = (
                representation_signals
                .validation_representation
            )

            alternative = job[
                "ckd_sensitivity_contract"
            ]["alternative_loss_weights"]

            route = "combined_kd"
            weights = Phase4LossWeights(
                supervised=float(
                    alternative["supervised"]
                ),
                prediction=float(
                    alternative["prediction"]
                ),
                representation=float(
                    alternative["representation"]
                ),
            )

            control_metadata = {
                "variant_id": job[
                    "ckd_sensitivity_contract"
                ]["variant_id"],
                "alternative_loss_weights": (
                    weights.normalized().to_mapping()
                ),
                "frozen_phase4_loss_weights": (
                    job[
                        "ckd_sensitivity_contract"
                    ][
                        "frozen_phase4_loss_weights"
                    ]
                ),
            }

        else:
            raise RuntimeError(
                "PHASE5_CONTROL_HANDLER_UNKNOWN:"
                + control
            )

        result = fit_phase4_student(
            route=route,
            train_values=train_weather,
            train_target=train_target,
            validation_values=validation_weather,
            validation_target=validation_target,
            contract=contract,
            seed=seed,
            loss_weights=weights,
            teacher_prediction_train=(
                None
                if prediction_teacher_train is None
                else torch.tensor(
                    prediction_teacher_train,
                    dtype=torch.float32,
                )
            ),
            teacher_prediction_validation=(
                None
                if prediction_teacher_validation is None
                else torch.tensor(
                    prediction_teacher_validation,
                    dtype=torch.float32,
                )
            ),
            teacher_representation_train=(
                None
                if representation_teacher_train is None
                else torch.tensor(
                    representation_teacher_train,
                    dtype=torch.float32,
                )
            ),
            teacher_representation_validation=(
                None
                if representation_teacher_validation
                is None
                else torch.tensor(
                    representation_teacher_validation,
                    dtype=torch.float32,
                )
            ),
        )

        prediction = result.prediction
        representation = result.representation
        state_dict = result.best_state_dict
        target_mean = result.target_mean
        target_scale = result.target_scale
        optimizer_steps = result.optimizer_steps
        parameter_delta = (
            result.parameter_delta_l2
        )
        fit_time = result.fit_time_seconds
        inference_time = (
            result.inference_time_seconds
        )
        loss_ledger = result.loss_ledger
        representation_dim = int(
            representation.shape[1]
        )

    output.mkdir(parents=True, exist_ok=True)

    checkpoint = output / "model.pt"
    preprocessor_path = (
        output / "preprocessor.joblib"
    )
    prediction_path = (
        output / "validation_prediction.npy"
    )
    representation_path = (
        output / "validation_representation.npy"
    )
    loss_path = output / "loss_ledger.json"

    torch.save(
        {
            "state_dict": state_dict,
            "control": control,
            "input_dim": int(
                train_weather.shape[1]
            ),
            "hidden": 32,
            "representation_dim": (
                representation_dim
            ),
            "target_mean": target_mean,
            "target_scale": target_scale,
        },
        checkpoint,
    )

    joblib.dump(preprocessor, preprocessor_path)
    np.save(prediction_path, prediction)
    np.save(representation_path, representation)

    _atomic_json(
        loss_path,
        {
            "control": control,
            "epochs": loss_ledger,
        },
    )

    replay_error = _replay_student(
        checkpoint=checkpoint,
        validation_values=validation_weather,
        reference_prediction=prediction,
    )

    if replay_error > 1e-6:
        raise RuntimeError(
            "PHASE5_CHECKPOINT_REPLAY_MISMATCH"
        )

    validation_mae = float(
        np.mean(
            np.abs(
                prediction
                - validation.target_yield.to_numpy(
                    float
                )
            )
        )
    )

    record = {
        "job_id": job["job_id"],
        "phase": "phase5",
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "control": control,
        "required_by_route": job[
            "required_by_route"
        ],
        "status": "ENGINEERING_ACCEPTED",
        "source_route_id": source["route_id"],
        "matched_s0_route_id": baseline[
            "route_id"
        ],
        "validation_mae": validation_mae,
        "optimizer_steps": optimizer_steps,
        "parameter_delta_l2": parameter_delta,
        "fit_time_seconds": fit_time,
        "inference_time_seconds": (
            inference_time
        ),
        "checkpoint_replay_max_abs_error": (
            replay_error
        ),
        "control_metadata": control_metadata,
        "checkpoint": str(
            checkpoint.relative_to(output_root)
        ),
        "checkpoint_sha256": _sha256(checkpoint),
        "preprocessor": str(
            preprocessor_path.relative_to(
                output_root
            )
        ),
        "preprocessor_sha256": _sha256(
            preprocessor_path
        ),
        "prediction_artifact": str(
            prediction_path.relative_to(
                output_root
            )
        ),
        "prediction_artifact_sha256": _sha256(
            prediction_path
        ),
        "representation_artifact": str(
            representation_path.relative_to(
                output_root
            )
        ),
        "representation_artifact_sha256": (
            _sha256(representation_path)
        ),
        "loss_ledger": str(
            loss_path.relative_to(output_root)
        ),
        "loss_ledger_sha256": _sha256(loss_path),
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_used": False,
        "scientific_status": (
            "CONTROL_EXECUTED_NOT_INTERPRETED"
        ),
    }

    _atomic_json(
        output / "job_record.json",
        record,
    )
    _atomic_json(
        output / "completion_marker.json",
        {
            "job_id": job["job_id"],
            "status": "ENGINEERING_ACCEPTED",
            "outer_test_used": False,
        },
    )

    return record


# STAGE8_V4_SOIL_PHASE5_DISPATCH
_execute_phase5_job_without_soil = (
    execute_phase5_job
)


def execute_phase5_job(
    *,
    job: dict[str, Any],
    frame: pd.DataFrame,
    output_root: Path,
    output: Path,
    teacher_lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
    contract: NeuralTrainingContract,
    teacher_signal_provider: (
        Callable[..., TeacherSignals]
    ) = infer_frozen_teacher_signals,
) -> dict[str, Any]:
    control = str(job["control"])

    if control in {
        "soil_representation_sensitivity_control",
        "missing_aware_ablation_suite_control",
    }:
        from .phase5_soil_execution import (
            execute_phase5_soil_control,
        )

        return execute_phase5_soil_control(
            job=job,
            frame=frame,
            output_root=output_root,
            output=output,
            contract=contract,
        )

    return (
        _execute_phase5_job_without_soil(
            job=job,
            frame=frame,
            output_root=output_root,
            output=output,
            teacher_lookup=(
                teacher_lookup
            ),
            contract=contract,
            teacher_signal_provider=(
                teacher_signal_provider
            ),
        )
    )
