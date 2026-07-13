from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .frozen_teacher import (
    FrozenTeacher,
    load_frozen_teacher,
)


@dataclass(frozen=True)
class TeacherSignals:
    train_prediction: np.ndarray | None
    validation_prediction: np.ndarray | None
    train_representation: np.ndarray | None
    validation_representation: np.ndarray | None
    teacher_candidate: str
    teacher_role: str
    teacher_checkpoint: str
    teacher_checkpoint_sha256: str


def _validate_signal_rows(
    *,
    name: str,
    signal: np.ndarray | None,
    expected_rows: int,
) -> None:
    if signal is None:
        return

    if len(signal) != expected_rows:
        raise RuntimeError(
            f"PHASE4_TEACHER_SIGNAL_ROW_MISMATCH:"
            f"{name}:{len(signal)}:{expected_rows}"
        )

    if not np.isfinite(signal).all():
        raise RuntimeError(
            f"PHASE4_TEACHER_SIGNAL_NONFINITE:{name}"
        )


def _joblib_prediction(
    *,
    teacher: FrozenTeacher,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    payload = teacher.payload

    if not isinstance(payload, dict):
        raise RuntimeError(
            "PHASE4_JOBLIB_TEACHER_PAYLOAD_INVALID"
        )

    model = payload.get("model")
    features = payload.get("features")

    if model is None or not isinstance(features, list):
        raise RuntimeError(
            "PHASE4_JOBLIB_TEACHER_FIELDS_MISSING"
        )

    missing = sorted(
        set(features).difference(train.columns)
    )
    if missing:
        raise RuntimeError(
            "PHASE4_TEACHER_FEATURES_MISSING:"
            + ",".join(missing)
        )

    train_prediction = np.asarray(
        model.predict(train[features]),
        dtype=float,
    )
    validation_prediction = np.asarray(
        model.predict(validation[features]),
        dtype=float,
    )

    return train_prediction, validation_prediction


def _load_neural_preprocessors(
    *,
    output_root: Path,
    teacher_record: dict[str, Any],
) -> dict[str, Any]:
    value = teacher_record.get("preprocessor")

    if not value:
        raise RuntimeError(
            "PHASE4_NEURAL_TEACHER_PREPROCESSOR_MISSING"
        )

    path = (output_root / str(value)).resolve()

    try:
        path.relative_to(output_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "PHASE4_TEACHER_PREPROCESSOR_OUTSIDE_ROOT"
        ) from exc

    if not path.is_file():
        raise RuntimeError(
            "PHASE4_TEACHER_PREPROCESSOR_FILE_MISSING"
        )

    payload = joblib.load(path)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "PHASE4_TEACHER_PREPROCESSOR_PAYLOAD_INVALID"
        )

    return payload


def _neural_values(
    *,
    frame: pd.DataFrame,
    preprocessors: dict[str, Any],
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    values: dict[str, torch.Tensor] = {}

    for modality, preprocessor in (
        preprocessors.items()
    ):
        feature_names = getattr(
            preprocessor,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            raise RuntimeError(
                "PHASE4_TEACHER_FEATURE_LINEAGE_MISSING:"
                + modality
            )

        features = [
            str(name)
            for name in feature_names
        ]

        if not features:
            raise RuntimeError(
                "PHASE4_TEACHER_MODALITY_FEATURES_EMPTY:"
                + modality
            )

        missing = sorted(
            set(features).difference(frame.columns)
        )
        if missing:
            raise RuntimeError(
                "PHASE4_TEACHER_MODALITY_FEATURES_MISSING:"
                + modality
                + ":"
                + ",".join(missing)
            )

        transformed = preprocessor.transform(
            frame.loc[:, features]
        )

        values[modality] = torch.tensor(
            transformed,
            dtype=torch.float32,
        )

    if not values:
        raise RuntimeError(
            "PHASE4_TEACHER_MODALITIES_EMPTY"
        )

    mask = torch.ones(
        len(frame),
        len(values),
        dtype=torch.float32,
    )

    return values, mask


def _neural_signals(
    *,
    teacher: FrozenTeacher,
    teacher_record: dict[str, Any],
    output_root: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    payload = teacher.payload

    if not isinstance(payload, dict):
        raise RuntimeError(
            "PHASE4_NEURAL_TEACHER_PAYLOAD_INVALID"
        )

    model = payload.get("model")
    if model is None:
        raise RuntimeError(
            "PHASE4_NEURAL_TEACHER_MODEL_MISSING"
        )

    preprocessors = _load_neural_preprocessors(
        output_root=output_root,
        teacher_record=teacher_record,
    )

    train_values, train_mask = _neural_values(
        frame=train,
        preprocessors=preprocessors,
    )
    validation_values, validation_mask = _neural_values(
        frame=validation,
        preprocessors=preprocessors,
    )

    model.eval()

    with torch.no_grad():
        train_scaled, train_auxiliary = model(
            train_values,
            train_mask,
        )
        validation_scaled, validation_auxiliary = model(
            validation_values,
            validation_mask,
        )

    target_mean = float(payload["target_mean"])
    target_scale = float(payload["target_scale"])

    train_prediction = (
        train_scaled * target_scale + target_mean
    ).detach().cpu().numpy()
    validation_prediction = (
        validation_scaled * target_scale + target_mean
    ).detach().cpu().numpy()

    train_representation = train_auxiliary.get(
        "representation"
    )
    validation_representation = (
        validation_auxiliary.get("representation")
    )

    if (
        train_representation is None
        or validation_representation is None
    ):
        raise RuntimeError(
            "PHASE4_TEACHER_REPRESENTATION_MISSING"
        )

    return (
        np.asarray(train_prediction, dtype=float),
        np.asarray(validation_prediction, dtype=float),
        train_representation.detach().cpu().numpy(),
        validation_representation.detach().cpu().numpy(),
    )


def infer_frozen_teacher_signals(
    *,
    output_root: Path,
    teacher_record: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> TeacherSignals:
    teacher = load_frozen_teacher(
        output_root=output_root,
        record=teacher_record,
    )

    suffix = teacher.checkpoint_path.suffix.lower()

    if suffix in {
        ".joblib",
        ".pkl",
        ".pickle",
        ".bin",
    }:
        (
            train_prediction,
            validation_prediction,
        ) = _joblib_prediction(
            teacher=teacher,
            train=train,
            validation=validation,
        )

        train_representation = None
        validation_representation = None

    elif suffix == ".pt":
        (
            train_prediction,
            validation_prediction,
            train_representation,
            validation_representation,
        ) = _neural_signals(
            teacher=teacher,
            teacher_record=teacher_record,
            output_root=output_root,
            train=train,
            validation=validation,
        )

    else:
        raise RuntimeError(
            "PHASE4_TEACHER_FORMAT_UNSUPPORTED"
        )

    _validate_signal_rows(
        name="train_prediction",
        signal=train_prediction,
        expected_rows=len(train),
    )
    _validate_signal_rows(
        name="validation_prediction",
        signal=validation_prediction,
        expected_rows=len(validation),
    )
    _validate_signal_rows(
        name="train_representation",
        signal=train_representation,
        expected_rows=len(train),
    )
    _validate_signal_rows(
        name="validation_representation",
        signal=validation_representation,
        expected_rows=len(validation),
    )

    return TeacherSignals(
        train_prediction=train_prediction,
        validation_prediction=validation_prediction,
        train_representation=train_representation,
        validation_representation=(
            validation_representation
        ),
        teacher_candidate=teacher.candidate,
        teacher_role=teacher.role,
        teacher_checkpoint=str(
            teacher.checkpoint_path.relative_to(
                output_root.resolve()
            )
        ),
        teacher_checkpoint_sha256=(
            teacher.checkpoint_sha256
        ),
    )
