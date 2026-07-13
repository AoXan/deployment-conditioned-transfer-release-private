from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class HarmonisedData:
    source: pd.DataFrame
    target: pd.DataFrame
    feature_columns: list[str]
    ledger: dict


@dataclass
class TransferContext:
    strategy: str
    source_train: pd.DataFrame
    target_train: pd.DataFrame
    target_validation: pd.DataFrame
    target_test: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    seed: int


def _validate_explicit_harmonisation_features(
    source: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    columns = list(feature_columns)

    if not columns:
        raise ValueError(
            "NO_EXPLICIT_HARMONISATION_FEATURES"
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "DUPLICATE_HARMONISATION_FEATURES:"
            f"{columns}"
        )

    missing_source = [
        column
        for column in columns
        if column not in source.columns
    ]
    missing_target = [
        column
        for column in columns
        if column not in target.columns
    ]

    if missing_source or missing_target:
        raise ValueError(
            "HARMONISATION_CONTRACT_COLUMNS_MISSING:"
            f"source={missing_source}:target={missing_target}"
        )

    non_numeric_source = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(
            source[column]
        )
    ]
    non_numeric_target = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(
            target[column]
        )
    ]

    if non_numeric_source or non_numeric_target:
        raise ValueError(
            "HARMONISATION_CONTRACT_COLUMNS_NON_NUMERIC:"
            f"source={non_numeric_source}:"
            f"target={non_numeric_target}"
        )

    return columns


def apply_harmonisation(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    enabled: bool,
    target_column: str,
    feature_columns: list[str],
) -> HarmonisedData:
    common = _validate_explicit_harmonisation_features(
        source,
        target,
        feature_columns,
    )
    source_out, target_out = source.copy(), target.copy()
    ledger = {
        "enabled": enabled,
        "feature_mapping": "EXPLICIT_CALLER_FEATURE_CONTRACT",
        "feature_columns": common,
        "unit_alignment": "TARGET_CONTRACT_REQUIRED",
        "temporal_aggregation": "PRESERVE_PRECOMPUTED_CUTOFF_SAFE_FEATURES",
        "missing_feature_handling": "FOLD_LOCAL_IMPUTATION",
    }
    if enabled:
        source_mean, source_std = float(source[target_column].mean()), float(source[target_column].std())
        target_mean, target_std = float(target[target_column].mean()), float(target[target_column].std())
        if source_std <= 0 or target_std <= 0:
            raise ValueError("INVALID_DOMAIN_TARGET_SCALE")
        source_out[target_column] = (source[target_column] - source_mean) / source_std
        target_out[target_column] = (target[target_column] - target_mean) / target_std
        ledger.update(
            {
                "target_scaling_fit_scope": "TRAIN_ONLY_BY_DOMAIN",
                "source_target_mean": source_mean,
                "source_target_std": source_std,
                "target_target_mean": target_mean,
                "target_target_std": target_std,
            }
        )
    else:
        ledger["target_scaling_fit_scope"] = "NOT_APPLICABLE"
    return HarmonisedData(source_out, target_out, common, ledger)


def _pipeline(columns: list[str], *, model: str, seed: int) -> Pipeline:
    pre = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), columns)])
    estimator = Ridge(alpha=1.0) if model == "ridge" else RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=1)
    return Pipeline([("preprocessor", pre), ("model", estimator)])


def historical_m1_predictions(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    weather_columns: list[str],
    soil_columns: list[str],
    ec_columns: list[str],
    method: str,
    condition: str,
    seed: int,
) -> dict:
    train = train.copy()
    test = test.copy()

    modality = {
        "weather": list(weather_columns),
        "soil": list(soil_columns),
        "ec": list(ec_columns),
    }

    declared_columns = [
        column
        for columns in modality.values()
        for column in columns
    ]

    if len(declared_columns) != len(set(declared_columns)):
        raise ValueError(
            "M1_DUPLICATE_CONTRACT_COLUMNS:"
            f"{declared_columns}"
        )

    missing_train = [
        column
        for column in declared_columns
        if column not in train.columns
    ]
    missing_test = [
        column
        for column in declared_columns
        if column not in test.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            "M1_CONTRACT_COLUMNS_MISSING:"
            f"train={missing_train}:test={missing_test}"
        )

    non_numeric_train = [
        column
        for column in declared_columns
        if not pd.api.types.is_numeric_dtype(
            train[column]
        )
    ]
    non_numeric_test = [
        column
        for column in declared_columns
        if not pd.api.types.is_numeric_dtype(
            test[column]
        )
    ]

    if non_numeric_train or non_numeric_test:
        raise ValueError(
            "M1_CONTRACT_COLUMNS_NON_NUMERIC:"
            f"train={non_numeric_train}:"
            f"test={non_numeric_test}"
        )

    missing_modality = {
        "synthetic_no_soil": "soil",
        "synthetic_no_weather": "weather",
    }.get(condition)

    available = {
        name: list(columns)
        for name, columns in modality.items()
        if columns and name != missing_modality
    }

    if not available:
        raise ValueError(
            "NO_AVAILABLE_MODALITY"
        )

    expected_available = {
        name: columns
        for name, columns in modality.items()
        if columns and name != missing_modality
    }

    if available != expected_available:
        raise ValueError(
            "M1_AVAILABLE_MODALITY_CONTRACT_MISMATCH:"
            f"actual={available}:"
            f"expected={expected_available}"
        )

    mask_ledger = {
        "condition": condition,
        "whole_modality": missing_modality,
        "random_cell_mask": False,
    }

    if condition == "synthetic_random_0_15":
        columns_to_mask = sorted(
            {
                column
                for values in available.values()
                for column in values
            }
        )

        rng = np.random.default_rng(seed)
        mask = (
            rng.random(
                (
                    len(test),
                    len(columns_to_mask),
                )
            )
            < 0.15
        )

        test[columns_to_mask] = (
            test[columns_to_mask]
            .astype(float)
            .mask(mask)
        )

        mask_ledger.update(
            {
                "random_cell_mask": True,
                "realised_fraction": float(
                    mask.mean()
                ),
            }
        )

    if method == "late_fusion":
        predictions = []
        availability = []
        actual_model_inputs = {}

        for name, columns in sorted(
            available.items()
        ):
            actual_columns = list(columns)

            if actual_columns != available[name]:
                raise ValueError(
                    "M1_LATE_FUSION_FEATURE_ORDER_MISMATCH:"
                    f"{name}:"
                    f"actual={actual_columns}:"
                    f"expected={available[name]}"
                )

            model = _pipeline(
                actual_columns,
                model="rf",
                seed=seed,
            )

            model.fit(
                train[actual_columns],
                train[target_column],
            )

            predictions.append(
                model.predict(
                    test[actual_columns]
                )
            )

            availability.append(
                test[actual_columns]
                .notna()
                .any(axis=1)
                .to_numpy(float)
            )

            actual_model_inputs[name] = (
                actual_columns
            )

        matrix = np.column_stack(
            predictions
        )
        weights = np.column_stack(
            availability
        )

        weights = np.where(
            weights.sum(
                axis=1,
                keepdims=True,
            )
            > 0,
            weights,
            1.0,
        )

        prediction = (
            (matrix * weights).sum(axis=1)
            / weights.sum(axis=1)
        )

        ledger = {
            "base_estimator": "random_forest",
            "fusion": "availability_equal_weight",
            "modalities": sorted(available),
            "actual_model_inputs": (
                actual_model_inputs
            ),
            "actual_flat_feature_order": [
                column
                for name in sorted(
                    actual_model_inputs
                )
                for column in (
                    actual_model_inputs[name]
                )
            ],
        }

    else:
        columns = sorted(
            {
                column
                for values in available.values()
                for column in values
            }
        )

        expected_columns = sorted(
            {
                column
                for values in expected_available.values()
                for column in values
            }
        )

        if columns != expected_columns:
            raise ValueError(
                "M1_EARLY_FUSION_FEATURE_ORDER_MISMATCH:"
                f"actual={columns}:"
                f"expected={expected_columns}"
            )

        train_x = train[columns].copy()
        test_x = test[columns].copy()

        model_input_columns = list(columns)

        if method == "missing_indicators":
            for column in columns:
                missing_column = (
                    f"{column}__missing"
                )

                train_x[missing_column] = (
                    train_x[column]
                    .isna()
                    .astype(int)
                )

                test_x[missing_column] = (
                    test_x[column]
                    .isna()
                    .astype(int)
                )

                model_input_columns.append(
                    missing_column
                )

        if list(train_x.columns) != (
            model_input_columns
        ):
            raise ValueError(
                "M1_TRAIN_MATRIX_COLUMN_ORDER_MISMATCH:"
                f"actual={list(train_x.columns)}:"
                f"expected={model_input_columns}"
            )

        if list(test_x.columns) != (
            model_input_columns
        ):
            raise ValueError(
                "M1_TEST_MATRIX_COLUMN_ORDER_MISMATCH:"
                f"actual={list(test_x.columns)}:"
                f"expected={model_input_columns}"
            )

        model = _pipeline(
            model_input_columns,
            model="ridge",
            seed=seed,
        )

        model.fit(
            train_x,
            train[target_column],
        )

        prediction = model.predict(
            test_x
        )

        ledger = {
            "base_estimator": "ridge",
            "fusion": "early",
            "method": method,
            "actual_contract_feature_order": (
                columns
            ),
            "actual_model_input_order": (
                model_input_columns
            ),
        }

    ledger.update(
        {
            "weather_columns": (
                modality["weather"]
            ),
            "soil_columns": (
                modality["soil"]
            ),
            "ec_columns": (
                modality["ec"]
            ),
            "available_modalities": (
                sorted(available)
            ),
            "feature_selection": (
                "EXPLICIT_DATASET_CONTRACT"
            ),
        }
    )

    return {
        "predictions": np.asarray(
            prediction
        ),
        "mask_ledger": mask_ledger,
        "model_ledger": ledger,
    }




def historical_m2_predictions(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    weather_columns: list[str],
    soil_columns: list[str],
    condition: str,
    seed: int,
) -> dict:
    """Faithful generic form of the historical fixed 0.5 pseudo-target blend.

    This is deliberately labelled as a historical control, not true KD.
    Feature selection is restricted to the explicit dataset contract.
    """
    train = train.copy()
    test = test.copy()

    weather = list(weather_columns)
    soil = list(soil_columns)

    if not weather or not soil:
        raise ValueError("M2_REQUIRES_WEATHER_AND_SOIL")

    missing_train = [
        column
        for column in weather + soil
        if column not in train.columns
    ]
    missing_test = [
        column
        for column in weather
        if column not in test.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            "M2_CONTRACT_COLUMNS_MISSING:"
            f"train={missing_train}:test={missing_test}"
        )

    non_numeric = [
        column
        for column in weather + soil
        if not pd.api.types.is_numeric_dtype(train[column])
    ]

    if non_numeric:
        raise ValueError(
            f"M2_CONTRACT_COLUMNS_NON_NUMERIC:{non_numeric}"
        )
    complete = train[weather + soil + [target_column]].notna().all(axis=1)
    if int(complete.sum()) < 5:
        raise ValueError("M2_INSUFFICIENT_COMPLETE_CASES")
    teacher = _pipeline(weather + soil, model="rf", seed=seed)
    teacher.fit(train.loc[complete, weather + soil], train.loc[complete, target_column])
    teacher_pseudo = teacher.predict(train[weather + soil])
    blended = 0.5 * train[target_column].to_numpy(float) + 0.5 * teacher_pseudo
    student = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=seed)),
        ]
    )
    student.fit(train[weather], blended)
    if condition == "synthetic_no_weather":
        test[weather] = test[weather].astype(float)
        test.loc[:, weather] = np.nan
    elif condition == "synthetic_random_0_15":
        rng = np.random.default_rng(seed)
        mask = rng.random((len(test), len(weather))) < 0.15
        test[weather] = test[weather].astype(float).mask(mask)
    return {
        "predictions": np.asarray(student.predict(test[weather]), dtype=float),
        "teacher_predictions_train": np.asarray(teacher_pseudo, dtype=float),
        "model_ledger": {
            "status": "HISTORICAL_CONTROL_ONLY",
            "teacher": "complete_case_random_forest",
            "student": "mlp_weather_only",
            "pseudo_target": "0.5*observed_target+0.5*teacher_prediction",
            "condition": condition,
            "weather_columns": weather,
            "soil_columns": soil,
            "feature_selection": "EXPLICIT_DATASET_CONTRACT",
        },
    }


def _fit_ridge(train: pd.DataFrame, test: pd.DataFrame, *, columns: list[str], target: str, seed: int) -> tuple[np.ndarray, Pipeline]:
    model = _pipeline(columns, model="ridge", seed=seed)
    model.fit(train[columns], train[target])
    return np.asarray(model.predict(test[columns]), dtype=float), model


def _base_result(context: TransferContext, predictions: np.ndarray, *, training_ids: list[str], ledger: dict, model: object) -> dict:
    if predictions.size != len(context.target_test) or not np.isfinite(predictions).all():
        raise RuntimeError("INVALID_TRANSFER_PREDICTIONS")
    return {
        "status": "EXECUTION_COMPLETE",
        "strategy": context.strategy,
        "predictions": predictions,
        "test_ids": context.target_test["sample_id"].astype(str).tolist(),
        "training_ids": list(map(str, training_ids)),
        "target_test_used_for_training": False,
        "harmonisation_ledger": ledger,
        "model": model,
    }


def _source_only(context: TransferContext) -> dict:
    prediction, model = _fit_ridge(
        context.source_train, context.target_test, columns=context.feature_columns, target=context.target_column, seed=context.seed
    )
    result = _base_result(
        context,
        prediction,
        training_ids=context.source_train.sample_id.astype(str).tolist(),
        ledger={"enabled": False, "target_scaling_fit_scope": "NOT_APPLICABLE"},
        model=model,
    )
    result["refit_target"] = False
    return result


def _ordinary(context: TransferContext) -> dict:
    # Source fit followed by target-train-only partial adaptation.  The target
    # validation partition is never fitted and is reserved for the caller's
    # frozen selection rule.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    source_x = scaler.fit_transform(imputer.fit_transform(context.source_train[context.feature_columns]))
    target_x = scaler.transform(imputer.transform(context.target_train[context.feature_columns]))
    model = SGDRegressor(loss="huber", alpha=1e-4, max_iter=1000, tol=1e-6, random_state=context.seed)
    model.fit(source_x, context.source_train[context.target_column])
    model.partial_fit(target_x, context.target_train[context.target_column])
    prediction = model.predict(scaler.transform(imputer.transform(context.target_test[context.feature_columns])))
    result = _base_result(
        context,
        np.asarray(prediction, dtype=float),
        training_ids=[*context.source_train.sample_id.astype(str), *context.target_train.sample_id.astype(str)],
        ledger={"enabled": False, "target_scaling_fit_scope": "NOT_APPLICABLE"},
        model={"imputer": imputer, "scaler": scaler, "estimator": model},
    )
    result["initialisation"] = "ROUTE_SPECIFIC_SOURCE_FIT"
    return result


def _harmonised(context: TransferContext) -> dict:
    harmonised = apply_harmonisation(
        context.source_train,
        context.target_train,
        enabled=True,
        target_column=context.target_column,
        feature_columns=list(context.feature_columns),
    )
    columns = list(harmonised.feature_columns)

    if columns != list(context.feature_columns):
        raise ValueError(
            "HARMONISATION_ACTUAL_FEATURE_ORDER_MISMATCH:"
            f"actual={columns}:"
            f"declared={list(context.feature_columns)}"
        )
    if not columns:
        raise ValueError("DECLARED_FEATURES_NOT_IN_HARMONISED_INTERSECTION")
    pooled = pd.concat([harmonised.source, harmonised.target], ignore_index=True)
    prediction_scaled, model = _fit_ridge(
        pooled, context.target_test, columns=columns, target=context.target_column, seed=context.seed
    )
    target_mean = harmonised.ledger["target_target_mean"]
    target_std = harmonised.ledger["target_target_std"]
    prediction = prediction_scaled * target_std + target_mean
    return _base_result(
        context,
        prediction,
        training_ids=[*context.source_train.sample_id.astype(str), *context.target_train.sample_id.astype(str)],
        ledger=harmonised.ledger,
        model=model,
    )


def _pooled(context: TransferContext) -> dict:
    pooled = pd.concat([context.source_train, context.target_train], ignore_index=True)
    prediction, model = _fit_ridge(
        pooled, context.target_test, columns=context.feature_columns, target=context.target_column, seed=context.seed
    )
    result = _base_result(
        context,
        prediction,
        training_ids=pooled.sample_id.astype(str).tolist(),
        ledger={"enabled": False, "target_scaling_fit_scope": "NOT_APPLICABLE"},
        model=model,
    )
    result["training_provenance"] = {"source_rows": len(context.source_train), "target_rows": len(context.target_train)}
    return result


def _scratch(context: TransferContext) -> dict:
    prediction, model = _fit_ridge(
        context.target_train, context.target_test, columns=context.feature_columns, target=context.target_column, seed=context.seed
    )
    result = _base_result(
        context,
        prediction,
        training_ids=context.target_train.sample_id.astype(str).tolist(),
        ledger={"enabled": False, "target_scaling_fit_scope": "NOT_APPLICABLE"},
        model=model,
    )
    result["initialisation"] = "RANDOM_MATCHED"
    return result


METHOD_HANDLERS = {
    "source_only": _source_only,
    "ordinary_transfer": _ordinary,
    "harmonised_transfer": _harmonised,
    "pooled_source_target": _pooled,
    "target_scratch": _scratch,
}
