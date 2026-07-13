from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.distillation_v4.universal_weather_v2.artifacts import load_checkpoint
from src.distillation_v4.universal_weather_v2.data import UniversalDatasetV2
from src.distillation_v4.universal_weather_v2.models import UniversalWeatherRegressorV2, UniversalWeatherEncoderV2
from src.distillation_v4.universal_weather_v2.training import evaluate_student, prepare_arrays, train_student_route_v2

from .artifacts import accept_artifacts, build_fingerprint, atomic_json, code_tree_hash, sha256_file, stable_hash
from .datasets import load_frame
from .methods import historical_m1_predictions, historical_m2_predictions
from .splits import build_contract_splits, deterministic_fraction_ids


def _dataset(frame: pd.DataFrame, path: Path, dataset_id: str, contract: dict) -> UniversalDatasetV2:
    return UniversalDatasetV2(
        dataset_id=dataset_id,
        frame=frame,
        weather_features=tuple(contract.get("weather_columns", [])),
        privileged_features=tuple(contract.get("soil_columns", [])),
        target_column=contract["target_column"],
        sample_id_column=contract["sample_id_column"],
        year_column=contract["year_column"],
        target_unit=contract["target_unit"],
        sample_unit=contract["sample_unit"],
        source_path=path,
    )


def _frames(frame: pd.DataFrame, split, contract: dict, cell: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = contract["sample_id_column"]
    indexed = frame.assign(_sample_id=frame[sample].astype(str)).set_index("_sample_id", drop=False)
    train = indexed.loc[list(split.train_ids)].copy()
    validation = indexed.loc[list(split.validation_ids)].copy()
    test = indexed.loc[list(split.test_ids)].copy()
    if cell["adaptation_mode"] == "few_shot_adaptation":
        chosen = deterministic_fraction_ids(train[sample].astype(str), [float(cell["adaptation_fraction"])], int(cell["seed"]))
        train = train.loc[train[sample].astype(str).isin(chosen[float(cell["adaptation_fraction"])])].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def _source_assets(config: dict, repository_root: Path, source_id: str) -> tuple[dict, dict, dict]:
    contract = config["source_datasets"][source_id]
    frame, path = load_frame(contract, repository_root)
    split = build_contract_splits(frame, contract)[0]
    registry = {
        source_id: {
            **contract,
            "path": str(path),
            "weather_features": list(contract.get("weather_columns", [])),
            "privileged_features": list(contract.get("soil_columns", [])),
        }
    }
    feature = {
        source_id: {
            "weather": contract.get("weather_columns", []),
            "soil": contract.get("soil_columns", []),
            "code_tree_hash": code_tree_hash(),
            "resolved_config_hash": stable_hash(config),
            "data_hash": sha256_file(path),
            "runtime": {"python": __import__("sys").version, "torch": torch.__version__},
        }
    }
    splits = {
        source_id: {
            "train_ids": list(split.train_ids),
            "validation_ids": list(split.validation_ids),
            "test_ids": list(split.test_ids),
            "split_fingerprint": split.id_hash,
        }
    }
    return registry, feature, splits


def _apply_weather_deployment_condition(arrays: dict, raw_test: pd.DataFrame, features: list[str], condition: str, seed: int) -> dict:
    ledger = {"condition": condition, "whole_weather_mask": False, "random_cell_fraction": 0.0}
    weather = arrays["test"]["weather"]
    if condition == "synthetic_no_weather":
        weather[:] = np.nan
        ledger["whole_weather_mask"] = True
    elif condition == "synthetic_random_0_15":
        rng = np.random.default_rng(seed)
        mask = rng.random(weather.shape) < 0.15
        weather[mask] = np.nan
        ledger["random_cell_fraction"] = float(mask.mean())
    elif condition == "natural":
        raw_missing = raw_test[features].isna().to_numpy()
        weather[raw_missing] = np.nan
        ledger["natural_missing_cells"] = int(raw_missing.sum())
    elif condition not in {"complete", "synthetic_no_soil"}:
        raise ValueError(f"UNSUPPORTED_DEPLOYMENT_CONDITION:{condition}")
    return ledger


def ensure_source_checkpoint(
    config: dict,
    repository_root: Path,
    output_root: Path,
    cell: dict,
    *,
    smoke: bool,
) -> Path:
    """Resolve an already completed route-specific source checkpoint.

    Target execution is forbidden from training or repairing source
    dependencies. Missing, incomplete, or unqualified dependencies fail
    closed and must be handled by the DAG dispatcher.
    """
    del config, repository_root

    source_id = str(cell["source_dataset"])
    route = str(cell["source_route"])
    seed = int(cell["seed"])

    directory = (
        output_root
        / ("smoke" if smoke else "jobs")
        / "source"
        / source_id
        / route
        / f"seed_{seed}"
    )

    checkpoint = directory / "student.pt"
    screening = directory / "screening.json"
    qualification = directory / "teacher_qualification.json"
    handler = directory / "remediation_source_handler.json"

    required = {
        "checkpoint": checkpoint,
        "screening": screening,
        "qualification": qualification,
        "handler": handler,
    }
    missing = [
        name
        for name, artifact in required.items()
        if not artifact.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "SOURCE_DEPENDENCY_NOT_READY:"
            f"{source_id}:{route}:{seed}:"
            + ",".join(missing)
        )

    screening_payload = json.loads(screening.read_text())
    qualification_payload = json.loads(
        qualification.read_text()
    )
    handler_payload = json.loads(handler.read_text())

    if (
        screening_payload.get("status")
        != "SCREENING_ACCEPTED"
    ):
        raise ValueError(
            "SOURCE_SCREENING_NOT_ACCEPTED:"
            f"{source_id}:{route}:{seed}"
        )

    if qualification_payload.get("status") not in {
        "TEACHER_QUALIFIED",
        "TEACHER_NOT_REQUIRED",
    }:
        raise ValueError(
            "SOURCE_TEACHER_QUALIFICATION_NOT_ACCEPTED:"
            f"{source_id}:{route}:{seed}"
        )

    declared_hash = handler_payload.get(
        "checkpoint_sha256"
    )
    actual_hash = sha256_file(checkpoint)

    if not declared_hash:
        raise ValueError(
            "SOURCE_CHECKPOINT_DECLARED_HASH_MISSING:"
            f"{source_id}:{route}:{seed}"
        )

    if declared_hash != actual_hash:
        raise ValueError(
            "SOURCE_CHECKPOINT_HASH_MISMATCH:"
            f"{source_id}:{route}:{seed}"
        )

    return checkpoint


def _load_state(checkpoint: Path) -> dict:
    model = UniversalWeatherRegressorV2(encoder=UniversalWeatherEncoderV2())
    payload = load_checkpoint(path=checkpoint, model=model, map_location=torch.device("cpu"), strict=True)
    return {key: value.detach().cpu().clone() for key, value in payload["state_dict"].items()}


def execute_target_job(
    config: dict,
    cell: dict,
    *,
    repository_root: Path,
    output_root: Path,
    smoke: bool,
) -> dict:
    checkpoint = None
    contract = config["datasets"][cell["dataset_id"]]
    frame, path = load_frame(contract, repository_root)
    split = next(item for item in build_contract_splits(frame, contract, seed=int(cell["seed"])) if item.split_id == cell["split_id"])
    train, validation, test = _frames(frame, split, contract, cell)
    if smoke:
        sample = contract["sample_id_column"]
        train = train.sort_values(sample).head(256).copy()
        validation = validation.sort_values(sample).head(64).copy()
        test = test.sort_values(sample).head(64).copy()
    target = contract["target_column"]
    method = cell["missing_modality_method"]
    if method in {"imputation", "missing_indicators", "late_fusion"}:
        outcome = historical_m1_predictions(
            train=train,
            test=test,
            target_column=target,
            weather_columns=list(
                contract.get("weather_columns", [])
            ),
            soil_columns=list(
                contract.get("soil_columns", [])
            ),
            ec_columns=list(
                contract.get("ec_columns", [])
            ),
            method=method,
            condition=cell["deployment_condition"],
            seed=int(cell["seed"]),
        )
        prediction = outcome["predictions"]
        training_ledger = outcome
    elif method == "teacher_student_m2":
        outcome = historical_m2_predictions(
            train=train,
            test=test,
            target_column=target,
            weather_columns=list(contract["weather_columns"]),
            soil_columns=list(contract["soil_columns"]),
            condition=cell["deployment_condition"],
            seed=int(cell["seed"]),
        )
        prediction = outcome["predictions"]
        training_ledger = outcome
    else:
        strategy = cell["transfer_strategy"]
        checkpoint = None if strategy == "target_scratch" else ensure_source_checkpoint(config, repository_root, output_root, cell, smoke=smoke)
        initial = None if checkpoint is None else _load_state(checkpoint)
        device = torch.device("cpu")
        dataset = _dataset(frame, path, cell["dataset_id"], contract)
        inverse_target = None
        strategy_ledger: dict = {"strategy": strategy}
        if strategy in {"pooled_source_target", "harmonised_transfer"}:
            source_contract = config["source_datasets"][cell["source_dataset"]]
            source_frame, source_path = load_frame(source_contract, repository_root)
            source_split = build_contract_splits(source_frame, source_contract, seed=int(cell["seed"]))[0]
            source_train, _, _ = _frames(source_frame, source_split, source_contract, {**cell, "adaptation_mode": "full_adaptation"})
            common = sorted(set(contract["weather_columns"]) & set(source_contract["weather_columns"]))
            if not common:
                raise ValueError("NO_RECOVERED_COMMON_FEATURE_INTERSECTION")
            source_target = source_contract["target_column"]
            source_train = source_train[[source_contract["sample_id_column"], source_target, *common]].copy()
            source_train = source_train.rename(columns={source_contract["sample_id_column"]: contract["sample_id_column"], source_target: target})
            target_parts = [train.copy(), validation.copy(), test.copy()]
            if strategy == "harmonised_transfer":
                source_mean, source_std = float(source_train[target].mean()), float(source_train[target].std())
                target_mean, target_std = float(train[target].mean()), float(train[target].std())
                if min(source_std, target_std) <= 0:
                    raise ValueError("INVALID_DOMAIN_TARGET_SCALE")
                source_train[target] = (source_train[target] - source_mean) / source_std
                for part in target_parts:
                    part[target] = (part[target] - target_mean) / target_std
                inverse_target = (target_mean, target_std)
                strategy_ledger["harmonisation"] = {
                    "enabled": True,
                    "feature_mapping": "EXACT_COMMON_FEATURE_INTERSECTION",
                    "features": common,
                    "target_scaling_fit_scope": "SOURCE_TRAIN_AND_TARGET_TRAIN_SEPARATELY",
                    "source_mean": source_mean,
                    "source_std": source_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "source_path": str(source_path),
                }
            train, validation, test = target_parts
            pooled_train = pd.concat([source_train, train[[contract["sample_id_column"], target, *common]]], ignore_index=True)
            dataset = UniversalDatasetV2(
                dataset_id=cell["dataset_id"], frame=frame, weather_features=tuple(common), privileged_features=tuple(),
                target_column=target, sample_id_column=contract["sample_id_column"], year_column=contract["year_column"],
                target_unit=contract["target_unit"], sample_unit=contract["sample_unit"], source_path=path,
            )
            arrays = prepare_arrays(dataset=dataset, train_frame=pooled_train, validation_frame=validation, test_frame=test)
            strategy_ledger["training_provenance"] = {
                "source_rows": len(source_train), "target_rows": len(train), "target_test_rows_in_training": 0
            }
        else:
            arrays = prepare_arrays(dataset=dataset, train_frame=train, validation_frame=validation, test_frame=test)
        deployment_ledger = _apply_weather_deployment_condition(
            arrays, test, list(dataset.weather_features), cell["deployment_condition"], int(cell["seed"])
        )
        if strategy == "source_only":
            if checkpoint is None:
                raise RuntimeError("SOURCE_CHECKPOINT_REQUIRED")
            model = UniversalWeatherRegressorV2(encoder=UniversalWeatherEncoderV2()).to(device)
            load_checkpoint(path=checkpoint, model=model, map_location=device, strict=True)
            _, prediction = evaluate_student(
                model=model, arrays=arrays["test"], feature_names=list(dataset.weather_features), device=device, batch_size=64
            )
            training_ledger = {"strategy": strategy, "target_train_used": False, "source_checkpoint": str(checkpoint), "deployment": deployment_ledger}
        else:
            result = train_student_route_v2(
                route="supervised",
                arrays=arrays,
                feature_names=list(dataset.weather_features),
                teacher=None,
                seed=int(cell["seed"]),
                epochs=2 if smoke else 80,
                patience=2 if smoke else 12,
                batch_size=64 if smoke else 256,
                learning_rate=1e-3 if strategy == "target_scratch" else 2e-4,
                device=device,
                checkpoint_path=output_root / ("smoke" if smoke else "jobs") / "target_checkpoints" / f"{cell['candidate_id']}.pt",
                metadata={"job": cell, "strategy": strategy, "smoke_only": smoke},
                initial_state_dict=None if strategy == "target_scratch" else initial,
            )
            prediction = result["test_prediction"]
            if inverse_target is not None:
                prediction = np.asarray(prediction) * inverse_target[1] + inverse_target[0]
            training_ledger = {
                **strategy_ledger,
                "history": result["history"],
                "source_checkpoint": None if checkpoint is None else str(checkpoint),
                "initialisation": "RANDOM_MATCHED" if checkpoint is None else "ROUTE_SPECIFIC_SOURCE_CHECKPOINT",
                "deployment": deployment_ledger,
            }
    job_dir = output_root / ("smoke" if smoke else "jobs") / "targets" / cell["candidate_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {"sample_id": test[contract["sample_id_column"]].astype(str), "y_true": test[target].astype(float), "y_pred": np.asarray(prediction, dtype=float)}
    )
    predictions.to_csv(job_dir / "predictions.csv", index=False)
    atomic_json(job_dir / "fold_ids.json", {"test_ids": test[contract["sample_id_column"]].astype(str).tolist(), "split_fingerprint": split.id_hash})
    checkpoint_value = training_ledger.get("source_checkpoint") if isinstance(training_ledger, dict) else None
    checkpoint_hash = sha256_file(Path(checkpoint_value)) if checkpoint_value and Path(checkpoint_value).is_file() else None
    fingerprint = build_fingerprint(
        cell,
        namespace=str(output_root),
        extra={"dataset_path": str(path), "split": split.id_hash, "checkpoint_hash": checkpoint_hash},
        resolved_config=config,
    )
    if not isinstance(training_ledger, dict):
        raise TypeError(
            "TRAINING_LEDGER_MUST_BE_DICT"
        )

    declared_weather = list(
        contract.get("weather_columns", [])
    )
    declared_soil = list(
        contract.get("soil_columns", [])
    )
    declared_ec = list(
        contract.get("ec_columns", [])
    )

    declared_all = (
        declared_weather
        + declared_soil
        + declared_ec
    )

    if len(declared_all) != len(set(declared_all)):
        raise ValueError(
            "DUPLICATE_DATASET_CONTRACT_FEATURES:"
            f"{declared_all}"
        )

    missing_runtime_columns = [
        column
        for column in declared_all
        if column not in train.columns
        or column not in test.columns
    ]

    if missing_runtime_columns:
        raise ValueError(
            "RUNTIME_CONTRACT_COLUMNS_MISSING:"
            f"{missing_runtime_columns}"
        )

    training_ledger["declared_feature_contract"] = {
        "weather_columns": declared_weather,
        "soil_columns": declared_soil,
        "ec_columns": declared_ec,
        "feature_order": declared_all,
        "feature_count": len(declared_all),
        "selection_policy": (
            "EXPLICIT_VERSIONED_DATASET_CONTRACT"
        ),
    }

    training_ledger["runtime_feature_assertion"] = {
        "declared_columns_present": True,
        "duplicate_columns": False,
        "dataset_id": cell["dataset_id"],
        "candidate_id": cell["candidate_id"],
    }

    atomic_json(job_dir / "training_ledger.json", training_ledger)
    atomic_json(
        job_dir / "manifest.json",
        {
            "job": cell,
            "fingerprint": fingerprint,
            "code_tree_hash": code_tree_hash(),
            "resolved_config_hash": stable_hash(config),
            "data_hash": sha256_file(path),
            "dataset_path": str(path),
            "split_fingerprint": split.id_hash,
            "namespace": str(output_root),
            "source_checkpoint": (
                str(checkpoint)
                if checkpoint is not None
                else None
            ),
            "source_checkpoint_hash": checkpoint_hash,
            "dependency_checkpoint_hash": checkpoint_hash,
            "expected_checkpoint_hash": checkpoint_hash,
            "actual_checkpoint_hash_loaded": checkpoint_hash,
            "plan_fingerprint": cell.get("plan_fingerprint"),
            "feature_contract_hash": stable_hash(
                {
                    "dataset_id": cell["dataset_id"],
                    "weather_columns": list(
                        contract.get("weather_columns", [])
                    ),
                    "soil_columns": list(
                        contract.get("soil_columns", [])
                    ),
                    "model_contract": cell.get("model_contract"),
                }
            ),
            "target_test_used_for_training": False,
            "target_test_used_for_selection": False,
            "outer_test_used_for_selection": False,
            "smoke_only": smoke,
        },
    )
    acceptance = accept_artifacts(job_dir, expected_fingerprint=fingerprint)
    return {"status": "COMPLETED", "job_id": cell["candidate_id"], "acceptance": acceptance, "smoke_only": smoke}
