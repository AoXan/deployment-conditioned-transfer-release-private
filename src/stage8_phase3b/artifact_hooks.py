from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .formal_bridge import FormalCodeBridge, write_json


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in arrays:
            continue
        for key, value in arrays[split_name].items():
            if value is not None:
                payload[f"{split_name}_{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)


def _write_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path)


def persist_replay_inputs(
    *,
    bridge: FormalCodeBridge,
    config: dict,
    job: dict,
    output_root: Path,
) -> dict:
    """Capture replay inputs by calling original Stage 8 v4 helper functions.

    This is deliberately outside the training step and does not modify model
    state.  It uses the formal implementation's loader/split/preprocessing and
    deployment-mask helpers so the persisted artifacts are traceable to the
    original execution path.
    """

    execution = bridge.execution_module()
    artifacts = bridge.artifacts_module()
    contract = config["datasets"][job["dataset_id"]]
    frame, dataset_path = execution.load_frame(contract, bridge.formal_worktree)
    split = next(
        item
        for item in execution.build_contract_splits(frame, contract, seed=int(job["seed"]))
        if item.split_id == job["split_id"]
    )
    train, validation, test = execution._frames(frame, split, contract, job)
    target = contract["target_column"]
    dataset = execution._dataset(frame, dataset_path, job["dataset_id"], contract)
    strategy_metadata: dict[str, Any] = {"strategy": job.get("transfer_strategy")}
    if job.get("transfer_strategy") in {"pooled_source_target", "harmonised_transfer"}:
        source_contract = config["source_datasets"][job["source_dataset"]]
        source_frame, source_path = execution.load_frame(source_contract, bridge.formal_worktree)
        source_split = execution.build_contract_splits(source_frame, source_contract, seed=int(job["seed"]))[0]
        source_train, _, _ = execution._frames(
            source_frame,
            source_split,
            source_contract,
            {**job, "adaptation_mode": "full_adaptation"},
        )
        common = sorted(set(contract["weather_columns"]) & set(source_contract["weather_columns"]))
        if not common:
            raise ValueError("NO_RECOVERED_COMMON_FEATURE_INTERSECTION")
        source_target = source_contract["target_column"]
        source_train = source_train[[source_contract["sample_id_column"], source_target, *common]].copy()
        source_train = source_train.rename(
            columns={source_contract["sample_id_column"]: contract["sample_id_column"], source_target: target}
        )
        target_parts = [train.copy(), validation.copy(), test.copy()]
        if job.get("transfer_strategy") == "harmonised_transfer":
            source_mean, source_std = float(source_train[target].mean()), float(source_train[target].std())
            target_mean, target_std = float(train[target].mean()), float(train[target].std())
            if min(source_std, target_std) <= 0:
                raise ValueError("INVALID_DOMAIN_TARGET_SCALE")
            source_train[target] = (source_train[target] - source_mean) / source_std
            for part in target_parts:
                part[target] = (part[target] - target_mean) / target_std
            strategy_metadata["harmonisation"] = {
                "enabled": True,
                "features": common,
                "source_mean": source_mean,
                "source_std": source_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "source_path": str(source_path),
            }
        train, validation, test = target_parts
        pooled_train = pd.concat([source_train, train[[contract["sample_id_column"], target, *common]]], ignore_index=True)
        dataset = execution.UniversalDatasetV2(
            dataset_id=job["dataset_id"],
            frame=frame,
            weather_features=tuple(common),
            privileged_features=tuple(),
            target_column=target,
            sample_id_column=contract["sample_id_column"],
            year_column=contract["year_column"],
            target_unit=contract["target_unit"],
            sample_unit=contract["sample_unit"],
            source_path=dataset_path,
        )
        arrays = execution.prepare_arrays(dataset=dataset, train_frame=pooled_train, validation_frame=validation, test_frame=test)
    else:
        arrays = execution.prepare_arrays(dataset=dataset, train_frame=train, validation_frame=validation, test_frame=test)
    deployment_ledger = execution._apply_weather_deployment_condition(
        arrays,
        test,
        list(dataset.weather_features),
        job["deployment_condition"],
        int(job["seed"]),
    )

    root = Path(output_root) / "jobs" / "replay_artifacts" / str(job["candidate_id"])
    root.mkdir(parents=True, exist_ok=True)

    sample_col = contract["sample_id_column"]
    split_rows = []
    for name, ids in (
        ("train", split.train_ids),
        ("validation", split.validation_ids),
        ("test", split.test_ids),
    ):
        split_rows.extend({"sample_id": str(sample_id), "split": name} for sample_id in ids)
    _write_frame(root / "fold_assignments.csv", pd.DataFrame(split_rows))
    _write_frame(root / "fold_test_ids.csv", pd.DataFrame({"sample_id": test[sample_col].astype(str)}))
    write_json(
        root / "split_contract.json",
        {
            "split_id": split.split_id,
            "algorithm": split.algorithm,
            "random_state": split.random_state,
            "grouping_field": split.grouping_field,
            "split_fingerprint": split.id_hash,
            "train_count": len(split.train_ids),
            "validation_count": len(split.validation_ids),
            "test_count": len(split.test_ids),
        },
    )
    write_json(
        root / "feature_order.json",
        {
            "weather": list(dataset.weather_features),
            "privileged": list(dataset.privileged_features),
            "target": dataset.target_column,
        },
    )
    write_json(
        root / "modality_groups.json",
        {
            "weather": list(dataset.weather_features),
            "soil": list(dataset.privileged_features),
            "auxiliary": list(contract.get("ec_columns", [])),
        },
    )
    write_json(root / "mask_ledger.json", deployment_ledger)
    _write_npz(root / "arrays" / "transformed_inputs.npz", arrays)
    _write_joblib(root / "preprocessors" / "weather_preprocessor.joblib", arrays["weather_preprocessor"])
    if arrays.get("privileged_preprocessor") is not None:
        _write_joblib(root / "preprocessors" / "privileged_preprocessor.joblib", arrays["privileged_preprocessor"])
    write_json(
        root / "replay_identity.json",
        {
            "formal_candidate_id": job["candidate_id"],
            "dataset_id": job["dataset_id"],
            "split_id": job["split_id"],
            "condition": job["deployment_condition"],
            "seed": int(job["seed"]),
            "dataset_path": str(dataset_path),
            "data_hash": artifacts.sha256_file(Path(dataset_path)),
        },
    )
    write_json(
        root / "model_constructor.json",
        {
            "model_class": "UniversalWeatherRegressorV2",
            "encoder_class": "UniversalWeatherEncoderV2",
            "constructor_source": "original_stage8_v4_default",
            "strategy_metadata": strategy_metadata,
        },
    )
    return {
        "status": "persisted",
        "artifact_root": str(root),
        "split_fingerprint": split.id_hash,
        "weather_feature_count": len(dataset.weather_features),
        "privileged_feature_count": len(dataset.privileged_features),
    }
