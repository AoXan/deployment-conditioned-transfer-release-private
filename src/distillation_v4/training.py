from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.distillation_program.next_models import ModalitySpecificRegressor, SharedDenseRegressor, count_capacity

from .artifacts import ArtifactStore
from .matrix import ScientificJob
from .repair_trainer import fit_repair_regressor
from .training_contract import NeuralTrainingContract


@dataclass(frozen=True)
class TrainingResult:
    engineering_status: str
    metrics: dict[str, float]
    runtime: dict[str, Any]


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _fingerprints(job: Any, frame: pd.DataFrame, features: list[str], model: dict[str, Any]) -> dict[str, str]:
    return {
        "code_tree": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": _sha_json(job.__dict__ if hasattr(job, "__dict__") else job),
        "data": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "view": _sha_json({"columns": list(frame), "rows": len(frame)}),
        "split": _sha_json(frame[["sample_id", "year"]].to_dict("records")),
        "feature": _sha_json(features),
        "model": _sha_json(model),
        "runtime": _sha_json({"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}),
    }


def _metrics(y: np.ndarray, prediction: np.ndarray, groups: pd.Series | None = None) -> dict[str, float]:
    result = {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
    }
    if groups is not None:
        grouped = pd.DataFrame({"y": y, "p": prediction, "group": groups.astype(str).to_numpy()})
        values = grouped.assign(ae=lambda x: abs(x.y - x.p)).groupby("group").ae.mean()
        result.update(environment_balanced_mae=float(values.mean()), worst_group_mae=float(values.max()), group_count=int(len(values)))
    return result


def _feature_sets(frame: pd.DataFrame, profile: str, seed: int) -> tuple[pd.DataFrame, list[str]]:
    weather = [name for name in frame if name.startswith("weather_")]
    soil = [name for name in frame if name.startswith("soil_")]
    values = frame.copy()
    for name in soil:
        values[f"{name}__missing"] = values[name].isna().astype(float)
    masks = [f"{name}__missing" for name in soil]
    if profile == "deployable":
        features = weather
    elif profile == "soil_only":
        features = soil
    elif profile == "weather_soil":
        features = weather + soil
    elif profile == "mask_only":
        features = weather + masks
    elif profile == "shuffled_soil":
        rng = np.random.default_rng(seed)
        for name in soil:
            values[name] = values[name].to_numpy()[rng.permutation(len(values))]
        features = weather + soil
    else:
        raise ValueError(f"UNKNOWN_INPUT_PROFILE:{profile}")
    if not features:
        raise ValueError(f"FEATURE_PROFILE_EMPTY:{profile}")
    return values, features


def _partitions(frame: pd.DataFrame, outer_test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_year = outer_test_year - 1
    train = frame.loc[frame.year < validation_year].copy()
    validation = frame.loc[frame.year.eq(validation_year)].copy()
    if train.empty or validation.empty:
        raise ValueError("DEVELOPMENT_PARTITION_EMPTY")
    return train, validation


def _write_result(output: Path, *, job: Any, frame: pd.DataFrame, validation: pd.DataFrame, prediction: np.ndarray, features: list[str], model_ledger: dict[str, Any], runtime: dict[str, Any], scope: str = "OUTER_TRAIN_INNER_VALIDATION_ONLY") -> TrainingResult:
    groups = validation["group"] if "group" in validation else validation["sample_id"]
    metrics = _metrics(validation.target_yield.to_numpy(float), prediction, groups)
    store = ArtifactStore(output)
    store.write_predictions(pd.DataFrame({"sample_id": validation.sample_id.astype(str), "y_true": validation.target_yield, "y_pred": prediction}))
    store.write_fold_ids(pd.DataFrame({"sample_id": validation.sample_id.astype(str), "split": "test", "evaluation_role": "inner_validation"}))
    store.write_metrics(metrics)
    manifest = {
        "job_id": getattr(job, "job_id", str(job)),
        "gate_data_scope": scope,
        "outer_test_role": "FINAL_ESTIMATION_ONLY",
        "outer_test_may_enable_downstream_jobs": False,
        "engineering_status": "EXECUTION_COMPLETE_UNVERIFIED",
        "fingerprints": _fingerprints(job, frame, features, model_ledger),
        "features": features,
        "model": model_ledger,
        "runtime": runtime,
    }
    store.write_manifest(manifest)
    store.accept()
    store.complete()
    return TrainingResult("ENGINEERING_ACCEPTED", metrics, runtime)


def _sklearn(job: ScientificJob, train: pd.DataFrame, validation: pd.DataFrame, features: list[str], output: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if job.model_family == "ridge":
        estimator = Ridge(alpha=1.0)
    elif job.model_family == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=job.seed, early_stopping=False)
    elif job.model_family == "random_forest":
        estimator = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, max_features=0.7, n_jobs=1, random_state=job.seed)
    else:
        raise ValueError(job.model_family)
    pipeline = Pipeline([("preprocess", ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), features)])), ("model", estimator)])
    start = time.perf_counter()
    pipeline.fit(train[features], train.target_yield)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    prediction = pipeline.predict(validation[features])
    inference = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["preprocess"], output / "preprocessor.joblib")
    joblib.dump(pipeline.named_steps["model"], output / "model.joblib")
    ledger = {"family": job.model_family, "parameters": None, "training_steps": None}
    runtime = {"wall_time_seconds": fit_time + inference, "fit_time_seconds": fit_time, "inference_time_seconds": inference, "optimizer_steps": None, "device": "cpu", "peak_memory_bytes": None, "runtime_evidence_status": "RUNTIME_EVIDENCE_INCOMPLETE"}
    return prediction, ledger, runtime


def _tensors(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Any]:
    weather = [name for name in features if name.startswith("weather_")]
    soil = [name for name in features if name.startswith("soil_") and not name.endswith("__missing")]
    groups = {"weather": weather}
    if soil:
        groups["soil"] = soil
    transformers: dict[str, Pipeline] = {}
    tr: dict[str, torch.Tensor] = {}
    va: dict[str, torch.Tensor] = {}
    for name, columns in groups.items():
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        tr[name] = torch.tensor(pipe.fit_transform(train[columns]), dtype=torch.float32)
        va[name] = torch.tensor(pipe.transform(validation[columns]), dtype=torch.float32)
        transformers[name] = pipe
    return tr, va, transformers


def _neural(job: ScientificJob, train: pd.DataFrame, validation: pd.DataFrame, features: list[str], output: Path, max_epochs: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(job.seed)
    tr, va, preprocessors = _tensors(train, validation, features)
    dims = {name: tensor.shape[1] for name, tensor in tr.items()}
    model = SharedDenseRegressor(dims, hidden=32) if job.variant == "neural_deployable" else ModalitySpecificRegressor(dims, hidden=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target = torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32)
    validation_target = torch.tensor(validation.target_yield.to_numpy(), dtype=torch.float32)
    train_mask = torch.ones(len(train), len(dims)); validation_mask = torch.ones(len(validation), len(dims))
    best: dict[str, Any] | None = None; patience = 5; stale = 0; steps = 0
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train(); optimizer.zero_grad()
        inputs = tr
        if job.model_family == "neural_masked" and "soil" in inputs:
            masked = {key: value.clone() for key, value in inputs.items()}; masked["soil"][::2] = 0.0; inputs = masked
        pred, _ = model(inputs, train_mask); loss = torch.nn.functional.huber_loss(pred, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); steps += 1
        model.eval()
        with torch.no_grad(): vp, _ = model(va, validation_mask); score = float(torch.mean(torch.abs(vp - validation_target)))
        if best is None or score < best["score"]:
            best = {"score": score, "state": deepcopy(model.state_dict()), "epoch": epoch}; stale = 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best["state"])
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    model.eval()
    with torch.no_grad(): prediction, auxiliary = model(va, validation_mask)
    inference = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dims": dims, "variant": job.variant, "selected_epoch": best["epoch"]}, output / "model.pt")
    joblib.dump(preprocessors, output / "preprocessor.joblib")
    np.save(output / "representations.npy", auxiliary["representation"].numpy())
    ledger = {"family": job.model_family, **count_capacity(model), "training_steps": steps, "selected_epoch": best["epoch"]}
    runtime = {"wall_time_seconds": fit_time + inference, "fit_time_seconds": fit_time, "inference_time_seconds": inference, "optimizer_steps": steps, "device": "cpu", "peak_memory_bytes": None, "runtime_evidence_status": "RUNTIME_EVIDENCE_INCOMPLETE"}
    return prediction.numpy(), ledger, runtime


def fit_phase2_job(job: ScientificJob, frame: pd.DataFrame, output: Path, *, max_epochs: int = 30) -> TrainingResult:
    outer_year = int(job.fold.split("_")[-1])
    development = frame.loc[frame.year < outer_year].copy()
    values, features = _feature_sets(development, job.input_profile, job.seed)
    train, validation = _partitions(values, outer_year)
    if job.model_family in {"neural", "neural_masked"}:
        prediction, ledger, runtime = _neural(job, train, validation, features, output, max_epochs)
    else:
        prediction, ledger, runtime = _sklearn(job, train, validation, features, output)
    return _write_result(output, job=job, frame=development, validation=validation, prediction=prediction, features=features, model_ledger=ledger, runtime=runtime)


def fit_student_route(*, route: str, frame: pd.DataFrame, outer_test_year: int, seed: int, teacher_prediction: np.ndarray | None, teacher_representation: np.ndarray | None, output: Path, max_epochs: int = 30) -> TrainingResult:
    development = frame.loc[frame.year < outer_test_year].copy()
    train, validation = _partitions(development, outer_test_year)
    features = [name for name in frame if name.startswith("weather_")]
    preprocess = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    x_train = torch.tensor(preprocess.fit_transform(train[features]), dtype=torch.float32)
    x_validation = torch.tensor(preprocess.transform(validation[features]), dtype=torch.float32)
    torch.manual_seed(seed)
    encoder = torch.nn.Sequential(torch.nn.Linear(len(features), 32), torch.nn.ReLU())
    head = torch.nn.Linear(32, 1)
    model = torch.nn.ModuleDict({"encoder": encoder, "head": head})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32)
    teacher_all = None if teacher_prediction is None else np.asarray(teacher_prediction)
    teacher_train = None if teacher_all is None else torch.tensor(teacher_all[train.index], dtype=torch.float32)
    representation_all = None if teacher_representation is None else np.asarray(teacher_representation)
    representation_train = None if representation_all is None else torch.tensor(representation_all[train.index], dtype=torch.float32)
    trajectory=[]; start=time.perf_counter()
    for epoch in range(max_epochs):
        optimizer.zero_grad(); representation=encoder(x_train); prediction=head(representation).squeeze(1); supervised=torch.nn.functional.huber_loss(prediction,y)
        imitation=torch.tensor(0.0)
        if route in {"prediction_kd", "combined_kd"}:
            if teacher_train is None: raise ValueError("QUALIFIED_TEACHER_REQUIRED")
            imitation=torch.nn.functional.mse_loss(prediction,teacher_train)
        alignment=torch.tensor(0.0)
        if route in {"representation_kd", "combined_kd"}:
            if representation_train is None: raise ValueError("QUALIFIED_REPRESENTATION_TEACHER_REQUIRED")
            alignment=torch.nn.functional.mse_loss(representation,representation_train)
        loss=supervised+imitation+alignment; loss.backward(); optimizer.step()
        trajectory.append({"epoch":epoch,"supervised":float(supervised.detach()),"prediction_imitation":float(imitation.detach()),"representation_alignment":float(alignment.detach()),"total":float(loss.detach())})
    fit_time=time.perf_counter()-start
    with torch.no_grad(): pred=head(encoder(x_validation)).squeeze(1).numpy()
    output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"route":route},output/"model.pt"); joblib.dump(preprocess,output/"preprocessor.joblib"); (output/"loss_ledger.json").write_text(json.dumps(trajectory,indent=2)+"\n")
    runtime={"wall_time_seconds":fit_time,"fit_time_seconds":fit_time,"inference_time_seconds":None,"optimizer_steps":max_epochs,"device":"cpu","peak_memory_bytes":None,"runtime_evidence_status":"RUNTIME_EVIDENCE_INCOMPLETE"}
    job={"job_id":f"{route}__test_{outer_test_year}__seed_{seed}","route":route,"seed":seed}
    return _write_result(output,job=job,frame=development,validation=validation,prediction=pred,features=features,model_ledger={"family":"student_mlp","route":route,"parameters":sum(p.numel() for p in model.parameters())},runtime=runtime)




def _repair_model(
    *,
    job: ScientificJob,
    dims: dict[str, int],
) -> torch.nn.Module:
    if job.variant == "neural_deployable":
        return SharedDenseRegressor(dims, hidden=32)
    return ModalitySpecificRegressor(dims, hidden=32)


def _naive_validation_mae(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> float:
    prediction = np.full(
        len(validation),
        float(train.target_yield.mean()),
        dtype=float,
    )
    return float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            prediction,
        )
    )


def _classical_validation_mae(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> float:
    pipeline = Pipeline(
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
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            features,
                        )
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(
        train[features],
        train.target_yield,
    )
    prediction = pipeline.predict(
        validation[features]
    )
    return float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            prediction,
        )
    )


def _checkpoint_replay_max_abs_error(
    *,
    job: ScientificJob,
    dims: dict[str, int],
    state_dict: dict[str, torch.Tensor],
    validation_values: dict[str, torch.Tensor],
    validation_mask: torch.Tensor,
    target_mean: float,
    target_scale: float,
    reference_prediction: np.ndarray,
) -> float:
    replay_model = _repair_model(
        job=job,
        dims=dims,
    )
    replay_model.load_state_dict(state_dict)
    replay_model.eval()

    with torch.no_grad():
        replay_scaled, _ = replay_model(
            validation_values,
            validation_mask,
        )
        replay_prediction = (
            replay_scaled * target_scale
            + target_mean
        ).detach().cpu().numpy()

    return float(
        np.max(
            np.abs(
                replay_prediction
                - np.asarray(reference_prediction)
            )
        )
    )

def _neural_repair(
    job: ScientificJob,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    output: Path,
    contract: NeuralTrainingContract,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(job.seed)

    tr, va, preprocessors = _tensors(
        train,
        validation,
        features,
    )
    dims = {
        name: tensor.shape[1]
        for name, tensor in tr.items()
    }

    model = _repair_model(
        job=job,
        dims=dims,
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

    if (
        job.model_family == "neural_masked"
        and "soil" in tr
    ):
        soil_index = tuple(dims).index("soil")
        train_mask[::2, soil_index] = 0.0
        tr = dict(tr)
        tr["soil"] = tr["soil"].clone()
        tr["soil"][::2] = 0.0

    train_target = torch.tensor(
        train.target_yield.to_numpy(),
        dtype=torch.float32,
    )
    validation_target = torch.tensor(
        validation.target_yield.to_numpy(),
        dtype=torch.float32,
    )

    total_started = time.perf_counter()

    result = fit_repair_regressor(
        model=model,
        train_values=tr,
        train_mask=train_mask,
        train_target=train_target,
        validation_values=va,
        validation_mask=validation_mask,
        validation_target=validation_target,
        contract=contract,
        seed=job.seed,
    )

    total_time = time.perf_counter() - total_started

    output.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": result.best_state_dict,
        "dims": dims,
        "variant": job.variant,
        "selected_epoch": result.selected_epoch,
        "target_scaler": {
            "mean": result.target_mean,
            "scale": result.target_scale,
            "fit_scope": "TRAIN_ONLY",
        },
        "training_contract": {
            "batch_size": contract.batch_size,
            "max_epochs": contract.max_epochs,
            "min_epochs": contract.min_epochs,
            "early_stopping_patience": (
                contract.early_stopping_patience
            ),
            "learning_rate": contract.learning_rate,
            "weight_decay": contract.weight_decay,
            "gradient_clip_norm": (
                contract.gradient_clip_norm
            ),
            "minimum_optimizer_steps": (
                contract.minimum_optimizer_steps
            ),
            "minimum_parameter_delta": (
                contract.minimum_parameter_delta
            ),
        },
    }

    torch.save(checkpoint, output / "model.pt")
    joblib.dump(
        preprocessors,
        output / "preprocessor.joblib",
    )

    (output / "training_history.json").write_text(
        json.dumps(result.history, indent=2) + "\n"
    )
    (output / "training_audit.json").write_text(
        json.dumps(
            result.audit,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "target_scaler.json").write_text(
        json.dumps(
            {
                "mean": result.target_mean,
                "scale": result.target_scale,
                "fit_scope": "TRAIN_ONLY",
                "inverse_transform_applied": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with torch.no_grad():
        _, auxiliary = model(
            va,
            validation_mask,
        )

    representation = auxiliary.get("representation")
    if representation is not None:
        np.save(
            output / "representations.npy",
            representation.detach().cpu().numpy(),
        )

    naive_mae = _naive_validation_mae(
        train,
        validation,
    )
    classical_mae = _classical_validation_mae(
        train,
        validation,
        features,
    )
    student_mae = float(
        mean_absolute_error(
            validation.target_yield.to_numpy(float),
            result.prediction,
        )
    )
    replay_error = _checkpoint_replay_max_abs_error(
        job=job,
        dims=dims,
        state_dict=result.best_state_dict,
        validation_values=va,
        validation_mask=validation_mask,
        target_mean=result.target_mean,
        target_scale=result.target_scale,
        reference_prediction=result.prediction,
    )
    prediction_std = float(
        np.std(result.prediction)
    )
    converged = bool(
        result.optimizer_steps
        >= contract.minimum_optimizer_steps
        and result.parameter_delta_l2
        >= contract.minimum_parameter_delta
        and np.isfinite(student_mae)
        and replay_error <= 1e-6
        and prediction_std > 1e-12
    )

    credibility_record = {
        "fold": job.fold,
        "seed": job.seed,
        "variant": job.variant,
        "job_id": job.job_id,
        "dataset": job.dataset,
        "student_mae": student_mae,
        "naive_mae": naive_mae,
        "classical_mae": classical_mae,
        "optimizer_steps": result.optimizer_steps,
        "parameter_delta_l2": result.parameter_delta_l2,
        "checkpoint_replay_max_abs_error": replay_error,
        "prediction_std": prediction_std,
        "converged": converged,
        "data_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_may_enable_downstream_jobs": False,
    }

    (output / "student_credibility_record.json").write_text(
        json.dumps(
            credibility_record,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    ledger = {
        "family": job.model_family,
        **count_capacity(model),
        "trainer": "stage8_v4_repair_minibatch_v1",
        "training_steps": result.optimizer_steps,
        "selected_epoch": result.selected_epoch,
        "parameter_delta_l2": result.parameter_delta_l2,
        "target_scaling": "train_standard",
        "target_scaler_fit_scope": "TRAIN_ONLY",
        "prediction_inverse_transformed": True,
        "checkpoint_replay_max_abs_error": replay_error,
        "student_credibility_recorded": True,
    }

    runtime = {
        "wall_time_seconds": total_time,
        "fit_time_seconds": result.fit_time_seconds,
        "inference_time_seconds": (
            result.inference_time_seconds
        ),
        "optimizer_steps": result.optimizer_steps,
        "device": "cpu",
        "peak_memory_bytes": None,
        "runtime_evidence_status": (
            "RUNTIME_EVIDENCE_COMPLETE_EXCEPT_PEAK_MEMORY"
        ),
        "includes_preprocessing": False,
        "includes_model_training": True,
        "includes_validation_selection": True,
        "includes_inference": True,
    }

    return result.prediction, ledger, runtime


def fit_phase2_job_repair(
    job: ScientificJob,
    frame: pd.DataFrame,
    output: Path,
    *,
    contract: NeuralTrainingContract,
) -> TrainingResult:
    if job.phase != "phase2":
        raise ValueError("PHASE2_JOB_REQUIRED")

    outer_year = int(job.fold.split("_")[-1])
    development = frame.loc[
        frame.year < outer_year
    ].copy()

    values, features = _feature_sets(
        development,
        job.input_profile,
        job.seed,
    )
    train, validation = _partitions(
        values,
        outer_year,
    )

    if job.model_family in {
        "neural",
        "neural_masked",
    }:
        prediction, ledger, runtime = _neural_repair(
            job,
            train,
            validation,
            features,
            output,
            contract,
        )
    else:
        prediction, ledger, runtime = _sklearn(
            job,
            train,
            validation,
            features,
            output,
        )

    return _write_result(
        output,
        job=job,
        frame=development,
        validation=validation,
        prediction=prediction,
        features=features,
        model_ledger=ledger,
        runtime=runtime,
    )
