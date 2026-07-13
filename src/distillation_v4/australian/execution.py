from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import atomic_json, fingerprint, sha256_file


NEURAL_COMPONENTS = {
    "shared_early_fusion", "modality_specific_late_fusion", "shared_neural_receiver",
    "random_representation", "shuffled_soil_neural", "supervised", "missing_aware",
    "prediction_kd", "representation_kd", "combined_kd", "fine_tune",
    "random_representation_transfer_control", "no_mask", "explicit_mask",
    "dropout_0.25", "dropout_0.5", "dropout_0.75", "soil_representation_raw",
    "soil_representation_pca", "soil_representation_pls",
    "shuffled_teacher_prediction_control",
    "matched_parameter_deployable_control", "matched_active_compute_deployable_control", "capacity_scaling_control",
}


def _neural_fit_predict(component: str, seed: int, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, deployable: list[str], soil: list[str], source_train: pd.DataFrame | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_imp = SimpleImputer().fit(train[deployable])
    x_scale = StandardScaler().fit(x_imp.transform(train[deployable]))
    sx_imp = SimpleImputer().fit(train[soil]) if soil else None
    sx_scale = StandardScaler().fit(sx_imp.transform(train[soil])) if soil else None
    def tensor(frame: pd.DataFrame, cols: list[str], imp, scale):
        return torch.tensor(scale.transform(imp.transform(frame[cols])), dtype=torch.float32)
    xtr, xva, xte = (tensor(f, deployable, x_imp, x_scale) for f in (train, valid, test))
    ytr = torch.tensor(train.target_yield.to_numpy(), dtype=torch.float32).view(-1, 1)
    yva = torch.tensor(valid.target_yield.to_numpy(), dtype=torch.float32).view(-1, 1)
    hidden = 32 if component == "matched_parameter_deployable_control" else (12 if component == "capacity_scaling_control" else 24)
    soil_mask_tensors = None
    str_ = sva = ste = None
    if soil:
        soil_mask_tensors = tuple(torch.tensor(f[soil].notna().any(axis=1).to_numpy(), dtype=torch.float32).view(-1, 1) for f in (train, valid, test))
        str_, sva, ste = (tensor(f, soil, sx_imp, sx_scale) for f in (train, valid, test))
        if component == "soil_representation_pca":
            from sklearn.decomposition import PCA
            transform = PCA(n_components=max(1, min(8, str_.shape[1], len(str_) - 1)), random_state=seed).fit(str_.numpy())
            str_, sva, ste = (torch.tensor(transform.transform(value.numpy()), dtype=torch.float32) for value in (str_, sva, ste))
        elif component == "soil_representation_pls":
            from sklearn.cross_decomposition import PLSRegression
            transform = PLSRegression(n_components=max(1, min(4, str_.shape[1], len(str_) - 1))).fit(str_.numpy(), ytr.numpy())
            str_, sva, ste = (torch.tensor(transform.transform(value.numpy()), dtype=torch.float32) for value in (str_, sva, ste))
    variable_soil = component in {"missing_aware", "no_mask", "explicit_mask"} or component.startswith("dropout_") or component.startswith("soil_representation_")
    if variable_soil and soil:
        def variable_input(x, s, mask): return torch.cat([x, s * mask] + ([] if component == "no_mask" else [mask]), 1)
        xtr_model, xva_model, xte_model = (variable_input(x, s, m) for x, s, m in zip((xtr, xva, xte), (str_, sva, ste), soil_mask_tensors))
    else:
        xtr_model, xva_model, xte_model = xtr, xva, xte
    student_encoder = nn.Sequential(nn.Linear(xtr_model.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
    student_head = nn.Linear(hidden, 1)
    student = nn.ModuleDict({"encoder": student_encoder, "head": student_head})
    teacher = None
    teacher_targets = teacher_latent = None
    if soil and component in {"prediction_kd", "representation_kd", "combined_kd", "fine_tune", "shared_early_fusion", "modality_specific_late_fusion", "shared_neural_receiver", "shuffled_soil_neural", "shuffled_teacher_prediction_control"}:
        if component == "shuffled_soil_neural":
            str_ = str_[torch.randperm(len(str_))]
        if component == "modality_specific_late_fusion":
            class LateEncoder(nn.Module):
                def __init__(self):
                    super().__init__(); self.weather = nn.Sequential(nn.Linear(xtr.shape[1], hidden // 2), nn.ReLU()); self.soil = nn.Sequential(nn.Linear(str_.shape[1], hidden // 2), nn.ReLU()); self.fuse = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
                def forward(self, value):
                    return self.fuse(torch.cat([self.weather(value[:, :xtr.shape[1]]), self.soil(value[:, xtr.shape[1]:])], 1))
            encoder = LateEncoder()
        else:
            encoder = nn.Sequential(nn.Linear(xtr.shape[1] + str_.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        teacher = nn.ModuleDict({"encoder": encoder, "head": nn.Linear(hidden, 1)})
        opt_t = torch.optim.AdamW(teacher.parameters(), lr=2e-3)
        teacher_best, teacher_state = float("inf"), None
        for _ in range(35):
            opt_t.zero_grad(); latent = teacher["encoder"](torch.cat([xtr, str_], 1)); loss = nn.functional.huber_loss(teacher["head"](latent), ytr); loss.backward(); opt_t.step()
            teacher.eval()
            with torch.no_grad(): teacher_val = float(nn.functional.l1_loss(teacher["head"](teacher["encoder"](torch.cat([xva, sva], 1))), yva))
            if teacher_val < teacher_best:
                teacher_best = teacher_val; teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
            teacher.train()
        teacher.load_state_dict(teacher_state)
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            teacher_latent = teacher["encoder"](torch.cat([xtr, str_], 1)); teacher_targets = teacher["head"](teacher_latent)
        if component == "shuffled_teacher_prediction_control":
            teacher_targets = teacher_targets[torch.randperm(len(teacher_targets))]
        if component in {"shared_early_fusion", "modality_specific_late_fusion", "shuffled_soil_neural"}:
            with torch.no_grad(): prediction = teacher["head"](teacher["encoder"](torch.cat([xte, ste], 1))).view(-1).numpy()
            return prediction, {"backend": "PYTORCH_REAL_FORWARD_BACKWARD", "architecture": "MODALITY_SPECIFIC_LATE_FUSION" if component == "modality_specific_late_fusion" else "EARLY_FUSION_TEACHER", "best_validation_mae": teacher_best, "teacher_frozen_after_selection": True, "soil_shuffled_train_only": component == "shuffled_soil_neural"}, {"teacher": teacher.state_dict()}
    if component in {"random_representation", "random_representation_transfer_control"}:
        teacher_latent = torch.randn((len(xtr), hidden))
    dropout = float(component.split("_")[-1]) if component.startswith("dropout_") else (0.5 if component == "missing_aware" else 0.0)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-4)
    source_pretraining_steps = 0
    if source_train is not None and len(source_train):
        source_x = tensor(source_train, deployable, x_imp, x_scale)
        if variable_soil and soil:
            source_s = tensor(source_train, soil, sx_imp, sx_scale)
            source_m = torch.tensor(source_train[soil].notna().any(axis=1).to_numpy(), dtype=torch.float32).view(-1, 1)
            source_x = variable_input(source_x, source_s, source_m)
        source_y = torch.tensor(source_train.target_yield.to_numpy(), dtype=torch.float32).view(-1, 1)
        for _ in range(20):
            opt.zero_grad(); source_pred = student["head"](student["encoder"](source_x)); source_loss = nn.functional.huber_loss(source_pred, source_y); source_loss.backward(); nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); source_pretraining_steps += 1
        # The target head is new: no row-level alignment or source head reuse is
        # claimed across crops/sample units.
        student["head"] = nn.Linear(hidden, 1)
        opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    trajectory = []
    for epoch in range(60):
        student.train(); opt.zero_grad(); used = xtr_model
        if dropout:
            used = xtr_model * (torch.rand((len(xtr_model), 1)) > dropout)
        latent = student["encoder"](used); pred = student["head"](latent)
        supervised = nn.functional.huber_loss(pred, ytr)
        imitation = nn.functional.mse_loss(pred, teacher_targets) if teacher_targets is not None and component in {"prediction_kd", "combined_kd", "shuffled_teacher_prediction_control"} else pred.new_tensor(0.0)
        representation = nn.functional.mse_loss(latent, teacher_latent) if teacher_latent is not None and component in {"representation_kd", "combined_kd", "fine_tune", "random_representation", "random_representation_transfer_control"} else pred.new_tensor(0.0)
        loss = supervised + 0.5 * imitation + 0.25 * representation
        loss.backward(); nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        student.eval()
        with torch.no_grad(): val = float(nn.functional.l1_loss(student["head"](student["encoder"](xva_model)), yva))
        trajectory.append({"epoch": epoch, "supervised": float(supervised), "imitation": float(imitation), "representation": float(representation), "validation_mae": val})
        if val < best - 1e-7:
            best, patience = val, 0
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        else:
            patience += 1
            if patience >= 8: break
    student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad(): prediction = student["head"](student["encoder"](xte_model)).view(-1).numpy()
    ledger = {"backend": "PYTORCH_REAL_FORWARD_BACKWARD", "architecture": "MODALITY_SPECIFIC_LATE_FUSION" if component == "modality_specific_late_fusion" else ("MISSING_AWARE_EXPLICIT_MASK" if variable_soil else "SHARED_DEPLOYABLE_ENCODER"), "hidden_size": hidden, "parameter_count": int(sum(p.numel() for p in student.parameters())), "source_pretraining_steps": source_pretraining_steps, "cross_sample_unit_rows_concatenated": False, "optimizer_steps": len(trajectory), "best_validation_mae": best, "epochs": len(trajectory), "loss_trajectory": trajectory, "teacher_frozen": teacher is not None, "supervised_and_imitation_separate": True}
    return prediction, ledger, {"student": student.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None}


def _model(component: str, seed: int):
    if component.startswith("hgb") or component in {"reference", "soil_direct"}:
        estimator = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, random_state=seed)
    elif component in {"shuffled_teacher_prediction_control", "missing_aware_ablation_suite_control"}:
        estimator = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=seed, n_jobs=1)
    else:
        estimator = Ridge(alpha=1.0)
    return Pipeline([("imputer", SimpleImputer(add_indicator=True)), ("scale", StandardScaler()), ("model", estimator)])


def execute_job(job: dict[str, Any], frame: pd.DataFrame, features: dict[str, list[str]], split: dict[str, Any], output: Path, *, source_train: pd.DataFrame | None = None) -> dict[str, Any]:
    component = job["component_id"]
    seed = int(job.get("seed", 101))
    ids = frame.sample_id.astype(str)
    train = frame[ids.isin(split["train_ids"])]
    valid = frame[ids.isin(split["validation_ids"])]
    test = frame[ids.isin(split["test_ids"])]
    features = {name: [column for column in columns if column in train and train[column].notna().any()] for name, columns in features.items()}
    feature_names = features["deployable"]
    if component in {"reference", "soil_direct", "hgb_privileged", "shared_early_fusion", "modality_specific_late_fusion"}:
        feature_names = sorted(set(feature_names + features["soil"]))
    if not feature_names or min(len(train), len(valid), len(test)) == 0:
        raise ValueError("EMPTY_FEATURE_OR_SPLIT_CONTRACT")
    if train.target_yield.nunique(dropna=True) < 2 or test.target_yield.nunique(dropna=True) < 2:
        raise ValueError("INVALID_CONSTANT_TARGET")
    started = time.monotonic()
    training_ledger: dict[str, Any]
    if component in NEURAL_COMPONENTS:
        prediction, training_ledger, checkpoint = _neural_fit_predict(component, seed, train, valid, test, features["deployable"], features["soil"], source_train=source_train)
        import torch
        output.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output / "checkpoint.pt")
    else:
        model = _model(component, seed)
        model.fit(train[feature_names], train.target_yield)
        prediction = np.asarray(model.predict(test[feature_names]), dtype=float)
        estimator = model.named_steps["model"]
        training_ledger = {"backend": "SKLEARN_REAL_FIT_PREDICT", "model_class": type(estimator).__name__, "fitted_examples": len(train), "feature_count": len(feature_names), "active_compute_proxy": {"max_iter": getattr(estimator, "max_iter", None), "n_estimators": getattr(estimator, "n_estimators", None)}}
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output / "checkpoint.joblib")
    if not np.isfinite(prediction).all() or np.unique(prediction).size < 2:
        raise RuntimeError("INVALID_CONSTANT_OR_NONFINITE_PREDICTIONS")
    pred = pd.DataFrame({"sample_id": test.sample_id.astype(str), "y_true": test.target_yield.astype(float), "y_pred": prediction})
    deployable_preprocessor = Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())]).fit(train[features["deployable"]])
    soil_preprocessor = Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())]).fit(train[features["soil"]]) if features["soil"] else None
    joblib.dump({"deployable": deployable_preprocessor, "soil": soil_preprocessor, "deployable_features": features["deployable"], "soil_features": features["soil"], "fit_ids": train.sample_id.astype(str).tolist()}, output / "preprocessor.joblib")
    pred.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame({"sample_id": split["test_ids"]}).to_csv(output / "fold_test_ids.csv", index=False)
    pd.concat([pd.DataFrame({"sample_id": split[f"{part}_ids"], "split": part}) for part in ("train", "validation", "test")], ignore_index=True).to_csv(output / "fold_assignments.csv", index=False)
    atomic_json(output / "split_contract.json", split)
    metrics = {"mae": float(mean_absolute_error(pred.y_true, pred.y_pred)), "rmse": float(mean_squared_error(pred.y_true, pred.y_pred) ** 0.5), "r2": float(r2_score(pred.y_true, pred.y_pred))}
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "training_ledger.json", training_ledger)
    contract = {"job": job, "split": {k: v for k, v in split.items() if not k.endswith("_ids")}, "features": feature_names, "seed": seed}
    execution_fingerprint = str(job.get("execution_fingerprint") or fingerprint(contract))
    manifest = {"schema_version": "stage8_v4_australian_artifact_v1", "job_id": job["job_id"], "attempt": job.get("attempt", 1), "base_fingerprint": job.get("base_fingerprint"), "dataset_id": job["dataset_id"], "source_dataset_id": job.get("source_dataset_id"), "fold": job["fold"], "axis": job["axis"], "seed": seed, "label_budget": job.get("label_budget", 1.0), "execution_fingerprint": execution_fingerprint, "fingerprints": job.get("fingerprints", {}), "protocol_tier": job["planned_protocol"], "sample_unit": job["sample_unit"], "target_unit": job["target_unit"], "fit_seconds": time.monotonic() - started, "component": component, "handler": training_ledger["backend"], "checkpoint": "checkpoint.pt" if component in NEURAL_COMPONENTS else "checkpoint.joblib", "preprocessor": "preprocessor.joblib", "teacher_lineage": {"source_dataset_id": job.get("source_dataset_id"), "frozen_teacher": training_ledger.get("teacher_frozen", False)}, "smoke_only": bool(job.get("smoke_only", False))}
    atomic_json(output / "manifest.json", manifest)
    required = ["predictions.csv", "fold_test_ids.csv", "fold_assignments.csv", "split_contract.json", "metrics.json", "training_ledger.json", "manifest.json", manifest["checkpoint"], "preprocessor.joblib"]
    atomic_json(output / "completion.json", {"status": "SMOKE_ONLY_NOT_SCIENTIFIC_EVIDENCE" if job.get("smoke_only") else "EXECUTION_COMPLETE", "execution_fingerprint": execution_fingerprint, "artifact_hashes": {name: sha256_file(output / name) for name in required}})
    return {"metrics": metrics, "manifest": manifest}
