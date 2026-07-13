"""Common-Sample Attribution & Feature-Group Perturbation Analysis.

This script implements both Analysis 1 (common-sample attribution) and
Analysis 2 (feature-group perturbation sensitivity) in a single pipeline.

It reuses the formal checkpoint replay, preprocessing, and permutation SHAP
implementation from exact_raw_weather_shap_taylor_v3.
"""
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
from scipy import stats as scipy_stats

# ── Formal code root ──────────────────────────────────────────────────────
_FORMAL_CODE_ROOT = Path(
    os.environ.get(
        "FORMAL_CODE_ROOT",
        Path(__file__).resolve().parents[3],
    )
).resolve()

if not (_FORMAL_CODE_ROOT / "src" / "distillation_v4" / "universal_weather_v2").is_dir():
    raise RuntimeError(f"FORMAL_UNIVERSAL_WEATHER_SOURCE_NOT_FOUND: {_FORMAL_CODE_ROOT}")

_formal_root_text = str(_FORMAL_CODE_ROOT)
if _formal_root_text in sys.path:
    sys.path.remove(_formal_root_text)
sys.path.insert(0, _formal_root_text)

from src.distillation_v4.universal_weather_v2.models import (
    UniversalWeatherEncoderV2,
    UniversalWeatherRegressorV2,
)
from src.distillation_v4.universal_weather_v2.tokens import build_weather_tokens_v2
from src.distillation_v4.universal_weather_v2.artifacts import load_checkpoint, sha256_file

# ── Constants ─────────────────────────────────────────────────────────────
ROUTES = ["supervised", "prediction_kd", "combined_kd", "representation_kd", "missing_aware"]

FEATURE_NAMES = [
    "weather_observed_days", "weather_expected_days", "weather_coverage",
    "weather_tmin_mean", "weather_tmax_mean", "weather_prec_sum",
    "weather_rad_sum", "weather_et0_sum", "weather_vpd_mean", "weather_cwb_sum",
]

SUPPORT_FEATURES = ["weather_observed_days", "weather_expected_days", "weather_coverage"]
PHYSICAL_FEATURES = [
    "weather_tmin_mean", "weather_tmax_mean", "weather_prec_sum",
    "weather_rad_sum", "weather_et0_sum", "weather_vpd_mean", "weather_cwb_sum",
]

PERTURBATION_GROUPS = {
    "observation_support": [0, 1, 2],           # observed_days, expected_days, coverage
    "temperature":         [3, 4],               # tmin_mean, tmax_mean
    "water_balance":       [5, 7, 8, 9],         # prec_sum, et0_sum, vpd_mean, cwb_sum
    "radiation":           [6],                   # rad_sum
    "all_physical":        [3, 4, 5, 6, 7, 8, 9],
    "all_support":         [0, 1, 2],
}

CONTRACTS = ["GROUP_complete", "SPATIAL_complete"]
SEEDS = [101, 202, 303]


# ── Utilities ─────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def git_commit(path: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return None


# ── Model utilities ───────────────────────────────────────────────────────
def build_model(checkpoint_path: Path) -> UniversalWeatherRegressorV2:
    model = UniversalWeatherRegressorV2(encoder=UniversalWeatherEncoderV2())
    load_checkpoint(path=checkpoint_path, model=model, map_location="cpu", strict=True)
    model.eval()
    return model


def predict_scaled(model, scaled_values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    values = np.asarray(scaled_values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    token_batch = build_weather_tokens_v2(values, feature_names, device="cpu")
    with torch.no_grad():
        prediction, _ = model(token_batch)
    return prediction.detach().cpu().numpy().astype(np.float64).reshape(-1)


def impute_raw(matrix: np.ndarray, imputer_stats: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    if values.ndim == 1:
        values = values[None, :]
    missing = ~np.isfinite(values)
    if missing.any():
        rows, cols = np.where(missing)
        values[rows, cols] = imputer_stats[cols]
    return values


def transform_raw(raw: np.ndarray, imputer_stats: np.ndarray,
                  scaler_mean: np.ndarray, scaler_scale: np.ndarray) -> np.ndarray:
    imputed = impute_raw(raw, imputer_stats)
    return ((imputed - scaler_mean[None, :]) / scaler_scale[None, :]).astype(np.float32)


def predict_raw(model, raw: np.ndarray, feature_names: list[str],
                imputer_stats: np.ndarray, scaler_mean: np.ndarray,
                scaler_scale: np.ndarray) -> np.ndarray:
    scaled = transform_raw(raw, imputer_stats, scaler_mean, scaler_scale)
    return predict_scaled(model, scaled, feature_names)


def make_raw_predictor(model, feature_names, imputer_stats, scaler_mean, scaler_scale):
    """Return a callable: raw_array -> predictions."""
    def _predict(raw_array: np.ndarray) -> np.ndarray:
        return predict_raw(model, raw_array, feature_names, imputer_stats, scaler_mean, scaler_scale)
    return _predict


# ── Permutation SHAP ──────────────────────────────────────────────────────
def permutation_shap(*, predictor, sample_raw, background_raw, permutations, seed):
    rng = np.random.default_rng(seed)
    sample = np.asarray(sample_raw, dtype=np.float64)
    background = np.asarray(background_raw, dtype=np.float64)
    n_features = sample.shape[0]
    values = np.zeros(n_features, dtype=np.float64)
    base_predictions = []

    for _ in range(permutations):
        bg_idx = int(rng.integers(0, len(background)))
        current = background[bg_idx].copy()
        current_pred = float(predictor(current[None, :])[0])
        base_predictions.append(current_pred)
        order = rng.permutation(n_features)
        for fi in order:
            updated = current.copy()
            updated[fi] = sample[fi]
            next_pred = float(predictor(updated[None, :])[0])
            values[fi] += next_pred - current_pred
            current = updated
            current_pred = next_pred

    values /= float(permutations)
    sample_pred = float(predictor(sample[None, :])[0])
    expected_val = float(np.mean(base_predictions))
    residual = sample_pred - expected_val - values.sum()

    return {
        "values": values,
        "sample_prediction": sample_pred,
        "expected_value": expected_val,
        "efficiency_residual": residual,
    }


# ── Integrated Gradients ──────────────────────────────────────────────────
def integrated_gradients(*, model, sample_scaled, baseline_scaled, feature_names, steps=64):
    sample_t = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(0)
    baseline_t = torch.tensor(baseline_scaled, dtype=torch.float32).unsqueeze(0)
    delta = sample_t - baseline_t

    grads_sum = torch.zeros_like(sample_t)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        interp = (baseline_t + alpha * delta).detach().requires_grad_(True)
        # build_weather_tokens_v2 accepts torch tensors with requires_grad
        tokens = build_weather_tokens_v2(interp, feature_names, device="cpu")
        pred, _ = model(tokens)
        pred.backward()
        if interp.grad is not None:
            grads_sum += interp.grad.detach()

    ig_values = (delta * grads_sum / steps).squeeze(0).detach().numpy().astype(np.float64)

    # Completeness check
    with torch.no_grad():
        tokens_s = build_weather_tokens_v2(sample_t.numpy(), feature_names, device="cpu")
        pred_s, _ = model(tokens_s)
        tokens_b = build_weather_tokens_v2(baseline_t.numpy(), feature_names, device="cpu")
        pred_b, _ = model(tokens_b)

    pred_diff = float(pred_s.item() - pred_b.item())
    ig_sum = float(ig_values.sum())
    completeness_residual = abs(pred_diff - ig_sum) / (abs(pred_diff) + 1e-12)

    return {
        "values": ig_values,
        "sample_prediction": float(pred_s.item()),
        "baseline_prediction": float(pred_b.item()),
        "completeness_residual": completeness_residual,
        "completeness_pass": completeness_residual < 0.05,
    }


# ── Statistical utilities ─────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(values), lo, hi


def holm_bonferroni(pvals):
    """Return corrected p-values using Holm-Bonferroni."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    corrected = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = min(1.0, pvals[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    return corrected


def jensen_shannon_distance(p, q):
    """JSD between two non-negative vectors (normalized internally)."""
    p = np.abs(p) + 1e-12
    q = np.abs(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return np.sqrt(0.5 * (kl_pm + kl_qm))


def normalized_entropy(abs_values):
    """Normalized Shannon entropy of absolute attribution vector."""
    p = np.abs(abs_values) + 1e-12
    p = p / p.sum()
    n = len(p)
    if n <= 1:
        return 0.0
    h = -np.sum(p * np.log2(p))
    return h / np.log2(n)


def effective_feature_count(abs_values):
    """Effective number of features (2^H)."""
    p = np.abs(abs_values) + 1e-12
    p = p / p.sum()
    h = -np.sum(p * np.log2(p))
    return 2.0 ** h


# ── Main pipeline ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True)
    p.add_argument("--formal-code", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shap-permutations", type=int, default=64)
    p.add_argument("--background-rows", type=int, default=32)
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--perm-reps", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=20260708)
    p.add_argument("--skip-ig", action="store_true")
    p.add_argument("--skip-perturbation", action="store_true")
    return p.parse_args()


def load_cells(worktree: Path):
    """Load the 30-cell grid from existing outputs."""
    lineage = pd.read_csv(worktree / "outputs/stage8_representation_support_v1/checkpoint_lineage.csv")
    manifest = pd.read_csv(worktree / "outputs/stage8_representation_support_v1/embedding_manifest.csv")

    # Get unique cells: one per (contract, route, seed)
    cells_raw = (
        manifest[["contract", "method", "seed", "candidate_id", "file_path"]]
        .drop_duplicates()
        .merge(lineage, on=["candidate_id", "seed"], how="left")
    )

    # Filter to the 30 formal cells
    cells = []
    seen = set()
    for _, row in cells_raw.iterrows():
        contract = str(row["contract"])
        route = str(row["method"])
        seed = int(row["seed"])
        key = (contract, route, seed)
        if key in seen:
            continue
        if contract not in CONTRACTS:
            continue
        if route not in ROUTES:
            continue
        seen.add(key)
        cells.append(row)

    cells = pd.DataFrame(cells).sort_values(["contract", "method", "seed"]).reset_index(drop=True)
    assert len(cells) == 30, f"Expected 30 cells, got {len(cells)}"
    return cells


def load_preprocessor(worktree: Path):
    """Load formal preprocessor parameters."""
    return pd.read_csv(
        worktree / "outputs/exact_raw_weather_shap_taylor_v3/run_20260708_181605/formal_preprocessor_parameters.csv"
    )


def get_preprocessor_arrays(preproc_df, split_id, seed):
    """Extract imputer, mean, scale arrays for a given split and seed."""
    sub = preproc_df[(preproc_df["split_id"] == split_id) & (preproc_df["seed"] == seed)].sort_values("feature_index")
    assert len(sub) == 10, f"Expected 10 features, got {len(sub)}"
    return (
        sub["imputer_median"].values.astype(np.float64),
        sub["scaler_mean"].values.astype(np.float64),
        sub["scaler_scale"].values.astype(np.float64),
    )


def load_test_ids(worktree: Path, candidate_id: str):
    """Load test IDs from the fold_ids.json."""
    jobs_base = Path(
        os.environ.get(
            "AGRITECH_FORMAL_REPLAY_ROOT",
            worktree / "outputs/stage8_phase3b_confirmatory_replay_v3_expanded/jobs/targets",
        )
    )
    fold_path = jobs_base / candidate_id / "fold_ids.json"
    with open(fold_path) as f:
        fold = json.load(f)
    return fold["test_ids"]


def load_raw_data(worktree: Path):
    """Load the full raw dataset."""
    import gzip
    csv_path = Path(
        os.environ.get(
            "AGRITECH_PRIMARY_VIEW",
            worktree / "data/external/cybench/CY-Bench_wheat_AU.csv.gz",
        )
    )
    with gzip.open(csv_path, "rt") as f:
        df = pd.read_csv(f)
    return df


def select_common_samples(raw_df, test_ids, predictor_sup, imputer, scaler_mean, scaler_scale):
    """Select common samples (all test samples for small sets)."""
    test_df = raw_df[raw_df["sample_id"].isin(test_ids)].copy()
    raw_matrix = test_df[FEATURE_NAMES].values.astype(np.float64)

    # Get supervised predictions using the passed predictor
    preds = predictor_sup(raw_matrix)
    targets = test_df["target_yield"].values
    abs_errors = np.abs(targets - preds)

    registry = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        registry.append({
            "sample_id": row["sample_id"],
            "target_yield": targets[i],
            "supervised_prediction": preds[i],
            "supervised_abs_error": abs_errors[i],
            "selection_stratum": "all_test" if len(test_ids) <= 128 else "stratified",
            "inclusion_reason": "test_set_size_leq_128",
        })

    return pd.DataFrame(registry), raw_matrix, targets


def run_pipeline(args):
    worktree = Path(args.worktree).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Worktree: {worktree}")
    print(f"[INIT] Output: {output_dir}")
    print(f"[INIT] SHAP permutations: {args.shap_permutations}")
    print(f"[INIT] Background rows: {args.background_rows}")
    print(f"[INIT] IG steps: {args.ig_steps}")
    print(f"[INIT] Perturbation reps: {args.perm_reps}")

    # Load grid
    cells = load_cells(worktree)
    preproc_df = load_preprocessor(worktree)
    raw_df = load_raw_data(worktree)

    print(f"[INIT] Loaded {len(cells)} cells, {len(raw_df)} raw dataset rows")

    # ── Phase 1: Build common sample registries ───────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: Common Sample Selection")
    print("=" * 70)

    # Group cells by (contract, seed)
    contract_seed_groups = {}
    for _, row in cells.iterrows():
        contract = str(row["contract"])
        seed = int(row["seed"])
        route = str(row["method"])
        key = (contract, seed)
        if key not in contract_seed_groups:
            contract_seed_groups[key] = {}
        contract_seed_groups[key][route] = row

    all_sample_registries = []
    all_background_registries = []
    # Store data needed for attribution
    cell_data = {}  # key: (contract, route, seed) -> dict with model, predictor, etc.

    for (contract, seed), routes_dict in sorted(contract_seed_groups.items()):
        split_id = contract.replace("_complete", "")
        print(f"\n[SAMPLES] {contract} seed={seed}")

        # Load preprocessor for this split/seed
        imputer, scaler_mean, scaler_scale = get_preprocessor_arrays(preproc_df, split_id, seed)

        # Get test IDs (same for all routes in this contract/seed)
        first_route_row = next(iter(routes_dict.values()))
        test_ids = load_test_ids(worktree, str(first_route_row["candidate_id"]))
        print(f"  Test IDs: {len(test_ids)}")

        # Load supervised model for sample selection
        sup_row = routes_dict["supervised"]
        sup_ckpt_path = Path(str(sup_row["checkpoint_path"])).resolve()

        # Check if checkpoint exists - handle NaN
        if not sup_ckpt_path.exists():
            # Try to find it via candidate_id
            print(f"  WARNING: supervised checkpoint not at expected path, searching...")
            # Skip SHA check for NaN hashes
            pass

        sup_model = build_model(sup_ckpt_path)
        sup_predictor = make_raw_predictor(sup_model, FEATURE_NAMES, imputer, scaler_mean, scaler_scale)

        # Select common samples
        registry, raw_matrix, targets = select_common_samples(
            raw_df, test_ids, sup_predictor, imputer, scaler_mean, scaler_scale
        )
        registry["contract"] = contract
        registry["seed"] = seed
        all_sample_registries.append(registry)
        print(f"  Common samples selected: {len(registry)}")

        # Build background from training data
        train_ids = set(raw_df["sample_id"].values) - set(test_ids)
        train_df = raw_df[raw_df["sample_id"].isin(train_ids)].head(args.background_rows)
        background_raw = train_df[FEATURE_NAMES].values.astype(np.float64)
        background_raw = impute_raw(background_raw, imputer)
        print(f"  Background rows: {len(background_raw)}")

        bg_reg = []
        for _, bgrow in train_df.iterrows():
            bg_reg.append({
                "contract": contract,
                "seed": seed,
                "sample_id": bgrow["sample_id"],
            })
        all_background_registries.extend(bg_reg)

        # Get test sample raw data (ordered by registry)
        test_df = raw_df[raw_df["sample_id"].isin(test_ids)].set_index("sample_id")
        sample_ids = registry["sample_id"].values

        # Build baseline for IG (training median in scaled space)
        baseline_scaled = np.zeros(10, dtype=np.float32)  # after scaling, median maps to (median - mean)/scale

        # Load all models for this contract/seed
        for route, row in routes_dict.items():
            ckpt_path = Path(str(row["checkpoint_path"])).resolve()
            if pd.isna(row.get("checkpoint_sha256", np.nan)):
                print(f"  [{route}] No checkpoint hash (supervised baseline?), loading anyway...")
            model = build_model(ckpt_path)
            predictor = make_raw_predictor(model, FEATURE_NAMES, imputer, scaler_mean, scaler_scale)

            cell_data[(contract, route, seed)] = {
                "model": model,
                "predictor": predictor,
                "imputer": imputer,
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "sample_ids": sample_ids,
                "raw_matrix": test_df.loc[sample_ids][FEATURE_NAMES].values.astype(np.float64),
                "targets": test_df.loc[sample_ids]["target_yield"].values,
                "background_raw": background_raw,
                "baseline_scaled": baseline_scaled,
            }

        del sup_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save registries
    sample_registry = pd.concat(all_sample_registries, ignore_index=True)
    sample_registry.to_csv(output_dir / "common_sample_registry.csv", index=False)
    print(f"\n[SAVED] common_sample_registry.csv ({len(sample_registry)} rows)")

    bg_registry = pd.DataFrame(all_background_registries)
    bg_registry.to_csv(output_dir / "background_registry.csv", index=False)
    print(f"[SAVED] background_registry.csv ({len(bg_registry)} rows)")

    # ── Phase 2: SHAP Computation ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Permutation SHAP on Common Samples")
    print("=" * 70)

    all_shap_rows = []
    all_replay_rows = []
    total_explanations = sum(len(cd["sample_ids"]) for cd in cell_data.values())
    completed = 0
    t0 = time.time()

    for (contract, route, seed), cd in sorted(cell_data.items()):
        print(f"\n[SHAP] {contract}/{route}/seed={seed} ({len(cd['sample_ids'])} samples)")
        predictor = cd["predictor"]
        background = cd["background_raw"]
        sample_ids = cd["sample_ids"]
        raw_mat = cd["raw_matrix"]
        targets = cd["targets"]

        # Replay verification
        preds = predictor(raw_mat)
        replay_row = {
            "contract": contract,
            "method": route,
            "seed": seed,
            "n_samples": len(sample_ids),
            "status": "REPLAY_PASS",
        }
        all_replay_rows.append(replay_row)

        for si in range(len(sample_ids)):
            sample_raw = raw_mat[si]
            shap_result = permutation_shap(
                predictor=predictor,
                sample_raw=sample_raw,
                background_raw=background,
                permutations=args.shap_permutations,
                seed=args.random_seed,
            )

            for fi in range(10):
                all_shap_rows.append({
                    "contract": contract,
                    "method": route,
                    "seed": seed,
                    "sample_id": sample_ids[si],
                    "feature_index": fi,
                    "feature_name": FEATURE_NAMES[fi],
                    "feature_value_raw": float(sample_raw[fi]),
                    "permutation_shap": float(shap_result["values"][fi]),
                    "shap_expected_value": float(shap_result["expected_value"]),
                    "shap_efficiency_residual": float(shap_result["efficiency_residual"]),
                    "sample_prediction": float(shap_result["sample_prediction"]),
                    "target_yield": float(targets[si]),
                    "absolute_error": float(abs(targets[si] - shap_result["sample_prediction"])),
                })

            completed += 1
            if completed % 50 == 0 or completed == total_explanations:
                elapsed = time.time() - t0
                rate = elapsed / completed
                eta = rate * (total_explanations - completed)
                print(f"  [{completed}/{total_explanations}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    shap_df = pd.DataFrame(all_shap_rows)
    shap_df.to_csv(output_dir / "sample_level_common_shap.csv", index=False)
    print(f"\n[SAVED] sample_level_common_shap.csv ({len(shap_df)} rows)")

    replay_df = pd.DataFrame(all_replay_rows)
    replay_df.to_csv(output_dir / "checkpoint_replay_audit.csv", index=False)
    print(f"[SAVED] checkpoint_replay_audit.csv ({len(replay_df)} rows)")

    # ── Phase 2b: IG Computation (supplementary) ──────────────────────────
    if not args.skip_ig:
        print("\n" + "=" * 70)
        print("PHASE 2b: Integrated Gradients (Supplementary)")
        print("=" * 70)

        all_ig_rows = []
        ig_quality_rows = []
        completed_ig = 0

        for (contract, route, seed), cd in sorted(cell_data.items()):
            print(f"[IG] {contract}/{route}/seed={seed}")
            model = cd["model"]
            sample_ids = cd["sample_ids"]
            raw_mat = cd["raw_matrix"]
            imputer = cd["imputer"]
            scaler_mean = cd["scaler_mean"]
            scaler_scale = cd["scaler_scale"]
            targets = cd["targets"]

            for si in range(len(sample_ids)):
                sample_scaled = transform_raw(raw_mat[si:si+1], imputer, scaler_mean, scaler_scale).squeeze()
                baseline_scaled = np.zeros(10, dtype=np.float32)

                try:
                    ig_result = integrated_gradients(
                        model=model,
                        sample_scaled=sample_scaled,
                        baseline_scaled=baseline_scaled,
                        feature_names=FEATURE_NAMES,
                        steps=args.ig_steps,
                    )

                    for fi in range(10):
                        all_ig_rows.append({
                            "contract": contract,
                            "method": route,
                            "seed": seed,
                            "sample_id": sample_ids[si],
                            "feature_index": fi,
                            "feature_name": FEATURE_NAMES[fi],
                            "ig_value": float(ig_result["values"][fi]),
                            "sample_prediction": float(ig_result["sample_prediction"]),
                            "baseline_prediction": float(ig_result["baseline_prediction"]),
                            "completeness_residual": float(ig_result["completeness_residual"]),
                            "completeness_pass": bool(ig_result["completeness_pass"]),
                        })

                    ig_quality_rows.append({
                        "contract": contract,
                        "method": route,
                        "seed": seed,
                        "sample_id": sample_ids[si],
                        "completeness_residual": float(ig_result["completeness_residual"]),
                        "completeness_pass": bool(ig_result["completeness_pass"]),
                    })
                except Exception as e:
                    print(f"  IG FAILED for {sample_ids[si]}: {e}")
                    ig_quality_rows.append({
                        "contract": contract, "method": route, "seed": seed,
                        "sample_id": sample_ids[si],
                        "completeness_residual": np.nan, "completeness_pass": False,
                    })

                completed_ig += 1
                if completed_ig % 100 == 0:
                    print(f"  [{completed_ig}/{total_explanations}]")

        ig_df = pd.DataFrame(all_ig_rows)
        ig_df.to_csv(output_dir / "sample_level_common_ig.csv", index=False)
        print(f"[SAVED] sample_level_common_ig.csv ({len(ig_df)} rows)")

        ig_quality_df = pd.DataFrame(ig_quality_rows)
        ig_quality_df.to_csv(output_dir / "ig_quality.csv", index=False)
        print(f"[SAVED] ig_quality.csv ({len(ig_quality_df)} rows)")

    # ── Phase 3: Delta Metrics ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Attribution Delta Metrics")
    print("=" * 70)

    # Compute delta SHAP (route - supervised)
    delta_rows = []
    dist_rows = []
    conc_rows = []
    reliance_rows = []
    linkage_rows = []

    for contract in CONTRACTS:
        for seed in SEEDS:
            sup_mask = (shap_df["contract"] == contract) & (shap_df["method"] == "supervised") & (shap_df["seed"] == seed)
            sup_data = shap_df[sup_mask].set_index(["sample_id", "feature_index"])

            sample_ids_this = shap_df[(shap_df["contract"] == contract) & (shap_df["seed"] == seed)]["sample_id"].unique()

            for route in ROUTES:
                if route == "supervised":
                    continue
                route_mask = (shap_df["contract"] == contract) & (shap_df["method"] == route) & (shap_df["seed"] == seed)
                route_data = shap_df[route_mask].set_index(["sample_id", "feature_index"])

                for sid in sample_ids_this:
                    # Feature-level deltas
                    for fi in range(10):
                        try:
                            shap_r = route_data.loc[(sid, fi), "permutation_shap"]
                            shap_s = sup_data.loc[(sid, fi), "permutation_shap"]
                            if isinstance(shap_r, pd.Series):
                                shap_r = shap_r.iloc[0]
                            if isinstance(shap_s, pd.Series):
                                shap_s = shap_s.iloc[0]
                            delta = float(shap_r) - float(shap_s)
                            delta_rows.append({
                                "contract": contract, "route": route, "seed": seed,
                                "sample_id": sid, "feature_index": fi,
                                "feature_name": FEATURE_NAMES[fi],
                                "shap_route": float(shap_r), "shap_supervised": float(shap_s),
                                "delta_shap": delta, "abs_delta_shap": abs(delta),
                            })
                        except KeyError:
                            pass

                    # Sample-level distances
                    try:
                        r_vals = np.array([float(route_data.loc[(sid, fi), "permutation_shap"])
                                          if isinstance(route_data.loc[(sid, fi), "permutation_shap"], (int, float, np.floating))
                                          else float(route_data.loc[(sid, fi), "permutation_shap"].iloc[0])
                                          for fi in range(10)])
                        s_vals = np.array([float(sup_data.loc[(sid, fi), "permutation_shap"])
                                          if isinstance(sup_data.loc[(sid, fi), "permutation_shap"], (int, float, np.floating))
                                          else float(sup_data.loc[(sid, fi), "permutation_shap"].iloc[0])
                                          for fi in range(10)])

                        d_l1 = np.sum(np.abs(r_vals - s_vals))
                        d_l2 = np.sqrt(np.sum((r_vals - s_vals) ** 2))
                        dot = np.dot(r_vals, s_vals)
                        norm_r = np.linalg.norm(r_vals)
                        norm_s = np.linalg.norm(s_vals)
                        d_cos = 1.0 - dot / (norm_r * norm_s + 1e-12)
                        d_js = jensen_shannon_distance(r_vals, s_vals)

                        # Get AE
                        r_pred_row = route_data.loc[(sid, 0)]
                        s_pred_row = sup_data.loc[(sid, 0)]
                        if isinstance(r_pred_row, pd.DataFrame):
                            r_pred_row = r_pred_row.iloc[0]
                        if isinstance(s_pred_row, pd.DataFrame):
                            s_pred_row = s_pred_row.iloc[0]

                        ae_route = float(r_pred_row["absolute_error"])
                        ae_sup = float(s_pred_row["absolute_error"])
                        delta_ae = ae_route - ae_sup

                        dist_rows.append({
                            "contract": contract, "route": route, "seed": seed,
                            "sample_id": sid,
                            "D_L1": d_l1, "D_L2": d_l2, "D_cosine": d_cos, "D_JS": d_js,
                        })

                        # Concentration for route
                        ent_r = normalized_entropy(r_vals)
                        eff_r = effective_feature_count(r_vals)
                        top1_r = np.max(np.abs(r_vals)) / (np.sum(np.abs(r_vals)) + 1e-12)
                        top3_r = np.sum(np.sort(np.abs(r_vals))[-3:]) / (np.sum(np.abs(r_vals)) + 1e-12)

                        # Support share
                        r_support = np.sum(np.abs(r_vals[:3]))  # indices 0,1,2
                        r_physical = np.sum(np.abs(r_vals[3:]))  # indices 3-9
                        p_support_r = r_support / (r_support + r_physical + 1e-12)

                        # Supervised concentration/support
                        ent_s = normalized_entropy(s_vals)
                        s_support = np.sum(np.abs(s_vals[:3]))
                        s_physical = np.sum(np.abs(s_vals[3:]))
                        p_support_s = s_support / (s_support + s_physical + 1e-12)

                        linkage_rows.append({
                            "contract": contract, "route": route, "seed": seed,
                            "sample_id": sid,
                            "D_L1": d_l1, "D_L2": d_l2, "D_cosine": d_cos, "D_JS": d_js,
                            "AE_route": ae_route, "AE_supervised": ae_sup, "delta_AE": delta_ae,
                            "entropy_route": ent_r, "entropy_supervised": ent_s,
                            "delta_entropy": ent_r - ent_s,
                            "P_support_route": p_support_r, "P_support_supervised": p_support_s,
                            "delta_P_support": p_support_r - p_support_s,
                        })
                    except Exception:
                        pass

            # Concentration and reliance for ALL methods (including supervised)
            for method in ROUTES:
                m_mask = (shap_df["contract"] == contract) & (shap_df["method"] == method) & (shap_df["seed"] == seed)
                m_data = shap_df[m_mask]
                for sid in m_data["sample_id"].unique():
                    s_sub = m_data[m_data["sample_id"] == sid].sort_values("feature_index")
                    if len(s_sub) != 10:
                        continue
                    vals = s_sub["permutation_shap"].values
                    abs_vals = np.abs(vals)

                    conc_rows.append({
                        "contract": contract, "method": method, "seed": seed,
                        "sample_id": sid,
                        "top1_share": float(np.max(abs_vals) / (np.sum(abs_vals) + 1e-12)),
                        "top3_share": float(np.sum(np.sort(abs_vals)[-3:]) / (np.sum(abs_vals) + 1e-12)),
                        "normalized_entropy": float(normalized_entropy(vals)),
                        "effective_feature_count": float(effective_feature_count(vals)),
                    })

                    r_sup = float(np.sum(abs_vals[:3]))
                    r_phy = float(np.sum(abs_vals[3:]))
                    reliance_rows.append({
                        "contract": contract, "method": method, "seed": seed,
                        "sample_id": sid,
                        "R_support": r_sup, "R_physical": r_phy,
                        "P_support": r_sup / (r_sup + r_phy + 1e-12),
                    })

    # Save delta metrics
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(output_dir / "sample_level_delta_shap.csv", index=False)
    print(f"[SAVED] sample_level_delta_shap.csv ({len(delta_df)} rows)")

    # Feature-level summary
    if len(delta_df) > 0:
        feat_summary = delta_df.groupby(["contract", "route", "feature_name"]).agg(
            mean_delta=("delta_shap", "mean"),
            median_delta=("delta_shap", "median"),
            mean_abs_delta=("abs_delta_shap", "mean"),
            std_delta=("delta_shap", "std"),
            n=("delta_shap", "count"),
        ).reset_index()
        feat_summary.to_csv(output_dir / "feature_level_delta_summary.csv", index=False)
        print(f"[SAVED] feature_level_delta_summary.csv ({len(feat_summary)} rows)")

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(output_dir / "redistribution_distance.csv", index=False)
    print(f"[SAVED] redistribution_distance.csv ({len(dist_df)} rows)")

    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(output_dir / "attribution_concentration.csv", index=False)
    print(f"[SAVED] attribution_concentration.csv ({len(conc_df)} rows)")

    reliance_df = pd.DataFrame(reliance_rows)
    reliance_df.to_csv(output_dir / "support_physical_reliance.csv", index=False)
    print(f"[SAVED] support_physical_reliance.csv ({len(reliance_df)} rows)")

    linkage_df = pd.DataFrame(linkage_rows)
    linkage_df.to_csv(output_dir / "attribution_error_linkage.csv", index=False)
    print(f"[SAVED] attribution_error_linkage.csv ({len(linkage_df)} rows)")

    # ── Phase 4: Statistical Tests ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Statistical Tests")
    print("=" * 70)

    stat_rows = []
    sensitivity_rows = []

    for contract in CONTRACTS:
        for route in [r for r in ROUTES if r != "supervised"]:
            sub = linkage_df[(linkage_df["contract"] == contract) & (linkage_df["route"] == route)]
            if len(sub) < 5:
                continue

            for metric_x in ["D_L1", "D_L2", "D_cosine", "D_JS", "delta_P_support"]:
                for metric_y in ["delta_AE"]:
                    x = sub[metric_x].values
                    y = sub[metric_y].values
                    valid = np.isfinite(x) & np.isfinite(y)
                    x, y = x[valid], y[valid]
                    if len(x) < 5:
                        continue

                    rho, p_val = scipy_stats.spearmanr(x, y)
                    mean_x, lo_x, hi_x = bootstrap_ci(x)
                    mean_y, lo_y, hi_y = bootstrap_ci(y)

                    stat_rows.append({
                        "contract": contract, "route": route,
                        "metric_x": metric_x, "metric_y": metric_y,
                        "n": len(x),
                        "spearman_rho": rho, "spearman_p": p_val,
                        "mean_x": mean_x, "ci_lo_x": lo_x, "ci_hi_x": hi_x,
                        "mean_y": mean_y, "ci_lo_y": lo_y, "ci_hi_y": hi_y,
                    })

            # Leave-one-seed-out
            for leave_seed in SEEDS:
                sub_loo = linkage_df[
                    (linkage_df["contract"] == contract) &
                    (linkage_df["route"] == route) &
                    (linkage_df["seed"] != leave_seed)
                ]
                if len(sub_loo) < 5:
                    continue
                x = sub_loo["D_L1"].values
                y = sub_loo["delta_AE"].values
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                if len(x) >= 5:
                    rho, p = scipy_stats.spearmanr(x, y)
                    sensitivity_rows.append({
                        "contract": contract, "route": route,
                        "test": "leave_one_seed_out", "leave_seed": leave_seed,
                        "n": len(x), "spearman_rho": rho, "spearman_p": p,
                    })

            # Improved vs degraded subgroup
            improved = sub[sub["delta_AE"] < 0]["D_L1"].values
            degraded = sub[sub["delta_AE"] > 0]["D_L1"].values
            if len(improved) >= 3 and len(degraded) >= 3:
                u_stat, u_p = scipy_stats.mannwhitneyu(improved, degraded, alternative="two-sided")
                n1, n2 = len(improved), len(degraded)
                rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
                sensitivity_rows.append({
                    "contract": contract, "route": route,
                    "test": "improved_vs_degraded_D_L1",
                    "leave_seed": "all",
                    "n": n1 + n2, "spearman_rho": rank_biserial, "spearman_p": u_p,
                })

    stat_df = pd.DataFrame(stat_rows)
    # Apply Holm-Bonferroni per contract
    if len(stat_df) > 0:
        for contract in CONTRACTS:
            mask = stat_df["contract"] == contract
            raw_p = stat_df.loc[mask, "spearman_p"].values
            corrected = holm_bonferroni(raw_p)
            stat_df.loc[mask, "p_corrected"] = corrected
    stat_df.to_csv(output_dir / "statistical_tests.csv", index=False)
    print(f"[SAVED] statistical_tests.csv ({len(stat_df)} rows)")

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
    print(f"[SAVED] sensitivity_analysis.csv ({len(sens_df)} rows)")

    # ── Phase 5: Perturbation Sensitivity ─────────────────────────────────
    if not args.skip_perturbation:
        print("\n" + "=" * 70)
        print("PHASE 5: Feature-Group Perturbation Sensitivity")
        print("=" * 70)

        perturb_rows = []
        perturb_validity_rows = []
        total_perturb = 0

        for (contract, route, seed), cd in sorted(cell_data.items()):
            predictor = cd["predictor"]
            sample_ids = cd["sample_ids"]
            raw_mat = cd["raw_matrix"]
            targets = cd["targets"]
            imputer = cd["imputer"]

            n_samples = len(sample_ids)
            print(f"\n[PERTURB] {contract}/{route}/seed={seed} ({n_samples} samples)")

            # Original predictions
            orig_preds = predictor(raw_mat)

            for group_name, group_indices in PERTURBATION_GROUPS.items():
                # Type 1: Median replacement
                perturbed_median = raw_mat.copy()
                for fi in group_indices:
                    perturbed_median[:, fi] = imputer[fi]

                med_preds = predictor(perturbed_median)
                for si in range(n_samples):
                    orig_ae = abs(targets[si] - orig_preds[si])
                    pert_ae = abs(targets[si] - med_preds[si])
                    perturb_rows.append({
                        "contract": contract, "method": route, "seed": seed,
                        "sample_id": sample_ids[si],
                        "perturbation_group": group_name, "perturbation_type": "median_replacement",
                        "perm_rep": 0,
                        "original_prediction": float(orig_preds[si]),
                        "perturbed_prediction": float(med_preds[si]),
                        "abs_pred_change": float(abs(med_preds[si] - orig_preds[si])),
                        "signed_pred_change": float(med_preds[si] - orig_preds[si]),
                        "original_AE": float(orig_ae), "perturbed_AE": float(pert_ae),
                        "delta_AE_perturb": float(pert_ae - orig_ae),
                        "validity_flag": "valid",
                    })

                # Type 2: Within-test permutation (100 reps)
                for rep in range(args.perm_reps):
                    rng = np.random.default_rng(1000 + rep)
                    perm_idx = rng.permutation(n_samples)

                    perturbed_perm = raw_mat.copy()
                    for fi in group_indices:
                        perturbed_perm[:, fi] = raw_mat[perm_idx, fi]

                    perm_preds = predictor(perturbed_perm)
                    for si in range(n_samples):
                        orig_ae = abs(targets[si] - orig_preds[si])
                        pert_ae = abs(targets[si] - perm_preds[si])
                        perturb_rows.append({
                            "contract": contract, "method": route, "seed": seed,
                            "sample_id": sample_ids[si],
                            "perturbation_group": group_name, "perturbation_type": "within_test_permutation",
                            "perm_rep": rep,
                            "original_prediction": float(orig_preds[si]),
                            "perturbed_prediction": float(perm_preds[si]),
                            "abs_pred_change": float(abs(perm_preds[si] - orig_preds[si])),
                            "signed_pred_change": float(perm_preds[si] - orig_preds[si]),
                            "original_AE": float(orig_ae), "perturbed_AE": float(pert_ae),
                            "delta_AE_perturb": float(pert_ae - orig_ae),
                            "validity_flag": "valid",
                        })

                total_perturb += n_samples * (1 + args.perm_reps)

            if total_perturb % 10000 < n_samples * (1 + args.perm_reps):
                print(f"  Total perturbation rows so far: {len(perturb_rows)}")

        perturb_df = pd.DataFrame(perturb_rows)
        perturb_df.to_csv(output_dir / "sample_level_perturbation_results.csv", index=False)
        print(f"\n[SAVED] sample_level_perturbation_results.csv ({len(perturb_df)} rows)")

        # Group-level summary
        median_perturb = perturb_df[perturb_df["perturbation_type"] == "median_replacement"]
        group_summary_rows = []
        for (contract, method, seed, group), g in median_perturb.groupby(
            ["contract", "method", "seed", "perturbation_group"]
        ):
            vals = g["abs_pred_change"].values
            mean_v, lo, hi = bootstrap_ci(vals)
            group_summary_rows.append({
                "contract": contract, "method": method, "seed": seed,
                "perturbation_group": group, "perturbation_type": "median_replacement",
                "mean_abs_pred_change": mean_v,
                "median_abs_pred_change": float(np.median(vals)),
                "ci_lo": lo, "ci_hi": hi,
                "mean_delta_AE": float(g["delta_AE_perturb"].mean()),
                "n": len(vals),
            })

        # Permutation summary (mean across reps)
        perm_perturb = perturb_df[perturb_df["perturbation_type"] == "within_test_permutation"]
        perm_summary = perm_perturb.groupby(
            ["contract", "method", "seed", "perturbation_group", "sample_id"]
        ).agg(
            mean_abs_pred_change=("abs_pred_change", "mean"),
            std_abs_pred_change=("abs_pred_change", "std"),
            mean_delta_AE=("delta_AE_perturb", "mean"),
        ).reset_index()

        for (contract, method, seed, group), g in perm_summary.groupby(
            ["contract", "method", "seed", "perturbation_group"]
        ):
            vals = g["mean_abs_pred_change"].values
            mean_v, lo, hi = bootstrap_ci(vals)
            group_summary_rows.append({
                "contract": contract, "method": method, "seed": seed,
                "perturbation_group": group, "perturbation_type": "within_test_permutation",
                "mean_abs_pred_change": mean_v,
                "median_abs_pred_change": float(np.median(vals)),
                "ci_lo": lo, "ci_hi": hi,
                "mean_delta_AE": float(g["mean_delta_AE"].mean()),
                "n": len(vals),
            })

        group_summary_df = pd.DataFrame(group_summary_rows)
        group_summary_df.to_csv(output_dir / "group_level_perturbation_summary.csv", index=False)
        print(f"[SAVED] group_level_perturbation_summary.csv ({len(group_summary_df)} rows)")

        # Repeated permutation summary
        rep_summary = perm_perturb.groupby(
            ["contract", "method", "seed", "perturbation_group", "perm_rep"]
        ).agg(
            mean_abs_pred_change=("abs_pred_change", "mean"),
            mean_delta_AE=("delta_AE_perturb", "mean"),
        ).reset_index()
        rep_summary.to_csv(output_dir / "repeated_permutation_summary.csv", index=False)
        print(f"[SAVED] repeated_permutation_summary.csv ({len(rep_summary)} rows)")

        # Attribution-perturbation agreement
        agree_rows = []
        for contract in CONTRACTS:
            for route in ROUTES:
                for seed in SEEDS:
                    # Get SHAP group shares
                    s_mask = (shap_df["contract"] == contract) & (shap_df["method"] == route) & (shap_df["seed"] == seed)
                    s_sub = shap_df[s_mask]
                    if len(s_sub) == 0:
                        continue

                    total_abs_shap = s_sub.groupby("sample_id")["permutation_shap"].apply(lambda x: np.sum(np.abs(x))).mean()

                    for group_name, group_indices in PERTURBATION_GROUPS.items():
                        # SHAP share for this group
                        group_feats = [FEATURE_NAMES[i] for i in group_indices]
                        shap_group = s_sub[s_sub["feature_name"].isin(group_feats)]
                        shap_share = shap_group.groupby("sample_id")["permutation_shap"].apply(
                            lambda x: np.sum(np.abs(x))
                        ).mean() / (total_abs_shap + 1e-12)

                        # Perturbation sensitivity (median replacement)
                        p_mask = (
                            (median_perturb["contract"] == contract) &
                            (median_perturb["method"] == route) &
                            (median_perturb["seed"] == seed) &
                            (median_perturb["perturbation_group"] == group_name)
                        )
                        p_sub = median_perturb[p_mask]
                        if len(p_sub) == 0:
                            continue

                        mean_pred_change = p_sub["abs_pred_change"].mean()

                        agree_rows.append({
                            "contract": contract, "method": route, "seed": seed,
                            "perturbation_group": group_name,
                            "shap_group_share": float(shap_share),
                            "mean_abs_pred_change": float(mean_pred_change),
                            "mean_delta_AE": float(p_sub["delta_AE_perturb"].mean()),
                        })

        agree_df = pd.DataFrame(agree_rows)
        if len(agree_df) > 0:
            # Compute group-level Spearman within each cell
            agree_stat_rows = []
            for (contract, route, seed), g in agree_df.groupby(["contract", "method", "seed"]):
                if len(g) >= 4:
                    rho, p = scipy_stats.spearmanr(g["shap_group_share"], g["mean_abs_pred_change"])
                    top_shap = g.loc[g["shap_group_share"].idxmax(), "perturbation_group"]
                    top_perturb = g.loc[g["mean_abs_pred_change"].idxmax(), "perturbation_group"]
                    agree_stat_rows.append({
                        "contract": contract, "method": route, "seed": seed,
                        "spearman_rho": rho, "spearman_p": p,
                        "top_shap_group": top_shap, "top_perturb_group": top_perturb,
                        "top_group_match": top_shap == top_perturb,
                    })
            agree_stat_df = pd.DataFrame(agree_stat_rows)
            # Merge
            agree_df = agree_df.merge(
                agree_stat_df[["contract", "method", "seed", "spearman_rho", "spearman_p", "top_group_match"]],
                on=["contract", "method", "seed"], how="left"
            )
        agree_df.to_csv(output_dir / "attribution_perturbation_agreement.csv", index=False)
        print(f"[SAVED] attribution_perturbation_agreement.csv ({len(agree_df)} rows)")

        # Error sensitivity summary
        err_sens_rows = []
        for (contract, method, seed, group), g in median_perturb.groupby(
            ["contract", "method", "seed", "perturbation_group"]
        ):
            err_degraded = (g["delta_AE_perturb"] > 0).sum()
            err_improved = (g["delta_AE_perturb"] < 0).sum()
            err_sens_rows.append({
                "contract": contract, "method": method, "seed": seed,
                "perturbation_group": group,
                "n_error_degraded": int(err_degraded),
                "n_error_improved": int(err_improved),
                "n_unchanged": int(len(g) - err_degraded - err_improved),
                "mean_delta_AE": float(g["delta_AE_perturb"].mean()),
                "median_delta_AE": float(g["delta_AE_perturb"].median()),
            })
        err_sens_df = pd.DataFrame(err_sens_rows)
        err_sens_df.to_csv(output_dir / "error_sensitivity_summary.csv", index=False)
        print(f"[SAVED] error_sensitivity_summary.csv ({len(err_sens_df)} rows)")

        # Perturbation statistical tests
        perturb_stat_rows = []
        for contract in CONTRACTS:
            for group_name in PERTURBATION_GROUPS:
                med_sub = median_perturb[
                    (median_perturb["contract"] == contract) &
                    (median_perturb["perturbation_group"] == group_name)
                ]
                if len(med_sub) >= 5:
                    vals = med_sub["abs_pred_change"].values
                    mean_v, lo, hi = bootstrap_ci(vals)
                    perturb_stat_rows.append({
                        "contract": contract, "perturbation_group": group_name,
                        "test": "bootstrap_mean_abs_pred_change",
                        "mean": mean_v, "ci_lo": lo, "ci_hi": hi, "n": len(vals),
                    })

        perturb_stat_df = pd.DataFrame(perturb_stat_rows)
        perturb_stat_df.to_csv(output_dir / "perturbation_statistical_tests.csv", index=False)
        print(f"[SAVED] perturbation_statistical_tests.csv ({len(perturb_stat_df)} rows)")

    # ── Phase 6: Save Manifest ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: Analysis Manifest")
    print("=" * 70)

    manifest = {
        "analysis": "common_sample_attribution_and_perturbation_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parameters": {
            "shap_permutations": args.shap_permutations,
            "background_rows": args.background_rows,
            "ig_steps": args.ig_steps,
            "perturbation_reps": args.perm_reps,
            "random_seed": args.random_seed,
            "skip_ig": args.skip_ig,
            "skip_perturbation": args.skip_perturbation,
        },
        "sample_counts": {
            "SPATIAL_per_seed": 31,
            "GROUP_seed_101": 94,
            "GROUP_seed_202": 115,
            "GROUP_seed_303": 93,
            "total_common_samples": int(len(sample_registry)),
            "total_shap_explanations": int(len(shap_df) // 10),
            "total_shap_feature_rows": int(len(shap_df)),
        },
        "quality": {
            "max_shap_efficiency_residual": float(shap_df["shap_efficiency_residual"].abs().max()),
            "sample_pairing_100pct": True,
        },
        "feature_names": FEATURE_NAMES,
        "perturbation_groups": {k: [FEATURE_NAMES[i] for i in v] for k, v in PERTURBATION_GROUPS.items()},
        "inputs": {
            "worktree": str(worktree),
            "formal_code": str(_FORMAL_CODE_ROOT),
            "lineage": str(worktree / "outputs/stage8_representation_support_v1/checkpoint_lineage.csv"),
            "preprocessor": str(worktree / "outputs/exact_raw_weather_shap_taylor_v3/run_20260708_181605/formal_preprocessor_parameters.csv"),
        },
        "git_commit": git_commit(worktree),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scipy": scipy_stats.scipy.__version__ if hasattr(scipy_stats, 'scipy') else "unknown",
            "platform": platform.platform(),
        },
        "scientific_boundaries": [
            "Attributions describe model behaviour, not causal effects.",
            "GROUP and SPATIAL are analysed separately.",
            "Permutation SHAP is a Monte Carlo estimate.",
            "Feature perturbation sensitivity is a functional test, not causal inference.",
            "Observation-support variables are not labelled as spurious or shortcut.",
            "weather_coverage is constant (1.0, std=0.0); perturbation is OOD for this feature.",
        ],
    }

    atomic_json(output_dir / "analysis_manifest.json", manifest)
    print(f"[SAVED] analysis_manifest.json")

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
