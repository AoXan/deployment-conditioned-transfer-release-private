#!/usr/bin/env python3
"""
extract_embeddings.py — Multi-layer embedding extraction from audited neural checkpoints.

Phase 2 of the Latent Representation & Support Compatibility Scientific Audit.
Extracts raw_input (10D), early_hidden (64D), shared_encoder (64D),
penultimate (128D), and prediction (1D) embeddings for every
(contract, method, seed) cell using PyTorch forward hooks.

Usage:
    python extract_embeddings.py                     # Full run
    python extract_embeddings.py --smoke             # Smoke: SPATIAL, seed 101
    python extract_embeddings.py --contract SPATIAL   # Single contract
    python extract_embeddings.py --seed 101           # Single seed
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and imports for the AgriTech model library
# ---------------------------------------------------------------------------
FORMAL_WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORMAL_WORKTREE))
sys.path.insert(0, str(FORMAL_WORKTREE / "src"))

import torch

from src.distillation_v4.universal_weather_v2.models import (
    UniversalWeatherRegressorV2,
    UniversalWeatherEncoderV2,
)
from src.distillation_v4.universal_weather_v2.artifacts import (
    load_checkpoint,
    sha256_file,
)
from src.distillation_v4.universal_weather_v2.training import (
    prepare_arrays,
    build_weather_tokens_v2,
)
from src.distillation_v4.universal_weather_v2_remediation.config import load_config
from src.distillation_v4.universal_weather_v2_remediation.datasets import load_frame
from src.distillation_v4.universal_weather_v2_remediation.splits import (
    build_contract_splits,
)
from src.distillation_v4.universal_weather_v2_remediation.execution import (
    _frames,
    _dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_ROOT = FORMAL_WORKTREE / "outputs/stage8_representation_support_v1"
LINEAGE_CSV = OUTPUT_ROOT / "checkpoint_lineage.csv"
DATASET_ID = "CY-Bench_wheat_AU"
CONTRACTS = ["SPATIAL", "GROUP"]
SEEDS = [101, 202, 303]
NEURAL_ROUTES = ["supervised", "prediction_kd", "combined_kd", "representation_kd", "missing_aware"]
LAYERS = ["raw_input", "early_hidden", "shared_encoder", "penultimate", "prediction"]
PRED_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Failure ledger
# ---------------------------------------------------------------------------
@dataclass
class FailureLedger:
    path: Path
    records: list[dict] = field(default_factory=list)

    def log(self, **kwargs: Any) -> None:
        kwargs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.records.append(kwargs)
        print(f"[FAILURE] {kwargs}", file=sys.stderr)

    def flush(self) -> None:
        if not self.records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in self.records for k in r})
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)


# ---------------------------------------------------------------------------
# Hook-based embedding extractor
# ---------------------------------------------------------------------------
class EmbeddingExtractor:
    """Registers forward hooks to capture intermediate activations."""

    def __init__(self, model: UniversalWeatherRegressorV2) -> None:
        self.model = model
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        # 1) early_hidden: input to model.encoder.transformer
        def hook_transformer_input(module, input, output):
            # input is (src, ...), src shape [B, num_tokens, 64]
            tokens = input[0]  # [B, T, 64]
            # We need to pool over tokens. Use mean pooling (simple; exact availability
            # weighting requires the padding mask which is not in input[0].)
            # Actually the second argument is src_key_padding_mask.
            # TransformerEncoder.forward(src, mask=None, src_key_padding_mask=None, ...)
            # So input = (src,) or (src, mask) or (src, mask, src_key_padding_mask, ...)
            # Let's capture raw tokens; we'll pool later with padding info
            self._activations["_early_tokens"] = tokens.detach()

        h = self.model.encoder.transformer.register_forward_hook(hook_transformer_input)
        self._hooks.append(h)

        # 2) shared_encoder: input to model.encoder.output (=pooled transformer output)
        def hook_output_input(module, input, output):
            # input to encoder.output is the pooled vector [B, 64]
            self._activations["shared_encoder"] = input[0].detach()

        h = self.model.encoder.output.register_forward_hook(hook_output_input)
        self._hooks.append(h)

        # 3) penultimate: input to model.head (=128-dim encoder output)
        def hook_head_input(module, input, output):
            self._activations["penultimate"] = input[0].detach()

        h = self.model.head.register_forward_hook(hook_head_input)
        self._hooks.append(h)

    def extract(
        self,
        weather_array: np.ndarray,
        feature_names: list[str],
        padding_mask: np.ndarray | None = None,
        availability: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run forward pass and return all layer embeddings as numpy arrays."""
        self._activations.clear()
        self.model.eval()

        weather_tensor = torch.as_tensor(weather_array, dtype=torch.float32)
        token_batch = build_weather_tokens_v2(weather_tensor, feature_names)

        with torch.no_grad():
            prediction_tensor, representation = self.model(token_batch)

        # raw_input = the scaled weather features
        raw_input = weather_array.copy()

        # early_hidden: pool the captured pre-transformer tokens
        early_tokens = self._activations["_early_tokens"]  # [B, T, 64]
        # Use availability-based pooling matching encoder logic
        pad_mask = token_batch.padding_mask  # [B, T], True = padded
        avail = token_batch.availability     # [B, T]
        usable = (~pad_mask & (avail > 0)).to(early_tokens.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = usable.sum(dim=1).clamp_min(1.0)  # [B, 1]
        early_hidden = ((early_tokens * usable).sum(dim=1) / denom).numpy()  # [B, 64]

        shared_encoder = self._activations["shared_encoder"].numpy()  # [B, 64]
        penultimate = self._activations["penultimate"].numpy()        # [B, 128]
        prediction = prediction_tensor.numpy().reshape(-1, 1)         # [B, 1]

        return {
            "raw_input": raw_input,
            "early_hidden": early_hidden,
            "shared_encoder": shared_encoder,
            "penultimate": penultimate,
            "prediction": prediction,
        }

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------
def load_lineage() -> pd.DataFrame:
    df = pd.read_csv(LINEAGE_CSV)
    return df


def build_manifest_job(lineage_row: pd.Series) -> dict:
    """Construct a minimal job dict compatible with _frames()."""
    return {
        "candidate_id": lineage_row["candidate_id"],
        "split_id": lineage_row["split_id"],
        "seed": int(lineage_row["seed"]),
        "transfer_strategy": lineage_row["transfer_strategy"],
        "missing_modality_method": lineage_row["missing_modality_method"],
        "source_route": lineage_row["source_route"],
        "source_dataset": lineage_row["source_dataset"],
        "adaptation_mode": "complete_adaptation",
        "adaptation_fraction": 1.0,
        "deployment_condition": "complete",
        "dataset_id": DATASET_ID,
    }


def validate_predictions(
    official_path: str,
    replayed_preds: np.ndarray,
    sample_ids: list[str],
) -> float:
    """Validate replayed predictions against official file. Returns max abs diff."""
    official = pd.read_csv(official_path)
    replay_df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_pred_replayed": replayed_preds,
    })
    merged = official.merge(replay_df, on="sample_id", how="inner")
    if len(merged) != len(official):
        raise ValueError(
            f"Sample alignment mismatch: official={len(official)}, merged={len(merged)}"
        )
    diff = np.abs(
        merged["y_pred"].astype(float).values - merged["y_pred_replayed"].values
    )
    return float(diff.max())


def run_extraction(args: argparse.Namespace) -> None:
    ledger = FailureLedger(OUTPUT_ROOT / "failure_ledger.csv")

    # Determine which contracts/seeds to run
    contracts = [args.contract] if args.contract else CONTRACTS
    seeds = [args.seed] if args.seed else SEEDS
    if args.smoke:
        contracts = ["SPATIAL"]
        seeds = [101]

    # Load config and dataset
    config_path = args.config
    schema_path = args.schema
    config = load_config(config_path, schema_path)
    contract_cfg = config["datasets"][DATASET_ID]
    frame, data_path = load_frame(contract_cfg, FORMAL_WORKTREE)
    ds = _dataset(frame, data_path, DATASET_ID, contract_cfg)
    feature_names = list(ds.weather_features)

    print(f"Dataset {DATASET_ID} loaded: {len(frame)} rows, {len(feature_names)} weather features")
    print(f"Contracts: {contracts}, Seeds: {seeds}, Routes: {NEURAL_ROUTES}")

    lineage = load_lineage()
    manifest_records: list[dict] = []
    total_cells = 0
    passed_cells = 0

    for split_id in contracts:
        for seed in seeds:
            # Build split once per (contract, seed)
            split_obj = next(
                s for s in build_contract_splits(frame, contract_cfg, seed=seed)
                if s.split_id == split_id
            )

            # We need a representative job for _frames. Use the first neural job.
            rep_row = lineage[
                (lineage["split_id"] == split_id)
                & (lineage["seed"] == seed)
                & (lineage["transfer_strategy"] == "ordinary_transfer")
            ].iloc[0]
            rep_job = build_manifest_job(rep_row)

            train_df, val_df, test_df = _frames(frame, split_obj, contract_cfg, rep_job)
            arrays = prepare_arrays(
                dataset=ds,
                train_frame=train_df,
                validation_frame=val_df,
                test_frame=test_df,
            )

            sample_id_col = contract_cfg["sample_id_column"]
            train_ids = train_df[sample_id_col].astype(str).tolist()
            test_ids = test_df[sample_id_col].astype(str).tolist()

            # Guard: zero overlap
            overlap = set(train_ids) & set(test_ids)
            if overlap:
                ledger.log(
                    contract=split_id, seed=seed,
                    error="TRAIN_TEST_OVERLAP",
                    detail=f"{len(overlap)} overlapping IDs",
                )
                ledger.flush()
                sys.exit(1)

            train_weather = arrays["train"]["weather"]
            test_weather = arrays["test"]["weather"]
            train_targets = arrays["train"]["target"]
            test_targets = arrays["test"]["target"]

            if args.smoke:
                # Limit train to 16 samples for smoke test
                n_train_smoke = min(16, len(train_weather))
                train_weather = train_weather[:n_train_smoke]
                train_targets = train_targets[:n_train_smoke]
                train_ids = train_ids[:n_train_smoke]

            print(f"\n{'='*60}")
            print(f"Split: {split_id} | Seed: {seed}")
            print(f"Train: {len(train_ids)} samples | Test: {len(test_ids)} samples")
            print(f"{'='*60}")

            for route in NEURAL_ROUTES:
                total_cells += 1
                cell_key = f"{split_id}/{route}/{seed}"
                print(f"\n--- Processing {cell_key} ---")

                # Find checkpoint in lineage
                mask = (
                    (lineage["split_id"] == split_id)
                    & (lineage["seed"] == seed)
                    & (lineage["source_route"] == route)
                    & (lineage["transfer_strategy"] == "ordinary_transfer")
                )
                matches = lineage[mask]
                if len(matches) == 0:
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="NO_LINEAGE_ENTRY",
                    )
                    continue
                if len(matches) > 1:
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="MULTIPLE_LINEAGE_ENTRIES",
                        detail=f"Found {len(matches)} entries",
                    )
                    continue

                row = matches.iloc[0]
                candidate_id = row["candidate_id"]
                checkpoint_path = Path(row["checkpoint_path"])
                expected_hash = row["checkpoint_sha256"]
                official_pred_path = row["official_predictions_path"]

                # Verify checkpoint exists and hash matches
                if not checkpoint_path.is_file():
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="CHECKPOINT_MISSING",
                        path=str(checkpoint_path),
                    )
                    continue

                actual_hash = sha256_file(checkpoint_path)
                if not actual_hash.startswith(expected_hash[:8]):
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="CHECKPOINT_HASH_MISMATCH",
                        expected=expected_hash[:16],
                        actual=actual_hash[:16],
                    )
                    continue

                # Load model
                try:
                    model = UniversalWeatherRegressorV2(
                        encoder=UniversalWeatherEncoderV2()
                    )
                    load_checkpoint(
                        path=checkpoint_path, model=model,
                        map_location="cpu", strict=True,
                    )
                except Exception as exc:
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="CHECKPOINT_LOAD_FAILED",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                extractor = EmbeddingExtractor(model)

                try:
                    # Extract train embeddings
                    train_embs = extractor.extract(train_weather, feature_names)
                    train_preds = train_embs["prediction"].flatten()

                    # Extract test embeddings
                    test_embs = extractor.extract(test_weather, feature_names)
                    test_preds = test_embs["prediction"].flatten()

                    # Validate test predictions against official
                    max_diff = validate_predictions(
                        official_pred_path, test_preds, test_ids,
                    )
                    if max_diff > PRED_TOLERANCE:
                        ledger.log(
                            contract=split_id, seed=seed, route=route,
                            error="PREDICTION_REPLAY_FAILED",
                            max_diff=max_diff,
                            tolerance=PRED_TOLERANCE,
                        )
                        extractor.remove_hooks()
                        continue

                    print(f"  Prediction replay OK (max diff: {max_diff:.2e})")

                    # Build metadata arrays
                    train_abs_errors = np.abs(train_targets - train_preds)
                    test_abs_errors = np.abs(test_targets - test_preds)

                    # Sanity checks on embeddings
                    for layer in LAYERS:
                        for split_name, emb_dict in [("train", train_embs), ("test", test_embs)]:
                            arr = emb_dict[layer]
                            if not np.isfinite(arr).all():
                                ledger.log(
                                    contract=split_id, seed=seed, route=route,
                                    layer=layer, split=split_name,
                                    error="NON_FINITE_EMBEDDING",
                                )
                                raise ValueError(f"Non-finite embedding: {cell_key}/{layer}/{split_name}")

                            if arr.std() < 1e-8:
                                # Log but don't fail — constant embeddings are suspicious
                                # but technically possible for prediction layer
                                if layer != "prediction":
                                    ledger.log(
                                        contract=split_id, seed=seed, route=route,
                                        layer=layer, split=split_name,
                                        error="CONSTANT_EMBEDDING_WARNING",
                                        std=float(arr.std()),
                                    )

                    # Save embeddings
                    sub = "smoke" if args.smoke else "embeddings"
                    emb_dir = OUTPUT_ROOT / sub / f"{split_id}_complete" / route / str(seed)
                    emb_dir.mkdir(parents=True, exist_ok=True)

                    save_dict = {}
                    for layer in LAYERS:
                        save_dict[f"{layer}_train"] = train_embs[layer]
                        save_dict[f"{layer}_test"] = test_embs[layer]

                    # Structured metadata
                    train_meta = np.array(
                        list(zip(train_ids, train_targets, train_preds, train_abs_errors)),
                        dtype=[
                            ("sample_id", "U64"),
                            ("target", "f4"),
                            ("prediction", "f4"),
                            ("abs_error", "f4"),
                        ],
                    )
                    test_meta = np.array(
                        list(zip(test_ids, test_targets, test_preds, test_abs_errors)),
                        dtype=[
                            ("sample_id", "U64"),
                            ("target", "f4"),
                            ("prediction", "f4"),
                            ("abs_error", "f4"),
                        ],
                    )
                    save_dict["metadata_train"] = train_meta
                    save_dict["metadata_test"] = test_meta

                    npz_path = emb_dir / "embeddings.npz"
                    np.savez_compressed(npz_path, **save_dict)
                    print(f"  Saved: {npz_path}")

                    # Manifest records
                    for layer in LAYERS:
                        for split_name, emb_dict, n_samples in [
                            ("train", train_embs, len(train_ids)),
                            ("test", test_embs, len(test_ids)),
                        ]:
                            arr = emb_dict[layer]
                            manifest_records.append({
                                "contract": f"{split_id}_complete",
                                "method": route,
                                "seed": seed,
                                "layer": layer,
                                "split": split_name,
                                "candidate_id": candidate_id,
                                "checkpoint_hash": actual_hash[:16],
                                "n_samples": n_samples,
                                "embedding_dim": arr.shape[1] if arr.ndim > 1 else 1,
                                "embedding_mean": float(arr.mean()),
                                "embedding_std": float(arr.std()),
                                "has_nan": bool(np.isnan(arr).any()),
                                "has_inf": bool(np.isinf(arr).any()),
                                "is_constant": bool(arr.std() < 1e-8),
                                "file_path": str(npz_path),
                            })

                    passed_cells += 1

                except Exception as exc:
                    ledger.log(
                        contract=split_id, seed=seed, route=route,
                        error="EXTRACTION_FAILED",
                        detail=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                finally:
                    extractor.remove_hooks()

    # Save manifest CSV
    manifest_path = OUTPUT_ROOT / ("smoke_embedding_manifest.csv" if args.smoke else "embedding_manifest.csv")
    pd.DataFrame(manifest_records).to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path}")

    # Save completion summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE: {passed_cells}/{total_cells} cells passed")
    print(f"{'='*60}")

    ledger.flush()

    if passed_cells < total_cells:
        print(f"WARNING: {total_cells - passed_cells} cells failed. See failure_ledger.csv", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    global FORMAL_WORKTREE, OUTPUT_ROOT, LINEAGE_CSV
    parser = argparse.ArgumentParser(description="Extract multi-layer embeddings from audited checkpoints")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: SPATIAL, seed 101, 16 train samples")
    parser.add_argument("--contract", choices=["SPATIAL", "GROUP"], help="Run single contract")
    parser.add_argument("--seed", type=int, choices=[101, 202, 303], help="Run single seed")
    parser.add_argument("--repository-root", type=Path, default=FORMAL_WORKTREE)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--lineage",
        type=Path,
        default=FORMAL_WORKTREE / "data_manifest/checkpoints/checkpoint_lineage.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=FORMAL_WORKTREE / "configs/publication/universal_weather_full_campaign_v2_remediation_v1.yaml",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=FORMAL_WORKTREE / "schemas/universal_weather_full_campaign_v2_remediation_v1.schema.json",
    )
    args = parser.parse_args()
    FORMAL_WORKTREE = args.repository_root.resolve()
    OUTPUT_ROOT = args.output.resolve()
    LINEAGE_CSV = args.lineage.resolve()
    args.config = args.config.resolve()
    args.schema = args.schema.resolve()
    run_extraction(args)


if __name__ == "__main__":
    main()
