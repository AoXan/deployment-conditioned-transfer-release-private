"""Shared, non-smoke contracts for Stage 8 formal route adapters."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

T1_WINDOWS = {
    2021: {"pretrain": range(2014, 2019), "adapt": 2019, "validation": 2020},
    2022: {"pretrain": range(2014, 2020), "adapt": 2020, "validation": 2021},
    2023: {"pretrain": range(2014, 2021), "adapt": 2021, "validation": 2022},
}


def deterministic_fraction_ids(ids: Iterable[str], fractions: list[float], seed: int) -> dict[float, list[str]]:
    ordered = sorted(set(map(str, ids)), key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    return {f: ordered[: max(1, int(np.ceil(len(ordered) * f)))] for f in sorted(fractions)}


def validate_formal_output(rows: list[dict], expected_ids: set[str]) -> list[str]:
    errors = []
    if any("SMOKE" in str(row.get("evidence_status", "")) for row in rows): errors.append("smoke_artifact")
    ids = {str(row.get("sample_id")) for row in rows}
    if ids != set(map(str, expected_ids)): errors.append("prediction_fold_id_mismatch")
    pred = np.asarray([row.get("y_pred", np.nan) for row in rows], dtype=float)
    target = np.asarray([row.get("y_true", np.nan) for row in rows], dtype=float)
    if not len(rows): errors.append("empty_predictions")
    if not np.isfinite(pred).all() or not np.isfinite(target).all(): errors.append("non_finite")
    if len(pred) > 1 and np.nanstd(pred) == 0: errors.append("constant_predictions")
    if len(target) > 1 and np.nanstd(target) == 0: errors.append("constant_target")
    return errors
