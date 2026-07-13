from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any


def read_ckd_sensitivity_contract(
    *,
    output: Path,
    dataset: str,
) -> dict[str, Any] | None:
    path = (
        output
        / "control/gates"
        / f"{dataset}__phase5_ckd_sensitivity.json"
    )

    if not path.is_file():
        return None

    payload = json.loads(path.read_text())

    if payload.get("decision") not in {
        "DEVELOPMENT_OPEN",
        "DEVELOPMENT_CONDITIONAL_OPEN",
    }:
        return None

    if payload.get("selection_scope") != (
        "OUTER_TRAIN_INNER_VALIDATION_ONLY"
    ):
        raise RuntimeError(
            "PHASE5_CKD_SENSITIVITY_SCOPE_INVALID"
        )

    if payload.get("outer_test_used") is not False:
        raise RuntimeError(
            "OUTER_TEST_PHASE5_CKD_SENSITIVITY_FORBIDDEN"
        )

    variant_id = payload.get("variant_id")
    weights = payload.get("alternative_loss_weights")

    if not isinstance(variant_id, str) or not variant_id:
        raise RuntimeError(
            "PHASE5_CKD_VARIANT_ID_MISSING"
        )

    if not isinstance(weights, dict):
        raise RuntimeError(
            "PHASE5_CKD_ALTERNATIVE_WEIGHTS_MISSING"
        )

    required = {
        "supervised",
        "prediction",
        "representation",
    }

    if set(weights) != required:
        raise RuntimeError(
            "PHASE5_CKD_ALTERNATIVE_WEIGHT_KEYS_INVALID"
        )

    normalized: dict[str, float] = {}

    for name in sorted(required):
        value = float(weights[name])

        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                "PHASE5_CKD_ALTERNATIVE_WEIGHT_INVALID:"
                + name
            )

        normalized[name] = value

    total = sum(normalized.values())

    normalized = {
        name: value / total
        for name, value in normalized.items()
    }

    frozen_phase4 = payload.get(
        "frozen_phase4_loss_weights"
    )

    if not isinstance(frozen_phase4, dict):
        raise RuntimeError(
            "PHASE5_CKD_FROZEN_BASE_WEIGHTS_MISSING"
        )

    frozen_normalized = {
        name: float(frozen_phase4[name])
        for name in sorted(required)
    }
    frozen_total = sum(frozen_normalized.values())

    if frozen_total <= 0.0:
        raise RuntimeError(
            "PHASE5_CKD_FROZEN_BASE_WEIGHT_INVALID"
        )

    frozen_normalized = {
        name: value / frozen_total
        for name, value in frozen_normalized.items()
    }

    if all(
        abs(
            normalized[name]
            - frozen_normalized[name]
        ) < 1e-12
        for name in required
    ):
        raise RuntimeError(
            "PHASE5_CKD_SENSITIVITY_VARIANT_IDENTICAL"
        )

    return {
        "variant_id": variant_id,
        "alternative_loss_weights": normalized,
        "frozen_phase4_loss_weights": (
            frozen_normalized
        ),
        "selection_scope": (
            "OUTER_TRAIN_INNER_VALIDATION_ONLY"
        ),
        "outer_test_used": False,
        "source": str(path.relative_to(output)),
    }
