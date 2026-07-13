from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DevelopmentDecision(Enum):
    OPEN = "DEVELOPMENT_OPEN"
    CONDITIONAL = "DEVELOPMENT_CONDITIONAL_OPEN"
    BLOCKED = "DEVELOPMENT_BLOCKED"
    INSUFFICIENT = "DEVELOPMENT_INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class GateResult:
    decision: DevelopmentDecision
    evidence: dict[str, Any]


def evaluate_development_gate(
    records: list[dict[str, Any]],
    *,
    data_scope: str,
    maximum_worst_group_harm: float = 0.0,
) -> GateResult:
    if data_scope != "OUTER_TRAIN_INNER_VALIDATION_ONLY":
        raise ValueError("OUTER_TEST_GATE_INPUT_FORBIDDEN")

    if not records:
        return GateResult(
            DevelopmentDecision.INSUFFICIENT,
            {"reason": "NO_DEVELOPMENT_RECORDS"},
        )

    effects = np.array(
        [float(record["effect"]) for record in records],
        dtype=float,
    )
    controls = np.array(
        [
            float(record.get("negative_control_effect", 0.0))
            for record in records
        ],
        dtype=float,
    )
    worst = np.array(
        [
            float(record.get("worst_group_change", 0.0))
            for record in records
        ],
        dtype=float,
    )

    if not (
        np.isfinite(effects).all()
        and np.isfinite(controls).all()
        and np.isfinite(worst).all()
    ):
        return GateResult(
            DevelopmentDecision.BLOCKED,
            {"reason": "NONFINITE_GATE_EVIDENCE"},
        )

    directions = effects < 0
    mean_effect = float(np.mean(effects))
    mean_control = float(np.mean(controls))
    maximum_harm = float(np.max(worst))
    all_worst_groups_safe = bool(
        np.all(worst <= maximum_worst_group_harm)
    )
    control_separation = mean_effect < mean_control

    if (
        directions.all()
        and mean_effect < 0
        and control_separation
        and all_worst_groups_safe
    ):
        decision = DevelopmentDecision.OPEN
        reason = "CONSISTENT_IMPROVEMENT_WITHOUT_WORST_GROUP_HARM"
    elif (
        directions.any()
        and mean_effect < 0
        and maximum_harm <= maximum_worst_group_harm
    ):
        decision = DevelopmentDecision.CONDITIONAL
        reason = "MEAN_IMPROVEMENT_WITH_INCONSISTENT_DIRECTION"
    elif mean_effect < 0 and maximum_harm > maximum_worst_group_harm:
        decision = DevelopmentDecision.BLOCKED
        reason = "WORST_GROUP_HARM_EXCEEDS_LIMIT"
    else:
        decision = DevelopmentDecision.BLOCKED
        reason = "NO_RELIABLE_DEVELOPMENT_IMPROVEMENT"

    return GateResult(
        decision,
        {
            "reason": reason,
            "mean_effect": mean_effect,
            "mean_negative_control_effect": mean_control,
            "directions_improved": int(directions.sum()),
            "records": len(records),
            "maximum_worst_group_harm": maximum_harm,
            "allowed_worst_group_harm": maximum_worst_group_harm,
            "all_worst_groups_safe": all_worst_groups_safe,
            "negative_control_separated": control_separation,
        },
    )
