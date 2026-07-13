from __future__ import annotations

import torch


ROUTE_LOSSES = {
    "supervised": (
        "supervised",
    ),
    "prediction_kd": (
        "supervised",
        "prediction",
    ),
    "representation_kd": (
        "supervised",
        "representation",
    ),
    "combined_kd": (
        "supervised",
        "prediction",
        "representation",
    ),
    "missing_aware": (
        "supervised",
        "missing_prediction_consistency",
        "missing_representation_consistency",
    ),
}


def universal_route_loss_v2(
    *,
    route: str,
    student_prediction: torch.Tensor,
    student_representation: torch.Tensor,
    target: torch.Tensor,
    prediction_teacher: torch.Tensor | None,
    representation_teacher: torch.Tensor | None,
    missing_prediction: torch.Tensor | None,
    missing_representation: torch.Tensor | None,
    weights: dict[str, float],
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    if route not in ROUTE_LOSSES:
        raise ValueError(
            f"UNKNOWN_UNIVERSAL_ROUTE:{route}"
        )

    losses = {
        "supervised": (
            torch.nn.functional.huber_loss(
                student_prediction,
                target,
            )
        )
    }

    if "prediction" in ROUTE_LOSSES[route]:
        if prediction_teacher is None:
            raise ValueError(
                "PREDICTION_TEACHER_REQUIRED"
            )

        losses["prediction"] = (
            torch.nn.functional.huber_loss(
                student_prediction,
                prediction_teacher.detach(),
            )
        )

    if "representation" in ROUTE_LOSSES[
        route
    ]:
        if representation_teacher is None:
            raise ValueError(
                "REPRESENTATION_TEACHER_REQUIRED"
            )

        losses["representation"] = (
            torch.nn.functional.mse_loss(
                student_representation,
                representation_teacher.detach(),
            )
        )

    if (
        "missing_prediction_consistency"
        in ROUTE_LOSSES[route]
    ):
        if missing_prediction is None:
            raise ValueError(
                "MISSING_PREDICTION_REQUIRED"
            )

        losses[
            "missing_prediction_consistency"
        ] = torch.nn.functional.huber_loss(
            missing_prediction,
            student_prediction.detach(),
        )

    if (
        "missing_representation_consistency"
        in ROUTE_LOSSES[route]
    ):
        if missing_representation is None:
            raise ValueError(
                "MISSING_REPRESENTATION_REQUIRED"
            )

        losses[
            "missing_representation_consistency"
        ] = torch.nn.functional.mse_loss(
            missing_representation,
            student_representation.detach(),
        )

    total = sum(
        losses[name]
        * float(
            weights.get(name, 1.0)
        )
        for name in ROUTE_LOSSES[route]
    )

    if not torch.isfinite(total):
        raise FloatingPointError(
            "NONFINITE_UNIVERSAL_ROUTE_LOSS"
        )

    return total, losses
