from __future__ import annotations

from enum import Enum


class TeacherRole(Enum):
    PREDICTION = "prediction"
    REPRESENTATION = "representation"


TREE_FAMILIES = {"ridge", "random_forest", "hist_gradient_boosting"}


def validate_teacher_role(*, model_family: str, role: TeacherRole) -> None:
    if role is TeacherRole.REPRESENTATION and model_family in TREE_FAMILIES:
        raise ValueError("TREE_REPRESENTATION_TEACHER_FORBIDDEN")
