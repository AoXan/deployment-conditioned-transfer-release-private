"""Validation split helpers."""

from __future__ import annotations

import random
from typing import Sequence, TypeVar


T = TypeVar("T")


class SplitError(ValueError):
    """Raised when validation splits cannot be created."""


def holdout_split(rows: Sequence[T], test_fraction: float, random_seed: int) -> tuple[list[T], list[T]]:
    """Create a deterministic row-level held-out validation split."""

    if not 0 < test_fraction < 1:
        raise SplitError("test_fraction must be between 0 and 1.")
    if len(rows) < 2:
        raise SplitError("At least two rows are required for a holdout split.")
    shuffled = list(rows)
    random.Random(random_seed).shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_fraction))
    test = shuffled[-test_size:]
    train = shuffled[:-test_size]
    if not train:
        raise SplitError("Holdout split left no training rows.")
    return train, test
