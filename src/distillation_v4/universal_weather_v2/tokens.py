from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class WeatherTokenBatchV2:
    values: torch.Tensor
    feature_ids: torch.Tensor
    time_ids: torch.Tensor
    availability: torch.Tensor
    padding_mask: torch.Tensor
    feature_names: tuple[str, ...]


def canonical_feature_name_v2(name: str) -> str:
    text = str(name).strip().lower()

    replacements = {
        "precipitation": "rain",
        "rainfall": "rain",
        "temperature": "temp",
        "maximum": "max",
        "minimum": "min",
        "average": "mean",
        "vapour": "vapor",
    }

    parts = [
        replacements.get(part, part)
        for part in re.sub(
            r"[^a-z0-9]+",
            "_",
            text,
        ).split("_")
        if part
    ]

    return "_".join(parts)


def feature_id_v2(
    name: str,
    vocabulary_size: int,
) -> int:
    digest = hashlib.sha256(
        canonical_feature_name_v2(name).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    ) % vocabulary_size


def time_id_v2(
    name: str,
    position: int,
    maximum_positions: int,
) -> int:
    canonical = canonical_feature_name_v2(
        name
    )

    patterns = [
        r"(?:week|wk)_(\d+)",
        r"(?:month|mon)_(\d+)",
        r"(?:day|doy)_(\d+)",
        r"(?:period|window|bin)_(\d+)",
        r"_(\d+)$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            canonical,
        )

        if match:
            return (
                int(match.group(1))
                % maximum_positions
            )

    return position % maximum_positions


def build_weather_tokens_v2(
    values: np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
    *,
    availability: np.ndarray | torch.Tensor | None = None,
    feature_vocabulary_size: int = 8192,
    maximum_time_positions: int = 1024,
    device: torch.device | str | None = None,
) -> WeatherTokenBatchV2:
    if isinstance(values, np.ndarray):
        values = np.ascontiguousarray(
            values
        )

    tensor = torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    )

    if tensor.ndim != 2:
        raise ValueError(
            "WEATHER_VALUES_MUST_BE_2D"
        )

    rows, columns = tensor.shape

    if columns != len(feature_names):
        raise ValueError(
            "WEATHER_FEATURE_NAME_COUNT_MISMATCH"
        )

    finite = torch.isfinite(tensor)

    if availability is None:
        observed = finite
    else:
        supplied = torch.as_tensor(
            availability,
            dtype=torch.float32,
            device=tensor.device,
        )

        if supplied.shape != tensor.shape:
            raise ValueError(
                "WEATHER_AVAILABILITY_SHAPE_MISMATCH"
            )

        observed = (
            supplied > 0
        ) & finite

    clean = torch.where(
        finite,
        tensor,
        torch.zeros_like(tensor),
    )

    feature_ids = torch.tensor(
        [
            feature_id_v2(
                name,
                feature_vocabulary_size,
            )
            for name in feature_names
        ],
        dtype=torch.long,
        device=tensor.device,
    ).unsqueeze(0).expand(
        rows,
        -1,
    )

    time_ids = torch.tensor(
        [
            time_id_v2(
                name,
                index,
                maximum_time_positions,
            )
            for index, name in enumerate(
                feature_names
            )
        ],
        dtype=torch.long,
        device=tensor.device,
    ).unsqueeze(0).expand(
        rows,
        -1,
    )

    return WeatherTokenBatchV2(
        values=clean,
        feature_ids=feature_ids,
        time_ids=time_ids,
        availability=observed.to(
            torch.float32
        ),
        padding_mask=torch.zeros(
            (rows, columns),
            dtype=torch.bool,
            device=tensor.device,
        ),
        feature_names=tuple(
            str(name)
            for name in feature_names
        ),
    )
