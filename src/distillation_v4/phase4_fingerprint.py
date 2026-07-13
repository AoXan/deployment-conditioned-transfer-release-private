from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def dataframe_fingerprint(
    frame: pd.DataFrame,
) -> str:
    ordered_columns = sorted(
        str(column)
        for column in frame.columns
    )
    ordered = frame.loc[:, ordered_columns]

    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=True,
        categorize=True,
    ).to_numpy()

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            ordered_columns,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        str(ordered.dtypes.astype(str).tolist()).encode(
            "utf-8"
        )
    )
    digest.update(row_hashes.tobytes())

    return digest.hexdigest()


def phase4_execution_fingerprint(
    *,
    route_record: dict[str, Any],
    training_contract: dict[str, Any],
    frame_fingerprint: str,
    route_manifest_fingerprint: str,
    code_contract: str,
) -> str:
    return canonical_fingerprint(
        {
            "route_record": route_record,
            "training_contract": training_contract,
            "frame_fingerprint": frame_fingerprint,
            "route_manifest_fingerprint": (
                route_manifest_fingerprint
            ),
            "code_contract": code_contract,
        }
    )
