from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .phase3_matrix import Phase3Job


def canonical_mapping_fingerprint(
    value: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phase3_execution_fingerprint(
    *,
    job: Phase3Job,
    training_contract: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    split_fingerprint: str,
    data_fingerprint: str,
    code_fingerprint: str,
) -> str:
    return canonical_mapping_fingerprint(
        {
            "dataset": job.dataset,
            "fold": job.fold,
            "seed": job.seed,
            "candidate": job.candidate,
            "model_family": job.model_family,
            "input_profile": job.input_profile,
            "deterministic": job.deterministic,
            "training_contract": dict(
                training_contract
            ),
            "feature_contract": dict(
                feature_contract
            ),
            "split_fingerprint": split_fingerprint,
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": code_fingerprint,
        }
    )
