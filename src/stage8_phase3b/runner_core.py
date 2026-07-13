from __future__ import annotations

from pathlib import Path
from typing import Iterable


REPLAY_STAGES = [
    "replay",
    "artifact_validation",
    "alignment",
    "reproduction",
    "reproduction_gate",
]

EXPLANATION_STAGES = [
    "method_component_ablation",
    "grouped_permutation",
    "formal_condition_comparison",
    "modality_sensitivity",
    "integrated_gradients",
    "shap",
    "attribution_stability",
    "final_synthesis",
]


def pipeline_stages(*, no_explanations: bool, explanations_only: bool) -> list[str]:
    if explanations_only:
        return ["explanation_eligibility", *EXPLANATION_STAGES]
    if no_explanations:
        return list(REPLAY_STAGES)
    return [*REPLAY_STAGES, *EXPLANATION_STAGES]


def explanation_eligibility(reproduction_record: dict) -> str:
    return "eligible" if reproduction_record.get("reproduction_status") == "PASS" else "blocked_reproduction_not_pass"


def selected_formal_ids(manifest: dict, *, subset: str | None = None, single_job: str | None = None) -> list[str]:
    ids = list(dict.fromkeys(manifest.get("all_formal_candidate_ids", [])))
    if single_job:
        if single_job not in ids:
            raise ValueError(f"SINGLE_JOB_NOT_IN_MANIFEST:{single_job}")
        return [single_job]
    if subset and subset in manifest.get("subsets", {}):
        subset_ids = list(dict.fromkeys(str(item) for item in manifest["subsets"][subset]))
        missing = [item for item in subset_ids if item not in ids]
        if missing:
            raise ValueError(f"SUBSET_IDS_NOT_IN_MANIFEST:{subset}:{missing}")
        return subset_ids
    if subset and subset not in {manifest.get("matrix"), "all"}:
        # Named subsets are resolved by the manifest builder.  The runner only
        # accepts a subset label matching the manifest matrix to avoid silently
        # selecting a different scientific matrix.
        raise ValueError(f"SUBSET_NOT_IN_MANIFEST:{subset}")
    return ids


def expected_jobs_match(manifest: dict, expected_target_jobs: int | None) -> None:
    if expected_target_jobs is None:
        return
    actual = int(manifest.get("expected_target_jobs", -1))
    if actual != int(expected_target_jobs):
        raise ValueError(f"EXPECTED_TARGET_JOB_MISMATCH:{actual}!={expected_target_jobs}")


def ensure_manifest_hash(manifest: dict, confirmation: str | None) -> None:
    expected = str(manifest.get("manifest_hash", "")).strip()
    if not expected:
        raise ValueError("MANIFEST_HASH_MISSING")
    if confirmation != expected:
        raise ValueError("REPLAY_MANIFEST_HASH_CONFIRMATION_REQUIRED")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")
