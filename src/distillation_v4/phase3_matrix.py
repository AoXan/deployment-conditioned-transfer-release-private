from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Phase3Job:
    job_id: str
    dataset: str
    fold: str
    seed: int
    component: str
    candidate: str
    model_family: str
    input_profile: str
    deterministic: bool
    scientific_roles: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["scientific_roles"] = list(
            self.scientific_roles
        )
        return value


def build_phase3_jobs(
    *,
    dataset: str,
    folds: Iterable[str],
    deterministic_seed: int,
    stochastic_seeds: Iterable[int],
) -> list[Phase3Job]:
    folds = tuple(str(fold) for fold in folds)
    stochastic_seeds = tuple(
        int(seed) for seed in stochastic_seeds
    )

    if len(folds) != 3:
        raise ValueError(
            "PHASE3_REQUIRES_EXACTLY_THREE_FOLDS"
        )
    if len(stochastic_seeds) != 3:
        raise ValueError(
            "PHASE3_REQUIRES_EXACTLY_THREE_STOCHASTIC_SEEDS"
        )
    if len(set(folds)) != len(folds):
        raise ValueError("PHASE3_FOLDS_NOT_UNIQUE")
    if len(set(stochastic_seeds)) != len(
        stochastic_seeds
    ):
        raise ValueError("PHASE3_SEEDS_NOT_UNIQUE")

    jobs: list[Phase3Job] = []

    def add(
        *,
        fold: str,
        seed: int,
        component: str,
        candidate: str,
        model_family: str,
        input_profile: str,
        deterministic: bool,
        scientific_roles: tuple[str, ...],
    ) -> None:
        job_id = "__".join(
            (
                dataset,
                "phase3",
                component,
                candidate,
                fold,
                f"seed_{seed}",
            )
        )
        jobs.append(
            Phase3Job(
                job_id=job_id,
                dataset=dataset,
                fold=fold,
                seed=seed,
                component=component,
                candidate=candidate,
                model_family=model_family,
                input_profile=input_profile,
                deterministic=deterministic,
                scientific_roles=scientific_roles,
            )
        )

    for fold in folds:
        # 3 jobs:
        # one deterministic prediction-only teacher.
        add(
            fold=fold,
            seed=deterministic_seed,
            component="prediction_teacher",
            candidate="hgb_privileged",
            model_family="hist_gradient_boosting",
            input_profile="weather_soil",
            deterministic=True,
            scientific_roles=("prediction",),
        )

        # 18 jobs:
        # two neural representation candidates × 3 seeds.
        for seed in stochastic_seeds:
            add(
                fold=fold,
                seed=seed,
                component="representation_teacher",
                candidate="shared_early_fusion",
                model_family="shared_dense",
                input_profile="weather_soil",
                deterministic=False,
                scientific_roles=(
                    "prediction",
                    "representation",
                ),
            )
            add(
                fold=fold,
                seed=seed,
                component="representation_teacher",
                candidate="modality_specific_late_fusion",
                model_family="modality_specific",
                input_profile="weather_soil",
                deterministic=False,
                scientific_roles=("representation",),
            )

        # 12 jobs:
        # deterministic same-family deployable control
        # plus stochastic shared neural receiver control.
        add(
            fold=fold,
            seed=deterministic_seed,
            component="matched_deployable_control",
            candidate="hgb_deployable",
            model_family="hist_gradient_boosting",
            input_profile="deployable",
            deterministic=True,
            scientific_roles=("prediction_control",),
        )

        for seed in stochastic_seeds:
            add(
                fold=fold,
                seed=seed,
                component="matched_deployable_control",
                candidate="shared_neural_receiver",
                model_family="shared_dense",
                input_profile="deployable",
                deterministic=False,
                scientific_roles=(
                    "prediction_control",
                    "representation_control",
                ),
            )

        # 18 jobs:
        # independently materialized qualification controls.
        for seed in stochastic_seeds:
            add(
                fold=fold,
                seed=seed,
                component="qualification_control",
                candidate="random_representation",
                model_family="representation_probe_control",
                input_profile="deployable",
                deterministic=False,
                scientific_roles=(
                    "representation_negative_control",
                ),
            )
            add(
                fold=fold,
                seed=seed,
                component="qualification_control",
                candidate="shuffled_soil_neural",
                model_family="modality_specific",
                input_profile="weather_shuffled_soil",
                deterministic=False,
                scientific_roles=(
                    "prediction_negative_control",
                    "representation_negative_control",
                ),
            )

    if len(jobs) != 51:
        raise RuntimeError(
            f"PHASE3_MATRIX_SIZE_MISMATCH:{len(jobs)}"
        )

    identifiers = [job.job_id for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            "PHASE3_MATRIX_DUPLICATE_JOB_ID"
        )

    return jobs
