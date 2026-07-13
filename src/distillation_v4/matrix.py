from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScientificJob:
    job_id: str
    dataset: str
    phase: str
    variant: str
    model_family: str
    fold: str
    seed: int
    deterministic: bool
    input_profile: str


FOLDS = ("test_2021", "test_2022", "test_2023")
STOCHASTIC_SEEDS = (101, 202, 303)


def _jobs(dataset: str, variants: list[tuple[str, str, bool, str]]) -> list[ScientificJob]:
    result: list[ScientificJob] = []
    for variant, family, deterministic, profile in variants:
        seeds = (101,) if deterministic else STOCHASTIC_SEEDS
        for fold in FOLDS:
            for seed in seeds:
                result.append(ScientificJob(
                    job_id=f"{dataset}__phase2__{variant}__{fold}__seed_{seed}",
                    dataset=dataset,
                    phase="phase2",
                    variant=variant,
                    model_family=family,
                    fold=fold,
                    seed=seed,
                    deterministic=deterministic,
                    input_profile=profile,
                ))
    return result


def phase2_jobs(dataset: str) -> list[ScientificJob]:
    variants = [
        # 4 deterministic variants x 3 folds = 12
        ("ridge_deployable", "ridge", True, "deployable"),
        ("hgb_deployable", "hist_gradient_boosting", True, "deployable"),
        ("hgb_soil_only", "hist_gradient_boosting", True, "soil_only"),
        ("hgb_weather_soil", "hist_gradient_boosting", True, "weather_soil"),
        # 2 stochastic RF variants x 3 folds x 3 seeds = 18
        ("rf_deployable", "random_forest", False, "deployable"),
        ("rf_weather_soil", "random_forest", False, "weather_soil"),
        # 3 neural variants x 3 folds x 3 seeds = 27
        ("neural_deployable", "neural", False, "deployable"),
        ("neural_modality_encoder", "neural", False, "weather_soil"),
        ("neural_masked_pretrain", "neural_masked", False, "weather_soil"),
        # 2 deterministic proxy controls x 3 folds = 6
        ("hgb_mask_only", "hist_gradient_boosting", True, "mask_only"),
        ("hgb_shuffled_soil", "hist_gradient_boosting", True, "shuffled_soil"),
    ]
    jobs = _jobs(dataset, variants)
    if len(jobs) != 63:
        raise AssertionError(f"PHASE2_MATRIX_SIZE:{len(jobs)}")
    return jobs
