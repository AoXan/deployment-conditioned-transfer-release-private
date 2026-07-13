from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "weather_observed_days": "Observed days",
    "weather_expected_days": "Expected days",
    "weather_coverage": "Weather coverage",
    "weather_tmin_mean": "Mean minimum temperature",
    "weather_tmax_mean": "Mean maximum temperature",
    "weather_prec_sum": "Total precipitation",
    "weather_rad_sum": "Total radiation",
    "weather_et0_sum": "Reference ET",
    "weather_vpd_mean": "Mean VPD",
    "weather_cwb_sum": "Climatic water balance",
}

METHOD_LABELS = {
    "supervised": "Supervised",
    "prediction_kd": "Prediction KD",
    "combined_kd": "Combined KD",
    "representation_kd": "Representation KD",
    "missing_aware": "Missing-aware",
}

METHOD_ORDER = [
    "supervised",
    "prediction_kd",
    "combined_kd",
    "representation_kd",
    "missing_aware",
]

CONTRACT_ORDER = [
    "GROUP_complete",
    "SPATIAL_complete",
]


def save_figure(
    figure: plt.Figure,
    directory: Path,
    stem: str,
) -> None:
    for extension in [
        "pdf",
        "svg",
        "png",
    ]:
        path = directory / (
            f"{stem}.{extension}"
        )

        figure.savefig(
            path,
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )


def ordered_features(
    frame: pd.DataFrame,
    value_column: str,
) -> list[str]:
    return (
        frame.groupby(
            "feature_name"
        )[value_column]
        .mean()
        .sort_values(
            ascending=True
        )
        .index
        .tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-output",
        required=True,
    )
    parser.add_argument(
        "--supplement-output",
        required=True,
    )

    args = parser.parse_args()

    base = Path(
        args.base_output
    ).resolve()

    supplement = Path(
        args.supplement_output
    ).resolve()

    figures = (
        supplement
        / "publication_figures"
    )
    data_directory = (
        figures
        / "figure_data"
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap = pd.read_csv(
        base
        / "sample_level_raw_shap_taylor.csv"
    )
    feature_seed = pd.read_csv(
        base
        / "feature_level_by_seed.csv"
    )
    delta = pd.read_csv(
        base
        / "kd_vs_supervised_summary.csv"
    )
    ig = pd.read_csv(
        supplement
        / "sample_level_integrated_gradients.csv"
    )
    agreement = pd.read_csv(
        supplement
        / "method_agreement_by_seed.csv"
    )
    quality = pd.read_csv(
        supplement
        / "pathwise_quality.csv"
    )
    interactions = pd.read_csv(
        supplement
        / "interaction_summary_by_seed.csv"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    global_importance = (
        feature_seed.groupby(
            [
                "contract",
                "method",
                "feature_name",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            standard_deviation=(
                "mean_abs_shap",
                "std",
            ),
        )
    )

    global_importance.to_csv(
        data_directory
        / "figure_1_global_shap_importance.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.0),
        sharex=False,
        constrained_layout=True,
    )

    for axis, contract in zip(
        axes,
        CONTRACT_ORDER,
    ):
        subset = global_importance[
            global_importance[
                "contract"
            ] == contract
        ]

        feature_order = ordered_features(
            subset,
            "mean_abs_shap",
        )

        positions = np.arange(
            len(feature_order)
        )
        width = 0.15

        for method_index, method in enumerate(
            METHOD_ORDER
        ):
            method_frame = (
                subset[
                    subset["method"]
                    == method
                ]
                .set_index(
                    "feature_name"
                )
                .reindex(
                    feature_order
                )
            )

            axis.barh(
                positions
                + (
                    method_index
                    - 2
                )
                * width,
                method_frame[
                    "mean_abs_shap"
                ],
                height=width,
                xerr=method_frame[
                    "standard_deviation"
                ].fillna(0),
                label=METHOD_LABELS[
                    method
                ],
                capsize=1.5,
                linewidth=0.4,
            )

        axis.set_yticks(
            positions
        )
        axis.set_yticklabels(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in feature_order
            ]
        )
        axis.set_xlabel(
            "Mean absolute permutation SHAP"
        )
        axis.set_title(
            contract.replace(
                "_complete",
                "",
            )
        )
        axis.axvline(
            0,
            linewidth=0.7,
        )
        axis.grid(
            axis="x",
            alpha=0.2,
            linewidth=0.5,
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.035,
        ),
    )

    save_figure(
        fig,
        figures,
        "figure_1_global_shap_importance",
    )
    plt.close(fig)

    delta_column = (
        "delta_mean_abs_shap_mean"
    )

    delta_data = delta[
        [
            "contract",
            "method",
            "feature_name",
            delta_column,
            (
                "delta_mean_abs_shap_"
                "positive_seed_count"
            ),
            (
                "delta_mean_abs_shap_"
                "negative_seed_count"
            ),
        ]
    ].copy()

    delta_data.to_csv(
        data_directory
        / "figure_2_kd_supervised_shap_delta.csv",
        index=False,
    )

    limit = float(
        np.nanmax(
            np.abs(
                delta_data[
                    delta_column
                ]
            )
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.7),
        constrained_layout=True,
    )

    image = None

    for axis, contract in zip(
        axes,
        CONTRACT_ORDER,
    ):
        subset = delta_data[
            delta_data[
                "contract"
            ] == contract
        ]

        methods = [
            method
            for method
            in METHOD_ORDER
            if method != "supervised"
        ]

        features = list(
            FEATURE_LABELS
        )

        matrix = (
            subset.pivot(
                index="method",
                columns="feature_name",
                values=delta_column,
            )
            .reindex(
                index=methods,
                columns=features,
            )
            .to_numpy()
        )

        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0,
                vmax=limit,
            ),
        )

        axis.set_xticks(
            np.arange(
                len(features)
            )
        )
        axis.set_xticklabels(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in features
            ],
            rotation=55,
            ha="right",
        )

        axis.set_yticks(
            np.arange(
                len(methods)
            )
        )
        axis.set_yticklabels(
            [
                METHOD_LABELS[
                    method
                ]
                for method
                in methods
            ]
        )

        axis.set_title(
            contract.replace(
                "_complete",
                "",
            )
        )

        for row_index, method in enumerate(
            methods
        ):
            for column_index, feature in enumerate(
                features
            ):
                record = subset[
                    (
                        subset["method"]
                        == method
                    )
                    & (
                        subset[
                            "feature_name"
                        ] == feature
                    )
                ]

                if record.empty:
                    continue

                positive = int(
                    record[
                        (
                            "delta_mean_abs_shap_"
                            "positive_seed_count"
                        )
                    ].iloc[0]
                )
                negative = int(
                    record[
                        (
                            "delta_mean_abs_shap_"
                            "negative_seed_count"
                        )
                    ].iloc[0]
                )

                axis.text(
                    column_index,
                    row_index,
                    f"{positive}/{negative}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes,
            shrink=0.82,
            pad=0.02,
        )
        colorbar.set_label(
            "KD − supervised mean |SHAP|"
        )

    save_figure(
        fig,
        figures,
        "figure_2_kd_supervised_shap_delta",
    )
    plt.close(fig)

    supervised = shap[
        shap["method"]
        == "supervised"
    ].copy()

    beeswarm_data = supervised[
        [
            "contract",
            "seed",
            "sample_id",
            "feature_name",
            "feature_value_raw",
            "permutation_shap",
        ]
    ].copy()

    beeswarm_data.to_csv(
        data_directory
        / "figure_3_supervised_shap_distribution.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.4),
        constrained_layout=True,
    )

    for axis, contract in zip(
        axes,
        CONTRACT_ORDER,
    ):
        subset = supervised[
            supervised[
                "contract"
            ] == contract
        ]

        order = (
            subset.groupby(
                "feature_name"
            )[
                "permutation_shap"
            ]
            .apply(
                lambda values: float(
                    np.abs(values).mean()
                )
            )
            .sort_values(
                ascending=True
            )
            .index
            .tolist()
        )

        generator = np.random.default_rng(
            20260708
        )

        for feature_position, feature in enumerate(
            order
        ):
            data = subset[
                subset[
                    "feature_name"
                ] == feature
            ]

            values = data[
                "permutation_shap"
            ].to_numpy()

            raw = data[
                "feature_value_raw"
            ].to_numpy()

            if np.nanmax(raw) > np.nanmin(raw):
                normalised = (
                    raw - np.nanmin(raw)
                ) / (
                    np.nanmax(raw)
                    - np.nanmin(raw)
                )
            else:
                normalised = np.zeros_like(
                    raw
                )

            jitter = generator.normal(
                0,
                0.08,
                len(values),
            )

            axis.scatter(
                values,
                feature_position
                + jitter,
                c=normalised,
                cmap="viridis",
                s=8,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )

        axis.axvline(
            0,
            linewidth=0.8,
        )
        axis.set_yticks(
            np.arange(
                len(order)
            )
        )
        axis.set_yticklabels(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in order
            ]
        )
        axis.set_xlabel(
            "Permutation SHAP contribution"
        )
        axis.set_title(
            contract.replace(
                "_complete",
                "",
            )
        )
        axis.grid(
            axis="x",
            alpha=0.15,
            linewidth=0.5,
        )

    save_figure(
        fig,
        figures,
        "figure_3_supervised_shap_beeswarm",
    )
    plt.close(fig)

    agreement_data = agreement.copy()

    agreement_data.to_csv(
        data_directory
        / "figure_4_method_agreement.csv",
        index=False,
    )

    fig, axis = plt.subplots(
        figsize=(7.0, 4.8),
        constrained_layout=True,
    )

    for contract in CONTRACT_ORDER:
        subset = agreement_data[
            agreement_data[
                "contract"
            ] == contract
        ]

        positions = np.arange(
            len(subset)
        )

        axis.scatter(
            positions,
            subset[
                "shap_ig_pearson"
            ],
            label=(
                contract.replace(
                    "_complete",
                    "",
                )
                + ": SHAP vs IG"
            ),
            s=30,
        )

        if (
            "shap_taylor_pearson"
            in subset.columns
        ):
            axis.scatter(
                positions,
                subset[
                    "shap_taylor_pearson"
                ],
                marker="x",
                label=(
                    contract.replace(
                        "_complete",
                        "",
                    )
                    + ": SHAP vs Taylor–Shapley"
                ),
                s=34,
            )

    axis.axhline(
        0,
        linewidth=0.7,
    )
    axis.set_ylim(
        -1.02,
        1.02,
    )
    axis.set_ylabel(
        "Pearson correlation"
    )
    axis.set_xlabel(
        "Contract–method–seed cell"
    )
    axis.set_title(
        "Agreement between attribution methods"
    )
    axis.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.5,
    )
    axis.legend(
        frameon=False,
        ncol=2,
    )

    save_figure(
        fig,
        figures,
        "figure_4_attribution_method_agreement",
    )
    plt.close(fig)

    valid_quality = quality[
        quality[
            "taylor_relative_residual"
        ] <= 0.05
    ]

    valid_keys = valid_quality[
        [
            "contract",
            "method",
            "seed",
            "sample_id",
        ]
    ]

    valid_interactions = (
        interactions.merge(
            valid_keys,
            on=[
                "contract",
                "method",
                "seed",
            ],
            how="inner",
        )
        if not valid_keys.empty
        else pd.DataFrame()
    )

    interaction_plot_data = (
        interactions.groupby(
            [
                "contract",
                "method",
                "feature_i",
                "feature_j",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_interaction=(
                "mean_abs_interaction",
                "mean",
            )
        )
    )

    interaction_plot_data.to_csv(
        data_directory
        / "figure_5_interaction_strength.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.0),
        constrained_layout=True,
    )

    for axis, contract in zip(
        axes,
        CONTRACT_ORDER,
    ):
        subset = interaction_plot_data[
            (
                interaction_plot_data[
                    "contract"
                ] == contract
            )
            & (
                interaction_plot_data[
                    "method"
                ] != "supervised"
            )
        ]

        features = list(
            FEATURE_LABELS
        )

        matrix = np.zeros(
            (
                len(features),
                len(features),
            )
        )

        counts = np.zeros_like(
            matrix
        )

        feature_index = {
            feature: index
            for index, feature
            in enumerate(features)
        }

        for row in subset.itertuples(
            index=False
        ):
            first = feature_index[
                row.feature_i
            ]
            second = feature_index[
                row.feature_j
            ]

            matrix[first, second] += (
                row.mean_abs_interaction
            )
            matrix[second, first] += (
                row.mean_abs_interaction
            )

            counts[first, second] += 1
            counts[second, first] += 1

        matrix = np.divide(
            matrix,
            counts,
            out=np.zeros_like(matrix),
            where=counts > 0,
        )

        image = axis.imshow(
            matrix,
            cmap="magma",
            aspect="equal",
        )

        axis.set_xticks(
            np.arange(
                len(features)
            )
        )
        axis.set_yticks(
            np.arange(
                len(features)
            )
        )
        axis.set_xticklabels(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in features
            ],
            rotation=60,
            ha="right",
        )
        axis.set_yticklabels(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in features
            ]
        )
        axis.set_title(
            contract.replace(
                "_complete",
                "",
            )
        )

        colorbar = fig.colorbar(
            image,
            ax=axis,
            fraction=0.045,
            pad=0.03,
        )
        colorbar.set_label(
            "Mean absolute pair interaction"
        )

    save_figure(
        fig,
        figures,
        "figure_5_pathwise_interaction_heatmap",
    )
    plt.close(fig)

    example_rows = []

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.8),
        constrained_layout=True,
    )

    for axis, contract in zip(
        axes,
        CONTRACT_ORDER,
    ):
        subset = shap[
            shap[
                "contract"
            ] == contract
        ]

        totals = (
            subset.groupby(
                [
                    "method",
                    "seed",
                    "sample_id",
                ],
                as_index=False,
            )
            .agg(
                total_abs_shap=(
                    "permutation_shap",
                    lambda values: float(
                        np.abs(values).sum()
                    ),
                ),
                absolute_error=(
                    "absolute_error",
                    "first",
                ),
            )
            .sort_values(
                [
                    "total_abs_shap",
                    "absolute_error",
                ],
                ascending=False,
            )
        )

        selected = totals.iloc[0]

        sample = subset[
            (
                subset["method"]
                == selected["method"]
            )
            & (
                subset["seed"]
                == selected["seed"]
            )
            & (
                subset[
                    "sample_id"
                ].astype(str)
                == str(
                    selected[
                        "sample_id"
                    ]
                )
            )
        ].copy()

        sample[
            "absolute_shap"
        ] = sample[
            "permutation_shap"
        ].abs()

        sample = sample.sort_values(
            "absolute_shap",
            ascending=True,
        )

        example_rows.append(
            sample
        )

        axis.barh(
            [
                FEATURE_LABELS[
                    feature
                ]
                for feature
                in sample[
                    "feature_name"
                ]
            ],
            sample[
                "permutation_shap"
            ],
        )

        axis.axvline(
            0,
            linewidth=0.8,
        )
        axis.set_xlabel(
            "Permutation SHAP contribution"
        )
        axis.set_title(
            (
                contract.replace(
                    "_complete",
                    "",
                )
                + " exemplar\n"
                + METHOD_LABELS[
                    str(
                        selected[
                            "method"
                        ]
                    )
                ]
                + f", seed {int(selected['seed'])}"
            )
        )
        axis.grid(
            axis="x",
            alpha=0.2,
            linewidth=0.5,
        )

    pd.concat(
        example_rows,
        ignore_index=True,
    ).to_csv(
        data_directory
        / "figure_6_exemplar_contributions.csv",
        index=False,
    )

    save_figure(
        fig,
        figures,
        "figure_6_exemplar_shap_contributions",
    )
    plt.close(fig)

    figure_manifest = {
        "status": "COMPLETED",
        "figures": sorted(
            path.name
            for path in figures.iterdir()
            if path.is_file()
        ),
        "figure_data": sorted(
            path.name
            for path
            in data_directory.iterdir()
            if path.is_file()
        ),
        "notes": [
            (
                "SHAP figures use the formally "
                "validated raw-weather attribution "
                "output."
            ),
            (
                "Error bars in Figure 1 are "
                "between-seed standard deviations."
            ),
            (
                "Heatmap cells in Figure 2 show "
                "positive/negative seed counts."
            ),
            (
                "Interaction plots must be interpreted "
                "with the Taylor–Shapley reconstruction "
                "quality results."
            ),
        ],
    }

    (
        figures
        / "figure_manifest.json"
    ).write_text(
        json.dumps(
            figure_manifest,
            indent=2,
            sort_keys=True,
        )
    )

    print("=" * 88)
    print(
        "PUBLICATION_FIGURES_COMPLETE"
    )
    print("Directory:", figures)
    print("=" * 88)


if __name__ == "__main__":
    main()
