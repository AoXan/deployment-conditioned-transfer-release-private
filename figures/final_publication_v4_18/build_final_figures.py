#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MIN_FONT_SIZE = 9.5
FIGURE_IDS = tuple(range(1, 7))
CANDIDATES = ("minimal", "claim-first", "integrated")
CONTRACTS = ("GROUP_complete", "SPATIAL_complete")
METHODS = (
    "local_scratch",
    "supervised",
    "prediction_kd",
    "combined_kd",
    "representation_kd",
    "missing_aware",
)
TRANSFER_ROUTES = METHODS[2:]
ATTRIBUTION_METHODS = ("supervised", *TRANSFER_ROUTES)
METHOD_LABELS = {
    "local_scratch": "Target scratch",
    "supervised": "Supervised transfer",
    "prediction_kd": "Prediction KD",
    "combined_kd": "Combined KD",
    "representation_kd": "Representation KD",
    "missing_aware": "Missing-aware",
}
METHOD_SHORT = {
    "local_scratch": "Scratch",
    "supervised": "Sup.",
    "prediction_kd": "Pred.",
    "combined_kd": "Comb.",
    "representation_kd": "Repr.",
    "missing_aware": "Miss.",
}
ROUTE_COLORS = {
    "local_scratch": "#6B7280",
    "supervised": "#0077BB",
    "prediction_kd": "#EE7733",
    "combined_kd": "#009988",
    "representation_kd": "#CC79A7",
    "missing_aware": "#DDAA33",
}
CONTRACT_COLORS = {
    "GROUP_complete": "#0077BB",
    "SPATIAL_complete": "#EE7733",
}
GROUPS = (
    "observation_support",
    "temperature",
    "water_balance",
    "radiation",
)
GROUP_LABELS = {
    "observation_support": "Obs.",
    "temperature": "Temp.",
    "water_balance": "Water",
    "radiation": "Rad.",
}
FEATURE_ORDER = (
    "weather_observed_days",
    "weather_expected_days",
    "weather_coverage",
    "weather_tmin_mean",
    "weather_tmax_mean",
    "weather_prec_sum",
    "weather_rad_sum",
    "weather_et0_sum",
    "weather_vpd_mean",
    "weather_cwb_sum",
)
FEATURE_LABELS = {
    "weather_observed_days": "Observed days",
    "weather_expected_days": "Expected days",
    "weather_coverage": "Coverage",
    "weather_tmin_mean": "Min. temp.",
    "weather_tmax_mean": "Max. temp.",
    "weather_prec_sum": "Precipitation",
    "weather_rad_sum": "Radiation",
    "weather_et0_sum": "ET0",
    "weather_vpd_mean": "VPD",
    "weather_cwb_sum": "Water balance",
}
FEATURE_SHORT_LABELS = {
    "weather_observed_days": "Obs. days",
    "weather_expected_days": "Exp. days",
    "weather_coverage": "Cov.",
    "weather_tmin_mean": "Tmin",
    "weather_tmax_mean": "Tmax",
    "weather_prec_sum": "Precip.",
    "weather_rad_sum": "Rad.",
    "weather_et0_sum": "ET0",
    "weather_vpd_mean": "VPD",
    "weather_cwb_sum": "WB",
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": MIN_FONT_SIZE,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": MIN_FONT_SIZE,
            "ytick.labelsize": MIN_FONT_SIZE,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_dir(root: Path) -> Path:
    return root / "figures/final_publication_v4_18"


def source_paths(root: Path) -> dict[str, Path]:
    base = source_dir(root)
    return {
        "performance": base / "source_performance_landscape_seed.csv",
        "scratch_status": base / "source_scratch_baseline_status.csv",
        "support_points": base / "source_support_delta_samples.csv",
        "support_summary": base / "source_support_delta_summary.csv",
        "feature_summary": base / "source_signed_feature_delta_summary.csv",
        "linkage_points": base / "source_linkage_pointcloud.csv",
        "linkage_summary": base / "source_linkage_summary.csv",
        "linkage_stability": base / "source_linkage_seed_stability.csv",
        "attribution": base / "source_attribution_group_shares.csv",
        "sensitivity": (
            base / "source_perturbation_normalized_sensitivity.csv"
        ),
        "agreement": base / "source_perturbation_agreement_disjoint.csv",
        "ood": base / "source_ood_quality_9d.csv",
        "early_split": base / "source_early_split_ranking.csv",
        "beeswarm": base / "source_supervised_shap_beeswarm.csv",
        "agreement_cells": base / "source_explanation_agreement_by_cell.csv",
        "experiment_map": base / "source_complete_experiment_map.csv",
        "manifest": base / "evidence_build_manifest.json",
    }


def validate_inputs(root: Path) -> dict[str, object]:
    paths = source_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_V418_FIGURE_SOURCES:{missing}")
    performance = pd.read_csv(paths["performance"])
    support = pd.read_csv(paths["support_points"])
    features = pd.read_csv(paths["feature_summary"])
    linkage = pd.read_csv(paths["linkage_points"])
    agreement = pd.read_csv(paths["agreement"])
    early_split = pd.read_csv(paths["early_split"])
    beeswarm = pd.read_csv(paths["beeswarm"])
    agreement_cells = pd.read_csv(paths["agreement_cells"])
    experiment_map = pd.read_csv(paths["experiment_map"])
    ood = pd.read_csv(paths["ood"])
    if set(performance["method"]) != set(METHODS):
        raise ValueError("PERFORMANCE_METHOD_SET_MISMATCH")
    if set(features["feature_name"]) != set(FEATURE_ORDER):
        raise ValueError("FEATURE_ORDER_SET_MISMATCH")
    if set(agreement["n_groups"]) != {4}:
        raise ValueError("AGREEMENT_NOT_FOUR_GROUP")
    if set(ood["n_dimensions"]) != {9}:
        raise ValueError("OOD_DIMENSION_NOT_NINE")
    if len(early_split) != 14 or int(early_split["ranking_reversal"].sum()) != 9:
        raise ValueError("EARLY_SPLIT_SOURCE_MISMATCH")
    if set(beeswarm["method"]) != {"supervised"}:
        raise ValueError("BEESWARM_NOT_SUPERVISED_ONLY")
    if "feature_value_colour" not in beeswarm or "feature_value_scale" not in beeswarm:
        raise ValueError("BEESWARM_COLOUR_SEMANTICS_MISSING")
    if not beeswarm.loc[
        ~beeswarm["constant_within_contract"].astype(bool),
        "feature_value_colour",
    ].between(0.0, 1.0).all():
        raise ValueError("BEESWARM_COLOUR_RANGE_INVALID")
    if set(agreement_cells["comparison"]) != {"SHAP--IG", "SHAP--Taylor"}:
        raise ValueError("AGREEMENT_COMPARISON_SET_MISMATCH")
    return {
        "performance_rows": int(len(performance)),
        "support_point_rows": int(len(support)),
        "signed_feature_rows": int(len(features)),
        "linkage_point_rows": int(len(linkage)),
        "agreement_rows": int(len(agreement)),
        "early_split_rows": int(len(early_split)),
        "beeswarm_rows": int(len(beeswarm)),
        "agreement_cell_rows": int(len(agreement_cells)),
        "experiment_map_rows": int(len(experiment_map)),
        "ood_dimensions": sorted(ood["n_dimensions"].unique().tolist()),
    }


def save_figure(fig: plt.Figure, output: Path, number: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output / f"figure_{number:02d}.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.08,
        1.05,
        text,
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        va="bottom",
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    fontsize: float | None = 8.5,
    wrap_chars: int | None = None,
) -> tuple[patches.FancyBboxPatch, matplotlib.text.Text]:
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(rect)
    point_size = MIN_FONT_SIZE if fontsize is None else fontsize
    # Word-boundary wrapping keeps labels legible at their final embedded size.
    chars = wrap_chars or max(12, int(width * 55 / (point_size / 8.0)))
    wrapped = "\n".join(
        textwrap.wrap(text, width=chars, break_long_words=False, break_on_hyphens=False)
    )
    label = ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        wrapped,
        ha="center",
        va="center",
        fontsize=point_size,
        linespacing=1.12,
    )
    return rect, label


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#4B5563",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
    )


def figure_1(output: Path) -> None:
    """Selected publication layout from two grid candidates, with containment checks."""
    fig = plt.figure(figsize=(7.2, 3.65))
    grid = fig.add_gridspec(1, 3, left=0.03, right=0.98, top=0.88, bottom=0.20, wspace=0.30, width_ratios=(0.85, 2.25, 0.90))
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Deployment partitions", fontsize=10, pad=7)
    ax.text(0.5, 0.95, "Conceptual motivation", ha="center", va="top", fontsize=8.5, color="#4B5563")
    nodes: list[tuple[patches.FancyBboxPatch, matplotlib.text.Text]] = []
    rows = ((0.72, "Random", "M1  >  M2  >  M3", "#6B7280"), (0.47, "GROUP", "M2  >  M3  >  M1", "#0077BB"), (0.22, "SPATIAL", "M3  >  M1  >  M2", "#EE7733"))
    for y, name, ranking, color in rows:
        nodes.append(_box(ax, (0.03, y), 0.37, 0.13, name, color=color, fontsize=8.0))
        ax.text(0.46, y + 0.065, ranking, ha="left", va="center", fontsize=8.0)

    ax = axes[1]
    ax.set_title("(b) Formal transfer comparison", fontsize=10, pad=7)
    ax.text(0.06, 0.91, "Local path", fontsize=8.3, color="#00796B", fontweight="bold")
    ax.text(0.57, 0.91, "Transfer path", fontsize=8.3, color="#005B96", fontweight="bold")
    nodes.append(_box(ax, (0.04, 0.71), 0.40, 0.11, "Target training data", color="#009988", fontsize=8.0))
    nodes.append(_box(ax, (0.04, 0.53), 0.40, 0.11, "Random initialisation", color="#009988", fontsize=8.0))
    nodes.append(_box(ax, (0.04, 0.35), 0.40, 0.11, "Supervised target training", color="#009988", fontsize=8.0))
    nodes.append(_box(ax, (0.04, 0.17), 0.40, 0.11, "Target-scratch model", color="#009988", fontsize=8.0))
    _arrow(ax, (0.23, 0.71), (0.23, 0.64), color="#009988")
    _arrow(ax, (0.23, 0.53), (0.23, 0.46), color="#009988")
    _arrow(ax, (0.23, 0.35), (0.23, 0.28), color="#009988")
    nodes.append(_box(ax, (0.56, 0.71), 0.40, 0.11, "Source data + source route", color="#0077BB", fontsize=8.0))
    nodes.append(_box(ax, (0.56, 0.53), 0.40, 0.11, "Source-trained checkpoint", color="#0077BB", fontsize=8.0))
    nodes.append(_box(ax, (0.56, 0.33), 0.40, 0.15, "Target data + supervised fine-tuning", color="#0077BB", fontsize=8.0))
    nodes.append(_box(ax, (0.56, 0.15), 0.40, 0.11, "Transferred target model", color="#0077BB", fontsize=8.0))
    _arrow(ax, (0.75, 0.71), (0.75, 0.64), color="#0077BB")
    _arrow(ax, (0.75, 0.53), (0.75, 0.46), color="#0077BB")
    _arrow(ax, (0.75, 0.35), (0.75, 0.28), color="#0077BB")
    nodes.append(_box(ax, (0.08, 0.02), 0.84, 0.10, "Matched GROUP / SPATIAL evaluation", color="#EE7733", fontsize=8.0))
    _arrow(ax, (0.23, 0.17), (0.32, 0.12), color="#009988")
    _arrow(ax, (0.75, 0.17), (0.68, 0.12), color="#0077BB")

    ax = axes[2]
    ax.set_title("(c) Observation layers", fontsize=10, pad=7)
    for y, label, color in ((0.72, "Predictive performance", "#4B5563"), (0.47, "Attribution redistribution", "#0077BB"), (0.22, "Finite OOD stress response", "#EE7733")):
        nodes.append(_box(ax, (0.04, y), 0.92, 0.15, label, color=color, fontsize=8.0, wrap_chars=18))
    footer = fig.text(0.5, 0.070, "Transfer outcomes and the empirical informativeness\nof behavioural diagnostics are deployment-conditioned.", ha="center", va="center", fontsize=9.0, color="#111827", linespacing=1.18)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for rect, label in nodes:
        rb = rect.get_window_extent(renderer).padded(-1.0)
        tb = label.get_window_extent(renderer)
        if not (rb.contains(*tb.get_points()[0]) and rb.contains(*tb.get_points()[1])):
            failures.append(label.get_text().replace("\n", " "))
    if failures:
        raise ValueError(f"FIGURE1_TEXT_OUTSIDE_NODE:{failures}")
    (output / "figure_01_layout_audit.json").write_text(
        json.dumps({"node_count": len(nodes), "font_min_pt": 8.0, "containment": "pass", "footer_lines": 2}, indent=2) + "\n",
        encoding="utf-8",
    )
    save_figure(fig, output, 1)


def _metric_table(
    ax: plt.Axes,
    frame: pd.DataFrame,
    contract: str,
    label: str,
    fixed_scratch: bool = False,
) -> None:
    cell = frame[frame["contract"] == contract]
    means = cell.groupby("method", observed=True)[["mae", "rmse", "r2"]].mean()
    stds = cell.groupby("method", observed=True)[["mae", "rmse", "r2"]].std()
    ax.axis("off")
    table = ax.table(
        cellText=[[""] * 4 for _ in METHODS],
        colLabels=["Route", "MAE", "RMSE", r"$R^2$"],
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    for row, method in enumerate(METHODS):
        table[(row + 1, 0)].get_text().set_text(METHOD_SHORT[method])
        table[(row + 1, 0)].set_facecolor("#F3F4F6" if method in {"local_scratch", "supervised"} else "white")
        for column, metric in enumerate(("mae", "rmse", "r2")):
            value = means.loc[method, metric]
            spread = stds.loc[method, metric]
            best = (
                value == means[metric].min()
                if metric != "r2"
                else value == means[metric].max()
            )
            cell = table[(row + 1, column + 1)]
            if fixed_scratch and method == "local_scratch":
                cell.get_text().set_text(f"{value:.3f}\n(fixed)")
            else:
                cell.get_text().set_text(f"{value:.3f}\n({spread:.3f})")
            cell.get_text().set_fontweight("bold" if best else "normal")
            cell.set_facecolor("white")
    for cell in table.get_celld().values():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.55)
    ax.set_title(label, fontsize=10.0, pad=3)


def _delta_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    routes = METHODS[1:] if column.endswith("scratch") else TRANSFER_ROUTES
    x = np.arange(len(routes))
    offsets = {"GROUP_complete": -0.13, "SPATIAL_complete": 0.13}
    for contract in CONTRACTS:
        cell = frame[
            (frame["contract"] == contract) & frame["method"].isin(routes)
        ]
        for index, route in enumerate(routes):
            values = cell[cell["method"] == route][column].to_numpy()
            ax.scatter(
                np.full(len(values), index + offsets[contract]),
                values,
                s=22,
                color=CONTRACT_COLORS[contract],
                marker="o" if contract == "GROUP_complete" else "D",
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
            )
            ax.plot(
                [index + offsets[contract] - 0.07, index + offsets[contract] + 0.07],
                [np.mean(values), np.mean(values)],
                color="#111827",
                lw=1.1,
            )
    ax.axhline(0, color="#4B5563", lw=0.8)
    ax.set_xticks(x, [METHOD_SHORT[m] for m in routes])
    ax.set_ylabel(r"$\Delta$MAE (t ha$^{-1}$)", fontsize=10.0)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9.8)


def _combined_delta_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Show both legal references without duplicating the route axis."""
    x = np.arange(len(TRANSFER_ROUTES))
    styles = (
        ("GROUP_complete", "delta_mae_vs_scratch", -0.24, "o", False, "GROUP vs scratch"),
        ("SPATIAL_complete", "delta_mae_vs_scratch", -0.08, "D", False, "SPATIAL vs fixed scratch"),
        ("GROUP_complete", "delta_mae_vs_supervised", 0.08, "o", True, "GROUP vs supervised"),
        ("SPATIAL_complete", "delta_mae_vs_supervised", 0.24, "D", True, "SPATIAL vs supervised"),
    )
    handles = []
    for contract, column, offset, marker, filled, label in styles:
        for index, route in enumerate(TRANSFER_ROUTES):
            values = frame[
                (frame["contract"] == contract) & (frame["method"] == route)
            ][column].to_numpy()
            color = CONTRACT_COLORS[contract]
            face = color if filled else "white"
            ax.scatter(
                np.full(len(values), index + offset), values, s=28,
                facecolor=face, edgecolor=color, marker=marker, linewidth=0.9,
                alpha=0.92, zorder=3,
            )
            ax.plot(
                [index + offset - 0.055, index + offset + 0.055],
                [np.mean(values), np.mean(values)], color="#111827", lw=1.2,
                zorder=4,
            )
        handles.append(
            Line2D(
                [0], [0], marker=marker, color="none", markerfacecolor=face,
                markeredgecolor=CONTRACT_COLORS[contract], markersize=6.5,
                markeredgewidth=0.9, label=label,
            )
        )
    ax.axhline(0, color="#4B5563", lw=0.85, zorder=1)
    ax.set_xticks(x, [METHOD_SHORT[m] for m in TRANSFER_ROUTES])
    ax.set_ylabel(r"$\Delta$MAE (t ha$^{-1}$; negative = improvement)")
    ax.set_title("(c) Run-level MAE differences", fontsize=9.5, y=1.37, pad=0)
    ax.tick_params(axis="both", labelsize=8.4)
    ax.legend(
        handles=handles, ncol=2, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.24), columnspacing=1.15, handletextpad=0.35,
        fontsize=7.7,
    )


def _figure_2_full(root: Path, output: Path) -> None:
    frame = pd.read_csv(source_paths(root)["performance"])
    scratch_status = pd.read_csv(source_paths(root)["scratch_status"])
    fixed_scratch = scratch_status.iloc[0]["classification"] == "fixed_reference_reused"
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.45))
    _metric_table(
        axes[0, 0],
        frame,
        "GROUP_complete",
        "(a) GROUP: mean (SD)",
    )
    _metric_table(
        axes[0, 1],
        frame,
        "SPATIAL_complete",
        "(b) SPATIAL: mean (SD; scratch fixed)",
        fixed_scratch=fixed_scratch,
    )
    _delta_panel(
        axes[1, 0],
        frame,
        "delta_mae_vs_scratch",
        "(c) Relative to scratch",
    )
    _delta_panel(
        axes[1, 1],
        frame,
        "delta_mae_vs_supervised",
        "(d) Beyond supervised transfer",
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CONTRACT_COLORS["GROUP_complete"],
            markeredgecolor="white",
            label="GROUP",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=CONTRACT_COLORS["SPATIAL_complete"],
            markeredgecolor="white",
            label="SPATIAL",
        ),
    ]
    axes[1, 1].legend(
        handles=legend,
        frameon=False,
        loc="upper right",
        fontsize=9.0,
    )
    fig.tight_layout(pad=0.45, h_pad=0.55, w_pad=0.55)
    save_figure(fig, output, 2)


def _support_distribution(
    ax: plt.Axes,
    points: pd.DataFrame,
    summary: pd.DataFrame,
    contract: str,
    title: str,
) -> None:
    rng = np.random.default_rng(20260709)
    for index, route in enumerate(TRANSFER_ROUTES):
        values = points[
            (points["contract"] == contract) & (points["route"] == route)
        ]["delta_P_support"].to_numpy()
        violin = ax.violinplot(
            values,
            positions=[index],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(ROUTE_COLORS[route])
            body.set_alpha(0.16)
            body.set_edgecolor("none")
        jitter = rng.uniform(-0.22, 0.22, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=7 if len(values) > 100 else 12,
            color=ROUTE_COLORS[route],
            alpha=0.20 if len(values) > 100 else 0.38,
            linewidth=0,
        )
        row = summary[
            (summary["contract"] == contract) & (summary["route"] == route)
        ].iloc[0]
        ax.errorbar(
            index,
            row["mean_delta_support_share"],
            yerr=[
                [row["mean_delta_support_share"] - row["ci_lo"]],
                [row["ci_hi"] - row["mean_delta_support_share"]],
            ],
            fmt="D",
            color="#111827",
            mfc="white",
            ms=4,
            capsize=2,
            lw=1.1,
            zorder=5,
        )
        ax.plot(
            [index - 0.18, index + 0.18],
            [row["median_delta_support_share"]] * 2,
            color="#111827",
            lw=1.2,
        )
    ax.axhline(0, color="#4B5563", lw=0.8)
    ax.set_xticks(
        range(len(TRANSFER_ROUTES)),
        [METHOD_SHORT[r] for r in TRANSFER_ROUTES],
    )
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.title.set_size(9.5)


def _signed_heatmap(
    ax: plt.Axes,
    summary: pd.DataFrame,
    contract: str,
    title: str,
    limit: float,
) -> None:
    cell = summary[summary["contract"] == contract]
    pivot = cell.pivot(
        index="route",
        columns="feature_name",
        values="mean_signed_delta",
    ).reindex(index=TRANSFER_ROUTES, columns=FEATURE_ORDER)
    image = ax.imshow(
        pivot.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(FEATURE_ORDER)),
        [FEATURE_LABELS[f] for f in FEATURE_ORDER],
        rotation=52,
        ha="right",
    )
    ax.set_yticks(
        range(len(TRANSFER_ROUTES)),
        [METHOD_SHORT[r] for r in TRANSFER_ROUTES],
    )
    ax.axvline(2.5, color="#111827", lw=1.1)
    ax.set_title(title)
    return image


def _beeswarm(
    ax: plt.Axes,
    frame: pd.DataFrame,
    contract: str,
    title: str,
    xlim: tuple[float, float],
) -> None:
    cell = frame[frame["contract"] == contract]
    rng = np.random.default_rng(20260710)
    for position, feature in enumerate(FEATURE_ORDER):
        rows = cell[cell["feature_name"] == feature].copy()
        values = rows["permutation_shap"].to_numpy(dtype=float)
        if rows["constant_within_contract"].all():
            colors = np.repeat("#9CA3AF", len(rows))
        else:
            colors = plt.cm.coolwarm(rows["feature_value_colour"].to_numpy(dtype=float))
        jitter = rng.uniform(-0.19, 0.19, len(rows))
        ax.scatter(
            values,
            position + jitter,
            s=7,
            c=colors,
            alpha=0.42,
            linewidth=0,
            rasterized=False,
        )
    ax.axvline(0, color="#4B5563", lw=0.75)
    ax.set_xlim(*xlim)
    ax.set_yticks(
        range(len(FEATURE_ORDER)),
        [FEATURE_SHORT_LABELS[f] for f in FEATURE_ORDER],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Permutation-SHAP contribution", fontsize=9.0)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.title.set_size(9.5)


def _figure_3_full(root: Path, output: Path) -> None:
    paths = source_paths(root)
    points = pd.read_csv(paths["support_points"])
    support = pd.read_csv(paths["support_summary"])
    beeswarm = pd.read_csv(paths["beeswarm"])
    limit = float(np.nanpercentile(np.abs(beeswarm["permutation_shap"]), 99.5))
    fig = plt.figure(figsize=(5.15, 4.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.12, 0.88),
        hspace=0.44,
        wspace=0.30,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
    ])
    _beeswarm(axes[0, 0], beeswarm, "GROUP_complete", "(a) GROUP reference", (-limit, limit))
    _beeswarm(axes[0, 1], beeswarm, "SPATIAL_complete", "(b) SPATIAL reference", (-limit, limit))
    _support_distribution(axes[1, 0], points, support, "GROUP_complete", "(c) GROUP redistribution")
    _support_distribution(axes[1, 1], points, support, "SPATIAL_complete", "(d) SPATIAL redistribution")
    axes[0, 1].set_yticklabels([])
    axes[1, 0].set_ylabel(r"$\Delta p_{\mathrm{support}}$", fontsize=9.0)
    axes[1, 1].set_ylabel(r"$\Delta p_{\mathrm{support}}$", fontsize=9.0)
    scalar = matplotlib.cm.ScalarMappable(cmap="coolwarm", norm=matplotlib.colors.Normalize(0, 1))
    fig.subplots_adjust(left=0.15, right=0.89, top=0.95, bottom=0.11)
    colorbar_ax = fig.add_axes([0.925, 0.625, 0.015, 0.255])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="vertical")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=8.2)
    colorbar.set_label("Feature-relative value", fontsize=8.2, labelpad=4)
    save_figure(fig, output, 3)


def _linkage_scatter(
    ax: plt.Axes,
    points: pd.DataFrame,
    summary: pd.DataFrame,
    contract: str,
    title: str,
) -> None:
    cell = points[
        (points["contract"] == contract)
        & (points["route"] == "prediction_kd")
    ]
    row = summary[
        (summary["contract"] == contract)
        & (summary["route"] == "prediction_kd")
    ].iloc[0]
    ax.scatter(
        cell["D_L1"],
        cell["delta_AE"],
        s=10 if len(cell) > 100 else 20,
        color=CONTRACT_COLORS[contract],
        alpha=0.35,
        linewidth=0,
    )
    ax.axhline(0, color="#4B5563", lw=0.8)
    annotation_x = -0.60 if contract == "GROUP_complete" else 1.06
    annotation_ha = "right" if contract == "GROUP_complete" else "left"
    ax.text(
        annotation_x,
        0.98,
        (
            rf"$\rho={row['rho']:.2f}$"
            "\n"
            rf"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
        ),
        transform=ax.transAxes,
        ha=annotation_ha,
        va="top",
        rotation=90,
        rotation_mode="anchor",
        fontsize=7.5,
        clip_on=False,
    )
    ax.set_xlabel(r"Redistribution $d_{\phi}$", fontsize=9.0)
    if contract == "GROUP_complete":
        ax.set_ylabel("")
    else:
        ax.set_ylabel(r"Error change $\Delta e$ (t ha$^{-1}$)", fontsize=9.0)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8.6)
    ax.title.set_size(9.5)


def _figure_4_full(root: Path, output: Path) -> None:
    paths = source_paths(root)
    points = pd.read_csv(paths["linkage_points"])
    summary = pd.read_csv(paths["linkage_summary"])
    stability = pd.read_csv(paths["linkage_stability"])
    fig, axes = plt.subplots(2, 2, figsize=(5.15, 4.15))
    _linkage_scatter(
        axes[0, 0],
        points,
        summary,
        "GROUP_complete",
        "(a)",
    )
    _linkage_scatter(
        axes[0, 1],
        points,
        summary,
        "SPATIAL_complete",
        "(b)",
    )
    plot = summary.sort_values(["contract", "route"]).reset_index(drop=True)
    y = np.arange(len(plot))
    for position, row in enumerate(plot.itertuples(index=False)):
        color = CONTRACT_COLORS[row.contract]
        axes[1, 0].errorbar(
            row.rho,
            position,
            xerr=[[row.rho - row.ci_lo], [row.ci_hi - row.rho]],
            fmt="o",
            color=color,
            ecolor=color,
            ms=4,
            capsize=2,
            lw=1.1,
        )
        if row.p_holm < 0.05:
            axes[1, 0].text(row.ci_hi + 0.025, position, "*", va="center")
    axes[1, 0].axvline(0, color="#4B5563", lw=0.8)
    axes[1, 0].set_yticks(
        y,
        [
            f"{'G' if r.contract == 'GROUP_complete' else 'S'} / {METHOD_SHORT[r.route]}"
            for r in plot.itertuples(index=False)
        ],
    )
    axes[1, 0].set_xlabel(r"Cluster-level Spearman $\rho$", fontsize=9.0)
    axes[1, 0].set_title("(c)", fontsize=9.5)
    axes[1, 0].tick_params(axis="both", labelsize=8.5)
    loo = stability[stability["stability_type"] == "leave_one_seed_out"]
    rows = [
        (contract, route)
        for contract in CONTRACTS
        for route in TRANSFER_ROUTES
    ]
    matrix = np.asarray(
        [
            [
                loo[
                    (loo["contract"] == contract)
                    & (loo["route"] == route)
                    & (loo["left_out_seed"] == seed)
                ]["rho"].iloc[0]
                for seed in (101, 202, 303)
            ]
            for contract, route in rows
        ]
    )
    axes[1, 1].imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(
                column,
                row,
                f"{matrix[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
            )
    axes[1, 1].set_xticks(
        range(3),
        [r"$-\mathrm{S1}$", r"$-\mathrm{S2}$", r"$-\mathrm{S3}$"],
    )
    axes[1, 1].set_yticks(
        range(len(rows)),
        [f"{'G' if c == 'GROUP_complete' else 'S'} / {METHOD_SHORT[r]}" for c, r in rows],
    )
    axes[1, 1].set_title("(d)", fontsize=9.5)
    axes[1, 1].tick_params(axis="both", labelsize=8.5)
    fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.35)
    save_figure(fig, output, 4)


def _figure_5_full(root: Path, output: Path) -> None:
    paths = source_paths(root)
    attribution = pd.read_csv(paths["attribution"])
    sensitivity = pd.read_csv(paths["sensitivity"])
    sensitivity = sensitivity[sensitivity["perturbation_group"].isin(GROUPS)]
    agreement = pd.read_csv(paths["agreement"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    selected = ("supervised", "prediction_kd")
    for ax, frame, value, title in (
        (
            axes[0, 0],
            attribution,
            "attribution_share",
            "(a) SHAP share",
        ),
        (
            axes[0, 1],
            sensitivity.rename(columns={"perturbation_group": "group"}),
            "standardized_sensitivity",
            "(b) Stress response",
        ),
    ):
        positions = np.arange(len(GROUPS))
        width = 0.18
        for method_index, method in enumerate(selected):
            for contract_index, contract in enumerate(CONTRACTS):
                cell = frame[
                    (frame["method"] == method)
                    & (frame["contract"] == contract)
                ]
                means = cell.groupby("group", observed=True)[value].mean()
                offset = (
                    (method_index * len(CONTRACTS) + contract_index) - 1.5
                ) * width
                ax.bar(
                    positions + offset,
                    [means.get(group, np.nan) for group in GROUPS],
                    width=width,
                    color=ROUTE_COLORS[method],
                    alpha=0.45 if contract_index == 0 else 0.9,
                    hatch="//" if contract_index == 0 else None,
                    edgecolor="#374151",
                    linewidth=0.4,
                )
        ax.set_xticks(positions, [GROUP_LABELS[g] for g in GROUPS])
        ax.set_title(title)
    axes[0, 0].set_ylabel("Attribution share")
    axes[0, 1].set_ylabel(
            "Prediction change / standardised\nperturbation distance",
        fontsize=8.5,
        labelpad=0,
    )
    order = (
        agreement.assign(
            contract_label=agreement["contract"].str.replace("_complete", "")
        )
        .sort_values(["contract_label", "method", "seed"])
        .reset_index(drop=True)
    )
    y = np.arange(len(order))
    axes[1, 0].scatter(
        order["spearman_rho"],
        y,
        c=[CONTRACT_COLORS[c] for c in order["contract"]],
        s=18,
        marker="o",
    )
    axes[1, 0].axvline(0, color="#4B5563", lw=0.8)
    axes[1, 0].set_yticks(
        y[::3],
        [
            f"{row.contract_label} / {METHOD_SHORT[row.method]}"
            for row in order.iloc[::3].itertuples()
        ],
    )
    axes[1, 0].set_xlabel("Four-group Spearman agreement")
    axes[1, 0].set_title("(c) Cell-wise rank agreement")
    match = (
        agreement.assign(
            contract_label=agreement["contract"].str.replace("_complete", "")
        )
        .groupby(["contract_label", "method"], observed=True)[
            "top_group_match"
        ]
        .mean()
        .reset_index()
    )
    for index, contract in enumerate(("GROUP", "SPATIAL")):
        cell = match[match["contract_label"] == contract].set_index("method")
        values = [
            cell.loc[method, "top_group_match"]
            for method in ATTRIBUTION_METHODS
        ]
        axes[1, 1].scatter(
            np.arange(len(ATTRIBUTION_METHODS)) + (-0.10 if contract == "GROUP" else 0.10),
            values,
            marker="o" if contract == "GROUP" else "D",
            color=CONTRACT_COLORS[f"{contract}_complete"],
            # Contract is encoded by the shared legend outside the axes.
            s=22,
        )
    axes[1, 1].axhline(0.25, color="#6B7280", ls="--", lw=0.9, label="Chance")
    axes[1, 1].set_xticks(
        range(len(ATTRIBUTION_METHODS)),
        [METHOD_SHORT[m] for m in ATTRIBUTION_METHODS],
        rotation=25,
    )
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylabel("Top-group match rate")
    axes[1, 1].set_title("(d) Top-group match")
    axes[1, 1].legend(frameon=False, ncol=1, loc="upper left")
    legend = [
        patches.Patch(
            facecolor=ROUTE_COLORS["supervised"],
            label="Supervised transfer",
        ),
        patches.Patch(
            facecolor=ROUTE_COLORS["prediction_kd"],
            label="Prediction KD",
        ),
        patches.Patch(
            facecolor="white",
            edgecolor="#374151",
            hatch="//",
            label="GROUP",
        ),
        patches.Patch(
            facecolor="#9CA3AF",
            edgecolor="#374151",
            label="SPATIAL",
        ),
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output, 5)


def figure_2(root: Path, output: Path) -> None:
    """Run-level effects only; exact performance values live in the LaTeX table."""
    frame = pd.read_csv(source_paths(root)["performance"])
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.35))
    _delta_panel(
        axes[0], frame, "delta_mae_vs_scratch", "(a) Relative to target scratch"
    )
    _delta_panel(
        axes[1], frame, "delta_mae_vs_supervised", "(b) Beyond supervised transfer"
    )
    axes[1].set_ylabel("")
    handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=CONTRACT_COLORS["GROUP_complete"],
            markeredgecolor="white", label="GROUP",
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=CONTRACT_COLORS["SPATIAL_complete"],
            markeredgecolor="white", label="SPATIAL",
        ),
    ]
    fig.legend(
        handles=handles, frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), fontsize=8.6,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.79, wspace=0.24)
    save_figure(fig, output, 2)


def figure_3(root: Path, output: Path) -> None:
    """Matched observation-support redistribution on sample clusters."""
    paths = source_paths(root)
    points = pd.read_csv(paths["support_points"])
    support = pd.read_csv(paths["support_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.55))
    _support_distribution(axes[0], points, support, "GROUP_complete", "(a) GROUP")
    _support_distribution(axes[1], points, support, "SPATIAL_complete", "(b) SPATIAL")
    axes[0].set_ylabel(r"$\Delta p_{\mathrm{support}}$", fontsize=9.2)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.84, wspace=0.22)
    save_figure(fig, output, 3)


def _draw_linkage_forest(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot = summary.sort_values(["contract", "route"]).reset_index(drop=True)
    y = np.arange(len(plot))
    for position, row in enumerate(plot.itertuples(index=False)):
        color = CONTRACT_COLORS[row.contract]
        ax.errorbar(
            row.rho,
            position,
            xerr=[[row.rho - row.ci_lo], [row.ci_hi - row.rho]],
            fmt="o",
            color=color,
            ecolor=color,
            ms=3.8,
            capsize=2,
            lw=1.0,
        )
        if row.p_holm < 0.05:
            ax.text(row.ci_hi + 0.025, position, "*", va="center", fontsize=9.0)
    ax.axvline(0, color="#4B5563", lw=0.8)
    ax.set_yticks(
        y,
        [
            f"{'G' if row.contract == 'GROUP_complete' else 'S'} / {METHOD_SHORT[row.route]}"
            for row in plot.itertuples(index=False)
        ],
    )
    ax.set_xlabel(r"Cluster-level Spearman $\rho$", fontsize=8.8)
    ax.set_title("")
    ax.text(0.02, 1.02, "(c)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.4)
    ax.tick_params(axis="both", labelsize=8.8)


def _draw_linkage_loo(ax: plt.Axes, stability: pd.DataFrame) -> None:
    loo = stability[stability["stability_type"] == "leave_one_seed_out"]
    rows = [(contract, route) for contract in CONTRACTS for route in TRANSFER_ROUTES]
    matrix = np.asarray(
        [
            [
                loo[
                    (loo["contract"] == contract)
                    & (loo["route"] == route)
                    & (loo["left_out_seed"] == seed)
                ]["rho"].iloc[0]
                for seed in (101, 202, 303)
            ]
            for contract, route in rows
        ]
    )
    ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column, row, f"{matrix[row, column]:+.2f}",
                ha="center", va="center", fontsize=8.8,
            )
    ax.set_xticks(range(3), [r"$-\mathrm{S1}$", r"$-\mathrm{S2}$", r"$-\mathrm{S3}$"])
    ax.set_yticks(
        range(len(rows)),
        [f"{'G' if c == 'GROUP_complete' else 'S'} / {METHOD_SHORT[r]}" for c, r in rows],
    )
    ax.set_title("")
    ax.text(0.02, 1.02, "(d)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.4)
    ax.tick_params(axis="both", labelsize=8.8)


def figure_4(root: Path, output: Path) -> None:
    """Compact, wide layout retaining point clouds, forest, and LOO robustness."""
    paths = source_paths(root)
    points = pd.read_csv(paths["linkage_points"])
    summary = pd.read_csv(paths["linkage_summary"])
    stability = pd.read_csv(paths["linkage_stability"])
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.1))
    _linkage_scatter(axes[0, 0], points, summary, "GROUP_complete", "(a)")
    _linkage_scatter(axes[0, 1], points, summary, "SPATIAL_complete", "(b)")
    _draw_linkage_forest(axes[1, 0], summary)
    _draw_linkage_loo(axes[1, 1], stability)
    fig.subplots_adjust(left=0.22, right=0.86, bottom=0.12, top=0.86, hspace=0.42, wspace=0.45)
    save_figure(fig, output, 4)


def _agreement_order(agreement: pd.DataFrame) -> pd.DataFrame:
    return (
        agreement.assign(contract_label=agreement["contract"].str.replace("_complete", ""))
        .sort_values(["contract_label", "method", "seed"])
        .reset_index(drop=True)
    )


def figure_5(root: Path, output: Path) -> None:
    """Agreement summaries; representative profiles are retained in the supplement."""
    agreement = pd.read_csv(source_paths(root)["agreement"])
    order = _agreement_order(agreement)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65), gridspec_kw={"width_ratios": (1.15, 1.0)})
    y = np.arange(len(order))
    axes[0].scatter(
        order["spearman_rho"], y,
        c=[CONTRACT_COLORS[c] for c in order["contract"]], s=17,
    )
    axes[0].axvline(0, color="#4B5563", lw=0.8)
    axes[0].set_yticks(
        y[::3],
        [f"{'G' if row.contract_label == 'GROUP' else 'S'} / {METHOD_SHORT[row.method]}" for row in order.iloc[::3].itertuples()],
    )
    axes[0].set_xlabel(r"Four-group Spearman $\rho$", fontsize=9.0)
    axes[0].set_title("(a)", fontsize=9.6)
    axes[0].tick_params(axis="both", labelsize=9.0)
    match = (
        agreement.assign(contract_label=agreement["contract"].str.replace("_complete", ""))
        .groupby(["contract_label", "method"], observed=True)["top_group_match"]
        .mean().reset_index()
    )
    for contract in ("GROUP", "SPATIAL"):
        cell = match[match["contract_label"] == contract].set_index("method")
        values = [cell.loc[method, "top_group_match"] for method in ATTRIBUTION_METHODS]
        axes[1].scatter(
            np.arange(len(ATTRIBUTION_METHODS)) + (-0.10 if contract == "GROUP" else 0.10),
            values,
            marker="o" if contract == "GROUP" else "D",
            color=CONTRACT_COLORS[f"{contract}_complete"],
            # Contract is encoded by the shared legend outside the axes.
            s=22,
        )
    axes[1].axhline(0.25, color="#6B7280", ls="--", lw=0.9)
    axes[1].set_xticks(
        range(len(ATTRIBUTION_METHODS)),
        [METHOD_SHORT[m] for m in ATTRIBUTION_METHODS], rotation=18,
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Top-group match rate", fontsize=9.0)
    axes[1].set_title("(b)", fontsize=9.6)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CONTRACT_COLORS["GROUP_complete"], markeredgecolor="white", label="GROUP"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=CONTRACT_COLORS["SPATIAL_complete"], markeredgecolor="white", label="SPATIAL"),
        Line2D([0], [0], color="#6B7280", ls="--", label="Chance (0.25)"),
    ]
    fig.legend(handles=legend, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=8.3)
    axes[1].tick_params(axis="both", labelsize=9.0)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.22, top=0.78, wspace=0.35)
    save_figure(fig, output, 5)


def _save_named(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)


def supplement_supervised_landscapes(root: Path, output: Path) -> None:
    beeswarm = pd.read_csv(source_paths(root)["beeswarm"])
    limit = float(np.nanpercentile(np.abs(beeswarm["permutation_shap"]), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65), sharey=True)
    _beeswarm(axes[0], beeswarm, "GROUP_complete", "(a) GROUP reference", (-limit, limit))
    _beeswarm(axes[1], beeswarm, "SPATIAL_complete", "(b) SPATIAL reference", (-limit, limit))
    axes[1].set_yticklabels([])
    scalar = matplotlib.cm.ScalarMappable(
        cmap="coolwarm", norm=matplotlib.colors.Normalize(0, 1)
    )
    fig.subplots_adjust(left=0.13, right=0.89, bottom=0.20, top=0.88, wspace=0.20)
    colorbar_ax = fig.add_axes([0.92, 0.29, 0.014, 0.45])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=8.8)
    colorbar.set_label("Feature-relative value", fontsize=8.8)
    _save_named(fig, output, "supplement_figure_supervised_landscapes")


def supplement_representative_stress(root: Path, output: Path) -> None:
    paths = source_paths(root)
    attribution = pd.read_csv(paths["attribution"])
    sensitivity = pd.read_csv(paths["sensitivity"])
    sensitivity = sensitivity[sensitivity["perturbation_group"].isin(GROUPS)]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.45))
    selected = ("supervised", "prediction_kd")
    for ax, frame, value, title in (
        (axes[0], attribution, "attribution_share", "(a) SHAP share"),
        (axes[1], sensitivity.rename(columns={"perturbation_group": "group"}), "standardized_sensitivity", "(b) Stress response"),
    ):
        positions = np.arange(len(GROUPS))
        width = 0.18
        for method_index, method in enumerate(selected):
            for contract_index, contract in enumerate(CONTRACTS):
                cell = frame[(frame["method"] == method) & (frame["contract"] == contract)]
                means = cell.groupby("group", observed=True)[value].mean()
                offset = ((method_index * len(CONTRACTS) + contract_index) - 1.5) * width
                ax.bar(
                    positions + offset,
                    [means.get(group, np.nan) for group in GROUPS],
                    width=width,
                    color=ROUTE_COLORS[method],
                    alpha=0.45 if contract_index == 0 else 0.9,
                    hatch="//" if contract_index == 0 else None,
                    edgecolor="#374151",
                    linewidth=0.4,
                )
        ax.set_xticks(positions, [GROUP_LABELS[g] for g in GROUPS])
        ax.set_title(title, fontsize=9.6)
        ax.tick_params(axis="both", labelsize=8.4)
    axes[0].set_ylabel("Attribution share", fontsize=9.0)
    axes[1].set_ylabel("Prediction change / standardised\nperturbation distance", fontsize=8.7)
    legend = [
        patches.Patch(facecolor=ROUTE_COLORS["supervised"], label="Supervised transfer"),
        patches.Patch(facecolor=ROUTE_COLORS["prediction_kd"], label="Prediction KD"),
        patches.Patch(facecolor="white", edgecolor="#374151", hatch="//", label="GROUP"),
        patches.Patch(facecolor="#9CA3AF", edgecolor="#374151", label="SPATIAL"),
    ]
    fig.legend(handles=legend, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8.2)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.76, wspace=0.28)
    _save_named(fig, output, "supplement_figure_representative_stress")


def figure_6(root: Path, output: Path) -> None:
    ood = pd.read_csv(source_paths(root)["ood"])
    summary = (
        ood.groupby("perturbation_group", observed=True)
        .agg(
            baseline_nn=("mean_baseline_nn_distance", "mean"),
            perturbed_nn=("mean_perturbed_nn_distance", "mean"),
            baseline_mahal=("mean_baseline_mahalanobis", "mean"),
            perturbed_mahal=("mean_perturbed_mahalanobis", "mean"),
        )
        .reindex(GROUPS)
    )
    nn_ratio = summary["perturbed_nn"] / summary["baseline_nn"]
    mahal_ratio = summary["perturbed_mahal"] / summary["baseline_mahal"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"wspace": 0.42})
    x = np.arange(len(GROUPS))
    axes[0].scatter(x - 0.09, nn_ratio, marker="o", color="#0077BB", label="Nearest neighbour", zorder=3)
    axes[0].scatter(x + 0.09, mahal_ratio, marker="D", color="#EE7733", label="9D Ledoit--Wolf", zorder=3)
    axes[0].axhline(1, color="#4B5563", lw=0.8)
    axes[0].set_xticks(x, [GROUP_LABELS[g] for g in GROUPS])
    axes[0].set_ylabel("Perturbed / baseline distance")
    axes[0].set_title("(a) OOD movement")
    axes[0].legend(frameon=False, loc="upper left")
    quality_names = ("IG", "Taylor")
    quality_values = (0.9893670886075949, 0.15555555555555556)
    bars = axes[1].bar(
        quality_names,
        quality_values,
        color=("#0077BB", "#EE7733"),
        width=0.55,
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Pass fraction")
    axes[1].set_title("(b) Explanation validity coverage")
    for bar, value in zip(bars, quality_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.82, wspace=0.42)
    save_figure(fig, output, 6)


def supplement_signed_feature_heatmap(root: Path, output: Path) -> None:
    summary = pd.read_csv(source_paths(root)["feature_summary"])
    limit = float(np.nanpercentile(np.abs(summary["mean_signed_delta"]), 98.5))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    image = _signed_heatmap(axes[0], summary, "GROUP_complete", "(a) GROUP signed change", limit)
    _signed_heatmap(axes[1], summary, "SPATIAL_complete", "(b) SPATIAL signed change", limit)
    axes[0].set_ylabel("Route")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.0)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.39, top=0.88, wspace=0.25)
    colorbar_ax = fig.add_axes([0.31, 0.10, 0.38, 0.04])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Mean signed Permutation-SHAP change from supervised transfer")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"supplement_figure_s2.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)


def supplement_explanation_agreement(root: Path, output: Path) -> None:
    cells = pd.read_csv(source_paths(root)["agreement_cells"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, contract in zip(axes, CONTRACTS):
        cell = cells[cells["contract"].eq(contract)]
        for method_index, method in enumerate(ATTRIBUTION_METHODS):
            x0 = method_index
            for comparison, marker, edge, offset in (("SHAP--IG", "o", None, -0.10), ("SHAP--Taylor", "o", "#6B7280", 0.10)):
                rows = cell[(cell["method"] == method) & (cell["comparison"] == comparison)]
                for seed_index, row in enumerate(rows.sort_values("seed").itertuples(index=False)):
                    x = x0 + offset + (seed_index - 1) * 0.035
                    if not bool(row.display_estimate):
                        ax.text(x, -0.95, "n<3", rotation=90, ha="center", va="bottom", fontsize=8.0, color="#6B7280")
                        continue
                    color = ROUTE_COLORS[method] if comparison == "SHAP--IG" else "white"
                    ax.errorbar(x, row.median_rank_rho, yerr=[[row.median_rank_rho - row.q1_rank_rho], [row.q3_rank_rho - row.median_rank_rho]], fmt=marker, color=ROUTE_COLORS[method], mfc=color, mec=edge or ROUTE_COLORS[method], ms=4.2, lw=0.9, capsize=1.5)
        ax.axhline(0, color="#4B5563", lw=0.75)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(range(len(ATTRIBUTION_METHODS)), [METHOD_SHORT[m] for m in ATTRIBUTION_METHODS], rotation=24, ha="right")
        ax.set_title(contract.replace("_complete", ""))
    axes[0].set_ylabel("Sample 10-feature Spearman\nrank agreement", fontsize=8.0)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#374151", linestyle="None", label="SHAP--IG, completeness-qualified"), Line2D([0], [0], marker="o", color="#6B7280", markerfacecolor="white", linestyle="None", label="SHAP--Taylor, 5% local-fidelity subset")], loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.23, top=0.76, wspace=0.10)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"supplement_figure_s4.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)


def integrated_behavioural_candidate(root: Path, output: Path) -> None:
    paths = source_paths(root)
    support_points = pd.read_csv(paths["support_points"])
    support_summary = pd.read_csv(paths["support_summary"])
    linkage_points = pd.read_csv(paths["linkage_points"])
    linkage_summary = pd.read_csv(paths["linkage_summary"])
    stability = pd.read_csv(paths["linkage_stability"])
    fig = plt.figure(figsize=(5.2, 4.75))
    grid = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.46, width_ratios=(1.0, 1.0, 1.12))
    axes = np.asarray(
        [[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)]
    )
    _support_distribution(axes[0, 0], support_points, support_summary, "GROUP_complete", "(a) G redistribution")
    _support_distribution(axes[0, 1], support_points, support_summary, "SPATIAL_complete", "(b) S redistribution")
    _draw_linkage_forest(axes[0, 2], linkage_summary)
    _linkage_scatter(axes[1, 0], linkage_points, linkage_summary, "GROUP_complete", "(d) G / Pred.")
    _linkage_scatter(axes[1, 1], linkage_points, linkage_summary, "SPATIAL_complete", "(e) S / Pred.")
    _draw_linkage_loo(axes[1, 2], stability)
    axes[0, 0].set_ylabel(r"$\Delta p_{\mathrm{support}}$", fontsize=8.2)
    axes[0, 1].set_ylabel(r"$\Delta p_{\mathrm{support}}$", fontsize=8.2)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.94)
    _save_named(fig, output, "candidate_integrated_behavioural")


def write_captions(output: Path) -> None:
    text = r"""# Final publication v4.14 figure captions

**Figure 1.** Study framing. Panel (a) is a conceptual motivation: deployment-oriented partitions can change observed model selection and it is not a pooled empirical result. Panel (b) compares target scratch with source-route initialisation followed by supervised target fine-tuning; both use the same target labels, contract, and matched evaluation. Panel (c) separates predictive performance, attribution redistribution, and finite OOD stress response.

**Figure 2.** Run-level transfer effects. Panel (a) shows MAE differences from target scratch; the SPATIAL comparison uses one fixed local scratch reference shared across transferred runs. Panel (b) shows differences from run-matched supervised transfer. Negative \(\Delta\)MAE denotes lower error and therefore improvement. Exact MAE, RMSE, and \(R^2\) summaries are reported in the manuscript table and Supplement.

**Figure 3.** Matched observation-support redistribution relative to supervised transfer. Panels (a-b) show sample-cluster distributions for GROUP and SPATIAL. Diamonds and intervals report cluster-bootstrap means and 95% bootstrap intervals; horizontal bars report medians. Pred., Comb., Repr., and Miss. denote prediction KD, combined KD, representation KD, and missing-aware training. The supervised-reference beeswarms are reported in the Supplement.

**Figure 4.** Relation between attribution redistribution and matched error change. Panels (a-b) show every sample cluster for prediction KD without a fitted line. Panel (c) reports eight route-by-contract Spearman associations with 95% bootstrap intervals; * denotes Holm-adjusted \(p<0.05\). Panel (d) reports leave-one-run-out estimates. G/S denote GROUP/SPATIAL; Pred., Comb., Repr., and Miss. denote prediction KD, combined KD, representation KD, and missing-aware training. \(-\mathrm{S1}\), \(-\mathrm{S2}\), and \(-\mathrm{S3}\) denote omission of run S1, S2, or S3.

**Figure 5.** Agreement between local attribution allocation and finite stress response. Panel (a) reports cell-wise four-group Spearman agreement for all route--contract--run cells. Panel (b) reports top-group match by route and contract; the dashed line is the 25% chance rate for four groups. Representative attribution and stress profiles are reported in the Supplement.

**Figure 6.** Validity boundaries. Panel (a) shows perturbed-to-baseline nearest-neighbour and stable nine-dimensional Ledoit--Wolf distance ratios for the four discrete feature groups. Panel (b) shows the fraction of explanations passing their distinct 5% numerical gates: Integrated Gradients completeness and local pathwise-Taylor reconstruction. Taylor remains a conditional diagnostic; qualified SHAP--IG and Taylor agreement are reported in the supplement.
"""
    text = text.replace(
        "Exact MAE, RMSE, and",
        "Exact MAE summaries are reported in the manuscript table; RMSE,",
    ).replace(
        " summaries are reported in the manuscript table and Supplement.",
        ", and run-level metrics are reported in the Supplement.",
    )
    (output / "captions.md").write_text(text, encoding="utf-8")


def write_manifest(root: Path, output: Path) -> None:
    paths = source_paths(root)
    figure_sources = {
        1: [],
        2: ["performance"],
        3: ["support_points", "support_summary"],
        4: ["linkage_points", "linkage_summary", "linkage_stability"],
        5: ["agreement"],
        6: ["ood", "manifest"],
        "supplement_s2": ["feature_summary"],
        "supplement_s4": ["agreement_cells"],
        "supplement_supervised_landscapes": ["beeswarm"],
        "supplement_representative_stress": ["attribution", "sensitivity"],
    }
    records = []
    for number, keys in figure_sources.items():
        for key in keys:
            path = paths[key]
            records.append(
                {
                    "figure": (
                        f"figure_{number:02d}"
                        if isinstance(number, int)
                        else str(number)
                    ),
                    "source_key": key,
                    "source_file": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "rows": (
                        len(pd.read_csv(path))
                        if path.suffix == ".csv"
                        else 1
                    ),
                    "statistical_unit": (
                        "sample_id cluster"
                        if key in {
                            "beeswarm",
                            "support_points",
                            "support_summary",
                            "feature_summary",
                            "linkage_points",
                            "linkage_summary",
                            "linkage_stability",
                        }
                        else "route-contract-seed cell"
                    ),
                    "minimum_font_pt": {
                        1: 8.0,
                        2: 9.0,
                        3: 8.2,
                        4: 8.3,
                        5: 8.5,
                        6: 8.0,
                    }.get(number, 8.0),
                }
            )
    with (output / "provenance_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_panel_integrity_manifest(root: Path, output: Path) -> None:
    source_names = (
        "source_performance_landscape_seed.csv",
        "source_supervised_shap_beeswarm.csv",
        "source_support_delta_samples.csv",
        "source_support_delta_summary.csv",
        "source_linkage_pointcloud.csv",
        "source_linkage_summary.csv",
        "source_linkage_seed_stability.csv",
        "source_perturbation_agreement_disjoint.csv",
        "source_attribution_group_shares.csv",
        "source_perturbation_normalized_sensitivity.csv",
    )
    v413 = root / "figures/final_publication_v4_13"
    v414 = root / "figures/final_publication_v4_16"
    parity = {
        name: {
            "v4_13": sha256_file(v413 / name),
            "v4_14": sha256_file(v414 / name),
            "equal": sha256_file(v413 / name) == sha256_file(v414 / name),
        }
        for name in source_names
    }
    if not all(item["equal"] for item in parity.values()):
        raise ValueError("V412_SOURCE_PARITY_FAILURE")
    manifest = {
        "selected_layout": "claim-first-v4.14",
        "source_parity": "byte-identical-to-v4.13",
        "figure_1": "byte-identical-to-v4.13",
        "figure_02": {
            "panels": ["run-level-vs-scratch", "run-level-vs-supervised"],
            "minimum_font_pt": 8.6,
        },
        "figure_03": {
            "panels": ["group-support-redistribution", "spatial-support-redistribution"],
            "minimum_font_pt": 8.8,
        },
        "figure_04": {
            "panels": ["group-point-cloud", "spatial-point-cloud", "eight-test-forest", "leave-one-run-out"],
            "minimum_font_pt": 8.8,
            "final_embedded_effective_minimum_pt": 8.0,
        },
        "figure_05": {
            "panels": ["cell-wise-rank-agreement", "top-group-match"],
            "minimum_font_pt": 8.8,
            "final_embedded_effective_minimum_pt": 8.0,
        },
        "moved_to_supplement": ["supervised-reference-beeswarms", "representative-shap-stress-profiles"],
        "source_hashes": parity,
    }
    (output / "panel_integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(root: Path, output: Path) -> None:
    validate_inputs(root)
    configure_style()
    figure_2(root, output)
    figure_3(root, output)
    figure_4(root, output)
    figure_5(root, output)
    supplement_supervised_landscapes(root, output)
    supplement_representative_stress(root, output)
    write_captions(output)
    write_manifest(root, output)
    write_panel_integrity_manifest(root, output)


def render_candidate(root: Path, output: Path, candidate: str) -> None:
    if candidate not in CANDIDATES:
        raise ValueError(f"UNKNOWN_CANDIDATE:{candidate}")
    validate_inputs(root)
    configure_style()
    figure_2(root, output)
    if candidate == "minimal":
        _figure_3_full(root, output)
        _figure_4_full(root, output)
        _figure_5_full(root, output)
    elif candidate == "claim-first":
        figure_3(root, output)
        figure_4(root, output)
        figure_5(root, output)
    else:
        integrated_behavioural_candidate(root, output)
        figure_5(root, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--candidate", choices=CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.validate_only:
        print(json.dumps(validate_inputs(root), indent=2, sort_keys=True))
        return
    if args.candidate:
        render_candidate(root, args.output_dir.resolve(), args.candidate)
        print(json.dumps({"candidate": args.candidate, **validate_inputs(root)}, indent=2, sort_keys=True))
        return
    build(root, args.output_dir.resolve())
    print(json.dumps(validate_inputs(root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
