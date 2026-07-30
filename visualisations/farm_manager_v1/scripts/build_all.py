#!/usr/bin/env python3
"""Build the versioned farm-manager visualisation suite from frozen sources."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from matplotlib import animation, patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "frozen_sources"
EXPORT = ROOT / "exports"
QA = ROOT / "qa"
STYLE = yaml.safe_load((ROOT / "config" / "visual_style.yaml").read_text())
VERSION = (ROOT / "VERSION").read_text().strip()

P = STYLE["palette"]
COLORS = {
    "ink": P["ink"],
    "green": P["green"],
    "sage": P["sage"],
    "wheat": P["wheat"],
    "sky": P["sky"],
    "rust": P["rust"],
    "plum": P["plum"],
    "grey": P["grey"],
    "pale": P["pale"],
    "white": P["white"],
}
W, H = STYLE["canvas"]["width_inches"], STYLE["canvas"]["height_inches"]
DPI = STYLE["canvas"]["dpi"]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "axes.edgecolor": COLORS["grey"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "savefig.facecolor": COLORS["white"],
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_dirs() -> None:
    for directory in (
        EXPORT / "png",
        EXPORT / "svg",
        EXPORT / "pdf",
        EXPORT / "animation",
        EXPORT / "interactive",
        QA,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def clean_outputs() -> None:
    for directory in (EXPORT, QA):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                path.unlink()


def title(fig: plt.Figure, headline: str, deck: str, evidence: str) -> None:
    fig.text(0.045, 0.945, headline, fontsize=25, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(0.045, 0.895, deck, fontsize=13, color="#4F5C59", va="top")
    fig.text(
        0.955,
        0.943,
        evidence.upper(),
        fontsize=9,
        color=COLORS["white"],
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": COLORS["green"], "ec": "none"},
    )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.045, 0.025, text, fontsize=8.5, color="#697571", va="bottom")
    fig.text(0.955, 0.025, f"Farm decision evidence · v{VERSION}", fontsize=8.5, color="#697571", ha="right")


def save_figure(fig: plt.Figure, slug: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(EXPORT / ext / f"{slug}.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


def mean_sd(frame: pd.DataFrame, contract: str, method: str, metric: str = "mae") -> tuple[float, float]:
    values = frame.loc[(frame["contract"] == contract) & (frame["method"] == method), metric]
    if values.empty:
        raise ValueError(f"Missing {contract}/{method}/{metric}")
    return float(values.mean()), float(values.std(ddof=1))


def v01_decision_chain() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        fig,
        "Local data, tested before farm decisions",
        "A traceable path from observations to action, with uncertainty kept visible.",
        "Product logic",
    )

    columns = [
        (
            0.055,
            "Farm data\nalready available",
            [
                ("Paddock boundaries", COLORS["green"]),
                ("Weather and season", COLORS["sky"]),
                ("Soil and terrain", COLORS["wheat"]),
                ("Management and\noutcomes", COLORS["plum"]),
            ],
        ),
        (
            0.38,
            "Checks before\na recommendation",
            [
                ("Predictive error", COLORS["green"]),
                ("Data coverage", COLORS["sky"]),
                ("Model reliance", COLORS["wheat"]),
                ("Scenario sensitivity", COLORS["plum"]),
            ],
        ),
        (
            0.705,
            "Decision outputs\nfor the manager",
            [
                ("Paddock risk map", COLORS["green"]),
                ("Confidence and data gaps", COLORS["sky"]),
                ("Targeted sampling plan", COLORS["wheat"]),
                ("Contract-aligned\nmodel choice", COLORS["plum"]),
            ],
        ),
    ]
    for col, heading, items in columns:
        ax.text(col, 0.81, heading, fontsize=14, fontweight="bold", va="center", linespacing=1.05)
        for idx, (label, color) in enumerate(items):
            y = 0.66 - idx * 0.13
            box = patches.FancyBboxPatch(
                (col, y - 0.042),
                0.24,
                0.082,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                fc=mpl.colors.to_rgba(color, 0.12),
                ec=color,
                lw=1.5,
            )
            ax.add_patch(box)
            ax.add_patch(patches.Rectangle((col, y - 0.042), 0.012, 0.082, color=color, lw=0))
            ax.text(col + 0.027, y, label, fontsize=11.5, va="center", linespacing=0.95)

    for x0, x1 in ((0.297, 0.372), (0.622, 0.697)):
        for y in (0.66, 0.53, 0.40, 0.27):
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops={"arrowstyle": "-|>", "color": COLORS["grey"], "lw": 1.8},
            )
    ax.text(
        0.5,
        0.11,
        "Start with a small local baseline. Expand only when additional data changes a real decision.",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["green"],
    )
    footer(fig, "Conceptual workflow. No farm-specific prediction is shown.")
    save_figure(fig, "v01_data_to_decision")


def v02_long_term_context() -> None:
    df = read_csv("sa_deep_time_dataset.csv").sort_values("year").copy()
    df["year"] = df["year"].astype(int)
    df["yield_roll"] = df["observed_yield_t_ha"].rolling(11, center=True, min_periods=5).median()
    df["rain_roll"] = df["growing_season_rainfall_mm"].rolling(11, center=True, min_periods=5).median()
    rain_lo, rain_hi = df["growing_season_rainfall_mm"].quantile([0.2, 0.8])

    fig, axes = plt.subplots(2, 1, figsize=(W, H), gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.22})
    title(
        fig,
        "A season is never just an average year",
        "South Australian wheat context shows why local season and paddock information matter.",
        "Exploratory public context",
    )
    ax = axes[0]
    ax.plot(df["year"], df["observed_yield_t_ha"], color=mpl.colors.to_rgba(COLORS["green"], 0.38), lw=1.2)
    ax.plot(df["year"], df["yield_roll"], color=COLORS["green"], lw=3, label="11-year median")
    ax.fill_between(df["year"], 0, df["observed_yield_t_ha"], color=mpl.colors.to_rgba(COLORS["sage"], 0.08))
    ax.set_ylabel("Observed yield (t/ha)")
    ax.set_xlim(df["year"].min(), df["year"].max())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2DF", lw=0.8)

    ax = axes[1]
    colors = np.where(
        df["growing_season_rainfall_mm"] <= rain_lo,
        COLORS["rust"],
        np.where(df["growing_season_rainfall_mm"] >= rain_hi, COLORS["sky"], COLORS["grey"]),
    )
    ax.vlines(df["year"], 0, df["growing_season_rainfall_mm"], color=colors, lw=1.6, alpha=0.75)
    ax.plot(df["year"], df["rain_roll"], color=COLORS["ink"], lw=2.2, label="11-year median rainfall")
    ax.axhspan(0, rain_lo, color=mpl.colors.to_rgba(COLORS["rust"], 0.06))
    ax.set_ylabel("Growing-season rainfall (mm)")
    ax.set_xlabel("Year")
    ax.set_xlim(df["year"].min(), df["year"].max())
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2DF", lw=0.8)
    ax.text(df["year"].min() + 2, rain_lo * 0.52, "Lower-rainfall fifth", color=COLORS["rust"], fontsize=10)
    footer(fig, "Regional historical context, 1890–2022. It is not a Seabrook forecast.")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.08, right=0.97)
    save_figure(fig, "v02_long_term_season_context")


def v03_contract_reversal() -> None:
    df = read_csv("source_performance_landscape_seed.csv")
    methods = [
        ("local_scratch", "Local baseline", COLORS["grey"]),
        ("supervised", "Supervised transfer", COLORS["sky"]),
        ("prediction_kd", "Prediction transfer", COLORS["green"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharex=True)
    title(
        fig,
        "The right model depends on the deployment question",
        "The same transfer route helps at unseen locations and harms under the GROUP contract.",
        "Formal result",
    )
    for ax, contract, heading in zip(
        axes,
        ("GROUP_complete", "SPATIAL_complete"),
        ("GROUP · held-out populations", "SPATIAL · held-out locations"),
    ):
        values = []
        for idx, (method, label, color) in enumerate(methods):
            mean, sd = mean_sd(df, contract, method)
            y = 2 - idx
            values.append(mean)
            ax.errorbar(mean, y, xerr=sd, fmt="o", ms=12, lw=2.2, color=color, capsize=5, zorder=3)
            ax.text(mean + 0.025, y, f"{mean:.3f}", va="center", fontsize=14, fontweight="bold", color=color)
        ax.plot([values[0], values[2]], [2, 0], color=COLORS["ink"], lw=1.4, alpha=0.5, zorder=1)
        change = values[2] - values[0]
        message = f"{abs(change):.3f} t/ha lower error" if change < 0 else f"{change:.3f} t/ha higher error"
        ax.text(
            0.5,
            0.08,
            message,
            transform=ax.transAxes,
            ha="center",
            fontsize=15,
            fontweight="bold",
            color=COLORS["green"] if change < 0 else COLORS["rust"],
        )
        ax.set_title(heading, loc="left", fontweight="bold", pad=16)
        ax.set_yticks([2, 1, 0], [m[1] for m in methods])
        ax.set_xlabel("Mean absolute error (t/ha) · lower is better")
        ax.set_xlim(0.55, 1.68)
        ax.grid(axis="x", color="#DDE2DF", lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].text(
        0.5,
        -0.13,
        "Relative error improves, while absolute spatial fit remains limited.",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=10,
        color=COLORS["rust"],
    )
    footer(fig, "Dots show three-run means; bars show run SD. SPATIAL scratch is a fixed reference.")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.97, wspace=0.30)
    save_figure(fig, "v03_deployment_contract_reversal")


def v04_data_availability() -> None:
    rose = read_csv("roseworthy_run_metrics.csv")
    claim = read_csv("claim_gate_seed_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, H), gridspec_kw={"width_ratios": [1.1, 1.05, 1.15]})
    title(
        fig,
        "Input availability changes the preferred learning route",
        "Three independent views show why deployment data must be part of model selection.",
        "Formal result",
    )

    ax = axes[0]
    route_order = ["scratch", "supervised", "combined_kd", "prediction_kd", "missing_aware", "representation_kd"]
    route_labels = ["Scratch", "Supervised", "Combined", "Prediction", "Missing-aware", "Representation"]
    means = rose.groupby(["route", "condition"], as_index=False)["mae"].mean()
    for i, (route, label) in enumerate(zip(route_order, route_labels)):
        comp = float(means.query("route == @route and condition == 'complete'")["mae"].iloc[0])
        no_soil = float(means.query("route == @route and condition == 'no_soil'")["mae"].iloc[0])
        ax.plot([comp, no_soil], [i, i], color="#C9D0CD", lw=2.5)
        ax.scatter(comp, i, color=COLORS["sky"], s=65, zorder=3)
        ax.scatter(no_soil, i, color=COLORS["green"], s=65, zorder=3)
    ax.set_yticks(range(len(route_labels)), route_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.455, 0.505)
    ax.set_xlabel("MAE (t/ha)")
    ax.set_title("Within-paddock: complete vs no soil", loc="left", fontsize=14, fontweight="bold")
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="o", ls="", color=COLORS["sky"], label="Complete"),
            mpl.lines.Line2D([], [], marker="o", ls="", color=COLORS["green"], label="No soil"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = axes[1]
    lf = claim.query("gate == 'late_fusion' and split_id == 'SPATIAL'").copy()
    condition_order = ["complete", "synthetic_random_0_15", "synthetic_no_weather"]
    x = np.arange(3)
    by_seed = []
    for condition in condition_order:
        group = lf[lf["condition"] == condition]
        by_seed.append(
            {
                "joint": group["primary_mae"].mean(),
                "avg": group["official_mae"].mean(),
                "best": group["secondary_mae"].mean(),
            }
        )
    for key, label, color in (("joint", "Joint RF", COLORS["green"]), ("avg", "Branch average", COLORS["sky"]), ("best", "Best branch", COLORS["wheat"])):
        ax.plot(x, [v[key] for v in by_seed], marker="o", ms=8, lw=2.4, label=label, color=color)
    ax.set_xticks(x, ["Complete", "15% random\nmissing", "No weather"])
    ax.set_ylabel("Mean MAE (t/ha)")
    ax.set_ylim(0.88, 1.075)
    ax.set_title("Local RF controls", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="upper center")

    ax = axes[2]
    pi = claim.query("gate == 'teacher_student' and split_id == 'GROUP' and condition == 'synthetic_no_soil'")
    seed_order = [101, 202, 303]
    values = {
        "Observed-target": [float(pi.loc[pi.seed == s, "primary_mae"].iloc[0]) for s in seed_order],
        "Pseudo-target": [float(pi.loc[pi.seed == s, "secondary_mae"].iloc[0]) for s in seed_order],
        "Fixed blend": [float(pi.loc[pi.seed == s, "official_mae"].iloc[0]) for s in seed_order],
    }
    xs = np.arange(3)
    for label, color in (("Observed-target", COLORS["sky"]), ("Pseudo-target", COLORS["wheat"]), ("Fixed blend", COLORS["green"])):
        ax.plot(xs, values[label], marker="o", ms=8, lw=2.4, color=color, label=label)
        ax.scatter([3], [np.mean(values[label])], s=90, marker="D", color=color)
    ax.axvline(2.5, color="#C9D0CD", lw=1)
    ax.set_xticks([0, 1, 2, 3], ["S1", "S2", "S3", "Mean"])
    ax.set_ylabel("MAE (t/ha)")
    ax.set_ylim(0.54, 0.98)
    ax.set_title("Privileged-information runs", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#E0E4E2", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "Settings retain their native protocols; values are not pooled across datasets.")
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.98, wspace=0.36)
    save_figure(fig, "v04_data_availability_value")


def v05_apsim_complementarity() -> None:
    df = read_csv("apsim_run_summary.csv")
    sensitivity = read_csv("apsim_sensitivity_range.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharex=True)
    title(
        fig,
        "APSIM and data-driven models are complementary",
        "Observed differences are modest relative to APSIM input sensitivity.",
        "Formal supplementary result",
    )
    labels = ["Local baseline", "Weather transfer", "APSIM constrained", "APSIM available"]
    keys = [
        ("data_driven", "local_scratch"),
        ("data_driven", "validation_selected_weather_transfer"),
        ("constrained_central", "APSIM"),
        ("available_central", "APSIM"),
    ]
    colors = [COLORS["grey"], COLORS["green"], COLORS["wheat"], COLORS["plum"]]
    for ax, contract in zip(axes, ("GROUP", "SPATIAL")):
        rows = []
        for condition, model in keys:
            row = df.query("contract == @contract and condition == @condition and model == @model").iloc[0]
            rows.append(row)
        y = np.arange(4)[::-1]
        for yi, row, color in zip(y, rows, colors):
            ax.errorbar(
                row["mae_mean"],
                yi,
                xerr=row["mae_sd"],
                fmt="o",
                ms=11,
                lw=2,
                capsize=4,
                color=color,
                zorder=3,
            )
            ax.text(row["mae_mean"] + 0.025, yi, f"{row['mae_mean']:.3f}", va="center", fontsize=13, fontweight="bold")
        sens = sensitivity.query("contract == @contract")
        smin, smax = float(sens["mae_min"].min()), float(sens["mae_max"].max())
        ax.axvspan(smin, smax, color=mpl.colors.to_rgba(COLORS["wheat"], 0.12), zorder=0)
        ax.text(
            (smin + smax) / 2,
            -0.65,
            f"APSIM sensitivity range {smin:.3f}–{smax:.3f}",
            ha="center",
            color="#7B6125",
            fontsize=10.5,
        )
        transfer = float(rows[1]["mae_mean"])
        constrained = float(rows[2]["mae_mean"])
        ax.plot([transfer, constrained], [2, 1], color=COLORS["ink"], lw=1.3, alpha=0.5)
        ax.set_yticks(y, labels)
        ax.set_title(contract, loc="left", fontsize=17, fontweight="bold")
        ax.set_xlabel("Mean absolute error (t/ha) · lower is better")
        ax.set_xlim(0.60, 1.50)
        ax.set_ylim(-0.9, 3.65)
        ax.grid(axis="x", color="#DDE2DF", lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        0.095,
        "All paired intervals for weather transfer vs constrained APSIM include zero; sensitivity settings change the ordering.",
        ha="center",
        fontsize=12,
        color=COLORS["rust"],
        fontweight="bold",
    )
    footer(fig, "APSIM under available deployment inputs, not the full capability of process modelling.")
    fig.subplots_adjust(top=0.80, bottom=0.21, left=0.14, right=0.97, wspace=0.32)
    save_figure(fig, "v05_apsim_complementarity")


def v06_model_trust() -> None:
    support = read_csv("source_support_delta_summary.csv")
    agreement = read_csv("source_perturbation_agreement_disjoint.csv")
    ood = read_csv("source_ood_quality_9d.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, H))
    title(
        fig,
        "A useful forecast needs more than one check",
        "We track predictive reliance, finite response, and distance from familiar training conditions.",
        "Formal diagnostics",
    )

    ax = axes[0]
    route_order = ["prediction_kd", "combined_kd", "representation_kd", "missing_aware"]
    labels = ["Prediction", "Combined", "Representation", "Missing-aware"]
    offsets = {"GROUP_complete": -0.12, "SPATIAL_complete": 0.12}
    colors = {"GROUP_complete": COLORS["sky"], "SPATIAL_complete": COLORS["green"]}
    for contract in offsets:
        rows = support.set_index(["contract", "route"]).loc[contract]
        vals = [rows.loc[r, "mean_delta_support_share"] for r in route_order]
        los = [rows.loc[r, "ci_lo"] for r in route_order]
        his = [rows.loc[r, "ci_hi"] for r in route_order]
        y = np.arange(4) + offsets[contract]
        ax.errorbar(
            vals,
            y,
            xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
            fmt="o",
            ms=8,
            lw=1.8,
            capsize=3,
            color=colors[contract],
            label=contract.split("_")[0],
        )
    ax.axvline(0, color=COLORS["ink"], lw=1)
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in observation-support share")
    ax.set_title("What information use shifted?", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    contract_stats = agreement.groupby("contract").agg(mean_rho=("spearman_rho", "mean"), match=("top_group_match", "mean"))
    x = np.arange(2)
    rhos = [contract_stats.loc[c, "mean_rho"] for c in ("GROUP_complete", "SPATIAL_complete")]
    matches = [contract_stats.loc[c, "match"] for c in ("GROUP_complete", "SPATIAL_complete")]
    ax.bar(x - 0.16, rhos, width=0.30, color=[COLORS["sky"], COLORS["green"]], label="Rank agreement")
    ax.bar(x + 0.16, matches, width=0.30, color=[mpl.colors.to_rgba(COLORS["sky"], 0.45), mpl.colors.to_rgba(COLORS["green"], 0.45)], label="Top-group match")
    ax.axhline(0, color=COLORS["ink"], lw=1)
    ax.axhline(0.25, color=COLORS["wheat"], lw=1.5, ls="--", label="Chance top match")
    ax.set_xticks(x, ["GROUP", "SPATIAL"])
    ax.set_ylim(-0.28, 0.52)
    ax.set_title("Did attribution and stress agree?", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[2]
    ood["nn_ratio"] = ood["mean_perturbed_nn_distance"] / ood["mean_baseline_nn_distance"]
    summary = ood.groupby("perturbation_group")["nn_ratio"].mean().sort_values()
    pretty = {
        "observation_support": "Observation support",
        "temperature": "Temperature",
        "water_balance": "Water balance",
        "radiation": "Radiation",
    }
    ax.barh(
        [pretty[x] for x in summary.index],
        summary.values,
        color=[COLORS["sage"], COLORS["sky"], COLORS["green"], COLORS["wheat"]],
    )
    ax.axvline(1, color=COLORS["ink"], lw=1.2)
    for yi, value in enumerate(summary.values):
        ax.text(value + 0.02, yi, f"{value:.2f}×", va="center", fontweight="bold")
    ax.set_xlim(0.85, max(summary.values) + 0.25)
    ax.set_xlabel("Nearest-neighbour distance ratio")
    ax.set_title("Did the stress leave familiar data?", loc="left", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.grid(axis="x" if ax is not axes[1] else "y", color="#E0E4E2", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "Diagnostics describe fitted-model behaviour; they are not causal agronomic effects.")
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.98, wspace=0.42)
    save_figure(fig, "v06_model_trust_and_scope")


def synthetic_paddock() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(8417)
    x = np.linspace(0, 100, 90)
    y = np.linspace(0, 70, 64)
    xx, yy = np.meshgrid(x, y)
    elevation = 0.05 * xx + 0.08 * yy + 2.8 * np.sin(xx / 15) * np.cos(yy / 13)
    score = (
        2.15
        + 0.48 * np.sin(xx / 16)
        + 0.34 * np.cos(yy / 11)
        - 0.22 * np.exp(-((xx - 72) ** 2 + (yy - 18) ** 2) / 180)
    )
    confidence = np.clip(
        0.94
        - 0.0042 * np.sqrt((xx - 45) ** 2 + (yy - 37) ** 2)
        + 0.05 * np.sin(xx / 12),
        0.45,
        0.96,
    )
    samples = np.column_stack([rng.uniform(8, 90, 22), rng.uniform(7, 63, 22)])
    return xx, yy, elevation, score, confidence, samples


def add_illustrative_badge(fig: plt.Figure) -> None:
    fig.text(
        0.955,
        0.885,
        "ILLUSTRATIVE PRODUCT PREVIEW\nNOT A FARM PREDICTION",
        ha="right",
        va="top",
        fontsize=9,
        color=COLORS["rust"],
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "fc": "#FFF7F3", "ec": COLORS["rust"], "lw": 1.2},
    )


def v07_paddock_preview() -> None:
    xx, yy, elevation, score, confidence, samples = synthetic_paddock()
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    title(
        fig,
        "What a farm-specific delivery could look like",
        "Prediction and confidence are shown separately so uncertainty remains visible.",
        "Illustrative preview",
    )
    add_illustrative_badge(fig)
    cmap = LinearSegmentedColormap.from_list("farm", ["#D8E3C9", COLORS["wheat"], COLORS["green"]])
    im = axes[0].contourf(xx, yy, score, levels=16, cmap=cmap)
    axes[0].contour(xx, yy, elevation, levels=8, colors=mpl.colors.to_rgba(COLORS["ink"], 0.26), linewidths=0.7)
    axes[0].scatter(samples[:, 0], samples[:, 1], s=22, c=COLORS["white"], edgecolor=COLORS["ink"], lw=0.8)
    axes[0].set_title("Predicted yield or feed-base surface", loc="left", fontsize=15, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.set_label("Illustrative value")

    im2 = axes[1].contourf(xx, yy, confidence, levels=np.linspace(0.45, 0.96, 14), cmap="Blues")
    low = confidence < 0.65
    axes[1].contourf(xx, yy, low, levels=[0.5, 1.5], colors="none", hatches=["////"])
    axes[1].scatter(samples[:, 0], samples[:, 1], s=22, c=COLORS["white"], edgecolor=COLORS["ink"], lw=0.8)
    axes[1].set_title("Confidence and targeted sampling zones", loc="left", fontsize=15, fontweight="bold")
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.040, pad=0.02)
    cb2.set_label("Illustrative confidence")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("Local easting")
        ax.set_ylabel("Local northing")
        ax.spines[["top", "right"]].set_visible(False)
    footer(fig, "Requires paddock boundaries, georeferenced outcomes, terrain, soil and management data.")
    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.07, right=0.96, wspace=0.20)
    save_figure(fig, "v07_paddock_product_preview")


def v08_data_partnership() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(
        fig,
        "Start small. Add data when it changes a decision",
        "A staged pilot reduces burden while making the value of each dataset visible.",
        "Partnership proposal",
    )
    phases = [
        (
            0.055,
            "STEP 1",
            "Establish the local\nbaseline",
            ["Paddock boundaries", "Crop or pasture type", "Basic season history"],
            ["Season context", "Baseline forecast", "Data-gap map"],
            COLORS["sage"],
        ),
        (
            0.375,
            "STEP 2",
            "Resolve spatial\ndifferences",
            ["Yield or biomass observations", "Soil tests / EM38", "Terrain and local weather"],
            ["Paddock risk zones", "Confidence surface", "Targeted sampling"],
            COLORS["sky"],
        ),
        (
            0.695,
            "STEP 3",
            "Connect management\nto outcomes",
            ["Sowing and fertiliser", "Grazing and stocking", "Feed, weight and cost records"],
            ["Scenario comparison", "Feed-pressure outlook", "Decision review"],
            COLORS["green"],
        ),
    ]
    for x, step, heading, inputs, outputs, color in phases:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.17),
                0.25,
                0.60,
                boxstyle="round,pad=0.016,rounding_size=0.018",
                fc=mpl.colors.to_rgba(color, 0.10),
                ec=color,
                lw=1.6,
            )
        )
        ax.text(x + 0.02, 0.72, step, fontsize=9, fontweight="bold", color=color, va="top")
        ax.text(x + 0.02, 0.685, heading, fontsize=12.5, fontweight="bold", color=COLORS["ink"], va="top", linespacing=1.0)
        ax.text(x + 0.02, 0.575, "DATA PROVIDED", fontsize=9, fontweight="bold", color=color)
        for i, line in enumerate(inputs):
            ax.text(x + 0.025, 0.535 - i * 0.052, f"• {line}", fontsize=10.5)
        ax.plot([x + 0.02, x + 0.23], [0.37, 0.37], color=mpl.colors.to_rgba(color, 0.45), lw=1)
        ax.text(x + 0.02, 0.335, "DELIVERED BACK", fontsize=9, fontweight="bold", color=color)
        for i, line in enumerate(outputs):
            ax.text(x + 0.025, 0.295 - i * 0.052, f"• {line}", fontsize=10.5)
    for x0, x1 in ((0.321, 0.37), (0.641, 0.69)):
        ax.annotate("", xy=(x1, 0.49), xytext=(x0, 0.49), arrowprops={"arrowstyle": "-|>", "lw": 2, "color": COLORS["grey"]})
    ax.text(
        0.5,
        0.105,
        "Farm data remain the basis for local validation—not a one-time input to a generic model.",
        ha="center",
        fontsize=15,
        color=COLORS["green"],
        fontweight="bold",
    )
    footer(fig, "Recommended pilot structure. Scope and governance should be agreed before data transfer.")
    save_figure(fig, "v08_data_partnership_ladder")


def build_animation() -> None:
    df = read_csv("sa_deep_time_dataset.csv").sort_values("year").copy()
    df["year"] = df["year"].astype(int)
    years = df["year"].to_numpy()
    yield_values = df["observed_yield_t_ha"].to_numpy()
    rain_values = df["growing_season_rainfall_mm"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.4), gridspec_kw={"hspace": 0.23})
    fig.patch.set_facecolor(COLORS["white"])
    fig.suptitle("South Australian season context", x=0.08, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.08, 0.91, "Regional history reveals why local season information matters.", fontsize=11, color="#4F5C59")
    yield_line, = axes[0].plot([], [], color=COLORS["green"], lw=2.6)
    yield_dot, = axes[0].plot([], [], "o", color=COLORS["green"], ms=7)
    rain_line, = axes[1].plot([], [], color=COLORS["sky"], lw=2.2)
    rain_dot, = axes[1].plot([], [], "o", color=COLORS["sky"], ms=7)
    year_text = fig.text(0.92, 0.91, "", ha="right", fontsize=18, fontweight="bold", color=COLORS["ink"])
    for ax, ylabel, ylim in (
        (axes[0], "Observed yield (t/ha)", (0, max(yield_values) * 1.12)),
        (axes[1], "Growing-season rainfall (mm)", (0, max(rain_values) * 1.10)),
    ):
        ax.set_xlim(years.min(), years.max())
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDE2DF")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Year")
    frame_indices = np.unique(np.linspace(1, len(df), 48, dtype=int))

    def update(frame_idx: int):
        end = frame_indices[frame_idx]
        yield_line.set_data(years[:end], yield_values[:end])
        yield_dot.set_data([years[end - 1]], [yield_values[end - 1]])
        rain_line.set_data(years[:end], rain_values[:end])
        rain_dot.set_data([years[end - 1]], [rain_values[end - 1]])
        year_text.set_text(str(years[end - 1]))
        return yield_line, yield_dot, rain_line, rain_dot, year_text

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=130, blit=True)
    ani.save(EXPORT / "animation" / "v09_long_term_season_story.gif", writer=animation.PillowWriter(fps=8), dpi=110)
    plt.close(fig)


def build_3d_preview() -> None:
    xx, yy, elevation, score, confidence, samples = synthetic_paddock()
    custom = np.stack([score, confidence], axis=-1)
    fig = go.Figure(
        data=[
            go.Surface(
                x=xx,
                y=yy,
                z=elevation,
                surfacecolor=score,
                customdata=custom,
                colorscale=[[0, "#D8E3C9"], [0.52, COLORS["wheat"]], [1, COLORS["green"]]],
                colorbar={"title": "Illustrative<br>yield"},
                hovertemplate="Easting %{x:.0f}<br>Northing %{y:.0f}<br>Elevation %{z:.1f}<br>Value %{customdata[0]:.2f}<br>Confidence %{customdata[1]:.0%}<extra></extra>",
                contours={"z": {"show": True, "usecolormap": False, "color": "rgba(30,43,42,0.25)", "width": 1}},
            )
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=samples[:, 0],
            y=samples[:, 1],
            z=np.interp(samples[:, 0], xx[0], elevation[elevation.shape[0] // 2]) + 2,
            mode="markers",
            marker={"size": 4, "color": COLORS["white"], "line": {"color": COLORS["ink"], "width": 1}},
            name="Illustrative observations",
            hovertemplate="Example observation location<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Illustrative 3D paddock decision surface<br><sup>Product preview—not a Seabrook prediction</sup>",
            "x": 0.04,
            "xanchor": "left",
        },
        scene={
            "xaxis_title": "Local easting",
            "yaxis_title": "Local northing",
            "zaxis_title": "Terrain elevation",
            "camera": {"eye": {"x": 1.45, "y": -1.6, "z": 1.1}},
            "aspectratio": {"x": 1.45, "y": 1.0, "z": 0.38},
        },
        paper_bgcolor=COLORS["white"],
        font={"family": "Arial, sans-serif", "color": COLORS["ink"], "size": 14},
        margin={"l": 10, "r": 10, "t": 85, "b": 10},
        annotations=[
            {
                "text": "Height = terrain · colour = selected decision layer · requires local farm data",
                "x": 0.04,
                "y": 0.02,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12, "color": "#596662"},
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Illustrative yield",
                        "method": "restyle",
                        "args": [
                            {
                                "surfacecolor": [score],
                                "colorscale": [[[0, "#D8E3C9"], [0.52, COLORS["wheat"]], [1, COLORS["green"]]]],
                                "colorbar.title": ["Illustrative<br>yield"],
                            },
                            [0],
                        ],
                    },
                    {
                        "label": "Confidence",
                        "method": "restyle",
                        "args": [
                            {
                                "surfacecolor": [confidence],
                                "colorscale": [[[0, "#F2D7CE"], [0.5, "#D9E8F0"], [1, COLORS["sky"]]]],
                                "colorbar.title": ["Illustrative<br>confidence"],
                            },
                            [0],
                        ],
                    },
                ],
                "direction": "right",
                "x": 0.52,
                "y": 1.06,
                "xanchor": "center",
                "showactive": True,
            }
        ],
    )
    fig.write_html(
        EXPORT / "interactive" / "v10_interactive_3d_paddock_preview.html",
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )

    mpl_fig = plt.figure(figsize=(W, H))
    ax = mpl_fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(xx, yy, elevation, facecolors=plt.get_cmap("YlGn")((score - score.min()) / (score.max() - score.min())), rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False)
    ax.view_init(elev=28, azim=-58)
    ax.set_xlabel("Local easting")
    ax.set_ylabel("Local northing")
    ax.set_zlabel("Terrain elevation")
    ax.set_title("Illustrative 3D paddock decision surface", loc="left", fontsize=22, fontweight="bold", pad=18)
    mpl_fig.text(0.07, 0.89, "Product preview—not a Seabrook prediction", color=COLORS["rust"], fontsize=13, fontweight="bold")
    footer(mpl_fig, "Height represents terrain; colour represents an illustrative decision layer.")
    save_figure(mpl_fig, "v10_3d_paddock_preview")


def build_contact_sheets() -> None:
    slugs = [
        "v01_data_to_decision",
        "v02_long_term_season_context",
        "v03_deployment_contract_reversal",
        "v04_data_availability_value",
        "v05_apsim_complementarity",
        "v06_model_trust_and_scope",
        "v07_paddock_product_preview",
        "v08_data_partnership_ladder",
        "v10_3d_paddock_preview",
    ]
    thumbs = []
    for slug in slugs:
        image = Image.open(EXPORT / "png" / f"{slug}.png").convert("RGB")
        thumb = ImageOps.contain(image, (620, 340), method=Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (640, 360), COLORS["white"])
        tile.paste(thumb, ((640 - thumb.width) // 2, (360 - thumb.height) // 2))
        thumbs.append(tile)
    canvas = Image.new("RGB", (1920, 1080), COLORS["white"])
    for idx, image in enumerate(thumbs):
        x = (idx % 3) * 640
        y = (idx // 3) * 360
        canvas.paste(image, (x, y))
    canvas.save(QA / "contact_sheet.png", quality=95)
    ImageOps.grayscale(canvas).save(QA / "contact_sheet_grayscale.png")


def build_manifests() -> None:
    sources = []
    for path in sorted(DATA.glob("*")):
        if path.is_file():
            sources.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    (QA / "source_manifest.json").write_text(json.dumps({"version": VERSION, "sources": sources}, indent=2) + "\n")

    outputs = []
    for path in sorted(EXPORT.rglob("*")):
        if path.is_file():
            outputs.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    (QA / "output_manifest.json").write_text(json.dumps({"version": VERSION, "outputs": outputs}, indent=2) + "\n")


def build_gallery() -> None:
    items = [
        ("v01_data_to_decision", "From farm data to a decision product", "Conceptual workflow"),
        ("v02_long_term_season_context", "Long-term South Australian season context", "Exploratory public context"),
        ("v03_deployment_contract_reversal", "Deployment-contract reversal", "Formal evidence"),
        ("v04_data_availability_value", "The value of available inputs", "Formal evidence"),
        ("v05_apsim_complementarity", "APSIM and data-driven complementarity", "Formal supplementary evidence"),
        ("v06_model_trust_and_scope", "Model behaviour and operating range", "Formal diagnostics"),
        ("v07_paddock_product_preview", "2D paddock product preview", "Illustrative preview"),
        ("v08_data_partnership_ladder", "Staged data partnership", "Partnership proposal"),
        ("v10_3d_paddock_preview", "3D paddock product preview", "Illustrative preview"),
    ]
    cards = []
    for slug, heading, badge in items:
        cards.append(
            f"""
            <article>
              <div class="meta">{html.escape(badge)}</div>
              <h2>{html.escape(heading)}</h2>
              <img src="png/{slug}.png" alt="{html.escape(heading)}">
              <nav><a href="svg/{slug}.svg">SVG</a><a href="pdf/{slug}.pdf">PDF</a></nav>
            </article>
            """
        )
    gallery = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Farm Manager Visual Evidence · v{VERSION}</title>
<style>
  :root {{ --ink:#1E2B2A; --green:#1F5A47; --wheat:#D5A33F; --pale:#F3F4EF; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:Arial,sans-serif; color:var(--ink); background:var(--pale); }}
  header {{ padding:52px max(5vw,28px) 34px; background:#fff; border-bottom:1px solid #d8dedb; }}
  h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,56px); font-weight:600; }}
  header p {{ margin:0; max-width:850px; font-size:18px; color:#52605c; }}
  main {{ width:min(1500px,94vw); margin:36px auto 80px; display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:26px; }}
  article {{ background:#fff; padding:20px; border:1px solid #d8dedb; box-shadow:0 10px 28px rgba(30,43,42,.07); }}
  h2 {{ font-size:21px; font-weight:600; margin:7px 0 16px; }}
  .meta {{ color:var(--green); text-transform:uppercase; font-size:11px; font-weight:700; letter-spacing:.08em; }}
  img {{ display:block; width:100%; height:auto; border:1px solid #edf0ee; }}
  nav {{ display:flex; gap:16px; margin-top:14px; }}
  a {{ color:var(--green); font-weight:600; text-decoration:none; }}
  .wide {{ grid-column:1/-1; }}
  .wide img {{ max-height:560px; object-fit:contain; background:#fff; }}
  footer {{ padding:30px; text-align:center; color:#68736f; }}
  @media(max-width:520px) {{ main {{ grid-template-columns:1fr; }} article {{ padding:12px; }} }}
</style>
</head>
<body>
<header><h1>Farm decision evidence</h1><p>Formal results, contextual history and clearly labelled product previews. Every asset is generated from versioned scripts and source manifests.</p></header>
<main>
{''.join(cards)}
<article class="wide"><div class="meta">Animation</div><h2>Long-term season story</h2><img src="animation/v09_long_term_season_story.gif" alt="Animated South Australian yield and rainfall history"></article>
<article class="wide"><div class="meta">Interactive 3D preview</div><h2>Terrain, illustrative yield and confidence</h2><p><a href="interactive/v10_interactive_3d_paddock_preview.html">Open the self-contained interactive view</a></p></article>
</main>
<footer>Version {VERSION} · Illustrative previews are not farm predictions.</footer>
</body></html>
"""
    (EXPORT / "farm_manager_visual_gallery.html").write_text(gallery)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-only", action="store_true", help="Remove generated files and exit.")
    args = parser.parse_args()
    ensure_dirs()
    if args.clean_only:
        clean_outputs()
        return
    configure_matplotlib()
    v01_decision_chain()
    v02_long_term_context()
    v03_contract_reversal()
    v04_data_availability()
    v05_apsim_complementarity()
    v06_model_trust()
    v07_paddock_preview()
    v08_data_partnership()
    build_animation()
    build_3d_preview()
    build_contact_sheets()
    build_gallery()
    build_manifests()
    print(json.dumps({"status": "PASS", "version": VERSION, "exports": len(list(EXPORT.rglob('*.*')))}, indent=2))


if __name__ == "__main__":
    main()
