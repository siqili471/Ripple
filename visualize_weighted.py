"""
visualize_weighted.py — Heatmap generation for weighted-score results.

Generates a heatmap grid with one subplot per alpha value.
Each cell shows delta_pct: the percentage change in weighted score
relative to neutral baseline.

Usage:
    python visualize_weighted.py
    python visualize_weighted.py --results results_weighted_7b/results_weighted.csv --label "Qwen2.5-7B"
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from config_weighted import ABILITIES, RESULTS_DIR, FIGURES_DIR


def generate_heatmaps(results_csv=None, label=None, out_dir=None):
    """Generate delta heatmaps from weighted-score results."""

    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "results_weighted.csv")
    if label is None:
        label = "Weighted"
    if out_dir is None:
        out_dir = FIGURES_DIR

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(results_csv)
    alphas = sorted(df["alpha"].unique())

    # Filter out alpha=0 since that's the baseline
    alphas_to_plot = [a for a in alphas if a != 0.0]

    if not alphas_to_plot:
        print("No non-zero alphas found. Nothing to plot.")
        return

    # Determine grid layout
    n = len(alphas_to_plot)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Find global color range for consistent scale
    vmax = 0
    for alpha in alphas_to_plot:
        sub = df[df["alpha"] == alpha]
        pivot = sub.pivot_table(
            index="discouraged_ability", columns="test_ability",
            values="delta_pct", aggfunc="first"
        )
        vmax = max(vmax, abs(pivot.values[np.isfinite(pivot.values)]).max())

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for i, alpha in enumerate(alphas_to_plot):
        ax = axes[i]
        sub = df[df["alpha"] == alpha]

        pivot = sub.pivot_table(
            index="discouraged_ability", columns="test_ability",
            values="delta_pct", aggfunc="first"
        )

        # Reorder to canonical ability order
        rows = [a for a in ABILITIES if a in pivot.index]
        cols = [a for a in ABILITIES if a in pivot.columns]
        pivot = pivot.reindex(index=rows, columns=cols)

        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn", norm=norm,
            annot=True, fmt=".0f", annot_kws={"size": 7},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Δ score (%)"},
        )
        ax.set_title(f"α = {alpha}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Ability", fontsize=10)
        ax.set_ylabel("Discouraged Ability", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Weighted Score Delta (%) — {label}\n"
        f"Red = score decreased   |   Green = score increased",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"heatmap_weighted_{label}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)

    # Also generate a raw score heatmap (not delta) for reference
    _generate_raw_score_heatmap(df, alphas_to_plot, label, out_dir, norm)


def _generate_raw_score_heatmap(df, alphas_to_plot, label, out_dir, norm):
    """Generate a heatmap showing raw weighted scores (not delta)."""

    n = len(alphas_to_plot) + 1  # +1 for baseline
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Baseline subplot
    ax = axes[0]
    sub = df[df["alpha"] == df["alpha"].iloc[0]]  # any alpha has baseline
    baseline_pivot = sub.pivot_table(
        index="discouraged_ability", columns="test_ability",
        values="baseline", aggfunc="first"
    )
    rows = [a for a in ABILITIES if a in baseline_pivot.index]
    cols = [a for a in ABILITIES if a in baseline_pivot.columns]
    baseline_pivot = baseline_pivot.reindex(index=rows, columns=cols)

    sns.heatmap(
        baseline_pivot, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".0f", annot_kws={"size": 7},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Weighted Score"},
    )
    ax.set_title("Baseline (neutral)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Test Ability", fontsize=10)
    ax.set_ylabel("Discouraged Ability", fontsize=10)

    # Alpha subplots
    for i, alpha in enumerate(alphas_to_plot):
        ax = axes[i + 1]
        sub = df[df["alpha"] == alpha]
        pivot = sub.pivot_table(
            index="discouraged_ability", columns="test_ability",
            values="score", aggfunc="first"
        )
        pivot = pivot.reindex(index=rows, columns=cols)

        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd",
            annot=True, fmt=".0f", annot_kws={"size": 7},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Weighted Score"},
        )
        ax.set_title(f"α = {alpha}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Ability", fontsize=10)
        ax.set_ylabel("Discouraged Ability", fontsize=10)

    for j in range(i + 2, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Raw Weighted Scores — {label}",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"heatmap_weighted_raw_{label}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    generate_heatmaps(args.results, args.label, args.out)
