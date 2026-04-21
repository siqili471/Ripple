"""
visualize.py — Heatmap visualisation of contrastive decoding results.

For each (model, prompt) combination, produces two figures:
  1. Absolute AUC heatmaps (one subplot per alpha)
  2. Delta heatmaps: AUC(alpha) - AUC(alpha=0.0 baseline)

Each subplot is an 18×18 heatmap where:
    rows    = discouraged ability
    columns = test ability
    value   = AUC score or delta

Usage:
    python visualize.py --results results/results.csv --label "Qwen2.5-7B"
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from config import ABILITIES, ALPHAS, FIGURES_DIR

BASELINE_ALPHA = 0.0

ALPHA_TITLES = {
    -1.5: "α = −1.5  (further discouragement)",
    -1.0: "α = −1.0  (discourage prompt only)",
     0.0: "α =  0.0  (neutral baseline)",
     0.5: "α =  0.5  (encouragement)",
}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ability_score_auc" in df.columns:
        df = df.rename(columns={"ability_score_auc": "auc"})
    return df


def build_matrix(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Return an 18×18 AUC matrix for one alpha value."""
    subset = df[np.isclose(df["alpha"], alpha)]
    matrix = subset.pivot(
        index="discouraged_ability",
        columns="test_ability",
        values="auc",
    )
    return matrix.reindex(index=ABILITIES, columns=ABILITIES)


def add_diagonal_boxes(ax, matrix):
    """Highlight diagonal cells (same ability discouraged and tested)."""
    row_labels = list(matrix.index)
    col_labels = list(matrix.columns)
    for ability in ABILITIES:
        if ability in row_labels and ability in col_labels:
            r = row_labels.index(ability)
            c = col_labels.index(ability)
            ax.add_patch(plt.Rectangle(
                (c, r), 1, 1,
                fill=False, edgecolor="black", lw=1.5, clip_on=False
            ))


def plot_heatmaps(csv_path: str, label: str, out_dir: str = FIGURES_DIR):
    """Generate absolute AUC heatmaps (2x2 grid, one per alpha)."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(csv_path)

    fig = plt.figure(figsize=(26, 22))
    fig.suptitle(
        f"Contrastive Decoding — Ability Suppression Heatmaps\n{label}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    all_values = df["auc"].dropna().values
    vmin, vmax = np.percentile(all_values, 2), np.percentile(all_values, 98)

    for idx, alpha in enumerate(ALPHAS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        matrix = build_matrix(df, alpha)

        sns.heatmap(
            matrix, ax=ax,
            vmin=vmin, vmax=vmax,
            cmap="RdYlGn",
            linewidths=0.3, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.75, "label": "AUC (ability score)"},
            mask=matrix.isna(),
        )
        ax.set_title(ALPHA_TITLES.get(alpha, f"α = {alpha}"), fontsize=11, pad=8)
        ax.set_xlabel("Test ability", fontsize=9)
        ax.set_ylabel("Discouraged ability", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        add_diagonal_boxes(ax, matrix)

    safe_label = label.replace(" ", "_").replace("/", "-")
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"heatmap_{safe_label}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_delta_heatmaps(csv_path: str, label: str, out_dir: str = FIGURES_DIR):
    """
    Generate delta heatmaps: AUC(alpha) - AUC(baseline=0.0).

    Positive (green) = better than neutral baseline
    Negative (red)   = worse than neutral baseline (discouragement worked)
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(csv_path)

    baseline = build_matrix(df, BASELINE_ALPHA)
    non_baseline_alphas = [a for a in ALPHAS if not np.isclose(a, BASELINE_ALPHA)]

    fig = plt.figure(figsize=(26, 11))
    fig.suptitle(
        f"Contrastive Decoding — Delta vs Neutral Baseline (α=0)\n{label}",
        fontsize=16, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.25)

    delta_matrices = [build_matrix(df, a) - baseline for a in non_baseline_alphas]
    all_deltas = np.concatenate([m.values.flatten() for m in delta_matrices])
    all_deltas = all_deltas[~np.isnan(all_deltas)]
    abs_max = np.percentile(np.abs(all_deltas), 98)
    vmin, vmax = -abs_max, abs_max

    for idx, alpha in enumerate(non_baseline_alphas):
        ax = fig.add_subplot(gs[0, idx])
        delta = build_matrix(df, alpha) - baseline

        sns.heatmap(
            delta, ax=ax,
            vmin=vmin, vmax=vmax,
            cmap="RdYlGn",
            center=0,
            linewidths=0.3, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.75, "label": "ΔAUC vs neutral baseline"},
            mask=delta.isna(),
        )
        title = ALPHA_TITLES.get(alpha, f"α = {alpha}")
        ax.set_title(f"{title}\n(red = worse than baseline)", fontsize=10, pad=8)
        ax.set_xlabel("Test ability", fontsize=9)
        ax.set_ylabel("Discouraged ability", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        add_diagonal_boxes(ax, delta)

    safe_label = label.replace(" ", "_").replace("/", "-")
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"heatmap_delta_{safe_label}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--label",   required=True, help="Experiment label")
    parser.add_argument("--out",     default=FIGURES_DIR, help="Output directory")
    args = parser.parse_args()

    plot_heatmaps(args.results, args.label, args.out)
    plot_delta_heatmaps(args.results, args.label, args.out)
