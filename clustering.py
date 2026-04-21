"""
clustering.py — Cluster abilities based on how similarly they are affected
by suppression across all discouraged abilities.

Two complementary clusterings:
  1. Cluster test_abilities by their "vulnerability profile"
     (how each test ability responds when different abilities are suppressed)
  2. Cluster discouraged_abilities by their "influence profile"
     (which test abilities each suppressed ability tends to affect)

Also produces a reordered heatmap showing the cluster structure clearly.

Usage:
    python clustering.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

RESULTS = {
    "Qwen2.5-3B":  "results_3b/results.csv",
    "Qwen2.5-7B":  "results_7b/results.csv",
    "Qwen2.5-14B": "results_14b/results.csv",
}
FIGURES_DIR    = "figures_analysis"
BASELINE_ALPHA = 0.0
CLUSTER_ALPHA  = -1.0   # use α=−1.0 (pure discourage) for clustering
N_CLUSTERS     = 4      # number of clusters to extract

ABILITIES = [
    "AS", "CEc", "CEe", "CL",
    "MCr", "MCt", "MCu", "MS",
    "QLl", "QLq", "SNs",
    "KNa", "KNc", "KNf", "KNn", "KNs",
    "AT", "VO",
]


def load_delta_matrix(csv_path: str, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    baseline = df[np.isclose(df["alpha"], BASELINE_ALPHA)].pivot(
        index="discouraged_ability", columns="test_ability", values="auc"
    ).reindex(index=ABILITIES, columns=ABILITIES)

    target = df[np.isclose(df["alpha"], alpha)].pivot(
        index="discouraged_ability", columns="test_ability", values="auc"
    ).reindex(index=ABILITIES, columns=ABILITIES)

    return target - baseline


def hierarchical_cluster(matrix: pd.DataFrame, axis: int = 1):
    """
    Perform hierarchical clustering on rows (axis=0) or columns (axis=1).
    Returns (linkage_matrix, ordered_labels).
    """
    data = matrix.values if axis == 1 else matrix.values.T
    # Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)
    labels = list(matrix.columns) if axis == 1 else list(matrix.index)

    Z = linkage(data, method="ward")
    order = dendrogram(Z, no_plot=True)["leaves"]
    ordered_labels = [labels[i] for i in order]
    return Z, ordered_labels, order


def plot_clustered_heatmap(delta: pd.DataFrame, model_name: str, out_dir: str):
    """
    Plot delta heatmap reordered by hierarchical clustering on both axes.
    """
    # Cluster columns (test abilities = vulnerability profile)
    Z_col, col_order, col_idx = hierarchical_cluster(delta, axis=1)
    # Cluster rows (discouraged abilities = influence profile)
    Z_row, row_order, row_idx = hierarchical_cluster(delta, axis=0)

    reordered = delta.loc[row_order, col_order]

    fig, axes = plt.subplots(1, 3, figsize=(28, 10),
                             gridspec_kw={"width_ratios": [1, 8, 1]})

    # Left dendrogram (rows)
    ax_row = axes[0]
    dendrogram(Z_row, orientation="left", ax=ax_row,
               labels=row_order, leaf_font_size=8)
    ax_row.set_title("Discouraged\nabilities", fontsize=9)
    ax_row.axis("off")

    # Main heatmap
    ax_main = axes[1]
    all_vals = delta.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    abs_max = np.percentile(np.abs(all_vals), 98)

    sns.heatmap(
        reordered, ax=ax_main,
        vmin=-abs_max, vmax=abs_max,
        cmap="RdYlGn", center=0,
        linewidths=0.3, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"size": 6},
        cbar_kws={"shrink": 0.75, "label": "ΔAUC vs neutral baseline"},
        mask=reordered.isna(),
    )
    ax_main.set_title(
        f"Clustered Delta Heatmap — {model_name}\n"
        f"α={CLUSTER_ALPHA} (discourage prompt only)  |  reordered by hierarchical clustering",
        fontsize=11, pad=10
    )
    ax_main.set_xlabel("Test ability (clustered)", fontsize=9)
    ax_main.set_ylabel("Discouraged ability (clustered)", fontsize=9)
    ax_main.tick_params(axis="x", labelsize=7, rotation=45)
    ax_main.tick_params(axis="y", labelsize=7, rotation=0)

    # Right dendrogram (columns)
    ax_col = axes[2]
    dendrogram(Z_col, orientation="right", ax=ax_col,
               labels=col_order, leaf_font_size=8)
    ax_col.set_title("Test\nabilities", fontsize=9)
    ax_col.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"clustered_heatmap_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cluster_profiles(delta: pd.DataFrame, model_name: str,
                          n_clusters: int, out_dir: str):
    """
    Extract N clusters of test abilities and plot the mean delta profile
    for each cluster — showing what each cluster 'looks like' when suppressed.
    """
    data = np.nan_to_num(delta.values.T, nan=0.0)   # shape: (18 test, 18 discouraged)
    Z = linkage(data, method="ward")
    cluster_ids = fcluster(Z, n_clusters, criterion="maxclust")

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)
    fig.suptitle(
        f"Mean Delta Profile per Cluster of Test Abilities\n{model_name}  |  α={CLUSTER_ALPHA}",
        fontsize=13, fontweight="bold"
    )

    for c in range(1, n_clusters + 1):
        members = [ABILITIES[i] for i, cid in enumerate(cluster_ids) if cid == c]
        profile = delta[members].mean(axis=1)   # mean delta across cluster members

        ax = axes[c - 1]
        colours = ["red" if v < 0 else "green" for v in profile.values]
        ax.barh(profile.index, profile.values, color=colours, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Cluster {c}\n{members}", fontsize=8, wrap=True)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Mean ΔAUC", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"cluster_profiles_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n[{model_name}] Clusters of test abilities (α={CLUSTER_ALPHA}):")
    for c in range(1, n_clusters + 1):
        members = [ABILITIES[i] for i, cid in enumerate(cluster_ids) if cid == c]
        print(f"  Cluster {c}: {members}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for model_name, csv_path in RESULTS.items():
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found")
            continue

        delta = load_delta_matrix(csv_path, CLUSTER_ALPHA)
        plot_clustered_heatmap(delta, model_name, FIGURES_DIR)
        plot_cluster_profiles(delta, model_name, N_CLUSTERS, FIGURES_DIR)


if __name__ == "__main__":
    main()
