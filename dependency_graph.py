"""
dependency_graph.py — Build and visualise ability dependency graphs.

For each model, builds a directed graph where an edge A→B means:
"suppressing ability A significantly reduces performance on ability B"

Uses the delta matrix (AUC relative to neutral baseline α=0).

Usage:
    python dependency_graph.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS = {
    "Qwen2.5-3B":  "results_3b/results.csv",
    "Qwen2.5-7B":  "results_7b/results.csv",
    "Qwen2.5-14B": "results_14b/results.csv",
}
FIGURES_DIR   = "figures_analysis"
BASELINE_ALPHA = 0.0
THRESHOLD      = -0.5   # delta < THRESHOLD counts as significant dependency
ALPHAS_TO_PLOT = [-1.5, -1.0]  # which alphas to draw graphs for

ABILITIES = [
    "AS", "CEc", "CEe", "CL",
    "MCr", "MCt", "MCu", "MS",
    "QLl", "QLq", "SNs",
    "KNa", "KNc", "KNf", "KNn", "KNs",
    "AT", "VO",
]

# Ability group colours for node colouring
ABILITY_GROUPS = {
    "Elemental":  ["AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl", "QLq", "SNs"],
    "Knowledge":  ["KNa", "KNc", "KNf", "KNn", "KNs"],
    "Extraneous": ["AT", "VO"],
}
GROUP_COLOURS = {
    "Elemental":  "#4C9BE8",
    "Knowledge":  "#E8874C",
    "Extraneous": "#7BC67B",
}

def ability_group(ability):
    for group, members in ABILITY_GROUPS.items():
        if ability in members:
            return group
    return "Other"


def load_delta_matrix(csv_path: str, alpha: float) -> pd.DataFrame:
    """Return 18x18 delta matrix: AUC(alpha) - AUC(baseline=0)."""
    df = pd.read_csv(csv_path)

    baseline = df[np.isclose(df["alpha"], BASELINE_ALPHA)].pivot(
        index="discouraged_ability", columns="test_ability", values="auc"
    ).reindex(index=ABILITIES, columns=ABILITIES)

    target = df[np.isclose(df["alpha"], alpha)].pivot(
        index="discouraged_ability", columns="test_ability", values="auc"
    ).reindex(index=ABILITIES, columns=ABILITIES)

    return target - baseline


def build_graph(delta: pd.DataFrame, threshold: float) -> nx.DiGraph:
    """
    Build directed graph from delta matrix.
    Edge A→B if delta[A][B] < threshold (suppressing A hurts B).
    """
    G = nx.DiGraph()
    G.add_nodes_from(ABILITIES)

    for discouraged in ABILITIES:
        for test in ABILITIES:
            if discouraged == test:
                continue
            val = delta.loc[discouraged, test]
            if pd.notna(val) and val < threshold:
                G.add_edge(discouraged, test, weight=abs(val))
    return G


def plot_dependency_graph(G: nx.DiGraph, title: str, out_path: str):
    """Draw and save the dependency graph."""
    fig, ax = plt.subplots(figsize=(16, 14))

    # Layout
    pos = nx.spring_layout(G, k=2.5, seed=42)

    # Node colours by group
    node_colours = [GROUP_COLOURS[ability_group(n)] for n in G.nodes()]

    # Edge widths scaled by weight
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        max_w = max(weights)
        edge_widths = [1 + 3 * (w / max_w) for w in weights]
    else:
        edge_widths = []

    # Out-degree = how many abilities this node suppresses when discouraged
    out_degrees = dict(G.out_degree())
    node_sizes = [300 + 150 * out_degrees.get(n, 0) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           alpha=0.6,
                           edge_color="gray",
                           arrows=True,
                           arrowsize=15,
                           connectionstyle="arc3,rad=0.1")

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=g)
        for g, c in GROUP_COLOURS.items()
    ]
    legend_handles.append(
        mpatches.Patch(color="white", label=f"Node size = out-degree\nThreshold = {THRESHOLD}")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_hub_abilities(G: nx.DiGraph, model_name: str, alpha: float):
    """Print the most influential (high out-degree) abilities."""
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    in_deg  = sorted(G.in_degree(),  key=lambda x: x[1], reverse=True)
    print(f"\n[{model_name}] α={alpha}  threshold={THRESHOLD}")
    print("  Most influential when suppressed (high out-degree):")
    for node, deg in out_deg[:5]:
        if deg > 0:
            targets = [v for _, v in G.out_edges(node)]
            print(f"    {node}: affects {deg} abilities → {targets}")
    print("  Most affected when others are suppressed (high in-degree):")
    for node, deg in in_deg[:5]:
        if deg > 0:
            sources = [u for u, _ in G.in_edges(node)]
            print(f"    {node}: affected by {deg} abilities ← {sources}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for model_name, csv_path in RESULTS.items():
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found")
            continue

        for alpha in ALPHAS_TO_PLOT:
            delta = load_delta_matrix(csv_path, alpha)
            G = build_graph(delta, THRESHOLD)

            alpha_str = str(alpha).replace("-", "neg").replace(".", "p")
            title = f"Ability Dependency Graph\n{model_name}  |  α={alpha}  |  threshold={THRESHOLD}"
            out_path = os.path.join(FIGURES_DIR, f"depgraph_{model_name}_alpha{alpha_str}.png")

            plot_dependency_graph(G, title, out_path)
            print_hub_abilities(G, model_name, alpha)


if __name__ == "__main__":
    main()
