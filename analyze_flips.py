"""
analyze_flips.py — Flip Analysis for the Ripple Effect

Core question: When we discourage ability X, does the model preferentially
get wrong on questions where X-level is high? If so, steering has
directionality — the "ripple" is caused by multi-ability overlap in
questions, not by random global degradation.

Analysis pipeline:
    1. Load detail CSV with per-question predictions under each condition
    2. For each (discouraged_ability, alpha), identify "flip" questions:
       - Positive→Negative flip: correct under neutral, wrong under discourage
       - Negative→Positive flip: wrong under neutral, correct under discourage
       - No flip: same outcome
    3. Compare the ability level profile of flipped vs non-flipped questions
    4. Build a "directionality matrix": for each discouraged ability,
       which ability levels best predict whether a question flips
    5. Visualize results

Usage:
    python analyze_flips.py --detail results_weighted_7b_v2/detail_predictions.csv --alpha -2.0 --label "Qwen-7B"
    python analyze_flips.py --detail results_weighted_7b_v2/detail_predictions.csv --alpha -3.0 --label "Qwen-7B"
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config_weighted import ABILITIES

FIGURES_DEFAULT = "figures_flip_analysis"


def load_and_prepare(detail_path: str):
    """
    Load detail CSV and split into neutral and discouraged conditions.

    Returns:
        neutral_df: DataFrame with neutral predictions (one row per question)
        disc_df: DataFrame with all discourage conditions
    """
    df = pd.read_csv(detail_path)

    # Separate neutral baseline from discourage conditions
    neutral_df = df[df["discouraged_ability"] == "NEUTRAL"].copy()
    disc_df = df[df["discouraged_ability"] != "NEUTRAL"].copy()

    print(f"Loaded {len(df)} total rows", flush=True)
    print(f"  Neutral: {len(neutral_df)} rows", flush=True)
    print(f"  Discouraged: {len(disc_df)} rows", flush=True)
    print(f"  Unique discouraged abilities: {sorted(disc_df['discouraged_ability'].unique())}", flush=True)
    print(f"  Alpha values: {sorted(disc_df['alpha'].unique())}", flush=True)

    return neutral_df, disc_df


def identify_flips(neutral_df: pd.DataFrame, disc_df: pd.DataFrame,
                   disc_ability: str, alpha: float):
    """
    For a given (discouraged_ability, alpha), classify each question into:
        - pos_to_neg: correct under neutral → wrong under discourage
        - neg_to_pos: wrong under neutral → correct under discourage
        - stayed_correct: correct under both
        - stayed_wrong: wrong under both

    Returns a DataFrame with one row per question and a 'flip_type' column.
    """
    # Get the relevant discourage condition
    cond = disc_df[
        (disc_df["discouraged_ability"] == disc_ability) &
        (disc_df["alpha"] == alpha)
    ].copy()

    if len(cond) == 0:
        return None

    # Merge neutral and discourage on question_idx
    neutral_sub = neutral_df[["question_idx", "correct"]].copy()
    neutral_sub = neutral_sub.rename(columns={"correct": "correct_neutral"})

    cond_sub = cond[["question_idx", "correct"]].copy()
    cond_sub = cond_sub.rename(columns={"correct": "correct_disc"})

    merged = neutral_sub.merge(cond_sub, on="question_idx", how="inner")

    # Classify flip type
    def classify(row):
        if row["correct_neutral"] == 1 and row["correct_disc"] == 0:
            return "pos_to_neg"
        elif row["correct_neutral"] == 0 and row["correct_disc"] == 1:
            return "neg_to_pos"
        elif row["correct_neutral"] == 1 and row["correct_disc"] == 1:
            return "stayed_correct"
        else:
            return "stayed_wrong"

    merged["flip_type"] = merged.apply(classify, axis=1)

    # Add level columns from the neutral df (they're the same for all conditions)
    level_cols = [f"level_{a}" for a in ABILITIES]
    level_data = neutral_df[["question_idx"] + level_cols].copy()
    merged = merged.merge(level_data, on="question_idx", how="left")

    return merged


def compute_directionality_matrix(neutral_df: pd.DataFrame,
                                  disc_df: pd.DataFrame,
                                  alpha: float):
    """
    Build the directionality matrix D[X, Y]:
        For each discouraged ability X, compute the mean level_Y of
        pos_to_neg flipped questions minus the mean level_Y of
        non-flipped questions (stayed_correct).

    If D[X, X] > D[X, Y!=X], steering X preferentially affects
    X-heavy questions → directionality exists.

    Returns:
        D: 18x18 DataFrame (rows=discouraged, cols=ability levels)
        flip_counts: dict of flip type counts per discouraged ability
    """
    D = pd.DataFrame(0.0, index=ABILITIES, columns=ABILITIES)
    D_pval = pd.DataFrame(1.0, index=ABILITIES, columns=ABILITIES)
    flip_counts = {}

    for disc_ability in ABILITIES:
        merged = identify_flips(neutral_df, disc_df, disc_ability, alpha)
        if merged is None:
            continue

        pos_to_neg = merged[merged["flip_type"] == "pos_to_neg"]
        stayed_correct = merged[merged["flip_type"] == "stayed_correct"]

        counts = merged["flip_type"].value_counts().to_dict()
        flip_counts[disc_ability] = counts

        if len(pos_to_neg) < 3 or len(stayed_correct) < 3:
            # Not enough flips to analyze
            continue

        for test_ability in ABILITIES:
            col = f"level_{test_ability}"

            mean_flipped = pos_to_neg[col].mean()
            mean_stayed = stayed_correct[col].mean()

            # Effect size: difference in means
            D.loc[disc_ability, test_ability] = mean_flipped - mean_stayed

            # Statistical test: Mann-Whitney U (non-parametric)
            try:
                stat, pval = stats.mannwhitneyu(
                    pos_to_neg[col].values,
                    stayed_correct[col].values,
                    alternative="greater"  # flipped questions have HIGHER level?
                )
                D_pval.loc[disc_ability, test_ability] = pval
            except ValueError:
                D_pval.loc[disc_ability, test_ability] = 1.0

    return D, D_pval, flip_counts


def compute_diagonal_dominance(D: pd.DataFrame):
    """
    For each row (discouraged ability X), check if D[X, X] is the
    largest value in that row, or at least above the row median.

    Returns a summary DataFrame.
    """
    results = []
    for disc_ability in ABILITIES:
        row_values = D.loc[disc_ability]
        diag_val = row_values[disc_ability]
        row_max = row_values.max()
        row_mean = row_values.mean()
        row_median = row_values.median()

        # Rank of diagonal within the row (1 = highest)
        rank = int((row_values >= diag_val).sum())

        results.append({
            "discouraged_ability": disc_ability,
            "diagonal_value": diag_val,
            "row_max": row_max,
            "row_mean": row_mean,
            "row_median": row_median,
            "diagonal_rank": rank,
            "is_max": diag_val == row_max,
            "above_mean": diag_val > row_mean,
            "above_median": diag_val > row_median,
        })

    return pd.DataFrame(results)


def plot_directionality_heatmap(D: pd.DataFrame, D_pval: pd.DataFrame,
                                alpha: float, label: str, out_dir: str):
    """Plot the directionality matrix as a heatmap with significance markers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Mark significant cells
    annot = D.round(2).astype(str)
    for i, row_ab in enumerate(ABILITIES):
        for j, col_ab in enumerate(ABILITIES):
            pval = D_pval.loc[row_ab, col_ab]
            val = D.loc[row_ab, col_ab]
            if pval < 0.001:
                annot.iloc[i, j] = f"{val:.2f}***"
            elif pval < 0.01:
                annot.iloc[i, j] = f"{val:.2f}**"
            elif pval < 0.05:
                annot.iloc[i, j] = f"{val:.2f}*"
            else:
                annot.iloc[i, j] = f"{val:.2f}"

    vmax = max(abs(D.values.min()), abs(D.values.max()))
    if vmax == 0:
        vmax = 1

    sns.heatmap(
        D, ax=ax, cmap="RdYlGn", center=0,
        vmin=-vmax, vmax=vmax,
        annot=annot, fmt="", annot_kws={"size": 7},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean level difference (flipped - stayed_correct)"},
    )

    # Highlight diagonal
    for i in range(len(ABILITIES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor="black", linewidth=2.5))

    ax.set_title(
        f"Directionality Matrix — {label} (α={alpha})\n"
        f"Each cell: mean ability level of flipped questions minus stayed-correct questions\n"
        f"Diagonal highlighted | * p<0.05, ** p<0.01, *** p<0.001",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Ability Level (of questions)", fontsize=11)
    ax.set_ylabel("Discouraged Ability", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"directionality_{label}_a{alpha}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_flip_profile(neutral_df: pd.DataFrame, disc_df: pd.DataFrame,
                      disc_ability: str, alpha: float, label: str, out_dir: str):
    """
    For a single discouraged ability, plot the level distribution
    of flipped vs non-flipped questions across all 18 abilities.
    """
    merged = identify_flips(neutral_df, disc_df, disc_ability, alpha)
    if merged is None:
        return

    pos_to_neg = merged[merged["flip_type"] == "pos_to_neg"]
    stayed_correct = merged[merged["flip_type"] == "stayed_correct"]

    if len(pos_to_neg) < 3:
        print(f"  Skipping {disc_ability}: only {len(pos_to_neg)} flips")
        return

    # Compute mean levels for each group
    means_flipped = []
    means_stayed = []
    for ab in ABILITIES:
        col = f"level_{ab}"
        means_flipped.append(pos_to_neg[col].mean())
        means_stayed.append(stayed_correct[col].mean())

    x = np.arange(len(ABILITIES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width / 2, means_flipped, width,
                    label=f"Flipped (correct→wrong, n={len(pos_to_neg)})",
                    color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width / 2, means_stayed, width,
                    label=f"Stayed correct (n={len(stayed_correct)})",
                    color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ABILITIES, rotation=45, ha="right")

    # Highlight the discouraged ability on x-axis
    disc_idx = ABILITIES.index(disc_ability)
    fig.canvas.draw()  # force render so tick labels exist
    ax.get_xticklabels()[disc_idx].set_fontweight("bold")
    ax.axvline(x=disc_idx, color="blue", linestyle="--", alpha=0.3, linewidth=2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"flip_profile_{disc_ability}_{label}_a{alpha}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal_summary(diag_df: pd.DataFrame, alpha: float,
                           label: str, out_dir: str):
    """Plot summary of diagonal dominance across all discouraged abilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: diagonal value vs row mean
    ax = axes[0]
    x = np.arange(len(diag_df))
    ax.bar(x, diag_df["diagonal_value"], color="#e74c3c", alpha=0.8,
           label="Diagonal (target ability)")
    ax.bar(x, diag_df["row_mean"], color="#95a5a6", alpha=0.5,
           label="Row mean (all abilities)")
    ax.set_xticks(x)
    ax.set_xticklabels(diag_df["discouraged_ability"], rotation=45, ha="right")
    ax.set_ylabel("Level difference (flipped - stayed)")
    ax.set_title("Diagonal vs Row Mean", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: diagonal rank distribution
    ax = axes[1]
    ranks = diag_df["diagonal_rank"]
    ax.hist(ranks, bins=range(1, 20), color="#3498db", alpha=0.8,
            edgecolor="white")
    ax.axvline(x=1, color="red", linestyle="--", label="Rank 1 (best)")
    ax.set_xlabel("Rank of diagonal within row (1=highest)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"How often is the diagonal the strongest signal?\n"
        f"Rank 1 count: {(ranks == 1).sum()}/{len(ranks)}",
        fontsize=12, fontweight="bold"
    )
    ax.legend()

    fig.suptitle(
        f"Directionality Summary — {label} (α={alpha})",
        fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"diagonal_summary_{label}_a{alpha}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def print_flip_summary(flip_counts: dict):
    """Print a table of flip counts per discouraged ability."""
    print("\n" + "=" * 70)
    print(f"{'Ability':<10} {'→Wrong':<10} {'→Correct':<10} "
          f"{'Stayed✓':<10} {'Stayed✗':<10} {'Flip Rate':<10}")
    print("-" * 70)
    for ab in ABILITIES:
        if ab not in flip_counts:
            continue
        c = flip_counts[ab]
        p2n = c.get("pos_to_neg", 0)
        n2p = c.get("neg_to_pos", 0)
        sc = c.get("stayed_correct", 0)
        sw = c.get("stayed_wrong", 0)
        total = p2n + n2p + sc + sw
        flip_rate = (p2n + n2p) / total * 100 if total > 0 else 0
        print(f"{ab:<10} {p2n:<10} {n2p:<10} {sc:<10} {sw:<10} {flip_rate:<.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Flip analysis for Ripple Effect")
    parser.add_argument("--detail", required=True, help="Path to detail_predictions.csv")
    parser.add_argument("--alpha", type=float, default=-2.0, help="Alpha value to analyze")
    parser.add_argument("--label", default="Model", help="Label for output files")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--profiles", action="store_true",
                        help="Also generate per-ability flip profile plots")
    args = parser.parse_args()

    if args.out is None:
        args.out = FIGURES_DEFAULT
    os.makedirs(args.out, exist_ok=True)

    # Load data
    neutral_df, disc_df = load_and_prepare(args.detail)

    # Check if requested alpha exists
    available_alphas = sorted(disc_df["alpha"].unique())
    if args.alpha not in available_alphas:
        print(f"Alpha {args.alpha} not found. Available: {available_alphas}")
        return

    print(f"\n--- Analyzing alpha={args.alpha} ---\n")

    # Step 1: Compute directionality matrix
    print("Computing directionality matrix...", flush=True)
    D, D_pval, flip_counts = compute_directionality_matrix(
        neutral_df, disc_df, args.alpha
    )

    # Step 2: Print flip summary
    print_flip_summary(flip_counts)

    # Step 3: Compute diagonal dominance
    diag_df = compute_diagonal_dominance(D)
    print("\nDiagonal dominance summary:")
    print(diag_df[["discouraged_ability", "diagonal_value", "row_mean",
                    "diagonal_rank", "is_max", "above_mean"]].to_string(index=False))

    count_is_max = diag_df["is_max"].sum()
    count_above_mean = diag_df["above_mean"].sum()
    print(f"\nDiagonal is row maximum: {count_is_max}/{len(diag_df)}")
    print(f"Diagonal above row mean: {count_above_mean}/{len(diag_df)}")

    # Step 4: Save directionality matrix CSV
    D.to_csv(os.path.join(args.out, f"directionality_{args.label}_a{args.alpha}.csv"))
    D_pval.to_csv(os.path.join(args.out, f"directionality_pval_{args.label}_a{args.alpha}.csv"))

    # Step 5: Plot directionality heatmap
    plot_directionality_heatmap(D, D_pval, args.alpha, args.label, args.out)

    # Step 6: Plot diagonal summary
    plot_diagonal_summary(diag_df, args.alpha, args.label, args.out)

    # Step 7: Optionally plot individual flip profiles
    if args.profiles:
        print("\nGenerating per-ability flip profiles...", flush=True)
        for disc_ability in ABILITIES:
            plot_flip_profile(neutral_df, disc_df, disc_ability,
                              args.alpha, args.label, args.out)

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
