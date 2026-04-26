"""
rescore.py — Re-compute weighted scores from detail_predictions.csv
using different scoring methods. No GPU needed.

Usage:
    python rescore.py --detail results_weighted_7b/detail_predictions.csv --method threshold --threshold 2 --label "Qwen-7B-thresh2"
    python rescore.py --detail results_weighted_7b/detail_predictions.csv --method top_only --label "Qwen-7B-toponly"
    python rescore.py --detail results_weighted_7b/detail_predictions.csv --method original --label "Qwen-7B-original"

Scoring methods:
    original   — Same as before: add all level values for correct answers
    threshold  — Only add level values >= threshold (default 2)
    top_only   — Only add the level of the highest ability for each question
    normalized — Divide each level by the sum of all levels for that question
"""

import argparse
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config_weighted import ABILITIES

FIGURES_DEFAULT = "figures_rescore"


def rescore(detail_df: pd.DataFrame, method: str, threshold: int = 2) -> pd.DataFrame:
    """
    Re-compute scores from per-question detail data.

    Returns a DataFrame with columns:
        discouraged_ability, alpha, test_ability, score, baseline, delta_pct
    """
    level_cols = [f"level_{a}" for a in ABILITIES]

    # Split neutral and discouraged
    neutral_df = detail_df[detail_df["discouraged_ability"] == "NEUTRAL"].copy()
    disc_df = detail_df[detail_df["discouraged_ability"] != "NEUTRAL"].copy()

    def compute_scores(sub_df: pd.DataFrame) -> dict:
        """Compute ability scores for a set of predictions."""
        scores = {a: 0.0 for a in ABILITIES}
        for _, row in sub_df.iterrows():
            if row["correct"] != 1:
                continue
            for ability in ABILITIES:
                level = row[f"level_{ability}"]
                if pd.isna(level):
                    continue
                level = int(level)

                if method == "original":
                    scores[ability] += level
                elif method == "threshold":
                    if level >= threshold:
                        scores[ability] += level
                elif method == "top_only":
                    # Only add if this ability has the max level for this question
                    levels = {a: int(row[f"level_{a}"]) for a in ABILITIES if pd.notna(row[f"level_{a}"])}
                    max_level = max(levels.values())
                    if level == max_level and level > 0:
                        scores[ability] += level
                elif method == "normalized":
                    levels = [int(row[f"level_{a}"]) for a in ABILITIES if pd.notna(row[f"level_{a}"])]
                    total = sum(levels)
                    if total > 0 and level > 0:
                        scores[ability] += level / total
        return scores

    # Compute neutral baseline
    baseline_scores = compute_scores(neutral_df)

    # Compute scores for each (discouraged_ability, alpha) pair
    results = []
    for (disc_ab, alpha), group in disc_df.groupby(["discouraged_ability", "alpha"]):
        scores = compute_scores(group)
        for test_ab in ABILITIES:
            score = scores[test_ab]
            baseline = baseline_scores[test_ab]
            if baseline > 0:
                delta_pct = (score - baseline) / baseline * 100
            else:
                delta_pct = 0.0
            results.append({
                "discouraged_ability": disc_ab,
                "alpha": alpha,
                "test_ability": test_ab,
                "score": score,
                "baseline": baseline,
                "delta_pct": delta_pct,
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Re-score from detail predictions")
    parser.add_argument("--detail", required=True, help="Path to detail_predictions.csv")
    parser.add_argument("--method", default="original",
                        choices=["original", "threshold", "top_only", "normalized"])
    parser.add_argument("--threshold", type=int, default=2, help="Threshold for threshold method")
    parser.add_argument("--label", default=None, help="Label for output files")
    parser.add_argument("--out", default=None, help="Output directory for figures")
    args = parser.parse_args()

    if args.label is None:
        args.label = f"{args.method}"
    if args.out is None:
        args.out = FIGURES_DEFAULT

    print(f"Loading detail predictions from {args.detail}...", flush=True)
    detail_df = pd.read_csv(args.detail)
    print(f"  {len(detail_df)} rows loaded", flush=True)

    print(f"Rescoring with method={args.method}, threshold={args.threshold}...", flush=True)
    results_df = rescore(detail_df, args.method, args.threshold)

    # Save rescored results
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, f"results_{args.label}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Rescored results saved to {results_path}", flush=True)

    # Generate heatmaps
    from visualize_weighted import generate_heatmaps
    generate_heatmaps(results_csv=results_path, label=args.label, out_dir=args.out)


if __name__ == "__main__":
    main()
