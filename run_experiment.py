"""
run_experiment.py — Main entry point for the Ripple experiment.

Steps:
  1. Load model
  2. Load classified question CSVs
  3. Build (or load cached) neutral logits
  4. Run contrastive decoding evaluation with checkpoint/resume
  5. Generate heatmap visualisations

Usage:
    # Full run
    python run_experiment.py --label "Qwen2.5-7B_Prompt1"

    # Visualise only (results CSV already exists)
    python run_experiment.py --label "Qwen2.5-7B_Prompt1" --visualize-only
"""

import argparse
import os
from config import RESULTS_DIR, FIGURES_DIR


def main():
    parser = argparse.ArgumentParser(description="Ripple: Contrastive Decoding Experiment")
    parser.add_argument(
        "--label", required=True,
        help="Experiment label used in filenames and plot titles, e.g. 'Qwen2.5-7B_Prompt1'"
    )
    parser.add_argument(
        "--visualize-only", action="store_true",
        help="Skip inference and only regenerate heatmaps from existing results CSV"
    )
    args = parser.parse_args()

    results_csv = os.path.join(RESULTS_DIR, "results.csv")

    # ── Visualise only ────────────────────────────────────────────────────────
    if args.visualize_only:
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"No results CSV found at {results_csv}")
        from visualize import plot_heatmaps
        plot_heatmaps(results_csv, label=args.label, out_dir=FIGURES_DIR)
        return

    # ── Full experiment run ───────────────────────────────────────────────────
    from model import load_model
    from evaluate import load_ability_dataframes, build_neutral_cache, run_evaluation
    from visualize import plot_heatmaps

    # 1. Load model
    load_model()

    # 2. Load data
    ability_dfs = load_ability_dataframes()

    # 3. Build/load neutral cache
    neutral_cache = build_neutral_cache(ability_dfs)

    # 4. Run evaluation (resumes automatically if partially done)
    run_evaluation(ability_dfs, neutral_cache)

    # 5. Visualise
    plot_heatmaps(results_csv, label=args.label, out_dir=FIGURES_DIR)


if __name__ == "__main__":
    main()
