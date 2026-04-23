"""
run_weighted.py — Entry point for the weighted-score experiment.

Usage:
    python run_weighted.py
    python run_weighted.py --visualize-only
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Ripple weighted-score experiment")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Skip inference, only regenerate heatmaps")
    args = parser.parse_args()

    # Patch config BEFORE anything imports model.py
    # model.py does "from config import MODEL_NAME" at module level,
    # so we must modify config's attributes before that import happens.
    import config_weighted as cfg
    import config
    config.MODEL_NAME = cfg.MODEL_NAME
    config.NEUTRAL_PROMPT = cfg.NEUTRAL_PROMPT
    config.TEMPERATURE = cfg.TEMPERATURE

    if not args.visualize_only:
        # Now it's safe to import model (which reads from config)
        from model import load_model
        from evaluate_weighted import load_questions, build_neutral_cache, run_evaluation

        # model.py binds MODEL_NAME at import via "from config import MODEL_NAME"
        # so we also need to patch model's own reference
        import model
        model.MODEL_NAME = cfg.MODEL_NAME

        load_model()
        df = load_questions()
        neutral_cache = build_neutral_cache(df)
        run_evaluation(df, neutral_cache)

    # Generate visualizations
    from visualize_weighted import generate_heatmaps
    generate_heatmaps()


if __name__ == "__main__":
    main()
