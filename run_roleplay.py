"""
run_roleplay.py — Entry point for the role-play prompt experiment.

Usage:
    python run_roleplay.py
    python run_roleplay.py --visualize-only
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Ripple role-play experiment")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Skip inference, only regenerate heatmaps")
    args = parser.parse_args()

    # Patch configs before any other imports
    import config_roleplay as cfg
    import config
    config.MODEL_NAME = cfg.MODEL_NAME
    config.NEUTRAL_PROMPT = cfg.NEUTRAL_PROMPT
    config.TEMPERATURE = cfg.TEMPERATURE

    # Also patch config_weighted so evaluate_weighted.py reads the right values
    import config_weighted
    config_weighted.MODEL_NAME = cfg.MODEL_NAME
    config_weighted.ALPHAS = cfg.ALPHAS
    config_weighted.NUM_QUESTIONS = cfg.NUM_QUESTIONS
    config_weighted.RANDOM_SEED = cfg.RANDOM_SEED
    config_weighted.ADELE_CSV = cfg.ADELE_CSV
    config_weighted.RESULTS_DIR = cfg.RESULTS_DIR
    config_weighted.FIGURES_DIR = cfg.FIGURES_DIR
    config_weighted.ABILITIES = cfg.ABILITIES
    config_weighted.NEUTRAL_PROMPT = cfg.NEUTRAL_PROMPT
    config_weighted.DISCOURAGE_PROMPTS = cfg.DISCOURAGE_PROMPTS

    if not args.visualize_only:
        from model import load_model
        from evaluate_weighted import load_questions, build_neutral_cache, run_evaluation

        import model
        model.MODEL_NAME = cfg.MODEL_NAME

        load_model()
        df = load_questions()
        neutral_cache = build_neutral_cache(df)
        run_evaluation(df, neutral_cache)

    from visualize_weighted import generate_heatmaps
    generate_heatmaps(
        results_csv=f"{cfg.RESULTS_DIR}/results_weighted.csv",
        label="RolePlay-7B",
        out_dir=cfg.FIGURES_DIR,
    )


if __name__ == "__main__":
    main()
