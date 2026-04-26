"""
config_roleplay.py — Configuration for the role-play prompt experiment.

Same weighted-score method, but uses role-play prompts instead of
direct prohibition prompts to test if specificity improves.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ── Experiment ───────────────────────────────────────────────────────────────
ALPHAS = [-3.0, -2.0, -1.5, -1.0, 0.0, 0.5]
TEMPERATURE = 0
NUM_QUESTIONS = 500
RANDOM_SEED = 42         # same seed → same 500 questions as before

# ── Paths ────────────────────────────────────────────────────────────────────
ADELE_CSV       = "data/ADeLe_batterry_v1dot0.csv"
RESULTS_DIR     = "results_roleplay_7b"
FIGURES_DIR     = "figures_roleplay_7b"

# ── Abilities ────────────────────────────────────────────────────────────────
ABILITIES = [
    "AS", "CEc", "CEe", "CL",
    "MCr", "MCt", "MCu", "MS",
    "QLl", "QLq", "SNs",
    "KNa", "KNc", "KNf", "KNn", "KNs",
    "AT", "VO",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
NEUTRAL_PROMPT = "You are a helpful assistant."

from prompts_roleplay import ROLEPLAY_DISCOURAGE_PROMPTS as DISCOURAGE_PROMPTS
