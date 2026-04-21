"""
evaluate.py — Main evaluation loop with checkpoint/resume support.

Runs the triple-nested loop:
    Outer : discouraged_ability  (18)
    Middle: test_ability         (18)
    Inner : alpha values         (4)

Results are saved incrementally after each discouraged_ability so that
a server crash never loses more than one outer iteration.
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    ABILITIES, ALPHAS, MAX_QUESTIONS,
    NEUTRAL_PROMPT, DISCOURAGE_PROMPTS,
    QUESTIONS_DIR, RESULTS_DIR,
)
from model import build_prompt, get_next_logits
from decoding import apply_contrastive_decoding, compute_auc


RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
NEUTRAL_CACHE_FILE = os.path.join(RESULTS_DIR, "neutral_cache.pkl")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ability_dataframes() -> dict[str, pd.DataFrame]:
    """Load up to MAX_QUESTIONS rows from each ability CSV."""
    csv_files = sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.csv")))
    csv_files = [f for f in csv_files if not os.path.basename(f) == "UG.csv"]

    ability_dfs = {}
    for path in csv_files:
        ability = os.path.basename(path).replace(".csv", "")
        if ability in ABILITIES:
            ability_dfs[ability] = pd.read_csv(path).head(MAX_QUESTIONS)

    print(f"Loaded {len(ability_dfs)} ability CSVs.", flush=True)
    return ability_dfs


# ── Neutral logit cache ───────────────────────────────────────────────────────

def build_neutral_cache(ability_dfs: dict) -> dict:
    """
    Pre-compute neutral logits for every question in every ability CSV.
    Cached to disk so re-runs skip this expensive step.
    """
    import pickle

    if os.path.exists(NEUTRAL_CACHE_FILE):
        print("Loading neutral cache from disk...", flush=True)
        with open(NEUTRAL_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Building neutral cache...", flush=True)
    neutral_cache = {}
    for ability, df in tqdm(ability_dfs.items(), desc="Neutral cache"):
        neutral_cache[ability] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {ability}", leave=False):
            prompt = build_prompt(NEUTRAL_PROMPT, row["question"])
            neutral_cache[ability].append(get_next_logits(prompt))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(NEUTRAL_CACHE_FILE, "wb") as f:
        pickle.dump(neutral_cache, f)
    print("Neutral cache saved.", flush=True)

    return neutral_cache


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_completed_discouraged() -> set:
    """Return the set of discouraged_ability values already saved to CSV."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    # A discouraged ability is complete only if all alphas x all test abilities are present
    expected = len(ALPHAS) * len(ABILITIES)
    counts = df.groupby("discouraged_ability").size()
    return set(counts[counts >= expected].index.tolist())


def append_results(rows: list[dict]):
    """Append a batch of result rows to the CSV (create if not exists)."""
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df_new.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, index=False)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(ability_dfs: dict, neutral_cache: dict):
    """
    Triple-nested evaluation loop with checkpoint/resume.
    Results are appended to RESULTS_CSV after each discouraged ability.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = load_completed_discouraged()
    if completed:
        print(f"Resuming — skipping {len(completed)} completed abilities: {sorted(completed)}", flush=True)

    for discouraged_ability, discourage_prompt in DISCOURAGE_PROMPTS.items():

        if discouraged_ability in completed:
            print(f"[SKIP] {discouraged_ability} already done.", flush=True)
            continue

        print(f"\n[DISCOURAGE] {discouraged_ability}", flush=True)

        # Cache discourage logits for all test ability CSVs
        discourage_cache = {}
        for test_ability, df in tqdm(
            ability_dfs.items(), desc=f"Discourage logits [{discouraged_ability}]"
        ):
            discourage_cache[test_ability] = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {test_ability}", leave=False):
                prompt = build_prompt(discourage_prompt, row["question"])
                discourage_cache[test_ability].append(get_next_logits(prompt))

        # Evaluate all (alpha, test_ability) pairs
        batch = []
        for alpha in ALPHAS:
            for test_ability, df in ability_dfs.items():

                # The difficulty column for this test_ability must exist in the CSV
                if test_ability not in df.columns:
                    batch.append({
                        "discouraged_ability": discouraged_ability,
                        "alpha": alpha,
                        "test_ability": test_ability,
                        "auc": np.nan,
                    })
                    continue

                difficulty_levels = []
                correct_labels = []

                for idx, (_, row) in enumerate(df.iterrows()):
                    # Skip rows where difficulty level is missing
                    if pd.isna(row[test_ability]):
                        continue
                    # Guard against malformed groundtruth
                    gt = str(row["groundtruth"]).strip()
                    if not gt:
                        continue
                    answer = gt[0].upper()
                    lp = discourage_cache[test_ability][idx]
                    ln = neutral_cache[test_ability][idx]
                    pred = apply_contrastive_decoding(lp, ln, alpha)
                    correct_labels.append(int(pred == answer))
                    difficulty_levels.append(int(row[test_ability]))

                auc = compute_auc(difficulty_levels, correct_labels)
                batch.append({
                    "discouraged_ability": discouraged_ability,
                    "alpha": alpha,
                    "test_ability": test_ability,
                    "auc": auc,
                })

        append_results(batch)
        print(f"  [SAVED] {discouraged_ability} → {RESULTS_CSV}", flush=True)

    print(f"\nAll done. Results in {RESULTS_CSV}", flush=True)
