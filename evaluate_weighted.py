"""
evaluate_weighted.py — Weighted-score evaluation method.

Instead of classifying each question into one ability and computing AUC
per ability test-set, this method:

1. Randomly samples NUM_QUESTIONS from the full ADeLe battery.
2. For each discouraged ability, runs contrastive decoding on ALL sampled
   questions (not per-ability subsets).
3. If the model answers correctly, the question's level values for ALL 18
   abilities are added to the corresponding ability scores.
4. The final 18×18 heatmap shows the delta between discouraged scores and
   neutral baseline scores (as percentage change).

Advantages:
- No hard classification of questions into abilities
- Same question pool across all discourage conditions → cleaner comparison
- Every question contributes to all abilities proportionally via level weights
- Faster than the original method (~9000 vs ~26000 discourage inferences)
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from config_weighted import (
    ABILITIES, ALPHAS, NUM_QUESTIONS, RANDOM_SEED,
    NEUTRAL_PROMPT, DISCOURAGE_PROMPTS,
    ADELE_CSV, RESULTS_DIR,
)
from model import build_prompt, get_next_logits
from decoding import apply_contrastive_decoding


RESULTS_CSV = os.path.join(RESULTS_DIR, "results_weighted.csv")
NEUTRAL_CACHE_FILE = os.path.join(RESULTS_DIR, "neutral_cache_weighted.pkl")


# ── Data loading ─────────────────────────────────────────────────────────────

UG_THRESHOLD = 75

def load_questions() -> pd.DataFrame:
    """
    Load the full ADeLe battery, keep only MCQs with UG >= 75,
    and randomly sample NUM_QUESTIONS rows.
    Uses RANDOM_SEED for reproducibility.

    Filtering criteria (same as prepare_data.py):
      - answer_format == "MC"
      - UG >= 75 (unguessability threshold)
    """
    df = pd.read_csv(ADELE_CSV)

    # Verify that all ability columns exist
    missing = [a for a in ABILITIES if a not in df.columns]
    if missing:
        raise ValueError(f"Missing ability columns in CSV: {missing}")

    total = len(df)

    # Filter: multiple-choice only
    df = df[df["answer_format"] == "MC"].copy()
    print(f"MC filter: {len(df)} / {total} questions", flush=True)

    # Filter: unguessability threshold
    df = df[df["UG"] >= UG_THRESHOLD].copy()
    print(f"UG >= {UG_THRESHOLD} filter: {len(df)} questions remain", flush=True)

    # Sample
    if NUM_QUESTIONS < len(df):
        df = df.sample(n=NUM_QUESTIONS, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        print(f"Warning: requested {NUM_QUESTIONS} but only {len(df)} available. Using all.")
        df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} questions for evaluation", flush=True)
    return df


# ── Neutral logit cache ──────────────────────────────────────────────────────

def build_neutral_cache(df: pd.DataFrame) -> list:
    """
    Pre-compute neutral logits for every question.
    Returns a list of tensors, one per question (same order as df).
    """
    if os.path.exists(NEUTRAL_CACHE_FILE):
        print("Loading neutral cache from disk...", flush=True)
        with open(NEUTRAL_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if len(cache) == len(df):
            return cache
        print("Cache size mismatch — rebuilding.", flush=True)

    print("Building neutral cache...", flush=True)
    cache = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Neutral logits"):
        prompt = build_prompt(NEUTRAL_PROMPT, row["question"])
        cache.append(get_next_logits(prompt))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(NEUTRAL_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("Neutral cache saved.", flush=True)
    return cache


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_completed_discouraged() -> set:
    """Return discouraged abilities already fully saved."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    expected = len(ALPHAS) * len(ABILITIES)
    counts = df.groupby("discouraged_ability").size()
    return set(counts[counts >= expected].index.tolist())


def append_results(rows: list[dict]):
    """Append result rows to CSV."""
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df_new.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, index=False)


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_weighted_scores(
    df: pd.DataFrame,
    predictions: list[str],
) -> dict[str, float]:
    """
    For each ability, sum the level values of correctly answered questions.

    Args:
        df: question dataframe with ability level columns
        predictions: list of predicted answers (same order as df)

    Returns:
        dict mapping ability name → weighted score
    """
    scores = {ability: 0.0 for ability in ABILITIES}

    for idx, (_, row) in enumerate(df.iterrows()):
        gt = str(row["groundtruth"]).strip()
        if not gt:
            continue
        answer = gt[0].upper()
        pred = predictions[idx]

        if pred == answer:
            # Correct: add level values for all abilities
            for ability in ABILITIES:
                level = row.get(ability, 0)
                if pd.notna(level):
                    scores[ability] += float(level)

    return scores


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(df: pd.DataFrame, neutral_cache: list):
    """
    For each discouraged ability:
      1. Compute discourage logits for all questions
      2. For each alpha, apply contrastive decoding → get predictions
      3. Compute weighted scores for all 18 abilities
      4. Save to CSV
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # First compute neutral baseline scores (alpha=0 equivalent)
    print("\nComputing neutral baseline predictions...", flush=True)
    from model import get_tokenizer
    tokenizer = get_tokenizer()

    neutral_preds = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Neutral preds")):
        gt = str(row["groundtruth"]).strip()
        if not gt:
            neutral_preds.append("")
            continue
        # At alpha=0, logits_final = logits_neutral (penalized term cancels out)
        ln = neutral_cache[idx]
        # ln shape: [1, vocab_size] — squeeze to [vocab_size] for argmax
        token_id = ln.squeeze(0).argmax(dim=-1).item()
        neutral_preds.append(tokenizer.decode([token_id]).strip().upper())

    neutral_scores = compute_weighted_scores(df, neutral_preds)
    print("\nNeutral baseline scores:", flush=True)
    for ability, score in neutral_scores.items():
        print(f"  {ability}: {score:.1f}", flush=True)

    # Now loop over discouraged abilities
    completed = load_completed_discouraged()
    if completed:
        print(f"\nResuming — skipping {len(completed)} completed: {sorted(completed)}", flush=True)

    for disc_ability, disc_prompt in DISCOURAGE_PROMPTS.items():

        if disc_ability in completed:
            print(f"[SKIP] {disc_ability} already done.", flush=True)
            continue

        print(f"\n[DISCOURAGE] {disc_ability}", flush=True)

        # Step 1: compute discourage logits for all questions
        disc_logits = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Disc logits [{disc_ability}]"):
            prompt = build_prompt(disc_prompt, row["question"])
            disc_logits.append(get_next_logits(prompt))

        # Step 2: for each alpha, get predictions and compute scores
        batch = []
        for alpha in ALPHAS:
            predictions = []
            for idx in range(len(df)):
                pred = apply_contrastive_decoding(
                    disc_logits[idx], neutral_cache[idx], alpha
                )
                predictions.append(pred)

            scores = compute_weighted_scores(df, predictions)

            # Store one row per (discouraged_ability, alpha, test_ability)
            for test_ability in ABILITIES:
                score = scores[test_ability]
                baseline = neutral_scores[test_ability]

                # Delta as percentage change from neutral baseline
                if baseline > 0:
                    delta_pct = (score - baseline) / baseline * 100
                else:
                    delta_pct = 0.0

                batch.append({
                    "discouraged_ability": disc_ability,
                    "alpha": alpha,
                    "test_ability": test_ability,
                    "score": score,
                    "baseline": baseline,
                    "delta_pct": delta_pct,
                })

        append_results(batch)
        print(f"  [SAVED] {disc_ability} → {RESULTS_CSV}", flush=True)

    print(f"\nAll done. Results in {RESULTS_CSV}", flush=True)
