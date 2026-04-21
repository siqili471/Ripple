"""
prepare_data.py — Filter ADeLe battery and classify questions by ability.

For each ability, selects questions where that ability has the highest
difficulty score (i.e., it is the "primary demand" of the question).
Only multiple-choice questions with UG >= 75 are kept.

Usage:
    python prepare_data.py
"""

import os
import pandas as pd
from config import ABILITIES, ADELE_CSV, QUESTIONS_DIR

UG_THRESHOLD = 75


def classify_questions(input_csv: str, output_dir: str):
    print(f"Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv)

    # Keep only multiple-choice questions above the unguessability threshold
    df = df[df["answer_format"] == "MC"]
    df = df[df["UG"] >= UG_THRESHOLD].reset_index(drop=True)
    print(f"MC questions with UG >= {UG_THRESHOLD}: {len(df)}", flush=True)

    def get_dominant_abilities(row) -> list:
        """Return abilities with the highest difficulty score for this question."""
        scores = {a: row[a] for a in ABILITIES if a in row}
        max_score = max(scores.values())
        return [a for a, s in scores.items() if s == max_score]

    df["dominant_abilities"] = df.apply(get_dominant_abilities, axis=1)

    os.makedirs(output_dir, exist_ok=True)

    for ability in ABILITIES:
        subset = df[df["dominant_abilities"].apply(lambda x: ability in x)]
        subset = subset.drop(columns=["dominant_abilities"])
        out_path = os.path.join(output_dir, f"{ability}.csv")
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  {ability}: {len(subset)} questions → {out_path}")

    print("Done.", flush=True)


if __name__ == "__main__":
    classify_questions(ADELE_CSV, QUESTIONS_DIR)
