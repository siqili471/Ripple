"""
decoding.py — Contrastive decoding and AUC computation.

Formula:
    logits_final = logits_neutral + α * (logits_neutral - logits_penalized)

    α = 0   → pure neutral prompt output
    α < 0   → discourage: push logits toward penalized distribution
    α > 0   → encourage:  push logits away from penalized distribution
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from model import get_model
from config import TEMPERATURE


def apply_contrastive_decoding(
    logits_penalized: torch.Tensor,
    logits_neutral: torch.Tensor,
    alpha: float,
) -> str:
    """
    Combine penalized and neutral logits via contrastive decoding and
    return the predicted answer token (single uppercase letter).

    Args:
        logits_penalized: next-token logits from the discourage prompt (lp)
        logits_neutral:   next-token logits from the neutral prompt    (ln)
        alpha:            steering strength
            α = 0  → pure neutral
            α < 0  → push toward discouraged distribution
            α > 0  → push away from discouraged distribution

    Returns:
        Predicted token as a single uppercase string, e.g. "A".
    """
    model = get_model()
    lp = logits_penalized.to(model.device)
    ln = logits_neutral.to(model.device)

    logits_final = ln + alpha * (ln - lp)

    if TEMPERATURE == 0:
        token_id = torch.argmax(logits_final, dim=-1).item()
    else:
        probs = torch.softmax(logits_final / TEMPERATURE, dim=-1)
        token_id = torch.multinomial(probs, 1).item()

    from model import get_tokenizer
    return get_tokenizer().decode([token_id]).strip().upper()


def _compute_sample_weights(difficulty_levels: list, correct_labels: list):
    """
    Weight each question inversely proportional to how many other questions
    share its difficulty level, so all difficulty levels contribute equally
    to the logistic fit.

    Also appends an anchor point at difficulty=20, label=0 to ensure the
    fitted curve converges to zero at high difficulty.
    """
    X = np.array(difficulty_levels).reshape(-1, 1)
    y = np.array(correct_labels, dtype=float)

    unique_levels, counts = np.unique(X, return_counts=True)
    freq_map = dict(zip(unique_levels, counts))
    weights = np.array([1.0 / freq_map[x[0]] for x in X])
    weights = weights / weights.sum() * len(unique_levels)

    anchor_weight = weights.sum()
    X_train = np.vstack([X, [[20]]])
    y_train = np.concatenate([y, [0]])
    sample_weights = np.concatenate([weights, [anchor_weight]])

    return X_train, y_train, sample_weights


def compute_auc(difficulty_levels: list, correct_labels: list) -> float:
    """
    Fit a logistic regression curve of accuracy vs. difficulty level,
    then return the area under that curve over [0, 10].

    This follows the ADeLe paper methodology: the AUC summarises the
    model's ability on a given dimension — higher = better performance
    across all difficulty levels.

    Returns np.nan if fitting fails or there is insufficient label variance.
    """
    if len(set(correct_labels)) < 2:
        return np.nan

    X_train, y_train, sample_weights = _compute_sample_weights(
        difficulty_levels, correct_labels
    )

    try:
        lr = LogisticRegression(random_state=42, max_iter=10000)
        lr.fit(X_train, y_train, sample_weight=sample_weights)
        x_eval = np.linspace(0, 10, 1001)
        p_correct = lr.predict_proba(x_eval.reshape(-1, 1))[:, 1]
        return float(np.trapz(p_correct, x_eval))
    except Exception as e:
        print(f"    [AUC] Logistic regression failed: {e}")
        return np.nan
