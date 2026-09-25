"""
Task quality and classification metrics for probabilistic decisions.
Includes Accuracy, Macro F1, Negative Log-Likelihood (NLL), and Brier Score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Mapping


def accuracy_score(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute top-1 classification accuracy."""
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return round(correct / len(targets), 6)


def brier_score_single(probabilities: Mapping[str, float], ground_truth: str) -> float:
    """
    Compute Brier score for a single multi-class probabilistic prediction.
    Brier = (1/K) * sum_{k=1}^K (P(O_k) - I(O_k == y*))^2
    Lower is better (0.0 is perfect, 1.0 is worst).
    """
    if not probabilities:
        return 1.0
    k = len(probabilities)
    sq_err_sum = sum(
        (prob - (1.0 if opt == ground_truth else 0.0)) ** 2
        for opt, prob in probabilities.items()
    )
    return round(sq_err_sum / k, 6)


def brier_score(predictions: Sequence[Mapping[str, float]], targets: Sequence[str]) -> float:
    """Compute mean Brier score across multiple examples."""
    if not targets:
        return 0.0
    total = sum(brier_score_single(prob_map, target) for prob_map, target in zip(predictions, targets))
    return round(total / len(targets), 6)


def negative_log_likelihood(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    eps: float = 1e-12,
) -> float:
    """Compute mean Negative Log-Likelihood (NLL) of the true option."""
    if not targets:
        return 0.0
    total_nll = 0.0
    for prob_map, target in zip(probabilities, targets):
        p_true = max(eps, prob_map.get(target, eps))
        total_nll += -math.log(p_true)
    return round(total_nll / len(targets), 6)


def macro_f1_score(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute unweighted Macro F1 score across all classes present."""
    from sklearn.metrics import f1_score
    if not targets:
        return 0.0
    labels = sorted(list(set(targets) | set(predictions)))
    return round(float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)), 6)
