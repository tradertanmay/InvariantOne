"""
Metamorphic robustness and permutation equivariance metrics for probabilistic decisions.
"""

from __future__ import annotations

import math
from typing import Mapping


def total_variation_distance(p1: Mapping[str, float], p2: Mapping[str, float]) -> float:
    """Compute Total Variation Distance between two categorical distributions."""
    all_keys = set(p1.keys()) | set(p2.keys())
    diff_sum = sum(abs(p1.get(k, 0.0) - p2.get(k, 0.0)) for k in all_keys)
    return round(0.5 * diff_sum, 6)


def jensen_shannon_divergence(p1: Mapping[str, float], p2: Mapping[str, float], eps: float = 1e-12) -> float:
    """Compute symmetric Jensen-Shannon Divergence between two probability distributions."""
    all_keys = set(p1.keys()) | set(p2.keys())

    # Clip and normalize
    q1 = {k: max(eps, p1.get(k, 0.0)) for k in all_keys}
    q2 = {k: max(eps, p2.get(k, 0.0)) for k in all_keys}
    s1, s2 = sum(q1.values()), sum(q2.values())
    q1 = {k: v / s1 for k, v in q1.items()}
    q2 = {k: v / s2 for k, v in q2.items()}

    m = {k: 0.5 * (q1[k] + q2[k]) for k in all_keys}

    kl1 = sum(q1[k] * math.log2(q1[k] / m[k]) for k in all_keys)
    kl2 = sum(q2[k] * math.log2(q2[k] / m[k]) for k in all_keys)

    jsd = 0.5 * (kl1 + kl2)
    return round(max(0.0, float(jsd)), 6)


def max_probability_delta(p1: Mapping[str, float], p2: Mapping[str, float]) -> float:
    """Compute maximum absolute probability delta across all options."""
    all_keys = set(p1.keys()) | set(p2.keys())
    if not all_keys:
        return 0.0
    return round(max(abs(p1.get(k, 0.0) - p2.get(k, 0.0)) for k in all_keys), 6)


def is_permutation_equivariant(
    p_orig: Mapping[str, float],
    p_remapped: Mapping[str, float],
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify numerical structural permutation equivariance.
    Returns True if max probability delta between canonical and remapped probabilities is <= tolerance.
    """
    delta = max_probability_delta(p_orig, p_remapped)
    return delta <= tolerance
