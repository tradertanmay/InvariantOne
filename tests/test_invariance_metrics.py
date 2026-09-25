"""
Unit tests for invariance, TVD, JSD, and max probability delta metrics.
"""

from invariantone.metrics.invariance import (
    is_permutation_equivariant,
    jensen_shannon_divergence,
    max_probability_delta,
    total_variation_distance,
)


def test_tvd_and_jsd_identical_distributions():
    p1 = {"A": 0.7, "B": 0.2, "C": 0.1}
    p2 = {"A": 0.7, "B": 0.2, "C": 0.1}

    assert total_variation_distance(p1, p2) == 0.0
    assert jensen_shannon_divergence(p1, p2) == 0.0
    assert max_probability_delta(p1, p2) == 0.0
    assert is_permutation_equivariant(p1, p2) is True


def test_tvd_and_jsd_disjoint_distributions():
    p1 = {"A": 1.0, "B": 0.0}
    p2 = {"A": 0.0, "B": 1.0}

    assert total_variation_distance(p1, p2) == 1.0
    assert jensen_shannon_divergence(p1, p2) == 1.0
    assert max_probability_delta(p1, p2) == 1.0
    assert is_permutation_equivariant(p1, p2) is False
