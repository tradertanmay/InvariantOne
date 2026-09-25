"""
Unit tests for calibration diagnostics module.
"""

from invariantone.metrics.calibration_diagnostics import (
    compute_full_calibration_diagnostics,
    compute_margin_distribution,
    compute_reliability_diagram,
)


def test_reliability_diagram():
    probs = [
        {"A": 0.9, "B": 0.1},
        {"A": 0.8, "B": 0.2},
        {"A": 0.4, "B": 0.6},
        {"A": 0.3, "B": 0.7},
    ]
    targets = ["A", "A", "B", "A"]  # 3 correct, 1 incorrect
    rel = compute_reliability_diagram(probs, targets, num_bins=5)
    assert len(rel["bins"]) == 5
    assert rel["total_samples"] == 4
    assert 0.0 <= rel["ece"] <= 1.0
    assert 0.0 <= rel["mce"] <= 1.0


def test_margin_distribution():
    scores = [
        {"A": 2.5, "B": 1.0},
        {"A": 0.5, "B": 1.5},
    ]
    targets = ["A", "A"]
    margins = compute_margin_distribution(scores, targets)
    assert margins["count"] == 2
    assert margins["min"] == -1.0
    assert margins["max"] == 1.5
    assert margins["fraction_positive"] == 0.5


def test_full_diagnostics():
    probs = [
        {"billing": 0.8, "tech": 0.2},
        {"billing": 0.1, "tech": 0.9},
    ]
    targets = ["billing", "tech"]
    diag = compute_full_calibration_diagnostics(probs, targets)
    assert "expected_calibration_error" in diag
    assert "reliability_bins" in diag
    assert "probability_margins" in diag
    assert "entropy_distribution" in diag
