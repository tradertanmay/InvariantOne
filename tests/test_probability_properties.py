"""
Tests for Mathematical Probability Properties in InvariantOne.
"""

from __future__ import annotations

import math
import pytest
import torch

from invariantone.calibration import calibrated_softmax
from invariantone.result import DecisionResult


def test_probability_distribution_properties():
    """Verifies that calibrated_softmax produces normalized, non-negative, finite probabilities."""
    scores = torch.tensor([-5.2, 0.4, 3.1, -1.8], dtype=torch.float32)
    probs = calibrated_softmax(scores, tau=1.0091)

    # 1. Sums to 1.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    # 2. Non-negative
    assert torch.all(probs >= 0.0)

    # 3. Finite
    assert torch.all(torch.isfinite(probs))

    # 4. Monotonicity: higher score must yield strictly higher probability
    assert probs[2] > probs[1] > probs[3] > probs[0]


def test_reject_nan_inf_scores():
    """Verifies that NaN or Inf in scores raises an explicit ValueError."""
    nan_tensor = torch.tensor([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        calibrated_softmax(nan_tensor)

    inf_tensor = torch.tensor([1.0, float("inf"), 2.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        calibrated_softmax(inf_tensor)

    nan_list = [1.0, float("nan"), 2.0]
    with pytest.raises(ValueError, match="not finite"):
        calibrated_softmax(nan_list)


def test_reject_nonpositive_temperature():
    """Verifies that non-positive calibration temperature raises ValueError."""
    scores = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        calibrated_softmax(scores, tau=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        calibrated_softmax(scores, tau=-1.5)


def test_decision_result_validation():
    """Verifies that DecisionResult raises ValueError if invariants are violated."""
    # Valid result
    res = DecisionResult(
        choice="Option B",
        choice_index=1,
        probabilities=[0.2, 0.8],
        scores=[-1.0, 1.0],
        options=["Option A", "Option B"],
    )
    assert res.choice == "Option B"
    assert res.choice_index == 1

    # Invalid choice index
    with pytest.raises(ValueError, match="out of bounds"):
        DecisionResult(
            choice="Option B",
            choice_index=5,
            probabilities=[0.2, 0.8],
            scores=[-1.0, 1.0],
            options=["Option A", "Option B"],
        )

    # Probabilities don't sum to 1.0
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        DecisionResult(
            choice="Option A",
            choice_index=0,
            probabilities=[0.2, 0.2],
            scores=[-1.0, -1.0],
            options=["Option A", "Option B"],
        )
