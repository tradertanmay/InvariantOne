"""
Tests for Dynamic-K Option Count Handling in InvariantOne.
"""

from __future__ import annotations

import pytest
import torch

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.calibration import calibrated_softmax
from invariantone.result import DecisionResult


def test_direct_comparative_head_dynamic_k():
    """Verifies that DirectComparativeHead handles arbitrary K >= 1 without reshaping errors."""
    head = DirectComparativeHead(hidden_size=2560, proj_dim=256, mode="absolute_only")
    head.eval()

    for k in [2, 3, 4, 5, 8, 12]:
        dummy_reps = torch.randn(k, 2560)
        with torch.no_grad():
            scores = head(dummy_reps)
        assert scores.shape == (k,), f"Expected shape ({k},), got {scores.shape}"

        probs = calibrated_softmax(scores)
        assert probs.shape == (k,)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_input_validation_k_less_than_two():
    """Verifies that fewer than 2 options raises ValueError."""
    from invariantone.config import InvariantOneConfig
    from invariantone.model import InvariantOne

    # Minimal mock instance to test validation logic
    cfg = InvariantOneConfig()
    head = DirectComparativeHead()
    dummy_model = torch.nn.Identity()
    dummy_tok = None

    model = InvariantOne(
        config=cfg,
        model=dummy_model,
        head=head,
        tokenizer=dummy_tok,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="at least 2"):
        model.decide(
            state="Valid state",
            question="Valid question",
            options=["Only one option"],
        )

    with pytest.raises(ValueError, match="state must be a non-empty string"):
        model.decide(
            state="",
            question="Valid question",
            options=["Option 1", "Option 2"],
        )

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        model.decide(
            state="Valid state",
            question="   ",
            options=["Option 1", "Option 2"],
        )

    with pytest.raises(ValueError, match="Option at index 1 must be a non-empty string"):
        model.decide(
            state="Valid state",
            question="Valid question",
            options=["Option 1", ""],
        )


def test_duplicate_options_produce_identical_scores():
    """
    Verifies documented semantics: mathematically independent branches produce
    identical representations and identical scores for duplicate option strings.
    """
    head = DirectComparativeHead(hidden_size=2560, proj_dim=256, mode="absolute_only")
    head.eval()

    # Create representation where option 0 and option 2 are identical
    rep_a = torch.randn(2560)
    rep_b = torch.randn(2560)
    reps = torch.stack([rep_a, rep_b, rep_a], dim=0)

    with torch.no_grad():
        scores = head(reps)

    assert torch.isclose(scores[0], scores[2], atol=1e-6)
