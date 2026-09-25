"""
Research Reproduction Tests for InvariantOne v1.

Verifies that the packaged InvariantOne runtime reproduces the exact research
outputs archived from the FINAL-HOLDOUT-V2 one-shot evaluation within expected
numerical precision tolerances.

This is a BLOCKING release test.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from invariantone.model import InvariantOne

# Reference items and predictions archived from the frozen FINAL-HOLDOUT-V2
# one-shot evaluation benchmark.
FROZEN_BENCHMARK_ITEMS = [
    {
        "id": "real_0",
        "state": "Why was my card payment reversed?",
        "question": "Determine the primary intent and routing category for this customer inquiry.",
        "options": [
            "reverted_card_payment?: Customer inquiry regarding reverted card payment?",
            "wrong_exchange_rate_for_cash_withdrawal: Customer inquiry regarding wrong exchange rate for cash withdrawal",
            "top_up_failed: Customer inquiry regarding top up failed",
            "activate_my_card: Customer inquiry regarding activate my card",
        ],
        "expected_choice_index": 0,
        "expected_scores": [3.8532, -5.9885, -6.0173, -5.9907],
        "expected_probs": [0.9998, 0.0001, 0.0001, 0.0001],
    },
    {
        "id": "real_1",
        "state": "How do I reverse a transaction on my account?",
        "question": "Determine the primary intent and routing category for this customer inquiry.",
        "options": [
            "change_pin: Customer inquiry regarding change pin",
            "request_refund: Customer inquiry regarding request refund",
            "transfer_not_received_by_recipient: Customer inquiry regarding transfer not received by recipient",
            "topping_up_by_card: Customer inquiry regarding topping up by card",
        ],
        "expected_choice_index": 2,
        "expected_scores": [-3.5006, -3.1231, -2.568, -6.4575],
        "expected_probs": [0.1989, 0.2892, 0.5013, 0.0106],
    },
]


@pytest.fixture(scope="module")
def loaded_model():
    """Loads InvariantOne once for reproduction test suite."""
    device = "cpu"
    dtype = "float32"
    return InvariantOne.from_pretrained(
        pretrained_model_name_or_path="checkpoints/invariantone-v1",
        device=device,
        dtype=dtype,
        strict=True,
    )


def test_research_reproduction_predictions_and_scores(loaded_model):
    """
    Verifies that the packaged runtime exactly reproduces:
    1. Selected option (argmax choice index)
    2. Raw comparative logits (within BF16 tolerance: atol=0.05)
    3. Calibrated probabilities (within atol=0.02)
    """
    for item in FROZEN_BENCHMARK_ITEMS:
        result = loaded_model.decide(
            state=item["state"],
            question=item["question"],
            options=item["options"],
        )

        # 1. Exact choice index match
        assert result.choice_index == item["expected_choice_index"], (
            f"Choice index mismatch for {item['id']}: "
            f"expected {item['expected_choice_index']}, got {result.choice_index}"
        )

        # 2. Raw scores reproduction
        for i, (actual_s, exp_s) in enumerate(zip(result.scores, item["expected_scores"])):
            assert abs(actual_s - exp_s) < 0.05, (
                f"Score mismatch at option {i} for {item['id']}: "
                f"expected {exp_s}, got {actual_s}"
            )

        # 3. Calibrated probability reproduction
        for i, (actual_p, exp_p) in enumerate(zip(result.probabilities, item["expected_probs"])):
            assert abs(actual_p - exp_p) < 0.02, (
                f"Probability mismatch at option {i} for {item['id']}: "
                f"expected {exp_p}, got {actual_p}"
            )
